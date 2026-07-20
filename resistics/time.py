"""
Classes and methods for storing and manipulating time data, including:

- The TimeMetadata model for defining metadata for TimeData
- The TimeData class for storing TimeData
- MTH5 ingestion into channel-labelled TimeData
- TimeData processors
"""

import types
from collections.abc import Callable
from typing import Annotated, Any, ClassVar, Literal, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
from loguru import logger
from pydantic import ConfigDict, Field, PositiveFloat, ValidationInfo, field_validator

from resistics.common import (
    History,
    Metadata,
    Record,
    ResisticsData,
    ResisticsProcess,
    WriteableMetadata,
    get_chan_type,
)
from resistics.errors import ProcessRunError
from resistics.sampling import (
    DateTimeLike,
    HighResDateTime,
    RSDateTime,
    RSTimeDelta,
    to_datetime,
)


class ChanMetadata(Metadata):
    """Channel metadata.

    Attributes
    ----------
    model_config : ConfigDict
        Pydantic assignment-validation configuration.
    name : str
        Channel component name.
    data_files : list[str] | None
        Legacy source filenames, absent for MTH5-backed channels.
    chan_type : str
        Electric, magnetic, auxiliary, or another MTH5 channel type.
    chan_source : str | None
        Component name in the source dataset.
    sensor : str
        Sensor type.
    serial : str
        Sensor serial identifier.
    gain1 : float
        Primary channel gain.
    gain2 : float
        Secondary channel gain.
    scaling : float
        Scaling applied to channel samples.
    chopper : bool
        Whether sensor chopper mode was enabled.
    dipole_dist : float
        Electric dipole length.
    sensor_calibration_file : str
        Explicit sensor calibration filename.
    instrument_calibration_file : str
        Explicit instrument calibration filename.
    mth5_metadata : dict[str, Any]
        MTH5 channel metadata preserved at ingestion.
    """

    model_config = ConfigDict(validate_assignment=True)

    name: str
    data_files: list[str] | None = None
    chan_type: str = ""
    chan_source: str | None = None
    sensor: str = ""
    serial: str = ""
    gain1: float = 1
    gain2: float = 1
    scaling: float = 1
    chopper: bool = False
    dipole_dist: float = 1
    sensor_calibration_file: str = ""
    instrument_calibration_file: str = ""
    mth5_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("data_files", mode="before")
    @classmethod
    def validate_data_files(cls, value: Any) -> list[str]:
        """Validate data files and convert to list if required"""
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("chan_type")
    @classmethod
    def validate_chan_type(cls, value: str, info: ValidationInfo) -> str:
        """Validate the channel type"""
        if value:
            return value
        try:
            return get_chan_type(info.data["name"])
        except ValueError:
            raise ValueError(
                f"Failed setting type for chan {info.data['name']}"
            ) from None

    def electric(self) -> bool:
        """True if the channel is an electric channel"""
        return self.chan_type == "electric"

    def magnetic(self) -> bool:
        """True if the channel is a magnetic channel"""
        return self.chan_type == "magnetic"


class TimeMetadata(WriteableMetadata):
    """Metadata for one MTH5-backed time-series run.

    Attributes
    ----------
    model_config : ConfigDict
        Pydantic assignment-validation configuration.
    fs : float
        Sampling frequency in hertz.
    chans : list[str]
        Ordered channel component names.
    n_chans : int
        Number of channels.
    n_samples : int
        Number of samples per channel.
    first_time : HighResDateTime
        Time of the first sample.
    last_time : HighResDateTime
        Time of the last sample.
    system : str
        Data-logger type.
    serial : str
        Data-logger serial identifier.
    wgs84_latitude : float
        Station latitude in WGS84.
    wgs84_longitude : float
        Station longitude in WGS84.
    easting : float
        Station easting in local coordinates.
    northing : float
        Station northing in local coordinates.
    elevation : float
        Station elevation.
    chans_metadata : dict[str, ChanMetadata]
        Metadata indexed by channel component.
    history : History
        Processing history.
    survey : str
        MTH5 survey identifier.
    station : str
        MTH5 station identifier.
    run : str
        MTH5 run identifier.
    mth5_metadata : dict[str, Any]
        MTH5 survey, station, and run metadata preserved at ingestion.
    """

    model_config = ConfigDict(validate_assignment=True)

    fs: float
    chans: list[str]
    n_chans: int = 0
    n_samples: int
    first_time: HighResDateTime
    last_time: HighResDateTime
    system: str = ""
    serial: str = ""
    wgs84_latitude: float = -999.0
    wgs84_longitude: float = -999.0
    easting: float = -999.0
    northing: float = -999.0
    elevation: float = -999.0
    chans_metadata: dict[str, ChanMetadata]
    history: History = History()
    survey: str = ""
    station: str = ""
    run: str = ""
    mth5_metadata: dict[str, Any] = Field(default_factory=dict)

    def __getitem__(self, chan: str) -> ChanMetadata:
        """
        Get channel metadata

        Parameters
        ----------
        chan : str
            The channel to get metadata for

        Returns
        -------
        ChanMetadata
            Metadata for the channel

        Examples
        --------
        >>> from resistics.testing import time_metadata_2chan
        >>> metadata = time_metadata_2chan()
        >>> chan_metadata = metadata["chan1"]
        >>> chan_metadata.summary()
        {
            'name': 'chan1',
            'data_files': ['example1.ascii'],
            'chan_type': 'electric',
            'chan_source': None,
            'sensor': '',
            'serial': '',
            'gain1': 1.0,
            'gain2': 1.0,
            'scaling': 1.0,
            'chopper': False,
            'dipole_dist': 1.0,
            'sensor_calibration_file': '',
            'instrument_calibration_file': '',
            'mth5_metadata': {}
        }
        """
        from resistics.common import check_chan

        check_chan(chan, self.chans)
        return self.chans_metadata[chan]

    @property
    def dt(self) -> float:
        """Get the sampling frequency"""
        return 1 / self.fs

    @property
    def duration(self) -> RSTimeDelta:
        """Get the duration of the recording"""
        return self.last_time - self.first_time

    @property
    def nyquist(self) -> float:
        """Get the nyquist frequency"""
        return self.fs / 2

    def get_chan_types(self) -> list[str]:
        """
        Get all the different channel types

        Returns
        -------
        List[str]
            A list of different channel types

        Examples
        --------
        >>> from resistics.testing import time_metadata_mt
        >>> metadata = time_metadata_mt()
        >>> metadata.get_chan_types()
        ['electric', 'magnetic']
        """
        chan_types = {x.chan_type for x in self.chans_metadata.values()}
        return sorted(chan_types)

    def get_chans_with_type(self, chan_type: str) -> list[str]:
        """
        Get channels with the given type

        Parameters
        ----------
        chan_type : str
            The channel type

        Returns
        -------
        List[str]
            A list of channels with the given channel type

        Examples
        --------
        >>> from resistics.testing import time_metadata_mt
        >>> metadata = time_metadata_mt()
        >>> metadata.get_chans_with_type("magnetic")
        ['Hx', 'Hy']
        """
        return [x for x in self.chans if self.chans_metadata[x].chan_type == chan_type]

    def get_electric_chans(self) -> list[str]:
        """
        Get list of electric channels

        Returns
        -------
        List[str]
            List of electric channels

        Examples
        --------
        >>> from resistics.testing import time_metadata_mt
        >>> metadata = time_metadata_mt()
        >>> metadata.get_electric_chans()
        ['Ex', 'Ey']
        """
        return [x for x in self.chans if self.chans_metadata[x].electric()]

    def get_magnetic_chans(self) -> list[str]:
        """
        Get list of magnetic channels

        Returns
        -------
        List[str]
            List of magnetic channels

        Examples
        --------
        >>> from resistics.testing import time_metadata_mt
        >>> metadata = time_metadata_mt()
        >>> metadata.get_magnetic_chans()
        ['Hx', 'Hy']
        """
        return [x for x in self.chans if self.chans_metadata[x].magnetic()]

    def any_electric(self) -> bool:
        """True if any channels are electric"""
        return len(self.get_electric_chans()) != 0

    def any_magnetic(self) -> bool:
        """True if any channels are magnetic"""
        return len(self.get_magnetic_chans()) != 0


def get_time_metadata(
    time_dict: dict[str, Any], chans_dict: dict[str, dict[str, Any]]
) -> TimeMetadata:
    """
    Get metadata for TimeData

    The time and channel dictionaries must have the TimeMetadata required
    fields. For more information about the required fields, see
    :class:`~TimeMetadata`

    Parameters
    ----------
    time_dict : Dict[str, Any]
        Dictionary with metadata for the whole dataset
    chans_dict : Dict[str, Dict[str, Any]]
        Dictionary of dictionaries with metadata for each channel

    Returns
    -------
    TimeMetadata
        Metadata for TimeData

    See Also
    --------
    TimeMetadata : The TimeMetadata class which is returned

    Examples
    --------
    >>> from resistics.time import get_time_metadata
    >>> time_dict = {
    ...     "fs": 10,
    ...     "n_samples": 100,
    ...     "chans": ["Ex", "Hy"],
    ...     "n_chans": 2,
    ...     "first_time": "2021-01-01 00:00:00",
    ...     "last_time": "2021-01-01 00:01:00"
    ... }
    >>> chans_dict = {
    ...     "Ex": {"name": "Ex", "data_files": "example.ascii"},
    ...     "Hy": {"name": "Hy", "data_files": "example2.ascii", "sensor": "MFS"}
    ... }
    >>> metadata = get_time_metadata(time_dict, chans_dict)
    >>> metadata.summary()
    {
        'file_info': None,
        'fs': 10.0,
        'chans': ['Ex', 'Hy'],
        'n_chans': 2,
        'n_samples': 100,
        'first_time': '2021-01-01 00:00:00.000000_000000_000000_000000',
        'last_time': '2021-01-01 00:01:00.000000_000000_000000_000000',
        'system': '',
        'serial': '',
        'wgs84_latitude': -999.0,
        'wgs84_longitude': -999.0,
        'easting': -999.0,
        'northing': -999.0,
        'elevation': -999.0,
        'chans_metadata': {
            'Ex': {
                'name': 'Ex',
                'data_files': ['example.ascii'],
                'chan_type': 'electric',
                'chan_source': None,
                'sensor': '',
                'serial': '',
                'gain1': 1.0,
                'gain2': 1.0,
                'scaling': 1.0,
                'chopper': False,
                'dipole_dist': 1.0,
                'sensor_calibration_file': '',
                'instrument_calibration_file': '',
                'mth5_metadata': {}
            },
            'Hy': {
                'name': 'Hy',
                'data_files': ['example2.ascii'],
                'chan_type': 'magnetic',
                'chan_source': None,
                'sensor': 'MFS',
                'serial': '',
                'gain1': 1.0,
                'gain2': 1.0,
                'scaling': 1.0,
                'chopper': False,
                'dipole_dist': 1.0,
                'sensor_calibration_file': '',
                'instrument_calibration_file': '',
                'mth5_metadata': {}
            }
        },
        'history': {'records': []},
        'survey': '',
        'station': '',
        'run': '',
        'mth5_metadata': {}
    }
    """
    chans = time_dict["chans"]
    chans_metadata = {chan: ChanMetadata(**chans_dict[chan]) for chan in chans}
    time_dict["chans_metadata"] = chans_metadata
    return TimeMetadata(**time_dict)


def adjust_time_metadata(
    metadata: TimeMetadata,
    fs: float,
    first_time: RSDateTime,
    n_samples: int,
) -> TimeMetadata:
    """
    Adjust time data metadata

    This is required if changes have been made to the sampling frequency, the
    time of the first sample of the number of samples. This might occur in
    processes such as resampling or decimating.

    .. warning::

        The metadata passed in will be changed in place. If the original
        metadata should be retained, pass through a deepcopy

    Parameters
    ----------
    metadata : TimeMetadata
        Metadata to adjust
    fs : float
        The sampling frequency
    first_time : RSDateTime
        The first time of the data
    n_samples : int
        The number of samples

    Returns
    -------
    TimeMetadata
        Adjusted metadata

    Examples
    --------
    >>> from resistics.sampling import to_datetime
    >>> from resistics.time import adjust_time_metadata
    >>> from resistics.testing import time_metadata_2chan
    >>> metadata = time_metadata_2chan(fs=10, first_time="2021-01-01 00:00:00", n_samples=101)
    >>> metadata.fs
    10.0
    >>> metadata.n_samples
    101
    >>> metadata.first_time
    attotime.objects.attodatetime(2021, 1, 1, 0, 0, 0, 0, 0)
    >>> metadata.last_time
    attotime.objects.attodatetime(2021, 1, 1, 0, 0, 10, 0, 0)
    >>> metadata = adjust_time_metadata(metadata, 20, to_datetime("2021-03-01 00:01:00"), 50)
    >>> metadata.fs
    20.0
    >>> metadata.n_samples
    50
    >>> metadata.first_time
    attotime.objects.attodatetime(2021, 3, 1, 0, 1, 0, 0, 0)
    >>> metadata.last_time
    attotime.objects.attodatetime(2021, 3, 1, 0, 1, 2, 450000, 0)
    """
    from resistics.sampling import to_timedelta

    metadata.fs = fs
    metadata.first_time = cast(HighResDateTime, first_time)
    metadata.n_samples = n_samples
    duration = to_timedelta(1 / fs) * (n_samples - 1)
    metadata.last_time = cast(HighResDateTime, first_time + duration)
    return metadata


class TimeData(ResisticsData):
    """
    Class for holding time data

    Samples are stored internally as an xarray object labelled by channel and
    time. The ``data`` property exposes its mutable NumPy array with shape:

    n_chans x n_samples

    Parameters
    ----------
    metadata : TimeMetadata
        Metadata for the TimeData
    data : np.ndarray | xr.DataArray | xr.Dataset
        Channel-labelled data or a NumPy array in channel/sample order.

    Examples
    --------
    >>> import numpy as np
    >>> from resistics.testing import time_metadata_2chan
    >>> from resistics.time import TimeData
    >>> data = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]]
    >>> time_data = TimeData(time_metadata_2chan(), np.array(data))
    >>> time_data.metadata.chans
    ['chan1', 'chan2']
    >>> time_data.get_chan("chan1")
    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
    >>> time_data["chan1"]
    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
    """

    def __init__(
        self,
        metadata: TimeMetadata,
        data: np.ndarray | xr.DataArray | xr.Dataset,
    ) -> None:
        """Initialise time data.

        Raises
        ------
        ValueError
            If channels are duplicated or the labelled data shape is invalid.
        """
        self.metadata = metadata
        if len(set(metadata.chans)) != len(metadata.chans):
            raise ValueError("TimeData channel names must be unique")
        if isinstance(data, xr.Dataset):
            missing = set(metadata.chans).difference(data.data_vars)
            if missing:
                raise ValueError(f"MTH5 dataset is missing channels {sorted(missing)}")
            array = data[metadata.chans].to_array(dim="channel")
            if "time" not in array.dims:
                raise ValueError("MTH5 dataset must have a time dimension")
            array = array.transpose("channel", "time")
        elif isinstance(data, xr.DataArray):
            if set(data.dims) != {"channel", "time"}:
                raise ValueError("TimeData arrays require channel and time dimensions")
            array = data.transpose("channel", "time").sel(channel=metadata.chans)
        else:
            values = np.asarray(data)
            timestamps = pd.date_range(
                start=pd.Timestamp(metadata.first_time.isoformat()),
                periods=metadata.n_samples,
                freq=pd.Timedelta(1 / metadata.fs, unit="s"),
            )
            array = xr.DataArray(
                values,
                dims=("channel", "time"),
                coords={"channel": metadata.chans, "time": timestamps},
            )
        expected_shape = (metadata.n_chans, metadata.n_samples)
        if array.shape != expected_shape:
            raise ValueError(
                f"TimeData shape {array.shape} != expected {expected_shape}"
            )
        self._array = array.assign_coords(channel=metadata.chans)
        self._chan_to_idx = {chan: idx for idx, chan in enumerate(metadata.chans)}
        logger.debug(f"Creating TimeData with data type {self.data.dtype}")

    @property
    def data(self) -> np.ndarray:
        """Return the mutable NumPy sample array in channel/time order."""
        return np.asarray(self._array.data)

    @property
    def dataset(self) -> xr.Dataset:
        """Return samples as an MTH5-compatible channel-variable dataset."""
        return self._array.to_dataset(dim="channel")

    def to_xarray(self) -> xr.DataArray:
        """Return the labelled channel/time sample array.

        Returns
        -------
        xr.DataArray
            Samples labelled by channel and time.
        """
        return self._array

    def __getitem__(self, chan: str) -> np.ndarray:
        """
        Get channel time data

        Parameters
        ----------
        chan : str
            The channel to get data for

        Returns
        -------
        np.ndarray
            pandas Series with channel data and datetime index
        """
        return self.get_chan(chan)

    def __setitem__(self, chan: str, chan_data: np.ndarray) -> None:
        """
        Set channel time data

        Parameters
        ----------
        chan : str
            The channel to set the data for
        chan_data : np.ndarray
            The new channel data
        """
        self.set_chan(chan, chan_data)

    def get_chan_index(self, chan: str) -> int:
        """
        Get the channel index in the data

        Parameters
        ----------
        chan : str
            The channel

        Returns
        -------
        int
            The index
        """
        return self._chan_to_idx[chan]

    def get_chan(self, chan: str) -> np.ndarray:
        """
        Get the time data for a channel

        Parameters
        ----------
        chan : str
            The channel for which to get the time data

        Returns
        -------
        np.ndarray
            pandas Series with channel data and datetime index
        """
        from resistics.common import check_chan

        check_chan(chan, self.metadata.chans)
        return np.asarray(self._array.sel(channel=chan).data)

    def set_chan(self, chan: str, chan_data: np.ndarray) -> None:
        """
        Set channel time data

        Parameters
        ----------
        chan : str
            The channel to set the data for
        chan_data : np.ndarray
            The new channel data

        Raises
        ------
        ValueError
            If the data has incorrect size
        ValueError
            If the data has incorrect dtype
        """
        from resistics.common import check_chan

        check_chan(chan, self.metadata.chans)
        if chan_data.size != self.metadata.n_samples:
            raise ValueError(
                f"Size {chan_data.size} != num samples {self.metadata.n_samples}"
            )
        if chan_data.dtype != self.data.dtype:
            raise ValueError(f"dtype {chan_data.dtype} != existing {self.data.dtype}")
        self._array.loc[{"channel": chan}] = chan_data

    def get_timestamps(
        self, samples: np.ndarray | None = None, estimate: bool = True
    ) -> np.ndarray | pd.DatetimeIndex:
        """
        Get an array of timestamps

        Parameters
        ----------
        samples : Optional[np.ndarray], optional
            If provided, timestamps are only returned for the specified samples,
            by default None
        estimate : bool, optional
            Flag for using estimates instead of high precision datetimes, by
            default True

        Returns
        -------
        Union[np.ndarray, pd.DatetimeIndex]
            The return dates. This will be a numpy array of RSDateTime objects
            if estimate is False, else it will be a pandas DatetimeIndex
        """
        from resistics.sampling import datetime_array, datetime_array_estimate

        fnc = datetime_array_estimate if estimate else datetime_array
        if samples is not None:
            return fnc(self.metadata.first_time, self.metadata.fs, samples=samples)
        return fnc(
            self.metadata.first_time,
            self.metadata.fs,
            n_samples=self.metadata.n_samples,
        )

    def subsection(self, from_time: DateTimeLike, to_time: DateTimeLike) -> "TimeData":
        """
        Get a subsection of the TimeData

        Returns a new TimeData object

        Parameters
        ----------
        from_time : DateTimeLike
            Start of subsection
        to_time : DateTimeLike
            End of subsection

        Returns
        -------
        TimeData
            Subsection as new TimeData
        """
        sub = Subsection(from_time=from_time, to_time=to_time)
        return sub.run(self)

    def subsamples(
        self, from_sample: int | None = None, to_sample: int | None = None
    ) -> "TimeData":
        """Get a subsample range of the TimeData

        Returns a new TimeData object

        Parameters
        ----------
        from_sample : Optional[int], optional
            The sample to go from, by default None. If not provided, this will
            be the first sample.
        to_sample : Optional[int], optional
            The sample to end at, by default None. If not provided, this will be
            the last sample

        Returns
        -------
        TimeData
            The subsamples as a new TimeData object
        """
        sub = Subsamples(from_sample=from_sample, to_sample=to_sample)
        return sub.run(self)

    def copy(self) -> "TimeData":
        """Get a deepcopy of the time data object"""
        return TimeData(
            self.metadata.model_copy(deep=True), self._array.copy(deep=True)
        )

    def plot(
        self,
        fig: go.Figure | None = None,
        chans: list[str] | None = None,
        color: str = "blue",
        legend: str = "TimeData",
        max_pts: int | None = 10_000,
    ) -> go.Figure:
        """
        Plot time series data

        Parameters
        ----------
        fig : Optional[go.Figure], optional
            A figure if appending the data to an existing plot, by default None
        chans : Optional[List[str]], optional
            Explicit definition of channels to plot, by default None
        color : str, optional
            The color for the data, by default "blue"
        legend : str, optional
            The legend group to use, by default "TimeData". This is more useful
            when plotting multiple TimeData
        max_pts : Optional[int], optional
            The maximum number of points for any channel plot before applying
            LTTB downsampling, by default 10_000. If set to None, no
            downsampling will be applied.

        Returns
        -------
        go.Figure
            Plotly Figure

        Raises
        ------
        ValueError
            If a figure is provided and channels have not been explicitly
            defined
        """
        from resistics.plot import apply_lttb, get_time_fig

        if fig is not None and chans is None:
            raise ValueError("If a figure is provided, then chans must be specified")
        if chans is None:
            chans = self.metadata.chans
        if fig is None:
            y_labels = {x: self.metadata.chans_metadata[x].chan_type for x in chans}
            fig = get_time_fig(chans, y_labels)

        # plot the data
        logger.info(f"Plotting time data, chans {chans}")
        for idx, chan in enumerate(chans):
            if chan not in self.metadata.chans:
                continue
            indices, data = apply_lttb(self.get_chan(chan), max_pts)
            timestamps = self.get_timestamps(samples=indices, estimate=True)
            scatter = go.Scattergl(
                x=timestamps,
                y=data,
                line={"color": color},
                name=legend,
                legendgroup=legend,
                showlegend=(idx == 0),
            )
            fig.add_trace(scatter, row=idx + 1, col=1)
        logger.info("Plotting completed")
        return fig

    def to_string(self) -> str:
        """Class details as a string"""
        outstr = f"{self.type_to_string()}\n"
        outstr += self.metadata.to_string()
        return outstr


class MTH5TimeReader(ResisticsProcess):
    """Read a selected MTH5 run group into channel-labelled :class:`TimeData`."""

    output_type: ClassVar[str] = "time_data"
    runtime_requirements: ClassVar[list[str]] = ["project", "run_batch"]

    def execute(self, inputs: dict[str, Any], context: Any) -> TimeData:
        """Read the concrete run selected by the current flow batch."""
        del inputs
        batch = context["run_batch"]
        return context["project"].read_run(
            batch["survey"],
            batch["station"],
            batch["run"],
            chans=context.get("channels"),
            from_time=context.get("from_time"),
            to_time=context.get("to_time"),
        )

    def run(
        self,
        run_group: Any,
        chans: list[str] | None = None,
        from_time: DateTimeLike | None = None,
        to_time: DateTimeLike | None = None,
        from_sample: int | None = None,
        to_sample: int | None = None,
        sample_rate: float | None = None,
    ) -> TimeData:
        """Read one bounded MTH5 run."""
        start = None if from_time is None else pd.Timestamp(from_time).isoformat()
        end = None if to_time is None else pd.Timestamp(to_time).isoformat()
        n_samples = None
        if from_sample is not None or to_sample is not None:
            if sample_rate is None:
                raise ValueError("sample_rate is required for MTH5 sample bounds")
            first = 0 if from_sample is None else from_sample
            full_data = _mth5_run_ts_to_time_data(run_group.to_runts(), chans=chans)
            start = str(
                pd.Timestamp(full_data.metadata.first_time.isoformat())
                + pd.to_timedelta(first / sample_rate, unit="s")
            )
            if to_sample is not None:
                n_samples = max(0, to_sample - first + 1)
        run_ts = run_group.to_runts(start=start, end=end, n_samples=n_samples)
        return _mth5_run_ts_to_time_data(run_ts, chans=chans)


def _mth5_metadata_dict(value: Any) -> dict[str, Any]:
    """Convert MTH5 metadata to a serialisable dictionary.

    Parameters
    ----------
    value : Any
        MTH5 metadata object or dictionary.

    Returns
    -------
    dict[str, Any]
        Single-level metadata dictionary.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        return value.to_dict(single=True)
    except TypeError:
        return value.to_dict()


def _nested_mth5_value(value: Any, *names: str, default: Any = "") -> Any:
    """Read a nested MTH5 metadata value without concrete-type coupling.

    Parameters
    ----------
    value : Any
        Root metadata object.
    *names : str
        Attribute path to follow.
    default : Any
        Value returned when the path is absent.

    Returns
    -------
    Any
        Nested value or ``default``.
    """
    for name in names:
        value = getattr(value, name, None)
        if value is None:
            return default
    return value


def _mth5_identifier(value: Any) -> str:
    """Return a normalised MTH5 metadata identifier.

    Parameters
    ----------
    value : Any
        MTH5 metadata object.

    Returns
    -------
    str
        Identifier, or an empty string when absent.
    """
    result = getattr(value, "id", "")
    return "" if result is None else str(result)


def _mth5_run_ts_to_time_data(run_ts: Any, chans: list[str] | None = None) -> TimeData:
    """Convert an MTH5 RunTS-like object to channel-labelled resistics data."""
    if hasattr(run_ts, "dataset"):
        dataset = run_ts.dataset
    elif hasattr(run_ts, "to_xarray"):
        dataset = run_ts.to_xarray()
    elif hasattr(run_ts, "to_dataset"):
        dataset = run_ts.to_dataset()
    else:
        raise NotImplementedError("Unable to convert MTH5 RunTS to a dataset")
    if chans is None:
        chans = list(getattr(dataset, "data_vars", []))
    else:
        dataset = dataset[chans]
    if not chans:
        raise ValueError("No channels found in MTH5 run")
    dataset = dataset[chans]
    if "time" not in dataset.coords or dataset.sizes.get("time", 0) == 0:
        raise ValueError("MTH5 run has no time samples")

    first_time = to_datetime(pd.Timestamp(dataset.coords["time"].values[0]))
    last_time = to_datetime(pd.Timestamp(dataset.coords["time"].values[-1]))
    fs = float(getattr(run_ts, "sample_rate", 0) or dataset.attrs.get("sample_rate", 0))
    if fs <= 0:
        raise ValueError("Unable to determine MTH5 run sample rate")

    run_metadata = getattr(run_ts, "run_metadata", None)
    station_metadata = getattr(run_ts, "station_metadata", None)
    survey_metadata = getattr(run_ts, "survey_metadata", None)
    channel_metadata = {
        str(getattr(value, "component", "")): value
        for value in getattr(run_metadata, "channels", [])
    }
    metadata = TimeMetadata(
        fs=fs,
        chans=chans,
        n_chans=len(chans),
        n_samples=dataset.sizes["time"],
        first_time=first_time,
        last_time=last_time,
        survey=_mth5_identifier(survey_metadata),
        station=_mth5_identifier(station_metadata),
        run=_mth5_identifier(run_metadata),
        system=str(_nested_mth5_value(run_metadata, "data_logger", "type")),
        serial=str(_nested_mth5_value(run_metadata, "data_logger", "id")),
        wgs84_latitude=float(
            _nested_mth5_value(station_metadata, "location", "latitude", default=-999)
        ),
        wgs84_longitude=float(
            _nested_mth5_value(station_metadata, "location", "longitude", default=-999)
        ),
        elevation=float(
            _nested_mth5_value(station_metadata, "location", "elevation", default=-999)
        ),
        mth5_metadata={
            "survey": _mth5_metadata_dict(survey_metadata),
            "station": _mth5_metadata_dict(station_metadata),
            "run": _mth5_metadata_dict(run_metadata),
        },
        chans_metadata={
            chan: ChanMetadata(
                name=chan,
                data_files=None,
                chan_source=chan,
                chan_type=str(
                    dataset[chan].attrs.get("type")
                    or getattr(channel_metadata.get(chan), "type", "unknown")
                ),
                sensor=str(
                    _nested_mth5_value(channel_metadata.get(chan), "sensor", "type")
                ),
                serial=str(
                    _nested_mth5_value(channel_metadata.get(chan), "sensor", "id")
                ),
                mth5_metadata={
                    **_mth5_metadata_dict(channel_metadata.get(chan)),
                    **dict(dataset[chan].attrs),
                },
            )
            for chan in chans
        },
    )
    return TimeData(metadata, dataset)


def new_time_data(
    time_data: TimeData,
    metadata: TimeMetadata | None = None,
    data: np.ndarray | None = None,
    record: Record | None = None,
) -> TimeData:
    """
    Get a new TimeData

    Values are taken from an existing TimeData where they are not explicitly
    specified. This is useful in a process where only some aspects of the
    TimeData have been changed

    Parameters
    ----------
    time_data : TimeData
        The existing TimeData
    metadata : Optional[TimeMetadata], optional
        A new TimeMetadata, by default None
    data : Optional[np.ndarray], optional
        New data, by default None
    record : Optional[Record], optional
        A new record to add, by default None

    Returns
    -------
    TimeData
        A new TimeData instance
    """
    if metadata is None:
        metadata = time_data.metadata.model_copy(deep=True)
    if data is None:
        data = np.array(time_data.data)
    if record is not None:
        metadata.history.add_record(record)
    return TimeData(metadata, data)


class TimeProcess(ResisticsProcess):
    """Base class for processes that transform time data."""

    input_types: ClassVar[dict[str, str]] = {"time_data": "time_data"}
    output_type: ClassVar[str] = "time_data"
    include_in_default_parameters: ClassVar[bool] = False

    def run(self, time_data: TimeData) -> TimeData:
        """Run the time processor"""
        raise NotImplementedError("Run not implemented in parent TimeProcess")


class Subsection(TimeProcess):
    """
    Get a subsection of time data

    Parameters
    ----------
    from_time : DateTimeLike
        Time to take subsection from
    to_time : DateTimeLike
        Time to take subsection to

    Examples
    --------
    .. plot::
        :width: 90%

        >>> import matplotlib.pyplot as plt
        >>> from resistics.testing import time_data_random
        >>> from resistics.time import Subsection
        >>> time_data = time_data_random(n_samples=1000)
        >>> print(time_data.metadata.first_time, time_data.metadata.last_time)
        2020-01-01 00:00:00 2020-01-01 00:01:39.9
        >>> process = Subsection(from_time="2020-01-01 00:00:25", to_time="2020-01-01 00:00:50.9")
        >>> subsection = process.run(time_data)
        >>> print(subsection.metadata.first_time, subsection.metadata.last_time)
        2020-01-01 00:00:25 2020-01-01 00:00:50.9
        >>> subsection.metadata.n_samples
        260
        >>> plt.plot(time_data.get_timestamps(), time_data["Ex"], label="full") # doctest: +SKIP
        >>> plt.plot(subsection.get_timestamps(), subsection["Ex"], label="sub") # doctest: +SKIP
        >>> plt.legend(loc=3) # doctest: +SKIP
        >>> plt.tight_layout() # doctest: +SKIP
        >>> plt.show() # doctest: +SKIP

    See Also
    --------
    Subsamples : Getting a section of data using samples rather than dates
    """

    from_time: DateTimeLike
    to_time: DateTimeLike

    def run(self, time_data: TimeData) -> TimeData:
        """
        Take a subsection from TimeData

        Parameters
        ----------
        time_data : TimeData
            TimeData to take subsection from

        Returns
        -------
        TimeData
            Subsection TimeData
        """
        from resistics.sampling import (
            datetimes_to_samples,
            samples_to_datetimes,
            to_datetime,
        )

        from_time = to_datetime(self.from_time)
        to_time = to_datetime(self.to_time)
        fs = time_data.metadata.fs
        first_time = time_data.metadata.first_time
        last_time = time_data.metadata.last_time
        logger.info(f"Data times {first_time!s} to {last_time!s}")
        logger.info(f"Taking subsection between {self.from_time} and {self.to_time}")
        # convert to samples
        from_sample, to_sample = datetimes_to_samples(
            fs, first_time, last_time, from_time, to_time
        )
        n_samples = to_sample - from_sample + 1
        # convert back to times as datetimes may not coincide with timestamps
        from_time, to_time = samples_to_datetimes(
            fs, first_time, from_sample, to_sample
        )
        messages = [f"Subsection from sample {from_sample} to {to_sample}"]
        messages.append(f"First time: {first_time!s} -> {from_time!s}")
        messages.append(f"Last time: {last_time!s} -> {to_time!s}")
        metadata = time_data.metadata.model_copy(deep=True)
        metadata = adjust_time_metadata(metadata, fs, from_time, n_samples=n_samples)
        data = np.array(time_data.data[:, from_sample : to_sample + 1])
        record = self._get_record(messages)
        return new_time_data(time_data, metadata=metadata, data=data, record=record)


class Subsamples(TimeProcess):
    """
    Get a subsamples of time data, an alternative to getting a subsection by
    times

    Parameters
    ----------
    from_sample : int
        Sample to begin from
    to_time : DateTimeLike
        Sample to end at

    Examples
    --------
    Taking subsample using positive sample numbers. Sample 0 is the first
    sample and sample -1 is the last sample.

    .. plot::
        :width: 90%

        >>> import matplotlib.pyplot as plt
        >>> from resistics.testing import time_data_random
        >>> from resistics.time import Subsamples
        >>> time_data = time_data_random(n_samples=300)
        >>> print(time_data.metadata.first_time, time_data.metadata.last_time)
        2020-01-01 00:00:00 2020-01-01 00:00:29.9
        >>> process = Subsamples(from_sample=10, to_sample=120)
        >>> subsample = process.run(time_data)
        >>> print(subsample.metadata.first_time, subsample.metadata.last_time)
        2020-01-01 00:00:01 2020-01-01 00:00:12
        >>> subsample.metadata.n_samples
        111
        >>> plt.plot(time_data.get_timestamps(), time_data["Ex"], label="full") # doctest: +SKIP
        >>> plt.plot(subsample.get_timestamps(), subsample["Ex"], label="sub") # doctest: +SKIP
        >>> plt.legend(loc=3) # doctest: +SKIP
        >>> plt.tight_layout() # doctest: +SKIP
        >>> plt.show() # doctest: +SKIP

    Another option is to use negative sample numbers which counts back from the
    end of the time data.

    .. plot::
        :width: 90%

        >>> import matplotlib.pyplot as plt
        >>> from resistics.testing import time_data_random
        >>> from resistics.time import Subsamples
        >>> time_data = time_data_random(n_samples=300)
        >>> print(time_data.metadata.first_time, time_data.metadata.last_time)
        2020-01-01 00:00:00 2020-01-01 00:00:29.9
        >>> process = Subsamples(from_sample=-100, to_sample=-50)
        >>> subsample = process.run(time_data)
        >>> print(subsample.metadata.first_time, subsample.metadata.last_time)
        2020-01-01 00:00:20 2020-01-01 00:00:25
        >>> subsample.metadata.n_samples
        51
        >>> plt.plot(time_data.get_timestamps(), time_data["Ex"], label="full") # doctest: +SKIP
        >>> plt.plot(subsample.get_timestamps(), subsample["Ex"], label="sub") # doctest: +SKIP
        >>> plt.legend(loc=3) # doctest: +SKIP
        >>> plt.tight_layout() # doctest: +SKIP
        >>> plt.show() # doctest: +SKIP

    If from_sample is not passed, it will be set to 0. If to_sample is not
    passed this will default to the last sample.

    .. plot::
        :width: 90%

        >>> import matplotlib.pyplot as plt
        >>> from resistics.testing import time_data_random
        >>> from resistics.time import Subsamples
        >>> time_data = time_data_random(n_samples=300)
        >>> print(time_data.metadata.first_time, time_data.metadata.last_time)
        2020-01-01 00:00:00 2020-01-01 00:00:29.9
        >>> process = Subsamples(to_sample=100)
        >>> subsample = process.run(time_data)
        >>> print(subsample.metadata.first_time, subsample.metadata.last_time)
        2020-01-01 00:00:00 2020-01-01 00:00:10
        >>> subsample.metadata.n_samples
        101
        >>> plt.plot(time_data.get_timestamps(), time_data["Ex"], label="full") # doctest: +SKIP
        >>> plt.plot(subsample.get_timestamps(), subsample["Ex"], label="sub") # doctest: +SKIP
        >>> plt.legend(loc=3) # doctest: +SKIP
        >>> plt.tight_layout() # doctest: +SKIP
        >>> plt.show() # doctest: +SKIP

    See Also
    --------
    Subsection : For taking a subsection using dates
    """

    from_sample: int | None = None
    to_sample: int | None = None

    def run(self, time_data: TimeData) -> TimeData:
        """
        Take a subsection from TimeData

        Parameters
        ----------
        time_data : TimeData
            TimeData to take subsection from

        Returns
        -------
        TimeData
            Subsection TimeData

        Raises
        ------
        ProcessRunError
            If from_sample is not less than to_sample
        ProcessRunError
            If from_sample is out of range
        ProcessRunError
            If to_sample is out of range
        """
        from resistics.sampling import check_sample, samples_to_datetimes

        # put in default values if either is None
        n_samples = time_data.metadata.n_samples
        from_sample = 0 if self.from_sample is None else self.from_sample
        to_sample = n_samples - 1 if self.to_sample is None else self.to_sample
        logger.info(f"From {from_sample} to {to_sample} from n_samples {n_samples}")
        # check negative sample values
        if from_sample < 0:
            from_sample = n_samples + from_sample
        if to_sample < 0:
            to_sample = n_samples + to_sample
        logger.info(f"Adjusted sample range {from_sample} to {to_sample}")
        # checks
        if from_sample >= to_sample:
            raise ProcessRunError(self.name, f"From {from_sample} not < to {to_sample}")
        if not check_sample(n_samples, from_sample):
            raise ProcessRunError(self.name, f"From sample {from_sample} out of range")
        if not check_sample(n_samples, to_sample):
            raise ProcessRunError(self.name, f"To sample {to_sample} out of range")
        # convert samples to datetimes
        n_subsamples = to_sample - from_sample + 1
        fs = time_data.metadata.fs
        first_time = time_data.metadata.first_time
        last_time = time_data.metadata.last_time
        from_time, to_time = samples_to_datetimes(
            time_data.metadata.fs, first_time, from_sample, to_sample
        )
        messages = [f"Taking subsample from {from_sample} to {to_sample}"]
        messages.append(f"First time: {first_time!s} -> {from_time!s}")
        messages.append(f"Last time: {last_time!s} -> {to_time!s}")
        metadata = time_data.metadata.model_copy(deep=True)
        metadata = adjust_time_metadata(metadata, fs, from_time, n_samples=n_subsamples)
        data = np.array(time_data.data[:, from_sample : to_sample + 1])
        record = self._get_record(messages)
        return new_time_data(time_data, metadata=metadata, data=data, record=record)


class InterpolateNans(TimeProcess):
    """
    Interpolate nan values in the data

    Preserve the data type of the input time data

    Examples
    --------
    >>> from resistics.testing import time_data_with_nans
    >>> from resistics.time import InterpolateNans
    >>> time_data = time_data_with_nans()
    >>> time_data["Hx"]
    array([nan,  2.,  3.,  5.,  1.,  2.,  3.,  4.,  2.,  6.,  7., nan, nan,
            4.,  3.,  2.], dtype=float32)
    >>> process = InterpolateNans()
    >>> time_data_new = process.run(time_data)
    >>> time_data_new["Hx"]
    array([2., 2., 3., 5., 1., 2., 3., 4., 2., 6., 7., 6., 5., 4., 3., 2.],
          dtype=float32)
    """

    include_in_default_parameters: ClassVar[bool] = True

    def run(self, time_data: TimeData) -> TimeData:
        """
        Interpolate nan values

        Parameters
        ----------
        time_data : TimeData
            TimeData to remove nan values from

        Returns
        -------
        TimeData
            TimeData with no nan values
        """
        logger.info(f"Removing nan values from channels {time_data.metadata.chans}")
        messages = []
        data = np.array(time_data.data)
        for chan in time_data.metadata.chans:
            idx = time_data.get_chan_index(chan)
            data[idx, :] = self._interpolate_nans(data[idx, :])
            messages.append(f"nan values removed from {chan}")
        record = self._get_record(messages)
        return new_time_data(time_data, data=data, record=record)

    def _interpolate_nans(self, chan_data: np.ndarray) -> np.ndarray:
        """
        Remove nans from an array

        Parameters
        ----------
        chan_data : np.ndarray
            The array

        Returns
        -------
        np.ndarray
            Array with nans removed
        """
        nan_bool = np.isnan(chan_data)
        if not np.any(nan_bool):
            return chan_data
        mask = np.ones(chan_data.size, bool)
        mask[nan_bool] = 0
        x = np.arange(chan_data.size)
        chan_data[nan_bool] = np.interp(x[nan_bool], x[mask], chan_data[mask])
        return chan_data


class RemoveMean(TimeProcess):
    """
    Remove channel mean value from each channel

    Preserve the data type of the input time data

    Examples
    --------
    >>> import numpy as np
    >>> from resistics.testing import time_data_simple
    >>> from resistics.time import RemoveMean
    >>> time_data = time_data_simple()
    >>> process = RemoveMean()
    >>> time_data_new = process.run(time_data)
    >>> time_data_new["Hx"]
    array([-2.5, -1.5, -0.5,  1.5, -2.5, -1.5, -0.5,  0.5, -1.5,  2.5,  3.5,
            2.5,  1.5,  0.5, -0.5, -1.5], dtype=float32)
    >>> hx_test = time_data["Hx"] - np.mean(time_data["Hx"])
    >>> hx_test
    array([-2.5, -1.5, -0.5,  1.5, -2.5, -1.5, -0.5,  0.5, -1.5,  2.5,  3.5,
            2.5,  1.5,  0.5, -0.5, -1.5], dtype=float32)
    >>> bool(np.all(hx_test == time_data_new["Hx"]))
    True
    """

    include_in_default_parameters: ClassVar[bool] = True

    def run(self, time_data: TimeData) -> TimeData:
        """
        Remove mean from TimeData

        Parameters
        ----------
        time_data : TimeData
            TimeData input

        Returns
        -------
        TimeData
            TimeData with mean removed
        """
        from resistics.common import array_to_string

        logger.info(f"Removing mean from channels {time_data.metadata.chans}")
        mean = np.mean(time_data.data, axis=1)
        data = time_data.data - mean[:, None]
        means_str = array_to_string(mean, precision=3)
        messages = [f"Removed means {means_str} for chans {time_data.metadata.chans}"]
        record = self._get_record(messages)
        return new_time_data(time_data, data=data, record=record)


class Add(TimeProcess):
    """
    Add values to channels

    Add can be used to add a constant value to all channels or values for
    specific channels can be provided.

    Add preserves the data type of the original data

    Parameters
    ----------
    add : Union[float, Dict[str, float]]
        Either a scalar to add to all channels or dictionary with values to
        add to each channel

    Examples
    --------
    Using a constant value for all channels passed as a scalar

    >>> from resistics.testing import time_data_ones
    >>> from resistics.time import Add
    >>> time_data = time_data_ones()
    >>> process = Add(add=5)
    >>> time_data_new = process.run(time_data)
    >>> time_data_new["Ex"] - time_data["Ex"]
    array([5., 5., 5., 5., 5., 5., 5., 5., 5., 5.], dtype=float32)
    >>> time_data_new["Ey"] - time_data["Ey"]
    array([5., 5., 5., 5., 5., 5., 5., 5., 5., 5.], dtype=float32)

    Variable values for the channels provided as a dictionary

    >>> time_data = time_data_ones()
    >>> process = Add(add={"Ex": 3, "Hy": -7})
    >>> time_data_new = process.run(time_data)
    >>> time_data_new["Ex"] - time_data["Ex"]
    array([3., 3., 3., 3., 3., 3., 3., 3., 3., 3.], dtype=float32)
    >>> time_data_new["Hy"] - time_data["Hy"]
    array([-7., -7., -7., -7., -7., -7., -7., -7., -7., -7.], dtype=float32)
    >>> time_data_new["Ey"] - time_data["Ey"]
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)
    """

    add: float | dict[str, float]

    def run(self, time_data: TimeData) -> TimeData:
        """
        Add values to the data

        Parameters
        ----------
        time_data : TimeData
            The input TimeData

        Returns
        -------
        TimeData
            TimeData with values added
        """
        add = self._get_add(time_data)
        logger.info(f"Added {add} to channels {time_data.metadata.chans}")
        data = time_data.data + add[:, None]
        messages = [f"Added {add} to channels {time_data.metadata.chans}"]
        record = self._get_record(messages)
        return new_time_data(time_data, data=data, record=record)

    def _get_add(self, time_data: TimeData) -> np.ndarray:
        """Make an array to add to the data"""
        add = np.zeros(shape=(time_data.metadata.n_chans), dtype=time_data.data.dtype)
        if isinstance(self.add, (float, int)):
            return add + self.add
        for chan in time_data.metadata.chans:
            if chan in self.add:
                idx = time_data.get_chan_index(chan)
                add[idx] = self.add[chan]
        return add


class Multiply(TimeProcess):
    """
    Multiply channels by values

    Multiply can be used to add a constant value to all channels or values for
    specific channels can be provided.

    Multiply preseves the original type of the time data

    Parameters
    ----------
    multiplier : Union[Dict[str, float], float]
        Either a float to multiply all channels with the same value or a
        dictionary to specify different values for each channel

    Examples
    --------
    Using a constant value for all channels passed as a scalar

    >>> from resistics.testing import time_data_ones
    >>> from resistics.time import Multiply
    >>> time_data = time_data_ones()
    >>> process = Multiply(multiplier=5)
    >>> time_data_new = process.run(time_data)
    >>> time_data_new["Ex"]/time_data["Ex"]
    array([5., 5., 5., 5., 5., 5., 5., 5., 5., 5.], dtype=float32)
    >>> time_data_new["Ey"]/time_data["Ey"]
    array([5., 5., 5., 5., 5., 5., 5., 5., 5., 5.], dtype=float32)

    Variable values for the channels provided as a dictionary

    >>> time_data = time_data_ones()
    >>> process = Multiply(multiplier={"Ex": 3, "Hy": -7})
    >>> time_data_new = process.run(time_data)
    >>> time_data_new["Ex"]/time_data["Ex"]
    array([3., 3., 3., 3., 3., 3., 3., 3., 3., 3.], dtype=float32)
    >>> time_data_new["Hy"]/time_data["Hy"]
    array([-7., -7., -7., -7., -7., -7., -7., -7., -7., -7.], dtype=float32)
    >>> time_data_new["Ey"]/time_data["Ey"]
    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], dtype=float32)
    """

    multiplier: float | dict[str, float]

    def run(self, time_data: TimeData) -> TimeData:
        """
        Multiply the channels

        Parameters
        ----------
        time_data : TimeData
            Input TimeData

        Returns
        -------
        TimeData
            TimeData with channels multiplied by the specified numbers
        """
        mult = self._get_mult(time_data)
        logger.info(f"Multipying channels {time_data.metadata.chans} by {mult}")
        data = time_data.data * mult[:, None]
        messages = [f"Multiplied channels {time_data.metadata.chans} by {mult}"]
        record = self._get_record(messages)
        return new_time_data(time_data, data=data, record=record)

    def _get_mult(self, time_data: TimeData) -> np.ndarray:
        """Make an array to multiply the data with"""
        mult = np.ones(shape=(time_data.metadata.n_chans), dtype=time_data.data.dtype)
        if isinstance(self.multiplier, (float, int)):
            return mult * self.multiplier
        for chan in time_data.metadata.chans:
            if chan in self.multiplier:
                idx = time_data.get_chan_index(chan)
                mult[idx] = self.multiplier[chan]
        return mult


class LowPass(TimeProcess):
    """
    Apply low pass filter

    Parameters
    ----------
    cutoff : float
        The cutoff for the low pass
    order : int, optional
        Order of the filter, by default 10

    Examples
    --------
    Low pass to remove 20 Hz from a time series sampled at 50 Hz

    .. plot::
        :width: 90%

        import matplotlib.pyplot as plt
        from resistics.testing import time_data_periodic
        from resistics.time import LowPass
        time_data = time_data_periodic([10, 50], fs=250, n_samples=100)
        process = LowPass(cutoff=30)
        filtered = process.run(time_data)
        plt.plot(time_data.get_timestamps(), time_data["chan1"], label="original")
        plt.plot(filtered.get_timestamps(), filtered["chan1"], label="filtered")
        plt.legend(loc=3)
        plt.tight_layout()
        plt.plot()
    """

    cutoff: float
    order: int = 10

    def run(self, time_data: TimeData) -> TimeData:
        """
        Apply the low pass filter

        Parameters
        ----------
        time_data : TimeData
            The input TimeData

        Returns
        -------
        TimeData
            The low pass filtered TimeData

        Raises
        ------
        ProcessRunError
            If cutoff > nyquist
        """
        from scipy.signal import butter, sosfiltfilt

        nyquist = time_data.metadata.nyquist
        if self.cutoff > nyquist:
            raise ProcessRunError(
                self.name, f"Cutoff {self.cutoff} > nyquist {nyquist}"
            )

        logger.info(f"Low pass filtering with cutoff {self.cutoff} Hz")
        normed = self.cutoff / nyquist
        sos = butter(self.order, normed, btype="lowpass", analog=False, output="sos")
        data = sosfiltfilt(sos, time_data.data, axis=1).astype(time_data.data.dtype)
        messages = [f"Low pass filtered data with cutoff {self.cutoff} Hz"]
        record = self._get_record(messages)
        return new_time_data(time_data, data=data, record=record)


class HighPass(TimeProcess):
    """
    High pass filter time data

    Parameters
    ----------
    cutoff : float
        Cutoff for the high pass filter
    order : int, optional
        Order of the filter, by default 10

    Examples
    --------
    High pass to remove 3 Hz from signal sampled at 50 Hz

    .. plot::
        :width: 90%

        import matplotlib.pyplot as plt
        from resistics.testing import time_data_periodic
        from resistics.time import HighPass
        time_data = time_data_periodic([10, 50], fs=250, n_samples=100)
        process = HighPass(cutoff=30)
        filtered = process.run(time_data)
        plt.plot(time_data.get_timestamps(), time_data["chan1"], label="original")
        plt.plot(filtered.get_timestamps(), filtered["chan1"], label="filtered")
        plt.legend(loc=3)
        plt.tight_layout()
        plt.plot()
    """

    cutoff: float
    order: int = 10

    def run(self, time_data: TimeData) -> TimeData:
        """
        Apply the high pass filter

        Parameters
        ----------
        time_data : TimeData
            The input TimeData

        Returns
        -------
        TimeData
            The high pass filtered TimeData

        Raises
        ------
        ProcessRunError
            If cutoff > nyquist
        """
        from scipy.signal import butter, sosfiltfilt

        nyquist = time_data.metadata.nyquist
        if self.cutoff > nyquist:
            raise ProcessRunError(
                self.name, f"Cutoff {self.cutoff} > nyquist {nyquist}"
            )

        logger.info(f"High pass filtering with cutoff {self.cutoff} Hz")
        normed = self.cutoff / nyquist
        sos = butter(self.order, normed, btype="highpass", analog=False, output="sos")
        data = sosfiltfilt(sos, time_data.data, axis=1).astype(time_data.data.dtype)
        messages = [f"High pass filtered data with cutoff {self.cutoff} Hz"]
        record = self._get_record(messages)
        return new_time_data(time_data, data=data, record=record)


class BandPass(TimeProcess):
    """
    Band pass filter time data

    Parameters
    ----------
    cutoff_low : float
        The low cutoff for the band pass filter
    cutoff_high : float
        The high cutoff for the band pass filter
    order : int, optional
        The order of the filter, by default 10

    Examples
    --------
    Band pass to isolate 12 Hz signal

    .. plot::
        :width: 90%

        import matplotlib.pyplot as plt
        from resistics.testing import time_data_periodic
        from resistics.time import BandPass
        time_data = time_data_periodic([10, 50], fs=250, n_samples=100)
        process = BandPass(cutoff_low=45, cutoff_high=55)
        filtered = process.run(time_data)
        plt.plot(time_data.get_timestamps(), time_data["chan1"], label="original")
        plt.plot(filtered.get_timestamps(), filtered["chan1"], label="filtered")
        plt.legend(loc=3)
        plt.tight_layout()
        plt.plot()
    """

    cutoff_low: float
    cutoff_high: float
    order: int = 10

    def run(self, time_data: TimeData) -> TimeData:
        """
        Apply the band pass filter

        Parameters
        ----------
        time_data : TimeData
            The input TimeData

        Returns
        -------
        TimeData
            The band pass filtered TimeData

        Raises
        ------
        ProcessRunError
            If cutoff_low > cutoff_high
        ProcessRunError
            If cutoff_high > nyquist
        """
        from scipy.signal import butter, sosfiltfilt

        nyquist = time_data.metadata.nyquist
        if self.cutoff_low > self.cutoff_high:
            raise ProcessRunError(
                self.name, f"Cutoff low {self.cutoff_low} > high {self.cutoff_high}"
            )
        if self.cutoff_high > nyquist:
            raise ProcessRunError(
                self.name, f"Cutoff high {self.cutoff_high} > nyquist {nyquist}"
            )

        logger.info(f"Band pass between {self.cutoff_low} and {self.cutoff_high} Hz")
        low = self.cutoff_low / nyquist
        high = self.cutoff_high / nyquist
        sos = butter(
            self.order, (low, high), btype="bandpass", analog=False, output="sos"
        )
        data = sosfiltfilt(sos, time_data.data, axis=1).astype(time_data.data.dtype)
        messages = [f"Band pass with cutoffs {self.cutoff_low},{self.cutoff_high} Hz"]
        record = self._get_record(messages)
        return new_time_data(time_data, data=data, record=record)


class Notch(TimeProcess):
    """
    Notch filter time data

    Parameters
    ----------
    notch : float
        The frequency to notch
    band : Optional[float], optional
        The bandwidth of the filter, by default None
    order : int, optional
        The order of the filter, by default 10

    Examples
    --------
    Notch to remove a 50 Hz signal, for example powerline noise

    .. plot::
        :width: 90%

        import matplotlib.pyplot as plt
        from resistics.testing import time_data_periodic
        from resistics.time import Notch
        time_data = time_data_periodic([10, 50], fs=250, n_samples=100)
        process = Notch(notch=50, band=10)
        filtered = process.run(time_data)
        plt.plot(time_data.get_timestamps(), time_data["chan1"], label="original")
        plt.plot(filtered.get_timestamps(), filtered["chan1"], label="filtered")
        plt.legend(loc=3)
        plt.tight_layout()
        plt.plot()
    """

    notch: float
    band: float | None = None
    order: int = 10

    def run(self, time_data: TimeData) -> TimeData:
        """
        Apply notch filter to TimeData

        Parameters
        ----------
        time_data : TimeData
            Input TimeData

        Returns
        -------
        TimeData
            Filtered TimeData

        Raises
        ------
        ProcessRunError
            If notch frequency > nyquist
        """
        from scipy.signal import butter, sosfiltfilt

        nyquist = time_data.metadata.nyquist
        if self.notch > nyquist:
            raise ProcessRunError(
                self.name, f"Notch frequency {self.notch} > nyquist {nyquist}"
            )

        band = 0.1 * time_data.metadata.fs if self.band is None else self.band
        low = (self.notch - (band / 2)) / nyquist
        high = (self.notch + (band / 2)) / nyquist
        if high > nyquist:
            logger.debug("Band high {high} > nyquist {nyquist}. Setting to nyquist")
            high = nyquist
        logger.info(f"Notch filtering at {self.notch} Hz with bandwidth {band} Hz")
        sos = butter(
            self.order, (low, high), btype="bandstop", analog=False, output="sos"
        )
        data = sosfiltfilt(sos, time_data.data, axis=1).astype(time_data.data.dtype)
        messages = [f"Notch filtered at {self.notch} Hz with bandwidth {band} Hz"]
        record = self._get_record(messages)
        return new_time_data(time_data, data=data, record=record)


class Resample(TimeProcess):
    """
    Resample TimeData

    Note that resampling is done on np.float64 data and this will lead to a
    temporary increase in memory usage. Once resampling is complete, the data is
    converted back to its original data type.

    Parameters
    ----------
    new_fs : int
        The new sampling frequency

    Examples
    --------
    Resample the data from 250 Hz to 50 Hz

    .. plot::
        :width: 90%

        >>> import matplotlib.pyplot as plt
        >>> from resistics.testing import time_data_periodic
        >>> from resistics.time import Resample
        >>> time_data = time_data_periodic([10, 50], fs=250, n_samples=200)
        >>> print(time_data.metadata.n_samples, time_data.metadata.first_time, time_data.metadata.last_time)
        200 2020-01-01 00:00:00 2020-01-01 00:00:00.796
        >>> process = Resample(new_fs=50)
        >>> resampled = process.run(time_data)
        >>> print(resampled.metadata.n_samples, resampled.metadata.first_time, resampled.metadata.last_time)
        40 2020-01-01 00:00:00 2020-01-01 00:00:00.78
        >>> plt.plot(time_data.get_timestamps(), time_data["chan1"], label="original") # doctest: +SKIP
        >>> plt.plot(resampled.get_timestamps(), resampled["chan1"], label="resampled") # doctest: +SKIP
        >>> plt.legend(loc=3) # doctest: +SKIP
        >>> plt.tight_layout() # doctest: +SKIP
        >>> plt.show() # doctest: +SKIP
    """

    new_fs: float

    def run(self, time_data: TimeData) -> TimeData:
        r"""
        Resample TimeData

        Resampling uses the polyphase method which does not assume periodicity
        Calculate the upsample rate and the downsampling rate and using
        polyphase filtering, the final sample rate is:

        .. math::

            (up / down) * original sample rate

        Therefore, to get a sampling frequency of resampFreq, want:

        .. math::

            (resampFreq / sampleFreq) * sampleFreq

        Use the fractions library to get up and down as integers which they are
        required to be.

        Parameters
        ----------
        time_data : TimeData
            Input TimeData

        Returns
        -------
        TimeData
            Resampled TimeData
        """
        from fractions import Fraction

        from scipy.signal import resample_poly

        fs = time_data.metadata.fs
        logger.info(f"Resampling data from {fs} Hz to {self.new_fs} Hz")
        # get the resampling fraction in its simplest form and resample
        frac = Fraction(self.new_fs / fs).limit_denominator()
        data = resample_poly(
            time_data.data.astype(np.float64),
            frac.numerator,
            frac.denominator,
            axis=1,
            padtype="mean",
        )
        data = data.astype(time_data.data.dtype)
        # adjust metadata
        n_samples = data.shape[1]
        metadata = time_data.metadata.model_copy(deep=True)
        metadata = adjust_time_metadata(
            metadata, self.new_fs, time_data.metadata.first_time, n_samples=n_samples
        )
        messages = [f"Resampled data from {fs} Hz to {self.new_fs} Hz"]
        messages.append(f"Resampled first time {metadata.first_time!s}")
        messages.append(f"Resampled last time {metadata.last_time!s}")
        record = self._get_record(messages)
        return new_time_data(time_data, metadata=metadata, data=data, record=record)


class Decimate(TimeProcess):
    """
    Decimate TimeData

    .. warning::

        Data is converted to np.float64 prior to decimation. This is going to
        cause a temporary increase in memory usage, but decimating np.float64
        delivers improved results.

        The decimated data is converted back to its original data type prior
        to being returned.

        The max_factor for a single decimation step is by default set as 3.
        When using np.float64 data, it is possible to use a larger decimation
        factor, up to 13, but this does again have an impact on results.

        For more information, see
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.decimate.html

    Parameters
    ----------
    factor : int
        The decimation factor

    Examples
    --------
    .. plot::
        :width: 90%

        >>> import matplotlib.pyplot as plt
        >>> from resistics.testing import time_data_periodic
        >>> from resistics.time import Decimate
        >>> time_data = time_data_periodic([10, 50], fs=250, n_samples=200)
        >>> print(time_data.metadata.n_samples, time_data.metadata.first_time, time_data.metadata.last_time)
        200 2020-01-01 00:00:00 2020-01-01 00:00:00.796
        >>> process = Decimate(factor=5)
        >>> decimated = process.run(time_data)
        >>> print(decimated.metadata.n_samples, decimated.metadata.first_time, decimated.metadata.last_time)
        40 2020-01-01 00:00:00 2020-01-01 00:00:00.78
        >>> plt.plot(time_data.get_timestamps(), time_data["chan1"], label="original") # doctest: +SKIP
        >>> plt.plot(decimated.get_timestamps(), decimated["chan1"], label="decimated") # doctest: +SKIP
        >>> plt.legend(loc=3) # doctest: +SKIP
        >>> plt.tight_layout() # doctest: +SKIP
        >>> plt.show() # doctest: +SKIP
    """

    factor: Annotated[int, Field(ge=1)]
    max_single_factor: Annotated[int, Field(ge=2)] = 3

    def run(self, time_data: TimeData) -> TimeData:
        """
        Decimate TimeData

        Parameters
        ----------
        time_data : TimeData
            Input TimeData

        Returns
        -------
        TimeData
            Decimated TimeData
        """
        from scipy.signal import decimate

        factors = self._get_downsample_factors(self.factor)
        n_steps = len(factors)
        logger.info(
            f"Decimating by {self.factor} in {n_steps} step(s) with factors {factors}"
        )
        messages = [
            f"Decimating by {self.factor} in {n_steps} step(s) with factors {factors}"
        ]
        # convert to float64 for decimation to avoid significant numerical issues
        data = time_data.data.astype(np.float64)
        for factor in factors:
            if factor == 1:
                continue
            data = decimate(data, factor, axis=1, zero_phase=True)
            messages.append(f"Data decimated by factor of {factor}")
        data = data.astype(time_data.data.dtype)
        # return new TimeData
        fs = time_data.metadata.fs
        new_fs = fs / self.factor
        messages.append(f"Sampling frequency adjusted from {fs} to {new_fs}")
        n_samples = data.shape[1]
        metadata = time_data.metadata.model_copy(deep=True)
        metadata = adjust_time_metadata(
            metadata, new_fs, time_data.metadata.first_time, n_samples=n_samples
        )
        record = self._get_record(messages)
        return new_time_data(time_data, metadata=metadata, data=data, record=record)

    def _get_downsample_factors(self, downsample_factor: int) -> list[int]:
        """Factorise a number to avoid too large a downsample factor

        Logic:

        - Perform a prime factorisation to get prime factors
        - Now want to combine factors to reduce the number of calls
        - Each single downsample factor must be <= self.max_single_factor

        Parameters
        ----------
        downsample_factor : int
            The number to factorise

        Returns
        -------
        List[int]
            The downsampling factors to use

        Notes
        -----
        There's a few pathological cases here that are being ignored. For
        example, what if the downsample factor is the product of two large
        primes.

        Examples
        --------
        A low value example

        >>> from resistics.time import Decimate
        >>> process = Decimate(factor=24)
        >>> process._get_downsample_factors(process.factor)
        [2, 2, 2, 3]
        >>> process._prime_factorisation(process.factor)
        [2, 2, 2, 3]

        An example with a higher value and a different maximum factor for any
        single decimation step

        >>> process = Decimate(factor=96, max_single_factor=13)
        >>> process._get_downsample_factors(process.factor)
        [8, 12]
        >>> process._prime_factorisation(process.factor)
        [2, 2, 2, 2, 2, 3]
        """
        if downsample_factor <= self.max_single_factor:
            return [downsample_factor]

        factors = self._prime_factorisation(downsample_factor)
        downsamples = []
        val = 1
        for factor in factors:
            if val * factor > self.max_single_factor:
                downsamples.append(val)
                val = 1
            val *= factor
        downsamples.append(val)
        return downsamples

    def _prime_factorisation(self, n: int) -> list[int]:
        """
        Factorise an integer into primes

        Parameters
        ----------
        n : int
            The integer to factorise

        Returns
        -------
        List[int]
            List of factors
        """
        import math

        prime_list = []
        # turn n into odd number
        while (n % 2) == 0:
            prime_list.append(2)
            n = n // 2
        if n == 1:
            return prime_list
        # odd divisors
        for ii in range(3, int(math.sqrt(n)) + 1, 2):
            while (n % ii) == 0:
                prime_list.append(ii)
                n = n // ii
        if n > 2:
            prime_list.append(n)
        return prime_list


class ShiftTimestamps(TimeProcess):
    """
    Shift timestamps. This method is usually used when there is an offset on the
    sampling, so that instead of coinciding with a second or an hour, they are
    offset from this.

    The function interpolates the original data onto the shifted timestamps.

    Parameters
    ----------
    shift : float
        The shift in seconds. This must be positive as data is never
        extrapolated

    Examples
    --------
    An example shifting timestamps for TimeData with a sample period of 20
    seconds (fs = 1/20 = 0.05 Hz) but with an offset of 10 seconds on the
    timestamps

    .. plot::
        :width: 90%

        >>> from resistics.testing import time_data_with_offset
        >>> from resistics.time import ShiftTimestamps
        >>> time_data = time_data_with_offset(offset=10, fs=1/20, n_samples=5)
        >>> [x.time().strftime('%H:%M:%S') for x in time_data.get_timestamps()]
        ['00:00:10', '00:00:30', '00:00:50', '00:01:10', '00:01:30']
        >>> process = ShiftTimestamps(shift=10, style="linear")
        >>> result = process.run(time_data)
        >>> [x.time().strftime('%H:%M:%S') for x in result.get_timestamps()]
        ['00:00:20', '00:00:40', '00:01:00', '00:01:20']
        >>> plt.plot(time_data.get_timestamps(), time_data["chan1"], "bo", label="original") # doctest: +SKIP
        >>> plt.plot(result.get_timestamps(), result["chan1"], "rd", label="shifted") # doctest: +SKIP
        >>> plt.legend(loc=4) # doctest: +SKIP
        >>> plt.grid() # doctest: +SKIP
        >>> plt.tight_layout() # doctest: +SKIP
        >>> plt.show() # doctest: +SKIP

    See Also
    --------
    CropTimestamps : Crop timestamps to a specified time unit
    """

    shift: PositiveFloat
    style: Literal["spline", "linear"] = "spline"

    def run(self, time_data: TimeData) -> TimeData:
        """
        Shift timestamps and interpolate data

        Parameters
        ----------
        time_data : TimeData
            Input TimeData

        Returns
        -------
        TimeData
            TimeData with shifted timestamps and data interpolated

        Raises
        ------
        ProcessRunError
            If the shift is greater than the sampling frequency. This method is
            not supposed to be used for resampling, but simply for removing an
            offset from timestamps
        """
        from resistics.sampling import to_timedelta

        metadata = time_data.metadata
        if self.shift >= metadata.dt:
            raise ProcessRunError(
                self.name, f"Shift {self.shift} not < sample period {metadata.dt}"
            )
        # calculate properties of shifted data
        norm_shift = self.shift / metadata.dt
        n_samples = metadata.n_samples - 1
        delta = to_timedelta(self.shift)
        first_time = metadata.first_time + delta
        last_time = metadata.last_time - to_timedelta(1 / metadata.fs) + delta

        logger.info(f"Data covers {metadata.first_time!s} to {metadata.last_time!s}")
        logger.info(f"New data covers {first_time!s} to {last_time!s}")
        messages = [f"First time: {metadata.first_time!s} -> {first_time!s}"]
        messages.append(f"Last time: {metadata.last_time!s} -> {last_time!s}")
        # shift data
        x = np.arange(0, metadata.n_samples, dtype=np.float64)
        x_shift = np.arange(0, n_samples, dtype=np.float64) + norm_shift
        fnc_map = {
            "linear": self._linear_interpolate,
            "spline": self._spline_interpolate,
        }
        data = fnc_map[self.style](x, x_shift, time_data)

        # update metadata
        metadata = time_data.metadata.model_copy(deep=True)
        metadata = adjust_time_metadata(
            metadata, metadata.fs, first_time, n_samples=n_samples
        )
        record = self._get_record(messages)
        return new_time_data(time_data, metadata=metadata, data=data, record=record)

    def _spline_interpolate(
        self, x: np.ndarray, x_shift: np.ndarray, time_data: TimeData
    ) -> np.ndarray:
        """
        Perform a spline interpolation

        Parameters
        ----------
        x : np.ndarray
            The sample numbers of the current timestamps.
        x_shift : np.ndarray
            The sample number of the requested timestamps.
        time_data : TimeData
            The time data

        Returns
        -------
        np.ndarray
            The values of the data at the new timestamps calculated via a spline
            interpolation
        """
        from scipy.interpolate import splev, splrep

        logger.info("Interpolating new values using spline interpoation")
        data = np.empty(
            shape=(time_data.metadata.n_chans, x_shift.size), dtype=time_data.data.dtype
        )
        for idx in range(time_data.metadata.n_chans):
            spl = splrep(x, time_data.data[idx], s=0)
            data[idx] = splev(x_shift, spl, der=0)
        return data

    def _linear_interpolate(
        self, x: np.ndarray, x_shift: np.ndarray, time_data: TimeData
    ) -> np.ndarray:
        """
        Perform a linear interpolation

        Parameters
        ----------
        x : np.ndarray
            The sample numbers of the current timestamps.
        x_shift : np.ndarray
            The sample number of the requested timestamps.
        time_data : TimeData
            The time data

        Returns
        -------
        np.ndarray
            The values of the data at the new timestamps calculated via a linear
            interpolation
        """
        from scipy.interpolate import interp1d

        logger.info("Interpolating new values using linear interpoation")
        interp_fnc = interp1d(x, time_data.data, axis=1, kind="linear", copy=False)
        return interp_fnc(x_shift)


class CropTimestamps(TimeProcess):
    """
    Crop timestamps to make them begin at the next second or minute or hour
    and end one sample before the nearest second, minute or hour.

    Input timestamps should be sampled coincidentally with the time
    unit. This can be achieved with :class:`~ShiftTimestamps`.

    Cropping works as follows:

    - First times are cropped to the next time unit or maintained if they are a
      whole time unit already
    - Last times are cropped to the previous time unit or maintained if they are
      a whole time unit already

    Parameters
    ----------
    time_unit : str
        The time unit to crop to given in pandas style time freq

    Examples
    --------
    An example cropping timestamps to the nearest minute.

    .. plot::
        :width: 90%

        >>> from resistics.testing import time_data_random
        >>> from resistics.time import CropTimestamps
        >>> time_data = time_data_random(n_samples=1000)
        >>> print(time_data.metadata.first_time, time_data.metadata.last_time)
        2020-01-01 00:00:00 2020-01-01 00:01:39.9
        >>> process = CropTimestamps(time_unit="T")
        >>> result = process.run(time_data)
        >>> print(result.metadata.first_time, result.metadata.last_time)
        2020-01-01 00:00:00 2020-01-01 00:01:00
        >>> plt.plot(time_data.get_timestamps(), time_data["Ex"], "bo-", label="original") # doctest: +SKIP
        >>> plt.plot(result.get_timestamps(), result["Ex"], "rd-", label="cropped") # doctest: +SKIP
        >>> plt.legend(loc=4) # doctest: +SKIP
        >>> plt.grid() # doctest: +SKIP
        >>> plt.tight_layout() # doctest: +SKIP
        >>> plt.show() # doctest: +SKIP

    See Also
    --------
    ShiftTimestamps : Calculate data values on shifted timestamps
    """

    time_unit: str = "s"

    def run(self, time_data: TimeData) -> TimeData:
        """Crop timestamps to the next time unit

        Parameters
        ----------
        time_data : TimeData
            The TimeData to crop

        Returns
        -------
        TimeData
            The cropped TimeData
        """
        from resistics.sampling import (
            datetimes_to_samples,
            samples_to_datetimes,
            to_datetime,
        )

        # get the from and to time based on the time unit
        fs = time_data.metadata.fs
        first_time = time_data.metadata.first_time
        last_time = time_data.metadata.last_time
        time_unit = "min" if self.time_unit == "T" else self.time_unit
        from_time = pd.Timestamp(first_time.isoformat()).ceil(time_unit)
        from_time = to_datetime(from_time)
        to_time = pd.Timestamp(last_time.isoformat()).floor(time_unit)
        to_time = to_datetime(to_time)
        # this is now essentially taking a subsection
        from_sample, to_sample = datetimes_to_samples(
            fs, first_time, last_time, from_time, to_time
        )
        n_samples = to_sample - from_sample + 1
        # convert back to times as datetimes may not coincide with timestamps
        from_time, to_time = samples_to_datetimes(
            fs, first_time, from_sample, to_sample
        )
        # output
        logger.info(f"Cropping timestamps to the '{self.time_unit}'")
        logger.debug(f"First timestamp {first_time!s} -> {from_time!s}")
        logger.debug(f"Last timestamp {last_time!s} -> {to_time!s}")
        messages = [f"Cropping timestamps to the {self.time_unit}"]
        messages.append(f"First time: {first_time!s} -> {from_time!s}")
        messages.append(f"Last time: {last_time!s} -> {to_time!s}")
        metadata = time_data.metadata.model_copy(deep=True)
        metadata = adjust_time_metadata(metadata, fs, from_time, n_samples=n_samples)
        data = np.array(time_data.data[:, from_sample : to_sample + 1])
        record = self._get_record(messages)
        return new_time_data(time_data, metadata=metadata, data=data, record=record)


def serialize_custom_fnc(fnc: Callable) -> str:
    """
    Serialize the custom functions

    This is not really reversible and recovering parameters from ApplyFunction
    is not supported

    Parameters
    ----------
    fnc : Callable
        Function to serialize

    Returns
    -------
    str
        serialized output
    """
    from resistics.common import array_to_string

    test = np.arange(3)
    input_vals = array_to_string(test, precision=2)
    output_vals = array_to_string(fnc(test), precision=2)
    return f"Custom function with result [{output_vals}] on [{input_vals}]"


class ApplyFunction(TimeProcess):
    """
    Apply a generic functions to the time data

    To be used with single argument functions that take the channel data array
    and a perform transformation on the data.

    Parameters
    ----------
    fncs : Dict[str, Callable]
        Dictionary of channel to callable

    Examples
    --------
    >>> import numpy as np
    >>> from resistics.testing import time_data_ones
    >>> from resistics.time import ApplyFunction
    >>> time_data = time_data_ones()
    >>> process = ApplyFunction(fncs={"Ex": lambda x: 2*x, "Hy": lambda x: 3*x*x - 5*x + 1})
    >>> result = process.run(time_data)
    >>> time_data["Ex"]
    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], dtype=float32)
    >>> result["Ex"]
    array([2., 2., 2., 2., 2., 2., 2., 2., 2., 2.])
    >>> time_data["Hy"]
    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], dtype=float32)
    >>> result["Hy"]
    array([-1., -1., -1., -1., -1., -1., -1., -1., -1., -1.])
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={
            types.LambdaType: serialize_custom_fnc,
            types.FunctionType: serialize_custom_fnc,
        },
    )

    fncs: dict[str, Callable]

    def run(self, time_data: TimeData) -> TimeData:
        """
        Apply functions to channel data

        Parameters
        ----------
        time_data : TimeData
            Input TimeData

        Returns
        -------
        TimeData
            Transformed TimeData
        """
        logger.info(f"Applying custom functions to channels {list(self.fncs.keys())}")
        messages = []
        data = np.empty(shape=time_data.data.shape)
        for chan, fnc in self.fncs.items():
            if chan in time_data.metadata.chans:
                messages.append(f"Applying custom function to {chan}")
                idx = time_data.get_chan_index(chan)
                data[idx] = fnc(time_data[chan])
        record = self._get_record(messages)
        return new_time_data(time_data, data=data, record=record)


# class Join(ResisticsProcess):
#     """
#     Join together time data

#     All time data passed must have the same sampling frequencies and only
#     channels that are common across all time data will be merged.

#     Gaps are filled with Nans. To fill in gaps, use InterpolateNans.

#     Examples
#     --------
#     In the below example, three time data instances are joined together. Note
#     that in the plot, offsets have been added to the data to make it easier to
#     visualise what is happening.

#     .. plot::
#         :width: 90%

#         >>> import matplotlib.pyplot as plt
#         >>> from resistics.testing import time_data_ones
#         >>> from resistics.time import Join, InterpolateNans
#         >>> time_data1 = time_data_ones(fs = 0.1, first_time = "2020-01-01 00:00:00", n_samples=6)
#         >>> time_data2 = time_data_ones(fs = 0.1, first_time = "2020-01-01 00:02:00", n_samples=6)
#         >>> time_data3 = time_data_ones(fs = 0.1, first_time = "2020-01-01 00:04:00", n_samples=6)
#         >>> joiner = Join()
#         >>> joined = joiner.run(time_data1, time_data2, time_data3)
#         >>> joined["Ex"]
#         array([ 1.,  1.,  1.,  1.,  1.,  1., nan, nan, nan, nan, nan, nan,  1.,
#                 1.,  1.,  1.,  1.,  1., nan, nan, nan, nan, nan, nan,  1.,  1.,
#                 1.,  1.,  1.,  1.])
#         >>> interpolater = InterpolateNans()
#         >>> interpolated = interpolater.run(joined)
#         >>> interpolated["Ex"]
#         array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#             1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
#         >>> fig = plt.figure()
#         >>> p = plt.plot(time_data1.get_x(), time_data1["Ex"], "bx-", lw=2)
#         >>> p = plt.plot(time_data2.get_x(), time_data2["Ex"] + 1, "co-", lw=2)
#         >>> p = plt.plot(time_data3.get_x(), time_data3["Ex"] + 2, "gd-", lw=2)
#         >>> p = plt.plot(joined.get_x(), joined["Ex"] + 3, "rs-", lw=2)
#         >>> p = plt.plot(interpolated.get_x(), interpolated["Ex"] + 4, "kp", lw=2)
#         >>> plt.yticks([1, 2, 3, 4, 5], ["Data1", "Data2", "Data3", "Joined", "Interpolated"]) # doctest: +SKIP
#         >>> plt.tight_layout() # doctest: +SKIP
#         >>> plt.show() # doctest: +SKIP
#     """

#     def run(self, *args: TimeData) -> TimeData:
#         """
#         Join TimeData

#         Pass TimeData as additional arguments. They will all be joined together
#         into a new TimeData object

#         Parameters
#         ----------
#         args : TimeData
#             TimeData to join, e.g. run(time_data1, time_data2, time_data3,...)

#         Returns
#         -------
#         TimeData
#             Joined TimeData
#         """
#         # from resistics.common import get_process_record, histories_to_parameters
#         from resistics.sampling import datetimes_to_samples, to_n_samples

#         fs = self._get_fs(args)
#         chans = self._get_chans(args)
#         logger.info(f"Joining data with sample frequency {fs} and channels {chans}")
#         first_times = [x.metadata.first_time for x in args]
#         last_times = [x.metadata.last_time for x in args]
#         first_time = min(first_times)
#         last_time = max(last_times)
#         n_samples = to_n_samples(last_time - first_time, fs)
#         data = np.empty(shape=(len(chans), n_samples))
#         data[:] = np.nan

#         # begin copying in the data
#         messages = [f"Joining data between {first_time}, {last_time}"]
#         messages.append(f"Sampling frequency {fs} Hz, channels {chans}")
#         for time_data in args:
#             first_time = time_data.metadata.first_time
#             last_time = time_data.metadata.last_time
#             from_sample, to_sample = datetimes_to_samples(
#                 fs, first_time, last_time, first_time, last_time
#             )
#             for idx, chan in enumerate(chans):
#                 data[idx, from_sample : to_sample + 1] = time_data[chan]
#         # process record
#         histories = [x.history for x in args]
#         parameters = histories_to_parameters(histories)
#         record = get_process_record(self.name, parameters, messages)
#         metadata = args[0].metadata.copy(deep=True)
#         metadata = adjust_time_metadata(metadata, fs, first_time, n_samples=n_samples)
#         return TimeData(metadata, data)

#     def _get_fs(self, time_data_col: Collection[TimeData]) -> float:
#         """
#         Get sampling frequency of all TimeData

#         Parameters
#         ----------
#         time_data_col : Collection[TimeData]
#             Collection of TimeData

#         Returns
#         -------
#         float
#             Sampling frequency in seconds

#         Raises
#         ------
#         ProcessRunError
#             If more than one unique sampling frequency found
#         """
#         from resistics.errors import ProcessRunError

#         fs = set([x.metadata.fs for x in time_data_col])
#         if len(fs) != 1:
#             raise ProcessRunError(self.name, f"> 1 sample frequency found {fs}")
#         fs = list(fs)[0]
#         return fs

#     def _get_chans(self, time_data_col: Collection[TimeData]) -> List[str]:
#         """
#         Find channels common across all TimeData

#         Parameters
#         ----------
#         time_data_col : Collection[TimeData]
#             Collection of TimeData

#         Returns
#         -------
#         List[str]
#             List of channels

#         Raises
#         ------
#         ProcessRunError
#             If no common channels are found
#         """
#         from resistics.errors import ProcessRunError

#         chans_list = [set(x.metadata.chans) for x in time_data_col]
#         chans = set.intersection(*chans_list)
#         if len(chans) == 0:
#             raise ProcessRunError(
#                 self.name, "Found no common channels amongst time data"
#             )
#         return sorted(list(chans))
