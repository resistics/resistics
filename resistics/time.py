"""
Classes and methods for storing and manipulating time data, including:

- The TimeMetadata model for defining metadata for TimeData
- The TimeData class for storing TimeData
- Implementations of time data readers for numpy and ascii formatted TimeData
- TimeData processors
"""
from loguru import logger
from typing import List, Dict, Union, Any, Tuple
from typing import Optional, Callable
import types
from pathlib import Path
from pydantic import validator, conint, PositiveFloat
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from resistics.errors import ProcessRunError
from resistics.common import ResisticsData, ResisticsProcess
from resistics.common import Metadata, WriteableMetadata
from resistics.common import History, Record, ResisticsWriter
from resistics.sampling import RSDateTime, RSTimeDelta, DateTimeLike
from resistics.sampling import HighResDateTime, datetime_to_string


class ChanMetadata(Metadata):
    """Channel metadata"""

    data_files: Optional[List[str]] = None
    """The data files"""
    sensor: str = ""
    """The name of the sensor"""
    serial: str = ""
    """The serial number of the sensor"""
    gain1: int = 1
    """Primary channel gain"""
    gain2: int = 1
    """Secondary channel gain"""
    scaling: float = 1
    """Scaling to apply to the data. May include the gains and other scaling"""
    hchopper: bool = False
    """Boolean flag for magnetic chopper on"""
    echopper: bool = False
    """Boolean flag for electric chopper on"""
    dx: float = 1
    """Dipole spacing x direction"""
    dy: float = 1
    """Dipole spacing y direction"""
    dz: float = 1
    """Dipole spacing z direction"""
    sensor_calibration_file: str = ""
    """Explicit name of sensor calibration file"""
    instrument_calibration_file: str = ""
    """Explicit name of instrument calibration file"""

    class Config:
        """pydantic configuration information"""

        validate_assignment = True

    @validator("data_files", pre=True)
    def validate_data_files(cls, value: Any) -> List[str]:
        """Validate data files and convert to list if required"""
        if isinstance(value, str):
            return [value]
        return value


class TimeMetadata(WriteableMetadata):
    """Time metadata"""

    fs: float
    """The sampling frequency"""
    chans: List[str]
    """List of channels"""
    n_chans: Optional[int] = None
    """The number of channels"""
    n_samples: int
    """The number of samples"""
    first_time: HighResDateTime
    """The datetime of the first sample"""
    last_time: HighResDateTime
    """The datetime of the last sample"""
    system: str = ""
    """The system used for recording"""
    serial: str = ""
    """Serial number of the system"""
    wgs84_latitude: float = -999.0
    """Latitude in WGS84"""
    wgs84_longitude: float = -999.0
    """Longitude in WGS84"""
    easting: float = -999.0
    """The easting of the site in local cartersian coordinates"""
    northing: float = -999.0
    """The northing of the site in local cartersian coordinates"""
    elevation: float = -999.0
    """The elevation of the site"""
    chans_metadata: Dict[str, ChanMetadata]
    """List of channel metadata"""
    history: History = History()
    """Processing history"""

    class Config:
        """pydantic configuration information"""

        json_encoders = {RSDateTime: datetime_to_string}
        validate_assignment = True

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
            'data_files': ['example1.ascii'],
            'sensor': '',
            'serial': '',
            'gain1': 1,
            'gain2': 1,
            'scaling': 1,
            'hchopper': False,
            'echopper': False,
            'dx': 1,
            'dy': 1,
            'dz': 1,
            'sensor_calibration_file': '',
            'instrument_calibration_file': ''
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


def get_time_metadata(
    time_dict: Dict[str, Any], chans_dict: Dict[str, Dict[str, Any]]
) -> TimeMetadata:
    """
    Get metadata for TimeData

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

    Examples
    --------
    >>> from resistics.time import get_time_metadata
    >>> time_dict = {"fs": 10, "n_samples": 100, "n_chans": 2, "first_time": "2021-01-01 00:00:00", "last_time": "2021-01-01 00:01:00"}
    >>> chans_dict = {"Ex": {"data_files": "example.ascii"}, "Hy": {"data_files": "example2.ascii", "sensor": "MFS"}}
    >>> metadata = get_time_metadata(time_dict, chans_dict)
    >>> metadata.fs
    10.0
    """
    if "chans" in time_dict:
        chans = time_dict["chans"]
    else:
        chans = sorted(list(chans_dict.keys()))
        time_dict["chans"] = chans
    chans_metadata = {chan: ChanMetadata(**meta) for chan, meta in chans_dict.items()}
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
    metadata.first_time = first_time
    metadata.n_samples = n_samples
    duration = to_timedelta(1 / fs) * (n_samples - 1)
    metadata.last_time = first_time + duration
    return metadata


class TimeData(ResisticsData):
    """
    Class for holding time data

    The data values are stored in an numpy array attribute named data. This has
    shape:

    n_chans x n_samples

    Parameters
    ----------
    metadata : TimeMetadata
        Metadata for the TimeData
    data : np.ndarray
        Numpy array of the data

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
        data: np.ndarray,
    ) -> None:
        """Initialise time data"""
        logger.debug(f"Creating TimeData with data type {data.dtype}")
        self.metadata = metadata
        self.data = data
        self._chan_to_idx: Dict[str, int] = {}
        for idx, chan in enumerate(self.metadata.chans):
            self._chan_to_idx[chan] = idx

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
        return self.data[self._chan_to_idx[chan]]

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
        self.data[self._chan_to_idx[chan]] = chan_data

    def get_timestamps(
        self, samples: Optional[np.ndarray] = None, estimate: bool = True
    ) -> Union[np.ndarray, pd.DatetimeIndex]:
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

    def copy(self) -> "TimeData":
        """Get a deepcopy of the time data object"""
        return TimeData(TimeMetadata(**self.metadata.dict()), np.array(self.data))

    def plot(
        self,
        fig: Optional[go.Figure] = None,
        subplots: Optional[List[str]] = None,
        subplot_columns: Optional[Dict[str, List[str]]] = None,
        max_pts: int = 10_000,
        label_prefix: str = "",
    ) -> go.Figure:
        """
        Plot time data

        Parameters
        ----------
        fig : go.Figure, optional
            Plotly figure, by default None. If no figure is provided, a new one
            will be created.
        subplots : Optional[List[str]], optional
            Subplots, by default None. To customise the plot, provide a list of
            subplots
        subplot_columns : Optional[Dict[str, List[str]]], optional
            Subplot columns defines which channels to plot for each subplot, by
            default None
        max_pts : int, optional
            Maximum number of points to plot, by default 10000. Data will be
            downsampled using lttb method.
        label_prefix : str, optional
            Prefix to add to legend labels, by default "".

        Returns
        -------
        go.Figure
            Plotly figure
        """
        from resistics.plot import figure_columns_as_lines, plot_columns_1d

        if subplots is None:
            subplots = self._get_subplots()
        if subplot_columns is None:
            subplot_columns = self._get_subplot_chans(subplots)
        if fig is None:
            y_labels = self._get_y_labels(subplot_columns)
            fig = figure_columns_as_lines(
                subplots=subplots, y_labels=y_labels, x_label="Datetime"
            )
        plot_columns_1d(
            fig,
            self,
            subplots,
            subplot_columns,
            max_pts=max_pts,
            label_prefix=label_prefix,
        )
        return fig

    def _get_subplots(self) -> List[str]:
        """Get list of subplots"""
        from resistics.common import any_electric, any_magnetic

        subplots = []
        if any_electric(self.metadata.chans):
            subplots.append("Electric")
        if any_magnetic(self.metadata.chans):
            subplots.append("Magnetic")
        return subplots

    def _get_subplot_chans(self, subplots: List[str]) -> Dict[str, List[str]]:
        """Get channels for each subplot"""
        from resistics.common import is_electric, is_magnetic

        subplot_columns = {}
        if "Electric" in subplots:
            subplot_columns["Electric"] = [
                x for x in self.metadata.chans if is_electric(x)
            ]
        if "Magnetic" in subplots:
            subplot_columns["Magnetic"] = [
                x for x in self.metadata.chans if is_magnetic(x)
            ]
        return subplot_columns

    def _get_y_labels(self, subplot_columns: Dict[str, List[str]]) -> Dict[str, str]:
        """Get subplot columns"""
        from resistics.common import is_electric, is_magnetic

        y_labels = {}
        for subplot, columns in subplot_columns.items():
            if len(columns) > 0 and is_electric(columns[0]):
                y_labels[subplot] = "mv/km"
            elif len(columns) > 0 and is_magnetic(columns[0]):
                y_labels[subplot] = "mV or nT"
            else:
                y_labels[subplot] = "Unknown"
        return y_labels

    def x_size(self) -> int:
        """
        For abstract plotting functions, return the size


        Returns
        -------
        int
            The x size, equal to the number of samples
        """
        return self.metadata.n_samples

    def get_x(self, samples: Optional[np.ndarray] = None) -> pd.DatetimeIndex:
        """
        For plotting, get x dimension, in this case times

        Parameters
        ----------
        samples : Union[np.ndarray, None], optional
            If provided, x values (timestamps) are only returned for the
            specified samples, by default None

        Returns
        -------
        pd.DatetimeIndex
            Timestamp array
        """
        return self.get_timestamps(samples=samples, estimate=True)

    def to_string(self) -> str:
        """Class details as a string"""
        outstr = f"{self.type_to_string()}\n"
        outstr += self.metadata.to_string()
        return outstr


class TimeReader(ResisticsProcess):

    apply_scalings: bool = True
    extension: Union[str, None] = None

    def run(
        self,
        dir_path: Path,
        metadata_only: Optional[bool] = False,
        metadata: Optional[TimeMetadata] = None,
        from_time: Optional[DateTimeLike] = None,
        to_time: Optional[DateTimeLike] = None,
        from_sample: Optional[int] = None,
        to_sample: Optional[int] = None,
    ) -> Union[TimeMetadata, TimeData]:
        """
        Read time series data

        Parameters
        ----------
        dir_path : Path
            The directory path
        metadata_only : Optional[bool], optional
            Read only the metadata, by default False
        metadata : Optional[TimeMetadata], optional
            Pass the metadata if its already been read in, by default None.
        from_time : Union[DateTimeLike, None], optional
            Timestamp to read from, by default None
        to_time : Union[DateTimeLike, None], optional
            Timestamp to read to, by default None
        from_sample : Union[int, None], optional
            Sample to read from, by default None
        to_sample : Union[int, None], optional
            Sample to read to, by default None

        Returns
        -------
        TimeData
            A TimeData instance
        """
        if metadata_only:
            logger.info(f"Reading time series metadata only from {dir_path}")
            return self.read_metadata(dir_path)
        logger.info(f"Reading time series data from {dir_path}")
        if metadata is None:
            logger.debug(f"Reading time series metadata from {dir_path}")
            metadata = self.read_metadata(dir_path)
        else:
            logger.debug("Using provided time series metadata")

        from_sample, to_sample = self._get_read_samples(
            metadata, from_time, to_time, from_sample, to_sample
        )
        logger.debug(f"Reading samples from {from_sample} to {to_sample}")
        time_data = self.read_data(dir_path, metadata, from_sample, to_sample)
        if self.apply_scalings:
            logger.debug("Applying scaling to time series data")
            return self.scale_data(time_data)
        return time_data

    def read_metadata(self, dir_path: Path) -> TimeMetadata:
        """
        Read time series data metadata

        Parameters
        ----------
        dir_path : Path
            The directory path of the time series data


        Raises
        ------
        NotImplementedError
            To be implemented in child classes
        """
        raise NotImplementedError(
            "read_metadata should be implemented in child classes"
        )

    def read_data(
        self, dir_path: Path, metadata: TimeMetadata, read_from: int, read_to: int
    ) -> TimeData:
        """
        Read raw data with minimal scalings applied

        Parameters
        ----------
        dir_path : path
            The directory path to read from
        metadata : TimeMetadata
            Time series data metadata
        read_from : int
            Sample to read data from
        read_to : int
            Sample to read data to

        Raises
        ------
        NotImplementedError
            To be implemented in child TimeReader classes
        """
        raise NotImplementedError("read_data needs to be implemented in child classes")

    def scale_data(self, time_data: TimeData) -> TimeData:
        """
        Scale data to physically meaningful units.

        For magnetotelluric data, this is assumed to be mV/km for electric
        channels, mV for magnetic channels (or nT for certain sensors)

        The base class assumes the data is already in the correct units and
        requires no scaling.

        Parameters
        ----------
        time_data : TimeData
            TimeData read in from file

        Returns
        -------
        TimeData
            TimeData scaled to give physically meaningful units
        """
        return time_data

    def _check_data_files(self, dir_path: Path, metadata: TimeMetadata) -> bool:
        """Check all data files in TimeMetadata exist"""
        from resistics.common import is_file

        chk = True
        for chan_metadata in metadata.chans_metadata.values():
            for data_file in chan_metadata.data_files:
                if not is_file(dir_path / data_file):
                    logger.debug(f"Data file {data_file} does not exist in {dir_path}")
                    chk = False
        return chk

    def _check_extensions(self, dir_path: Path, metadata: TimeMetadata) -> bool:
        """Check the data files have the correct extensions"""
        chk = True
        for chan_metadata in metadata.chans_metadata.values():
            for data_file in chan_metadata.data_files:
                if (dir_path / data_file).suffix != self.extension:
                    logger.debug(f"Extension of {data_file} != {self.extension}")
                    chk = False
        return chk

    def _get_read_samples(
        self,
        metadata: TimeMetadata,
        from_time: Optional[DateTimeLike] = None,
        to_time: Optional[DateTimeLike] = None,
        from_sample: Optional[int] = None,
        to_sample: Optional[int] = None,
    ) -> Tuple[int, int]:
        """
        Get samples to read from a mixture of from and to times or from and to samples.

        Times and samples can be used together. However, any provided times take priority over matching provided samples.

        Parameters
        ----------
        metadata : TimeMetadata
            Time series data metadata
        from_time : Union[DateTimeLike, None], optional
            Timestamp to read from, by default None
        to_time : Union[DateTimeLike, None], optional
            Timestamp to read to, by default None
        from_sample : Union[int, None], optional
            Sample to read from, by default None
        to_sample : Union[int, None], optional
            Sample to read to, by default None

        Returns
        -------
        read_from : int
            Sample to read from
        read_to : int
            Sample to read to
        """
        from resistics.sampling import check_sample, to_datetime
        from resistics.sampling import from_time_to_sample, to_time_to_sample

        n_samples = metadata.n_samples

        read_to = n_samples - 1
        if from_time is not None:
            read_from = from_time_to_sample(
                metadata.fs,
                metadata.first_time,
                metadata.last_time,
                to_datetime(from_time),
            )
        elif from_sample is not None:
            check_sample(n_samples, from_sample)
            read_from = from_sample
        else:
            read_from = 0

        if to_time is not None:
            read_to = to_time_to_sample(
                metadata.fs,
                metadata.first_time,
                metadata.last_time,
                to_datetime(to_time),
            )
        elif to_sample is not None:
            check_sample(n_samples, to_sample)
            read_to = to_sample
        else:
            read_to = n_samples - 1

        if read_from >= read_to:
            raise ValueError(f"From sample {read_from} >= to sample {read_to}")
        return read_from, read_to

    def _get_return_metadata(
        self, metadata: TimeMetadata, read_from: int, read_to: int
    ) -> TimeMetadata:
        """
        Get metadata to return

        Parameters
        ----------
        metadata : TimeMetadata
            Time series data metadata
        read_from : int
            Sample to read from
        read_to : int
            Sample to read to

        Returns
        -------
        TimeMetadata
            TimeMetadata for the TimeData
        """
        from resistics.sampling import sample_to_datetime

        from_time = sample_to_datetime(
            metadata.fs,
            metadata.first_time,
            read_from,
        )
        n_read = read_to - read_from + 1
        metadata = TimeMetadata(**metadata.dict())
        return adjust_time_metadata(metadata, metadata.fs, from_time, n_read)


class TimeReaderJSON(TimeReader):
    """Base class for TimeReaders that use a resistics JSON header"""

    def read_metadata(self, dir_path: Path) -> TimeMetadata:
        """
        Read the time series data metadata and return

        Parameters
        ----------
        dir_path : Path
            Path to time series data directory

        Returns
        -------
        TimeMetadata
            Metadata for time series data

        Raises
        ------
        MetadataReadError
            If the headers cannot be parsed
        TimeDataReadError
            If the data files do not match the expected extension
        """
        from resistics.errors import MetadataReadError, TimeDataReadError

        if self.extension is None:
            raise TimeDataReadError(dir_path, "No data file extension defined")

        metadata_path = dir_path / "metadata.json"
        try:
            metadata = TimeMetadata.parse_file(metadata_path)
        except KeyError:
            raise MetadataReadError(metadata_path, "No metadata found in metadata file")

        if not self._check_data_files(dir_path, metadata):
            raise TimeDataReadError(dir_path, "All data files do not exist")
        if not self._check_extensions(dir_path, metadata):
            raise TimeDataReadError(dir_path, f"Data file suffix not {self.extension}")
        return metadata


class TimeReaderAscii(TimeReaderJSON):
    """
    Class for reading Ascii data

    Ascii data expected to be one file per channel. Each file should have a
    single column of data with no header lines. Assumed to have a newline
    delimiter between values.
    """

    extension: str = ".ascii"

    def read_data(
        self, dir_path: Path, metadata: TimeMetadata, read_from: int, read_to: int
    ) -> TimeData:
        """
        Read data from Ascii files

        Parameters
        ----------
        dir_path : path
            The directory path to read from
        metadata : TimeMetadata
            Time series data metadata
        read_from : int
            Sample to read data from
        read_to : int
            Sample to read data to

        Returns
        -------
        TimeData
            TimeData

        Raises
        ------
        ValueError
            If metadata is None
        """
        dtype = np.float32
        n_samples = read_to - read_from + 1
        data = np.empty(shape=(len(metadata.chans), n_samples), dtype=dtype)

        logger.info(f"Reading data from {dir_path}")
        messages = [f"Reading raw data from {dir_path}"]
        messages.append(f"Sampling frequency {metadata.fs} Hz")
        for idx, chan in enumerate(metadata.chans):
            chan_path = dir_path / metadata[chan].data_files[0]
            messages.append(f"Reading data for {chan} from {chan_path}")
            data[idx] = np.loadtxt(
                chan_path,
                dtype=dtype,
                delimiter="\n",
                skiprows=read_from,
                max_rows=n_samples,
            )
        metadata = self._get_return_metadata(metadata, read_from, read_to)
        messages.append(f"From sample, time: {read_from}, {str(metadata.first_time)}")
        messages.append(f"To sample, time: {read_to}, {str(metadata.last_time)}")
        metadata.history.add_record(self._get_record(messages))
        logger.info(f"Data successfully read from {dir_path}")
        return TimeData(metadata, data)


class TimeReaderNumpy(TimeReaderJSON):
    """
    Class for reading Numpy data

    This is expected to be a single data file for all channels. The ordering is
    assumed to be the same as the channels definition in the metadata.
    """

    extension: str = ".npy"

    def read_data(
        self, dir_path: Path, metadata: TimeMetadata, read_from: int, read_to: int
    ) -> TimeData:
        """
        Read raw data saved in numpy data

        Parameters
        ----------
        dir_path : path
            The directory path to read from
        metadata : TimeMetadata
            Time series data metadata
        read_from : int
            Sample to read data from
        read_to : int
            Sample to read data to

        Returns
        -------
        TimeData
            TimeData

        Raises
        ------
        ValueError
            If metadata is None
        """
        import numpy as np

        messages = [f"Reading raw data from {dir_path}"]
        messages.append(f"Sampling frequency {metadata.fs} Hz")
        data_path = dir_path / "data.npy"
        data = np.load(data_path, mmap_mode="r")[:, read_from : read_to + 1]
        metadata = self._get_return_metadata(metadata, read_from, read_to)
        messages.append(f"From sample, time: {read_from}, {str(metadata.first_time)}")
        messages.append(f"To sample, time: {read_to}, {str(metadata.last_time)}")
        metadata.history.add_record(self._get_record(messages))
        logger.info(f"Data successfully read from {dir_path}")
        return TimeData(metadata, data)


class TimeWriterNumpy(ResisticsWriter):
    """
    Write out time data in numpy binary format

    Data is written out as a single data file including all channels
    """

    def run(self, dir_path: Path, time_data: TimeData) -> None:
        """
        Write out TimeData

        Parameters
        ----------
        dir_path : Path
            The directory path to write to
        time_data : TimeData
            TimeData to write out

        Raises
        ------
        WriteError
            If unable to write to the directory
        """
        from resistics.errors import WriteError

        if not self._check_dir(dir_path):
            WriteError(dir_path, "Unable to write to directory, check logs")
        logger.info(f"Writing time numpy data to {dir_path}")
        metadata_path = dir_path / "metadata.json"
        data_path = dir_path / "data.npy"
        np.save(data_path, time_data.data)
        metadata = time_data.metadata.copy()
        for chan in time_data.metadata.chans:
            metadata[chan].data_files = [data_path.name]
        metadata.history.add_record(self._get_record(dir_path, type(time_data)))
        metadata.write(metadata_path)


class TimeWriterAscii(ResisticsWriter):
    """
    Write out time data in ascii format
    """

    def run(self, dir_path: Path, time_data: TimeData) -> None:
        """
        Write out TimeData

        Parameters
        ----------
        dir_path : Path
            The directory path to write to
        time_data : TimeData
            TimeData to write out

        Raises
        ------
        WriteError
            If unable to write to the directory
        """
        from resistics.errors import WriteError

        if not self._check_dir(dir_path):
            WriteError(dir_path, "Unable to write to directory, check logs")
        logger.info(f"Writing time ASCII data to {dir_path}")
        metadata_path = dir_path / "metadata.json"
        metadata = time_data.metadata.copy()
        for chan in time_data.metadata.chans:
            chan_path = dir_path / f"{chan.lower()}.ascii"
            np.savetxt(chan_path, time_data[chan], fmt="%.6f", newline="\n")
            metadata[chan].data_files = [chan_path.name]
        metadata.history.add_record(self._get_record(dir_path, type(time_data)))
        metadata.write(metadata_path)


def new_time_data(
    time_data: TimeData,
    metadata: Optional[TimeMetadata] = None,
    data: Optional[np.ndarray] = None,
    record: Optional[Record] = None,
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
        metadata = TimeMetadata(**time_data.metadata.dict())
    if data is None:
        data = np.array(time_data.data)
    if record is not None:
        metadata.history.add_record(record)
    return TimeData(metadata, data)


class Subsection(ResisticsProcess):
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
        >>> plt.plot(time_data.get_x(), time_data["Ex"], label="full") # doctest: +SKIP
        >>> plt.plot(subsection.get_x(), subsection["Ex"], label="sub") # doctest: +SKIP
        >>> plt.legend(loc=3) # doctest: +SKIP
        >>> plt.tight_layout() # doctest: +SKIP
        >>> plt.show() # doctest: +SKIP
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
        from resistics.sampling import datetimes_to_samples, samples_to_datetimes
        from resistics.sampling import to_datetime

        logger.info(f"Taking subsection between {self.from_time} and {self.to_time}")
        from_time = to_datetime(self.from_time)
        to_time = to_datetime(self.to_time)
        fs = time_data.metadata.fs
        first_time = time_data.metadata.first_time
        last_time = time_data.metadata.last_time
        # convert to samples
        from_sample, to_sample = datetimes_to_samples(
            fs, first_time, last_time, from_time, to_time
        )
        n_samples = to_sample - from_sample + 1
        # convert back to times as datetimes may not coincide with timestamps
        from_time, to_time = samples_to_datetimes(
            fs, first_time, from_sample, to_sample
        )
        messages = [f"Subection from sample {from_sample} to {to_sample}"]
        messages.append(f"Adjusted times {str(from_time)} to {str(to_time)}")
        metadata = time_data.metadata.copy()
        metadata = adjust_time_metadata(metadata, fs, from_time, n_samples=n_samples)
        data = np.array(time_data.data[:, from_sample : to_sample + 1])
        record = self._get_record(messages)
        return new_time_data(time_data, metadata=metadata, data=data, record=record)


class InterpolateNans(ResisticsProcess):
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


class RemoveMean(ResisticsProcess):
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
    >>> np.all(hx_test == time_data_new["Hx"])
    True
    """

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


class Add(ResisticsProcess):
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

    add: Union[float, Dict[str, float]]

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
        if isinstance(self.add, float) or isinstance(self.add, int):
            return add + self.add
        for chan in time_data.metadata.chans:
            if chan in self.add:
                idx = time_data.get_chan_index(chan)
                add[idx] = self.add[chan]
        return add


class Multiply(ResisticsProcess):
    """
    Multiply channels by values

    Multiply can be used to add a constant value to all channels or values for
    specific channels can be provided.

    Multiply preseves the original type of the time data

    Parameters
    ----------
    multiply : Union[Dict[str, float], float]
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

    multiplier: Union[float, Dict[str, float]]

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
        if isinstance(self.multiplier, float) or isinstance(self.multiplier, int):
            return mult * self.multiplier
        for chan in time_data.metadata.chans:
            if chan in self.multiplier:
                idx = time_data.get_chan_index(chan)
                mult[idx] = self.multiplier[chan]
        return mult


class LowPass(ResisticsProcess):
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
        plt.plot(time_data.get_x(), time_data["chan1"], label="original")
        plt.plot(filtered.get_x(), filtered["chan1"], label="filtered")
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


class HighPass(ResisticsProcess):
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
        plt.plot(time_data.get_x(), time_data["chan1"], label="original")
        plt.plot(filtered.get_x(), filtered["chan1"], label="filtered")
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


class BandPass(ResisticsProcess):
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
        plt.plot(time_data.get_x(), time_data["chan1"], label="original")
        plt.plot(filtered.get_x(), filtered["chan1"], label="filtered")
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


class Notch(ResisticsProcess):
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
        plt.plot(time_data.get_x(), time_data["chan1"], label="original")
        plt.plot(filtered.get_x(), filtered["chan1"], label="filtered")
        plt.legend(loc=3)
        plt.tight_layout()
        plt.plot()
    """

    notch: float
    band: Optional[float] = None
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


class Resample(ResisticsProcess):
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
        >>> plt.plot(time_data.get_x(), time_data["chan1"], label="original") # doctest: +SKIP
        >>> plt.plot(resampled.get_x(), resampled["chan1"], label="resampled") # doctest: +SKIP
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
        from scipy.signal import resample_poly
        from fractions import Fraction

        fs = time_data.metadata.fs
        logger.info(f"Resampling data from {fs} Hz to {self.new_fs} Hz")
        # get the resampling fraction in its simplest form and resample
        frac = Fraction(self.new_fs / fs).limit_denominator()
        data = resample_poly(
            time_data.data.astype(np.float64),
            frac.numerator,
            frac.denominator,
            axis=1,
            window="hamming",
            padtype="mean",
        )
        data = data.astype(time_data.data.dtype)
        # adjust headers and
        n_samples = data.shape[1]
        metadata = time_data.metadata.copy()
        metadata = adjust_time_metadata(
            metadata, self.new_fs, time_data.metadata.first_time, n_samples=n_samples
        )
        messages = [f"Resampled data from {fs} Hz to {self.new_fs} Hz"]
        record = self._get_record(messages)
        return new_time_data(time_data, metadata=metadata, data=data, record=record)


class Decimate(ResisticsProcess):
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
        >>> plt.plot(time_data.get_x(), time_data["chan1"], label="original") # doctest: +SKIP
        >>> plt.plot(decimated.get_x(), decimated["chan1"], label="decimated") # doctest: +SKIP
        >>> plt.legend(loc=3) # doctest: +SKIP
        >>> plt.tight_layout() # doctest: +SKIP
        >>> plt.show() # doctest: +SKIP
    """

    factor: conint(ge=1)
    max_single_factor: conint(ge=2) = 3

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
        metadata = time_data.metadata.copy()
        metadata = adjust_time_metadata(
            metadata, new_fs, time_data.metadata.first_time, n_samples=n_samples
        )
        record = self._get_record(messages)
        return new_time_data(time_data, metadata=metadata, data=data, record=record)

    def _get_downsample_factors(self, downsample_factor: int) -> List[int]:
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

    def _prime_factorisation(self, n: int) -> List[int]:
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


class ShiftTimestamps(ResisticsProcess):
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
        >>> [x.time().strftime('%H:%M:%S') for x in time_data.get_x()]
        ['00:00:10', '00:00:30', '00:00:50', '00:01:10', '00:01:30']
        >>> process = ShiftTimestamps(shift=10)
        >>> result = process.run(time_data)
        >>> [x.time().strftime('%H:%M:%S') for x in result.get_x()]
        ['00:00:20', '00:00:40', '00:01:00', '00:01:20']
        >>> plt.plot(time_data.get_x(), time_data["chan1"], "bo", label="original") # doctest: +SKIP
        >>> plt.plot(result.get_x(), result["chan1"], "rd", label="shifted") # doctest: +SKIP
        >>> plt.legend(loc=4) # doctest: +SKIP
        >>> plt.grid() # doctest: +SKIP
        >>> plt.tight_layout() # doctest: +SKIP
        >>> plt.show() # doctest: +SKIP
    """

    shift: PositiveFloat

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
        from scipy.interpolate import interp1d

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

        logger.info(
            f"Data covers {str(metadata.first_time)} to {str(metadata.last_time)}"
        )
        logger.info(f"New data covers {str(first_time)} to {str(last_time)}")
        messages = [f"First time: {str(metadata.first_time)} -> {str(first_time)}"]
        messages.append(f"Last time: {str(metadata.last_time)} -> {str(last_time)}")

        # shift data
        x = np.arange(0, metadata.n_samples, dtype=time_data.data.dtype)
        x_shift = np.arange(0, n_samples, dtype=time_data.data.dtype) + norm_shift
        interp_fnc = interp1d(x, time_data.data, axis=1, copy=False)
        data = interp_fnc(x_shift)
        metadata = time_data.metadata.copy()
        metadata = adjust_time_metadata(
            metadata, metadata.fs, first_time, n_samples=n_samples
        )
        record = self._get_record(messages)
        return new_time_data(time_data, metadata=metadata, data=data, record=record)


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
#         metadata = args[0].metadata.copy()
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


class ApplyFunction(ResisticsProcess):
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

    fncs: Dict[str, Callable]

    class Config:

        arbitrary_types_allowed = True
        json_encoders = {
            types.LambdaType: serialize_custom_fnc,
            types.FunctionType: serialize_custom_fnc,
        }

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
