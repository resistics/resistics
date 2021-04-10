"""
Classes for storing and manipulating time data
"""
from pathlib import Path
from logging import getLogger
from typing import Collection, List, Dict, Union, Any, Tuple
from typing import Optional, Callable, Type
import numpy as np
import pandas as pd

from resistics.common import DatasetHeaders, ResisticsData, ResisticsProcess
from resistics.common import ProcessHistory, ProcessRecord
from resistics.calibrate import CalibrationReader, CalibrationData

logger = getLogger(__name__)


dataset_header_specs = {
    "fs": {"type": float, "default": None},
    "dt": {"type": float, "default": None},
    "n_chans": {"type": int, "default": None},
    "n_samples": {"type": int, "default": None},
    "first_time": {"type": pd.Timestamp, "default": None},
    "last_time": {"type": pd.Timestamp, "default": None},
    "system": {"type": str, "default": ""},
    "wgs84_latitude": {"type": float, "default": -999.0},
    "wgs84_longitude": {"type": float, "default": -999.0},
    "easting": {"type": float, "default": -999.0},
    "northing": {"type": float, "default": -999.0},
    "elevation": {"type": float, "default": -999.0},
}


chan_header_specs = {
    "data_files": {"type": str, "default": None},
    "sensor": {"type": str, "default": ""},
    "serial": {"type": str, "default": ""},
    "gain1": {"type": int, "default": 1},
    "gain2": {"type": int, "default": 1},
    "scaling": {"type": float, "default": 1},
    "hchopper": {"type": bool, "default": False},
    "echopper": {"type": bool, "default": False},
    "dx": {"type": float, "default": 1},
    "dy": {"type": float, "default": 1},
    "dz": {"type": float, "default": 1},
    "sensor_calibration_file": {"type": str, "default": ""},
    "instrument_calibration_file": {"type": str, "default": ""},
}


def template_time_headers(
    chans: List[str],
    overwrite_dataset: Optional[Dict[str, Any]] = None,
    overwrite_chans: Optional[Dict[str, Dict[str, Any]]] = None,
) -> DatasetHeaders:
    """
    Get TimeHeaders populated with defaults

    Parameters
    ----------
    chans : List[str]
        The channels to add
    overwrite_dataset : Optional[Dict[str, Any]], optional
        Any defaults to overwrite for the dataset headers, by default None, by default None
    overwrite_chans : Optional[Dict[str, Dict[str, Any]]], optional
        Any defaults to specify for the chans, by default None. These must be specified on a per channel basis, giving a dictionary of dictionaries.

    Returns
    -------
    DatasetHeaders
        DataHeaders with default values for time data where no overwrites are provided
    """
    from resistics.common import template_dataset_headers

    return template_dataset_headers(
        chans,
        dataset_header_specs,
        chan_header_specs,
        overwrite_dataset,
        overwrite_chans,
    )


def get_time_headers(
    dataset_headers: Dict[str, Any],
    chan_headers: Dict[str, Any],
) -> DatasetHeaders:
    """
    Get the DatasetHeaders for the time series

    Parameters
    ----------
    dataset_headers : Dict[str, Any]
        Headers for the dataset
    chan_headers : Dict[str, Any]
        Headers for the channels

    Returns
    -------
    DatasetHeaders
        The headers
    """
    return DatasetHeaders(
        dataset_headers, chan_headers, dataset_header_specs, chan_header_specs
    )


def adjust_headers_times(
    headers: DatasetHeaders,
    fs: float,
    first_time: pd.Timestamp,
    n_samples: Optional[int] = None,
    last_time: Optional[pd.Timestamp] = None,
) -> DatasetHeaders:
    """
    Adjust header time values and number of samples

    This is required if changes have been made to the sampling frequency, the number of samples, the first time or the last time. This might occur in processes such as resampling or decimating. One of last_time or n_samples must be provided.

    Note that the headers passed in will be changed in place

    Parameters
    ----------
    headers : DatasetHeaders
        The headers, this will be changed in place
    fs : float
        The sampling frequency
    first_time : pd.Timestamp
        The first time
    n_samples : Optional[int], optional
        The number of samples, by default None
    last_time : Optional[pd.Timestamp], optional
        The last time, by default None

    Returns
    -------
    DatasetHeaders
        The headers with the values adjusted

    Raises
    ------
    ValueError
        If both last_time and n_samples are not provided
    """
    if n_samples is None and last_time is None:
        raise ValueError("Both n_samples and last_time cannot be None")
    headers["fs"] = fs
    headers["dt"] = 1 / fs
    headers["first_time"] = first_time
    if n_samples is not None and last_time is not None:
        headers["last_time"] = last_time
        headers["n_samples"] = n_samples
    elif n_samples is not None:
        headers["last_time"] = first_time + pd.Timedelta(1 / fs, "s") * (n_samples - 1)
        headers["n_samples"] = n_samples
    else:
        headers["last_time"] = last_time
        headers["n_samples"] = (
            int((last_time - first_time) / pd.Timedelta(1 / fs, "s")) + 1
        )
    return headers


class TimeData(ResisticsData):
    """Class for holding time data"""

    def __init__(
        self,
        headers: DatasetHeaders,
        chans: List[str],
        data: np.ndarray,
        history: ProcessHistory,
    ) -> None:
        """Initialise TimeData"""
        self.headers = headers
        self.chans = chans
        self.data = data
        self.history = history
        self._chan_to_idx: Dict[str, int] = {}
        for idx, chan in enumerate(self.chans):
            self._chan_to_idx[chan] = idx

    def __iter__(self):
        """
        Return the channel iterator

        Returns
        -------
        list_iterator
            An iterator for the channels
        """
        return iter(self.chans)

    @property
    def n_chans(self) -> int:
        """
        Returns the number of channels

        Returns
        -------
        int
            The number of channels in spectra data
        """
        return len(self.chans)

    @property
    def n_samples(self) -> int:
        """
        Get the number of samples in the data

        Returns
        -------
        int
            The number of samples
        """
        return self.headers["n_samples"]

    @property
    def fs(self) -> float:
        """
        Get the sampling frequency

        Returns
        -------
        float
            TimeData sampling frequency
        """
        return self.headers["fs"]

    @property
    def dt(self) -> float:
        """
        Get the sampling period

        Returns
        -------
        float
            TimeData sampling period
        """
        return self.headers["dt"]

    @property
    def nyquist(self) -> float:
        """
        Get the nyquist frequency of the spectra data

        Returns
        -------
        nyquist : float
            The nyquist frequency in Hz
        """
        return self.fs / 2.0

    @property
    def first_time(self) -> pd.Timestamp:
        """
        Returns the timestamp of the first sample

        Returns
        -------
        datetime
            The timestamp of the first sample
        """
        return self.headers["first_time"]

    @property
    def last_time(self) -> pd.Timestamp:
        """
        Returns the timestamp of the last sample

        Returns
        -------
        datetime
            The time stamp of the last sample
        """
        return self.headers["last_time"]

    @property
    def duration(self) -> pd.Timedelta:
        """
        Duration of the recording

        This is the time between the first and last samples

        Returns
        -------
        float
            The duration in seconds
        """
        return self.last_time - self.first_time

    def get_timestamps(
        self, samples: Union[np.ndarray, None] = None
    ) -> pd.DatetimeIndex:
        """
        Get an array of timestamps

        Parameters
        ----------
        samples : Union[np.ndarray, None], optional
            If provided, timestamps are only returned for the specified samples, by default None

        Returns
        -------
        pd.DatetimeIndex
            The return dates
        """
        if samples is None:
            return pd.date_range(
                start=self.first_time,
                freq=pd.Timedelta(self.dt, "s"),
                periods=self.n_samples,
            )
        else:
            return pd.to_datetime(
                samples * pd.Timedelta(self.dt, "s") + self.first_time
            )

    def x_size(self) -> int:
        """
        For abstract plotting functions, return the size


        Returns
        -------
        int
            The x size, equal to the number of samples
        """
        return self.n_samples

    def get_x(self, samples: Union[np.ndarray, None] = None) -> pd.DatetimeIndex:
        """
        For plotting, get x dimension, in this case times

        Parameters
        ----------
        indices : Union[np.ndarray, None], optional
            If provided, x values (timestamps) are only returned for the specified samples, by default None

        Returns
        -------
        pd.DatetimeIndex
            Timestamp array
        """
        return self.get_timestamps(samples=samples)

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

        check_chan(chan, self.chans)
        return self.data[self._chan_to_idx[chan]]

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

    def set_chan(self, chan: str, chan_data: np.ndarray) -> None:
        """
        Set channel time data

        Parameters
        ----------
        chan : str
            The channel to set the data for
        chan_data : np.ndarray
            The new channel data
        """
        from resistics.common import check_chan

        check_chan(chan, self.chans)
        assert chan_data.size == self.n_samples
        assert chan_data.dtype == self.data.dtype
        self.data[self._chan_to_idx[chan]] = chan_data

    def subsection(self, from_time: pd.Timestamp, to_time: pd.Timestamp) -> "TimeData":
        """
        Get a subsection of the TimeData

        Returns a new TimeData object

        Parameters
        ----------
        from_time : pd.Timestamp
            Start of subsection
        to_time : pd.Timestamp
            End of subsection
        """
        sub = Subsection(from_time, to_time)
        return sub.run(self)

    def copy(self):
        """
        Get a copy of the time data object

        Returns
        -------
        TimeData
            A copy of the time data object
        """
        return TimeData(
            self.headers.copy(),
            list(self.chans),
            np.array(self.data),
            self.history.copy(),
        )

    def plot(
        self,
        fig=None,
        subplots: Union[List[str], None] = None,
        subplot_columns: Union[Dict[str, List[str]], None] = None,
        max_pts=10_000,
        label_prefix="",
    ) -> Any:
        """
        Plot time data

        Parameters
        ----------
        fig : [type], optional
            Plotly figure, by default None. If no figure is provided, a new one will be created.
        subplots : Union[List[str], None], optional
            Subplots, by default None. To customise the plot, provide a list of subplots
        subplot_columns : Union[Dict[str, List[str]], None], optional
            Subplot columns defines which channels to plot for each subplot, by default None
        max_pts : int, optional
            Maximum number of points to plot, by default 10000. Data will be downsampled using lttb method.

        Returns
        -------
        fig
            Plotly figure
        """
        from resistics.common import is_electric, is_magnetic
        from resistics.plot import figure_columns_as_lines, plot_columns_1d

        if subplots is None:
            subplots = ["Electric channels", "Magnetic channels"]
        if subplot_columns is None:
            subplot_columns = {
                "Electric channels": [x for x in self.chans if is_electric(x)],
                "Magnetic channels": [x for x in self.chans if is_magnetic(x)],
            }
        if fig is None:
            y_labels = {}
            for subplot, columns in subplot_columns.items():
                if len(columns) > 0 and is_electric(columns[0]):
                    y_labels[subplot] = "mv/km"
                elif len(columns) > 0 and is_magnetic(columns[0]):
                    y_labels[subplot] = "mV or nT"
                else:
                    y_labels[subplot] = "Unknown"
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

    def to_string(self) -> str:
        """
        Class information as a list of strings

        Returns
        -------
        out : List[str]
            List of strings with information
        """
        outstr = f"{self.type_to_string()}\n"
        outstr += f"Sampling frequency [Hz] = {self.fs}\n"
        outstr += f"Sample rate [s] = {self.dt}\n"
        outstr += f"Number of samples = {self.n_samples}\n"
        outstr += f"Number of channels = {self.n_chans}\n"
        outstr += f"Channels = {self.chans}\n"
        outstr += f"First sample time = {self.first_time}\n"
        outstr += f"Last sample time = {self.last_time}\n"
        outstr += self.history.to_string()
        return outstr


class TimeReader(ResisticsProcess):
    def __init__(self, dir_path: Path) -> None:
        """Initialise with path to data directory

        Parameters
        ----------
        dataPath : str
            Path to data directory
        """
        from resistics.common import assert_dir

        assert_dir(dir_path)
        self.dir_path = dir_path
        self.headers: Union[None, DatasetHeaders] = None
        self.apply_scalings: bool = True

    def parameters(self) -> Dict[str, Any]:
        """
        Get process parameters

        Returns
        -------
        Dict[str, Any]
            Returns a dictionary of key TimeReader parameters
        """
        return {
            "dir_path": str(self.dir_path.absolute()),
            "apply_scalings": self.apply_scalings,
        }

    def check(self):
        """
        Initial check to make sure the data folder can be read

        Parameters
        ----------
        dir_path : Path
            The data directory

        Raises
        ------
        NotImplementedError
            To be implemented in child classes
        """
        raise NotImplementedError("Checks should be implemented in child classes")

    def read_headers(self):
        """
        Read time series data file headers

        Raises
        ------
        NotImplementedError
            To be implemented in child classes
        """
        raise NotImplementedError(
            "read_headers needs to be implemented in child classes"
        )

    def run(
        self,
        from_time: Union[pd.Timestamp, None] = None,
        to_time: Union[pd.Timestamp, None] = None,
        from_sample: Union[int, None] = None,
        to_sample: Union[int, None] = None,
    ) -> TimeData:
        """
        Read time data

        Parameters
        ----------
        from_time : Union[pd.Timestamp, None], optional
            Timestamp to read from, by default None
        to_time : Union[pd.Timestamp, None], optional
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
        from_sample, to_sample = self._get_read_samples(
            from_time, to_time, from_sample, to_sample
        )
        time_data = self.read_data(from_sample, to_sample)
        if self.apply_scalings:
            return self.scale_data(time_data)
        return time_data

    def read_data(self, read_from: int, read_to: int) -> TimeData:
        """
        Read raw data with minimal scalings applied

        Parameters
        ----------
        read_from : int
            Sample to read data from
        read_to : int
            Sample to read data to

        Returns
        -------
        TimeData
            A TimeData instance

        Raises
        ------
        NotImplementedError
            To be implemented in child TimeReader classes
        """
        raise NotImplementedError("read_data needs to be implemented in child classes")

    def scale_data(self, time_data: TimeData) -> TimeData:
        """
        Scale data to physically meaningful units.

        - For magnetotelluric data, this is assumed to be mV/km for electric channels, mV for magnetic channels (or nT for certain sensors)

        The base class assumes the data is already in the correct units and requires no scaling.

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

    def _get_read_samples(
        self,
        from_time: Union[pd.Timestamp, None] = None,
        to_time: Union[pd.Timestamp, None] = None,
        from_sample: Union[int, None] = None,
        to_sample: Union[int, None] = None,
    ) -> Tuple[int, int]:
        """
        Get samples to read from a mixture of from and to times or from and to samples.

        Times and samples can be used together. However, any provided times take priority over matching provided samples.

        Parameters
        ----------
        from_time : Union[pd.Timestamp, None], optional
            Timestamp to read from, by default None
        to_time : Union[pd.Timestamp, None], optional
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
        from resistics.math import (
            from_time_to_sample,
            check_from_sample,
            to_time_to_sample,
            check_to_sample,
        )

        n_samples = self.headers["n_samples"]
        read_from = 0
        read_to = n_samples - 1

        if from_time is not None:
            read_from = from_time_to_sample(
                self.headers["fs"],
                self.headers["first_time"],
                self.headers["last_time"],
                from_time,
            )
        elif from_sample is not None:
            read_from = check_from_sample(self.headers["n_samples"], from_sample)

        if to_time is not None:
            read_to = to_time_to_sample(
                self.headers["fs"],
                self.headers["first_time"],
                self.headers["last_time"],
                to_time,
            )
        elif to_sample is not None:
            read_to = check_to_sample(n_samples, to_sample)

        return read_from, read_to

    def _get_return_headers(self, read_from: int, read_to: int) -> DatasetHeaders:
        """
        Get headers to return

        Parameters
        ----------
        read_from : int
            Sample to read from
        read_to : int
            Sample to read to

        Returns
        -------
        DatasetHeaders
            DatasetHeaders for the TimeData
        """
        from resistics.math import samples_to_datetimes

        from_time, to_time = samples_to_datetimes(
            self.headers["fs"],
            self.headers["first_time"],
            read_from,
            read_to,
        )
        headers = self.headers.copy()
        headers = adjust_headers_times(
            headers, headers["fs"], from_time, last_time=to_time
        )
        return headers


class TimeReaderJSON(TimeReader):
    """
    Base class for TimeReaders that use a resistics JSON header
    """

    def read_headers(self, header_path: Path) -> DatasetHeaders:
        """
        Read the time series data headers and return

        Parameters
        ----------
        header_path : Path
            Header

        Returns
        -------
        DatasetHeaders
            Dataset headers for the time data

        Raises
        ------
        HeaderReadError
            If the wrong type of headers are found
        """
        from resistics.errors import HeaderReadError
        from resistics.common import json_to_headers

        headers = json_to_headers(header_path)
        if not isinstance(headers, DatasetHeaders):
            raise HeaderReadError(
                header_path, f"Expected type {DatasetHeaders}, got type {type(headers)}"
            )
        # put it through get time headers to make sure everything is formatted correctly
        headers_dict = headers.to_dict()
        return get_time_headers(headers_dict["dataset"], headers_dict["channel"])


class TimeReaderAscii(TimeReaderJSON):
    """
    Class for reading Ascii data

    Ascii data expected to be one file per channel. Each file should have a single column of data with no header lines. Assumed to have a newline delimiter between values.
    """

    def check(self) -> bool:
        """
        Checks before reading a dataset

        Returns
        -------
        bool
            True if all checks are passed and data can be read
        """
        from resistics.common import is_file

        header_path = self.dir_path / "headers.json"
        if not is_file(header_path):
            logger.error(f"Header path {header_path} does not exist")
            return False
        try:
            headers = self.read_headers(header_path)
        except:
            logger.error(f"Unable to read header data in {header_path}")
            return False

        self.headers = headers
        chk_files = True
        for chan in self.headers.chans:
            chan_path = self.dir_path / self.headers[chan, "data_files"]
            if chan_path.suffix != ".ascii":
                logger.error(f"{chan_path.name} has incorrect suffix (require .ascii)")
                chk_files = False
            if not is_file(chan_path):
                logger.error(f"Unable to find {chan} data file {chan_path.name}")
                chk_files = False
        if not chk_files:
            return False
        logger.info(f"Passed checks and successfully read headers from {header_path}")
        return True

    def read_data(self, read_from: int, read_to: int) -> TimeData:
        """
        Read data from Ascii files

        Parameters
        ----------
        read_from : int
            Sample to read data from
        read_to : int
            Sample to read data to

        Returns
        -------
        TimeData
            TimeData
        """
        import numpy as np

        assert self.headers is not None

        logger.info(f"Reading data from {self.dir_path}")
        dtype = np.float32
        chans = self.headers.chans
        n_samples = read_to - read_from + 1

        messages = [f"Reading raw data from {self.dir_path}"]
        messages.append(f"Sampling rate {self.headers['fs']} Hz")
        messages.append(f"Reading samples {read_from} to {read_to}")
        data = np.empty(shape=(len(chans), n_samples))
        for idx, chan in enumerate(chans):
            chan_path = self.dir_path / self.headers[chan, "data_files"]
            messages.append(f"Reading data for {chan} from {chan_path}")
            data[idx] = np.loadtxt(
                chan_path,
                dtype=dtype,
                delimiter="\n",
                skiprows=read_from,
                max_rows=n_samples,
            )
        headers = self._get_return_headers(read_from, read_to)
        messages.append(f"Time range {headers['first_time']} to {headers['last_time']}")
        record = self._get_process_record(messages)
        logger.info(f"Data successfully read from {self.dir_path}")
        return TimeData(headers, chans, data, ProcessHistory([record]))


class TimeReaderNumpy(TimeReaderJSON):
    """
    Class for reading Numpy data

    This is expected to be one file per channel and of data type float32
    """

    def check(self) -> bool:
        """
        Checks before reading a dataset

        Returns
        -------
        bool
            True if all checks are passed and data can be read
        """
        from resistics.common import is_file

        header_path = self.dir_path / "headers.json"
        if not is_file(header_path):
            logger.error(f"Header path {header_path} does not exist")
            return False
        try:
            headers = self.read_headers(header_path)
        except:
            logger.error(f"Unable to read header data in {header_path}")
            return False

        self.headers = headers
        chk_files = True
        for chan in self.headers.chans:
            chan_path = self.dir_path / self.headers[chan, "data_files"]
            if chan_path.suffix != ".npy":
                logger.error(f"{chan_path.name} has incorrect suffix (require .npy)")
                chk_files = False
            if not is_file(chan_path):
                logger.error(f"Unable to find {chan} data file {chan_path.name}")
                chk_files = False
        if not chk_files:
            return False
        logger.info(f"Passed checks and successfully read headers from {header_path}")
        return True

    def read_data(self, read_from: int, read_to: int) -> TimeData:
        """
        Read raw data saved in numpy data

        Parameters
        ----------
        read_from : int
            Sample to read data from
        read_to : int
            Sample to read data to

        Returns
        -------
        TimeData
            TimeData
        """
        import numpy as np

        assert self.headers is not None

        chans = self.headers.chans
        n_samples = read_to - read_from + 1

        messages = [f"Reading raw data from {self.dir_path}"]
        messages.append(f"Sampling rate {self.headers['fs']} Hz")
        messages.append(f"Reading samples {read_from} to {read_to}")
        data = np.empty(shape=(len(chans), n_samples))
        for idx, chan in enumerate(self.headers.chans):
            chan_path = self.dir_path / self.headers[chan, "data_files"]
            messages.append(f"Reading data for {chan} from {chan_path}")
            data[idx] = np.load(chan_path, mmap_mode="r")[read_from : read_to + 1]
        headers = self._get_return_headers(read_from, read_to)
        messages.append(f"Time range {headers['first_time']} to {headers['last_time']}")
        record = self._get_process_record(messages)
        logger.info(f"Data successfully read from {self.dir_path}")
        return TimeData(headers, chans, data, ProcessHistory([record]))


class TimeWriter(ResisticsProcess):
    """
    Base class for writing time series data
    """

    def __init__(
        self, dir_path: Path, create_path: bool = False, overwrite: bool = False
    ):
        """
        Initialise the TimeData writer

        Parameters
        ----------
        dir_path : Path
            Directory to write to
        create_path : bool, optional
            Boolean flag for creating the path if it does not exist, by default False
        overwrite : bool, optional
            Boolean flag for overwriting the existing data, by default False
        """
        self.dir_path = dir_path
        self.create_path = create_path
        self.overwrite = overwrite

    def check(self) -> bool:
        from resistics.errors import ProcessCheckError

        if not self.dir_path.exists() and not self.create_path:
            raise ProcessCheckError(
                self.name,
                f"Directory {self.dir_path} not exist and create_path is False",
            )
        return True


class TimeWriterNumpy(TimeWriter):
    """
    Write out data in numpy binary format
    """

    def run(self, time_data: TimeData) -> None:
        """
        Write out TimeData

        Parameters
        ----------
        time_data : TimeData
            TimeData to write out
        """
        from resistics.common import headers_to_json

        header_path = self.dir_path / "headers.json"
        headers = time_data.headers.copy()
        for chan in time_data:
            chan_path = self.dir_path / f"{chan.lower()}.npy"
            np.save(chan_path, time_data[chan])
            headers[chan, "data_files"] = chan_path.name
        headers_to_json(headers, header_path)


class TimeWriterAscii(TimeWriter):
    """
    Write out data in ascii format
    """

    def run(self, time_data: TimeData) -> None:
        """
        Write out TimeData

        Parameters
        ----------
        time_data : TimeData
            TimeData to write out
        """
        from resistics.common import headers_to_json

        header_path = self.dir_path / "headers.json"
        headers = time_data.headers.copy()
        for chan in time_data:
            chan_path = self.dir_path / f"{chan.lower()}.ascii"
            np.savetxt(chan_path, time_data[chan], fmt="%.6f", newline="\n")
            headers[chan, "data_files"] = chan_path.name
        headers_to_json(headers, header_path)


def new_time_data(
    time_data: TimeData,
    headers: Optional[DatasetHeaders] = None,
    chans: Optional[List[str]] = None,
    data: Optional[np.ndarray] = None,
    record: Optional[ProcessRecord] = None,
) -> TimeData:
    """
    Get a new TimeData

    Values are taken from an existing TimeData where they are not explicitly specified. This is useful in a process where only some aspects of the TimeData have been changed

    Parameters
    ----------
    time_data : TimeData
        The existing TimeData
    headers : Optional[DatasetHeaders], optional
        A new headers, by default None
    chans : Optional[List[str]], optional
        A new list of chans, by default None
    data : Optional[np.ndarray], optional
        New data, by default None
    record : Optional[ProcessRecord], optional
        A new record to add, by default None

    Returns
    -------
    TimeData
        A new TimeData instance
    """
    if headers is None:
        headers = time_data.headers.copy()
    if chans is None:
        chans = list(time_data.chans)
    if data is None:
        data = np.array(time_data.data)
    history = time_data.history.copy()
    if record is not None:
        history.add_record(record)
    return TimeData(headers, chans, data, history)


class Subsection(ResisticsProcess):
    """
    Get a subsection of time data
    """

    def __init__(self, from_time: pd.Timestamp, to_time: pd.Timestamp):
        self.from_time = from_time
        self.to_time = to_time

    def run(self, time_data: TimeData) -> TimeData:
        from resistics.math import datetimes_to_samples, samples_to_datetimes

        logger.info(f"Taking subsection between {self.from_time} and {self.to_time}")
        fs = time_data.fs
        first_time = time_data.first_time
        last_time = time_data.last_time
        # convert to samples
        from_sample, to_sample = datetimes_to_samples(
            fs, first_time, last_time, self.from_time, self.to_time
        )
        n_samples = to_sample - from_sample + 1
        # convert back to times as datetimes may not coincide with timestamps
        from_time, to_time = samples_to_datetimes(
            fs, first_time, from_sample, to_sample
        )
        messages = [f"Subection from sample {from_sample} to {to_sample}"]
        messages.append(f"Adjusted times {from_time} to {to_time}")
        headers = time_data.headers.copy()
        headers = adjust_headers_times(
            headers, fs, from_time, n_samples=n_samples, last_time=to_time
        )
        data = np.array(time_data.data[:, from_sample : to_sample + 1])
        record = self._get_process_record(messages)
        return new_time_data(time_data, headers=headers, data=data, record=record)


class InterpolateNans(ResisticsProcess):
    """
    Interpolate nan values in the data

    Examples
    --------
    .. doctest::

        >>> from resistics.testing import time_data_with_nans
        >>> from resistics.time import InterpolateNans
        >>> time_data = time_data_with_nans()
        >>> time_data["Hx"]
        array([nan,  2.,  3.,  5.,  1.,  2.,  3.,  4.,  2.,  6.,  7., nan, nan,
            4.,  3.,  2.])
        >>> processor = InterpolateNans()
        >>> time_data_new = processor.run(time_data)
        >>> time_data_new["Hx"]
        array([2., 2., 3., 5., 1., 2., 3., 4., 2., 6., 7., 6., 5., 4., 3., 2.])
        >>>
    """

    def run(self, time_data: TimeData) -> TimeData:
        logger.info(f"Removing nan values from channels {time_data.chans}")
        messages = []
        data = np.array(time_data.data)
        for chan in time_data:
            idx = time_data.get_chan_index(chan)
            data[idx, :] = self._interpolate_nans(data[idx, :])
            messages.append(f"nan values removed from {chan}")
        record = self._get_process_record(messages)
        return new_time_data(time_data, data=data, record=record)

    def _interpolate_nans(self, chan_data: np.ndarray) -> np.ndarray:
        nan_bool = np.isnan(chan_data)
        if not np.any(nan_bool):
            return chan_data
        mask = np.ones(chan_data.size, np.bool)
        mask[nan_bool] = 0
        x = np.arange(chan_data.size)
        chan_data[nan_bool] = np.interp(x[nan_bool], x[mask], chan_data[mask])
        return chan_data


class RemoveMean(ResisticsProcess):
    """
    Remove channel mean value from each channel

    Examples
    --------
    .. doctest::

        >>> import numpy as np
        >>> from resistics.testing import time_data_simple
        >>> from resistics.time import RemoveMean
        >>> time_data = time_data_simple()
        >>> processor = RemoveMean()
        >>> time_data_new = processor.run(time_data)
        >>> time_data_new["Hx"]
        array([-2.5, -1.5, -0.5,  1.5, -2.5, -1.5, -0.5,  0.5, -1.5,  2.5,  3.5,
            2.5,  1.5,  0.5, -0.5, -1.5])
        >>> hx_test = time_data["Hx"] - np.mean(time_data["Hx"])
        >>> hx_test
        array([-2.5, -1.5, -0.5,  1.5, -2.5, -1.5, -0.5,  0.5, -1.5,  2.5,  3.5,
            2.5,  1.5,  0.5, -0.5, -1.5])
        >>> np.all(hx_test == time_data_new["Hx"])
        True
    """

    def run(self, time_data: TimeData) -> TimeData:
        from resistics.common import array_to_string

        logger.info(f"Removing mean from channels {time_data.chans}")
        mean = np.mean(time_data.data, axis=1)
        data = time_data.data - mean[:, None]
        messages = [
            f"Removed means {array_to_string(mean, precision=2)} for chans {time_data.chans}"
        ]
        record = self._get_process_record(messages)
        return new_time_data(time_data, data=data, record=record)


class Add(ResisticsProcess):
    """
    Add values to channels

    Add can be used to add a constant value to all channels or values for specific channels can be provided.

    Examples
    --------
    Using a constant value for all channels passed as a scalar

    .. doctest::

        >>> from resistics.testing import time_data_ones
        >>> from resistics.time import Add
        >>> time_data = time_data_ones()
        >>> adder = Add(5)
        >>> time_data_new = adder.run(time_data)
        >>> time_data_new["Ex"] - time_data["Ex"]
        array([5., 5., 5., 5., 5., 5., 5., 5., 5., 5.])
        >>> time_data_new["Ey"] - time_data["Ey"]
        array([5., 5., 5., 5., 5., 5., 5., 5., 5., 5.])

    Variable values for the channels provided as a dictionary

    .. doctest::

        >>> from resistics.testing import time_data_ones
        >>> from resistics.time import Add
        >>> time_data = time_data_ones()
        >>> adder = Add({"Ex": 3, "Hy": -7})
        >>> time_data_new = adder.run(time_data)
        >>> time_data_new["Ex"] - time_data["Ex"]
        array([3., 3., 3., 3., 3., 3., 3., 3., 3., 3.])
        >>> time_data_new["Hy"] - time_data["Hy"]
        array([-7., -7., -7., -7., -7., -7., -7., -7., -7., -7.])
        >>> time_data_new["Ey"] - time_data["Ey"]
        array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    """

    def __init__(self, add: Union[float, Dict[str, float]]):
        """
        Initialise

        Parameters
        ----------
        add : Union[float, Dict[str, float]]
            Either a scalar to add to all channels or dictionary with values to add to each channel
        """
        self.add = add

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
        logger.info(f"Added {add} to channels {time_data.chans}")
        data = time_data.data + add[:, None]
        messages = [f"Added {add} to channels {time_data.chans}"]
        record = self._get_process_record(messages)
        return new_time_data(time_data, data=data, record=record)

    def _get_add(self, time_data: TimeData) -> np.ndarray:
        add = np.zeros(shape=(time_data.n_chans))
        if isinstance(self.add, float) or isinstance(self.add, int):
            return add + self.add
        for chan in time_data.chans:
            if chan in self.add:
                idx = time_data.get_chan_index(chan)
                add[idx] = self.add[chan]
        return add


class Multiply(ResisticsProcess):
    """
    Multiply channels by values

    Multiply can be used to add a constant value to all channels or values for specific channels can be provided.

    Examples
    --------
    Using a constant value for all channels passed as a scalar

    .. doctest::

        >>> from resistics.testing import time_data_ones
        >>> from resistics.time import Multiply
        >>> time_data = time_data_ones()
        >>> multiplier = Multiply(5)
        >>> time_data_new = multiplier.run(time_data)
        >>> time_data_new["Ex"]/time_data["Ex"]
        array([5., 5., 5., 5., 5., 5., 5., 5., 5., 5.])
        >>> time_data_new["Ey"]/time_data["Ey"]
        array([5., 5., 5., 5., 5., 5., 5., 5., 5., 5.])

    Variable values for the channels provided as a dictionary

    .. doctest::

        >>> from resistics.testing import time_data_ones
        >>> from resistics.time import Multiply
        >>> time_data = time_data_ones()
        >>> multiplier = Multiply({"Ex": 3, "Hy": -7})
        >>> time_data_new = multiplier.run(time_data)
        >>> time_data_new["Ex"]/time_data["Ex"]
        array([3., 3., 3., 3., 3., 3., 3., 3., 3., 3.])
        >>> time_data_new["Hy"]/time_data["Hy"]
        array([-7., -7., -7., -7., -7., -7., -7., -7., -7., -7.])
        >>> time_data_new["Ey"]/time_data["Ey"]
        array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    """

    def __init__(self, multiply: Union[Dict[str, float], float]):
        """
        Multiply channels with values

        Parameters
        ----------
        multiply : Union[Dict[str, float], float]
            Either a float to multiply all channels with the same value or a dictionary to specify different values for each channel
        """
        self.multiply = multiply

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
        logger.info(f"Multipying channels {time_data.chans} by {mult}")
        data = time_data.data * mult[:, None]
        messages = [f"Multiplied channels {time_data.chans} by {mult}"]
        record = self._get_process_record(messages)
        return new_time_data(time_data, data=data, record=record)

    def _get_mult(self, time_data: TimeData) -> np.ndarray:
        mult = np.ones(shape=(time_data.n_chans))
        if isinstance(self.multiply, float) or isinstance(self.multiply, int):
            return mult * self.multiply
        for chan in time_data.chans:
            if chan in self.multiply:
                idx = time_data.get_chan_index(chan)
                mult[idx] = self.multiply[chan]
        return mult


class LowPass(ResisticsProcess):
    """
    Apply low pass filter

    Examples
    --------
    .. plot::
        :context:close-figs

        >>> from resistics.testing import time_data_random
    """

    def __init__(self, cutoff: float, order: int = 10):
        """
        Initalise

        Parameters
        ----------
        cutoff : float
            The cutoff for the low pass
        order : int, optional
            Order of the filter, by default 10
        """
        self.cutoff = cutoff
        self.order = order

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
        """
        from scipy.signal import butter, sosfiltfilt

        logger.info(f"Low pass filtering with cutoff {self.cutoff} Hz")
        normed = self.cutoff / time_data.nyquist
        sos = butter(self.order, normed, btype="lowpass", analog=False, output="sos")
        data = sosfiltfilt(sos, time_data.data, axis=1)
        messages = [f"Low pass filtered data with cutoff {self.cutoff} Hz"]
        record = self._get_process_record(messages)
        return new_time_data(time_data, data=data, record=record)


class HighPass(ResisticsProcess):
    """
    High pass filter time data

    Examples
    --------
    """

    def __init__(self, cutoff: float, order: int = 10):
        """
        Initalise

        Parameters
        ----------
        cutoff : float
            Cutoff for the high pass filter
        order : int, optional
            Order of the filter, by default 10
        """
        self.cutoff = cutoff
        self.order = order

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
        """
        from scipy.signal import butter, sosfiltfilt

        logger.info(f"High pass filtering with cutoff {self.cutoff} Hz")
        normed = self.cutoff / time_data.nyquist
        sos = butter(self.order, normed, btype="highpass", analog=False, output="sos")
        data = sosfiltfilt(sos, time_data.data, axis=1)
        messages = [f"High pass filtered data with cutoff {self.cutoff} Hz"]
        record = self._get_process_record(messages)
        return new_time_data(time_data, data=data, record=record)


class BandPass(ResisticsProcess):
    """
    Band pass filter time data

    Examples
    --------
    """
    def __init__(self, cutoff_low: float, cutoff_high: float, order: int = 10):
        """
        Initialise

        Parameters
        ----------
        cutoff_low : float
            The low cutoff for the band pass filter
        cutoff_high : float
            The high cutoff for the band pass filter
        order : int, optional
            The order of the filter, by default 10
        """
        self.cutoff_low = cutoff_low
        self.cutoff_high = cutoff_high
        self.order = order

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
        """
        from scipy.signal import butter, sosfiltfilt

        logger.info(f"Band pass between {self.cutoff_low} and {self.cutoff_high} Hz")
        low = self.cutoff_low / time_data.nyquist
        high = self.cutoff_high / time_data.nyquist
        sos = butter(
            self.order, (low, high), btype="bandpass", analog=False, output="sos"
        )
        data = sosfiltfilt(sos, time_data.data, axis=1)
        messages = [f"Band pass with cutoffs {self.cutoff_low},{self.cutoff_high} Hz"]
        record = self._get_process_record(messages)
        return new_time_data(time_data, data=data, record=record)


class Notch(ResisticsProcess):
    """
    Notch filter time data

    Examples
    --------


    """
    def __init__(self, notch: float, band: Optional[float] = None, order: int = 10):
        """
        Initialise

        Parameters
        ----------
        notch : float
            The frequency to notch
        band : Optional[float], optional
            The bandwidth of the filter, by default None
        order : int, optional
            The order of the filter, by default 10
        """
        self.notch = notch
        self.band = None
        self.order = order

    def run(self, time_data: TimeData) -> TimeData:
        from scipy.signal import iirfilter, sosfiltfilt

        band = 0.1 * time_data.fs if self.band is None else self.band
        low = (self.notch - (band / 2)) / time_data.nyquist
        high = (self.notch + (band / 2)) / time_data.nyquist
        logger.info(f"Notch filtering at {self.notch} Hz with bandwidth {band} Hz")
        sos = iirfilter(
            self.order, (low, high), btype="bandstop", analog=False, ftype="bessel"
        )
        data = sosfiltfilt(sos, time_data.data, axis=1)
        messages = [f"Notch filtered at {self.notch} Hz with bandwidth {band} Hz"]
        record = self._get_process_record(messages)
        return new_time_data(time_data, data=data, record=record)


class Resample(ResisticsProcess):
    def __init__(self, new_fs: int):
        self.new_fs = new_fs

    def parameters(self):
        return {"new_fs": self.new_fs}

    def run(self, time_data: TimeData) -> TimeData:
        """Resample time data

        Resample the data using the polyphase method which does not assume periodicity
        Calculate the upsample and then the downsampling rate and using polyphase filtering, the final sample rate is:
        (up / down) * original sample rate
        Therefore, to get a sampling frequency of resampFreq, want:
        (resampFreq / sampleFreq) * sampleFreq
        Use the fractions library to get up and down as integers which they are required to be.
        """
        from scipy.signal import resample_poly
        from fractions import Fraction

        fs = time_data.fs
        logger.info(f"Resampling data from {fs} Hz to {self.new_fs} Hz")
        # get the resampling fraction in its simplest form
        frac = Fraction(self.new_fs / fs).limit_denominator()
        data = resample_poly(time_data.data, frac.numerator, frac.denominator, axis=1)
        # adjust headers and
        n_samples = data.shape[1]
        headers = time_data.headers.copy()
        headers = adjust_headers_times(
            headers, self.new_fs, time_data.first_time, n_samples=n_samples
        )
        messages = [f"Resampled data from {fs} Hz to {self.new_fs} Hz"]
        record = self._get_process_record(messages)
        return new_time_data(time_data, headers=headers, data=data, record=record)


class Decimate(ResisticsProcess):
    def __init__(self, factor: int):
        if factor < 2:
            raise ValueError(f"Decimation factor {factor} must be greater than two")
        self.factor = factor

    def run(self, time_data: TimeData) -> TimeData:
        """
        Decimate TimeData
        """
        from scipy.signal import decimate

        factors = self._get_downsample_factors(self.factor)
        logger.info(f"Decimating by {self.factor} in {len(factors)} steps")
        messages = [f"Decimating by {self.factor} in {len(factors)} steps"]
        data = time_data.data
        for factor in factors:
            data = decimate(data, factor, axis=1, zero_phase=True)
            messages.append(f"Data decimated by factor of {factor}")
        # return new TimeData
        new_fs = time_data.fs / self.factor
        messages.append(f"Sampling frequency adjusted from {time_data.fs} to {new_fs}")
        n_samples = data.shape[1]
        headers = time_data.headers.copy()
        headers = adjust_headers_times(
            headers, new_fs, time_data.first_time, n_samples=n_samples
        )
        record = self._get_process_record(messages)
        return new_time_data(time_data, headers=headers, data=data, record=record)

    def _get_downsample_factors(self, n: int) -> List[int]:
        """Factorise a number to avoid too large a downsample factor

        logic: on the last value of f, val*f is tested
        if this is greater than 13, the previous val is added, which leaves one factor leftover
        if not greater than 13, then this is not added either
        so append the last value. the only situation in which this fails is the last factor itself is over 13.

        Parameters
        ----------
        number : int
            The number to factorise

        Returns
        -------
        List[int]
            The downsampling factors to use

        Notes
        -----
        There's a few pathological cases here that are being ignored. For example, what if the downsample factor is the product of two primes greater than 13.
        """
        from resistics.math import prime_factorisation

        if n <= 13:
            return [n]

        factors = prime_factorisation(n)
        downsamples = []
        val = 1
        for factor in factors:
            if val * factor > 13:
                downsamples.append(val)
                val = 1
            val *= factor
        downsamples.append(val)
        return downsamples


class InterpolateTimestamps(ResisticsProcess):
    def run(self, time_data: TimeData) -> TimeData:
        """Interpolate data to be on the second

        Interpolates the sampling so that it coincides with full seconds. The function also shifts the start point to the next full second. This function will truncate the data to the previous full second.

        .. warning::

            Do not use this method on data recording with a sampling frequency of less than 1Hz

        Parameters
        ----------
        data : Dict
            Dictionary with channel as keys and data as values
        sampleFreq : float
            Sampling frequency of the data
        startTime : datetime
            Time of first sample

        Returns
        -------
        data : Dict
            Dictionary with channel as keys and data as values
        """
        from resistics.math import round_up_time, round_down_time
        from scipy.interpolate import interp1d

        messages = []
        dt = pd.Timedelta(1 / time_data.fs, "s")
        first_up = round_up_time(time_data.first_time, dt)
        samples_up = (first_up - time_data.first_time) / dt
        if samples_up.is_integer():
            # sampling already correct
            return time_data
        last_down = round_down_time(time_data.last_time, dt)
        logger.info(f"Interpolating timestamps to between {first_up} and {last_down}")
        messages = [f"First time adjusted from {time_data.first_time} to {first_up}"]
        messages.append(f"Last time adjusted from {time_data.last_time} to {last_down}")
        new_x = pd.date_range(start=first_up, end=last_down, freq=dt).astype(int)
        n_samples = len(new_x)
        x = time_data.get_timestamps().astype(int)
        interp_fnc = interp1d(x, time_data.data, axis=1, copy=False)
        data = interp_fnc(new_x)
        headers = time_data.headers.copy()
        headers = adjust_headers_times(
            headers, time_data.fs, first_up, n_samples=n_samples, last_time=last_down
        )
        record = self._get_process_record(messages)
        return new_time_data(time_data, headers=headers, data=data, record=record)


class Join(ResisticsProcess):
    """
    Join together time data

    All time data passed must have the same sampling frequencies and only channels that are common across all time data will be merged.

    Gaps are filled with Nans. To fill in gaps, use InterpolateNans.

    Examples
    --------
    In the below example, three time data instances are joined together. Note that in the plot, offsets have been added to the data to make it easier to visualise what is happening.

    .. plot::
        :context: close-figs

        .. testsetup::

            >>> import matplotlib.pyplot as plt

        .. doctest::

            >>> from resistics.testing import time_data_ones
            >>> from resistics.time import Join, InterpolateNans
            >>> time_data1 = time_data_ones(fs = 0.1, first_time = "2020-01-01 00:00:00", n_samples=6)
            >>> time_data2 = time_data_ones(fs = 0.1, first_time = "2020-01-01 00:02:00", n_samples=6)
            >>> time_data3 = time_data_ones(fs = 0.1, first_time = "2020-01-01 00:04:00", n_samples=6)
            >>> joiner = Join()
            >>> joined = joiner.run(time_data1, time_data2, time_data3)
            >>> joined["Ex"]
            array([ 1.,  1.,  1.,  1.,  1.,  1., nan, nan, nan, nan, nan, nan,  1.,
                    1.,  1.,  1.,  1.,  1., nan, nan, nan, nan, nan, nan,  1.,  1.,
                    1.,  1.,  1.,  1.])
            >>> interpolater = InterpolateNans()
            >>> interpolated = interpolater.run(joined)
            >>> interpolated["Ex"]
            array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
            >>> fig = plt.figure()
            >>> p = plt.plot(time_data1.get_x(), time_data1["Ex"], "bx-", lw=2)
            >>> p = plt.plot(time_data2.get_x(), time_data2["Ex"] + 1, "co-", lw=2)
            >>> p = plt.plot(time_data3.get_x(), time_data3["Ex"] + 2, "gd-", lw=2)
            >>> p = plt.plot(joined.get_x(), joined["Ex"] + 3, "rs-", lw=2)
            >>> p = plt.plot(interpolated.get_x(), interpolated["Ex"] + 4, "kp", lw=2)
            >>> plt.yticks([1, 2, 3, 4, 5], ["Data1", "Data2", "Data3", "Joined", "Interpolated"]) # doctest: +SKIP
            >>> plt.tight_layout() # doctest: +SKIP
            >>> plt.show() # doctest: +SKIP
    """

    def run(self, *args: TimeData) -> TimeData:
        from resistics.common import get_process_record, histories_to_parameters
        from resistics.math import datetimes_to_samples

        fs = self._get_fs(args)
        chans = self._get_chans(args)
        logger.info(f"Joining data with sample frequency {fs} and channels {chans}")
        dt = pd.Timedelta(1 / fs, "s")
        first_times = [x.first_time for x in args]
        last_times = [x.last_time for x in args]
        first_time = min(first_times)
        last_time = max(last_times)
        n_samples = int((last_time - first_time) / dt) + 1
        data = np.empty(shape=(len(chans), n_samples))
        data[:] = np.nan

        # begin copying in the data
        messages = [f"Joining data between {first_time}, {last_time}"]
        messages.append(f"Sampling frequency {fs} Hz, channels {chans}")
        for time_data in args:
            from_sample, to_sample = datetimes_to_samples(
                fs, first_time, last_time, time_data.first_time, time_data.last_time
            )
            for idx, chan in enumerate(chans):
                data[idx, from_sample : to_sample + 1] = time_data[chan]
        # process record
        histories = [x.history for x in args]
        parameters = histories_to_parameters(histories)
        record = get_process_record(self.name, parameters, messages)
        headers = args[0].headers.copy()
        headers = adjust_headers_times(
            headers, fs, first_time, n_samples=n_samples, last_time=last_time
        )
        return TimeData(headers, chans, data, ProcessHistory([record]))

    def _get_fs(self, time_data_col: Collection[TimeData]) -> float:
        from resistics.errors import ProcessRunError

        fs = set([x.fs for x in time_data_col])
        if len(fs) != 1:
            raise ProcessRunError(self.name, f"> 1 sample frequency found {fs}")
        fs = list(fs)[0]
        return fs

    def _get_chans(self, time_data_col: Collection[TimeData]) -> List[str]:
        from resistics.errors import ProcessRunError

        chans_list = [set(x.chans) for x in time_data_col]
        chans = set.intersection(*chans_list)
        if len(chans) == 0:
            raise ProcessRunError(
                self.name, "Found no common channels amongst time data"
            )
        return sorted(list(chans))


class Calibrate(ResisticsProcess):
    def __init__(
        self,
        cal_folder: Path,
        cal_file_fnc: Callable,
        cal_readers: List[Type[CalibrationReader]],
    ):
        """
        Initialise

        Parameters
        ----------
        cal_folder : Path
            Folder with calibration files
        cal_file_fnc : Callable
            A function that can provide the calibration file give the time data headers and a channel
        cal_readers : List[Type[CalibrationReader]]
            The calibration data readers to use to try and read the calibration files
        """
        self.cal_folder = cal_folder
        self.cal_file_fnc = cal_file_fnc
        self.cal_readers = cal_readers

    def parameters(self) -> Dict[str, Any]:
        """
        Return the processor parameters

        Returns
        -------
        Dict[str, Any]
            The parameters
        """
        readers = [x.name for x in self.cal_readers]
        return {
            "cal_folder": str(self.cal_folder.absolute()),
            "cal_file_fnc": str(self.cal_file_fnc.__name__),
            "cal_readers": readers,
        }

    def run(self, time_data: TimeData) -> TimeData:
        """
        Calibrate TimeData

        Parameters
        ----------
        time_data : TimeData
            TimeData to calibrate

        Returns
        -------
        TimeData
            Calibrated data

        Raises
        ------
        ProcessRunError
            If the cal_file_fnc causes an exception
        """
        from resistics.errors import ProcessRunError

        logger.info("Calibration data")
        messages = []
        data = np.array(time_data.data)
        for chan in time_data:
            try:
                cal_file = self.cal_file_fnc(time_data.headers, chan)
            except:
                raise ProcessRunError(self.name, "Calibration function caused an error")
            if cal_file is None:
                logger.info(f"No calibration for channel {chan}")
                continue
            cal_path = self.cal_folder / cal_file
            if not cal_path.exists():
                logger.warning(f"{chan} calibration file, {cal_path}, does not exist")
            cal_data = self._get_cal_data(cal_path)
            if cal_data is None:
                logger.error(f"No reader was able to read {cal_path}. Continuing...")
                continue
            idx = time_data.get_chan_index(chan)
            data[idx, :] = self._calibrate(time_data.fs, data[idx, :], cal_data)
            messages.append(f"Calibrated {chan} with data from {cal_file}")
        record = self._get_process_record(messages)
        return new_time_data(time_data, data=data, record=record)

    def _get_cal_data(self, cal_path: Path) -> Union[CalibrationData, None]:
        """
        Get calibration data from the calibration file

        Parameters
        ----------
        cal_path : Path
            Path to the calibration file

        Returns
        -------
        Union[CalibrationData, None]
            CalibrationData if a read was successful, else None
        """
        for cal_reader in self.cal_readers:
            reader = cal_reader(cal_path)
            if reader.check():
                return reader.run()
            return None

    def _calibrate(
        self, fs: float, chan_data: np.ndarray, cal_data: CalibrationData
    ) -> np.ndarray:
        """
        Calibrate a channel

        This is essentially a deconvolution, which means a division in frequency domain.

        Parameters
        ----------
        fs : float
            The sampling frequency
        chan_data : np.ndarray
            Channel data
        cal_data : CalibrationData
            CalibrationData instance with calibration information

        Returns
        -------
        np.ndarray
            Calibrated data
        """
        from resistics.math import pad_to_power2, fft, ifft, frequency_array

        size = chan_data.size
        chan_data = np.pad(chan_data, (0, pad_to_power2(size)), "constant")
        fft_data = fft(chan_data)
        freqs = frequency_array(fs, fft_data.size)
        transfunc = self._interpolate(freqs, cal_data)
        fft_data = fft_data / transfunc["complex"].values
        inverse = ifft(fft_data, size)
        return inverse

    def _interpolate(
        self, freqs: np.ndarray, cal_data: CalibrationData
    ) -> pd.DataFrame:
        """
        Interpolate the calibration data to the same frequencies as the time data

        Parameters
        ----------
        freqs : np.ndarray
            The frequencies in the time data
        cal_data : CalibrationData
            The calibration data

        Returns
        -------
        pd.DataFrame
            The data interpolated to the frequencies and with an additional column, complex, which is the complex values for the magnitude and phase combinations.
        """
        combined_index = np.concatenate(cal_data.df.index.values, freqs)
        df = cal_data.df.set_index(combined_index)
        df = df.sort_index().interpolate(method="index").ffill().bfill()
        df = df.loc[freqs]
        df["complex"] = df["magnitude"].values * np.exp(1j * df["phase"].values)
        return df


class ApplyFunction(ResisticsProcess):
    """
    Apply a generic functions to the time data

    Best used with simple lambda functions to perform transformations on the data

    Examples
    --------
    .. doctest::

        >>> import numpy as np
        >>> from resistics.testing import time_data_ones
        >>> time_data = time_data
    """

    def __init__(self, fncs: Dict[str, Callable]):
        self.fncs = fncs

    def parameters(self) -> Dict[str, Any]:
        from resistics.common import array_to_string

        params = {}
        test = np.arange(3)
        input_vals = array_to_string(test, precision=2)
        for chan, fnc in self.fncs.items():
            output_vals = array_to_string(fnc(test), precision=2)
            chan_string = (
                f"Custom function with result [{output_vals}] on [{input_vals}]"
            )
            params[chan] = chan_string
        return params

    def check(self) -> bool:
        test = np.arange(10)
        for chan, fnc in self.fncs.items():
            try:
                fnc(test)
            except:
                logger.error(
                    f"Function for {chan} failed on numeric data. Check failed"
                )
                return False
        return True

    def run(self, time_data: TimeData) -> TimeData:
        logger.info(f"Applying custom functions to channels {list(self.fncs.keys())}")
        messages = []
        data = np.empty(shape=time_data.data.shape)
        for chan, fnc in self.fncs.items():
            if chan in time_data:
                messages.append(f"Applying custom function to {chan}")
                idx = time_data.get_chan_index(chan)
                data[idx] = fnc(time_data[chan])
        record = self._get_process_record(messages)
        return new_time_data(time_data, data=data, record=record)
