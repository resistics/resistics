"""
Module for time data decimation including classes and for the following

- Definition of DecimationParameters
- Performing decimation on time data
"""
from loguru import logger
from typing import Any, Optional, Tuple, Union, Dict, List
from pathlib import Path
from pydantic import validator
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from resistics.sampling import HighResDateTime
from resistics.common import ResisticsProcess, ResisticsModel, ResisticsData
from resistics.common import ResisticsWriter, History, Metadata, WriteableMetadata
from resistics.time import TimeData


class DecimationParameters(ResisticsModel):
    """
    Decimation parameters

    Parameters
    ----------
    fs : float
        Sampling frequency Hz
    n_levels : int
        Number of levels
    per_level : int
        Evaluation frequencies per level
    min_samples : int
        Number of samples to under which to quit decimating
    eval_df : pd.DataFrame
        The DataFrame with the decimation information

    Examples
    --------
    >>> from resistics.decimate import DecimationSetup
    >>> dec_setup = DecimationSetup(n_levels=3, per_level=2)
    >>> dec_params = dec_setup.run(128)
    >>> type(dec_params)
    <class 'resistics.decimate.DecimationParameters'>
    >>> print(dec_params.to_dataframe())
                         0          1     fs  factors  increments
    decimation level
    0                 32.0  22.627417  128.0        1           1
    1                 16.0  11.313708   64.0        2           2
    2                  8.0   5.656854   32.0        4           2
    >>> dec_params[2]
    [8.0, 5.65685424949238]
    >>> dec_params[2,1]
    5.65685424949238
    >>> dec_params.get_total_factor(2)
    4
    >>> dec_params.get_incremental_factor(2)
    2
    """

    fs: float
    n_levels: int
    per_level: int
    min_samples: int
    eval_freqs: List[float]
    dec_factors: List[int]
    dec_increments: Optional[List[int]] = None
    dec_fs: Optional[List[float]] = None

    @validator("dec_increments", always=True)
    def set_dec_increments(cls, value, values):
        """Initialise decimation increments if not provided"""
        if value is None:
            divisor = np.ones(shape=(values["n_levels"]), dtype=int)
            divisor[1:] = values["dec_factors"][:-1]
            return np.divide(values["dec_factors"], divisor).astype(int).tolist()
        return value

    @validator("dec_fs", always=True)
    def set_dec_fs(cls, value, values):
        """Initialise decimation sampling frequencies if not provided"""
        if value is None:
            factors = np.array(values["dec_factors"]).astype(float)
            return (values["fs"] * np.reciprocal(factors)).tolist()
        return value

    def __getitem__(self, args: Union[int, Tuple[int, int]]):
        """Get the evaluation frequency for level and evaluation frequency index"""
        if isinstance(args, int):
            return self.get_eval_freqs(args)
        level, idx = args
        return self.get_eval_freq(level, idx)

    def check_level(self, level: int):
        """Check level"""
        if level < 0 or level >= self.n_levels:
            raise ValueError(f"Level {level} not 0 <= level < {self.n_levels}")

    def check_eval_idx(self, idx: int):
        """Check evaluation frequency index"""
        if idx < 0 or idx >= self.per_level:
            raise ValueError(f"Index {idx} not 0 <= index < {self.per_level}")

    def get_eval_freqs(self, level: int) -> List[float]:
        """
        Get the evaluation frequencies for a level

        Parameters
        ----------
        level : int
            The decimation level

        Returns
        -------
        List[float]
            List of evaluation frequencies
        """
        self.check_level(level)
        index_from = self.per_level * level
        return self.eval_freqs[index_from : index_from + self.per_level]

    def get_eval_freq(self, level: int, idx: int) -> float:
        """
        Get an evaluation frequency

        Parameters
        ----------
        level : int
            The level
        idx : int
            Evaluation frequency index

        Returns
        -------
        float
            The evaluation frequency
        """
        self.check_level(level)
        self.check_eval_idx(idx)
        idx = self.per_level * level + idx
        return self.eval_freqs[idx]

    def get_fs(self, level: int) -> float:
        """
        Get sampling frequency for level

        Parameters
        ----------
        level : int
            The decimation level

        Returns
        -------
        float
            Sampling frequency Hz
        """
        self.check_level(level)
        return self.dec_fs[level]

    def get_total_factor(self, level: int) -> int:
        """
        Get total decimation factor for a level

        Parameters
        ----------
        level : int
            The level

        Returns
        -------
        int
            The decimation factor
        """
        self.check_level(level)
        return self.dec_factors[level]

    def get_incremental_factor(self, level: int) -> int:
        """
        Get incremental decimation factor

        Parameters
        ----------
        level : int
            The level

        Returns
        -------
        int
            The incremental decimation factor
        """
        self.check_level(level)
        return self.dec_increments[level]

    def to_numpy(self) -> np.ndarray:
        """Get evaluation frequencies as a 2-D array"""
        return np.array(self.eval_freqs).reshape(self.n_levels, self.per_level)

    def to_dataframe(self) -> pd.DataFrame:
        """Provide decimation parameters as DataFrame"""
        df = pd.DataFrame(data=self.to_numpy())
        df["fs"] = self.dec_fs
        df["factors"] = self.dec_factors
        df["increments"] = self.dec_increments
        df.index.name = "decimation level"
        return df


class DecimationSetup(ResisticsProcess):
    """
    Process to calculate decimation parameters

    Parameters
    ----------
    n_levels : int, optional
        Number of decimation levels, by default 8
    per_level : int, optional
        Number of frequencies per level, by default 5
    min_samples : int, optional
        Number of samples to under which to quit decimating
    div_factor : int, optional
        Minimum division factor for decimation, by default 2.
    eval_freqs : Optional[List[float]], optional
        Explicit definition of evaluation frequencies as a flat list, by
        default None. Must be of size n_levels * per_level

    Examples
    --------
    >>> from resistics.decimate import DecimationSetup
    >>> dec_setup = DecimationSetup(n_levels=3, per_level=2)
    >>> dec_params = dec_setup.run(128)
    >>> print(dec_params.to_dataframe())
                         0          1     fs  factors  increments
    decimation level
    0                 32.0  22.627417  128.0        1           1
    1                 16.0  11.313708   64.0        2           2
    2                  8.0   5.656854   32.0        4           2
    """

    n_levels: int = 8
    per_level: int = 5
    min_samples: int = 256
    div_factor: int = 2
    eval_freqs: Optional[List[float]] = None

    def run(self, fs: float) -> DecimationParameters:
        """
        Run DecimationSetup

        Parameters
        ----------
        fs : float
            Sampling frequency, Hz

        Returns
        -------
        DecimationParameters
            Decimation parameterisation
        """
        eval_freqs = self._get_eval_freqs(fs)
        factors = self._get_decimation_increments(fs, eval_freqs)
        return DecimationParameters(
            fs=fs,
            n_levels=self.n_levels,
            per_level=self.per_level,
            min_samples=self.min_samples,
            eval_freqs=eval_freqs.tolist(),
            dec_factors=factors,
        )

    def _get_eval_freqs(self, fs: float) -> np.ndarray:
        """
        Get evaluation frequencies

        Parameters
        ----------
        fs : float
            Sampling frequency, Hz

        Returns
        -------
        np.ndarray
            Evaluation frequencies as a flat array

        Raises
        ------
        ValueError
            If size of evaluation frequencies != n_levels * per_level
        ValueError
            If any evaluation frequency is > nyquist
        """
        from resistics.math import get_eval_freqs

        nyquist = fs / 2
        n_freqs = self.n_levels * self.per_level
        if self.eval_freqs is None:
            return get_eval_freqs(fs, n_freqs=n_freqs)
        eval_freqs = np.array(self.eval_freqs)
        if eval_freqs.size != n_freqs:
            raise ValueError(f"Size eval freqs {eval_freqs.size} != n_freqs {n_freqs}")
        if np.any(eval_freqs > nyquist):
            raise ValueError(f"Found frequencies {eval_freqs} > nyquist {nyquist}")
        return eval_freqs

    def _get_decimation_increments(
        self, fs: float, eval_freqs: np.ndarray
    ) -> List[int]:
        """
        Calculate decimation factors

        Uses evaluation frequencies, number levels and number of evaluation
        frequencies per level to calculate decimation factors with reference to
        the original sampling frequency.

        Parameters
        ----------
        fs : float
            Sampling frequency, Hz
        eval_freqs : np.ndarray
            Evaluation frequency as flat array

        Returns
        -------
        increments : List[int]
            The decimation factors referenced to the sampling frequency
        """
        eval_freqs = eval_freqs.reshape(self.n_levels, self.per_level)
        increments = []
        for ilevel in range(0, self.n_levels):
            factor = self._get_decimation_factor(fs, eval_freqs[ilevel, 0])
            increments.append(factor)
        return increments

    def _get_decimation_factor(self, fs: float, f_max: float) -> int:
        """
        Get decimation factor

        Given the maximum evaluation frequency in the decimation level, the
        method aims to find a suitable sampling frequency that satisfies

        - level sampling frequency >= 4 * max level evaluation frequency,

        and returns the decimation factor for this suitable sampling frequency

        Parameters
        ----------
        fs : float
            Sampling frequency, Hz
        f_max : float
            Maximum evaluation frequency in the decimation level

        Returns
        -------
        int
            The decimation factor
        """
        target_fs = f_max * 4
        factor = 1
        while True:
            test_fs = fs / self.div_factor
            if test_fs >= target_fs:
                fs = test_fs
                factor *= self.div_factor
            else:
                break
        return factor


class DecimatedLevelMetadata(Metadata):
    """Metadata for a decimation level"""

    fs: float
    n_samples: int
    first_time: HighResDateTime
    last_time: HighResDateTime

    @property
    def dt(self):
        return 1 / self.fs


class DecimatedMetadata(WriteableMetadata):
    """Metadata for DecimatedData"""

    fs: List[float]
    chans: List[str]
    n_chans: Optional[int] = None
    n_levels: int
    system: str = ""
    wgs84_latitude: float = -999.0
    wgs84_longitude: float = -999.0
    easting: float = -999.0
    northing: float = -999.0
    elevation: float = -999.0
    levels_metadata: List[DecimatedLevelMetadata] = []
    history: History = History()

    class Config:

        extra = "ignore"


class DecimatedData(ResisticsData):
    """
    Data class for storing decimated data

    Parameters
    ----------
    metadata : DecimatedMetadata
        The metadata
    data : Dict[int, TimeData]
        The decimated time data

    Examples
    --------
    .. plot::
        :width: 90%

        >>> import matplotlib.pyplot as plt
        >>> from resistics.testing import time_data_random
        >>> from resistics.decimate import DecimationSetup, Decimator
        >>> time_data = time_data_random(fs=256, n_samples=10_000)
        >>> dec_params = DecimationSetup(n_levels=4, per_freq=4).run(time_data.metadata.fs)
        >>> decimator = Decimator(**dec_params.dict())
        >>> dec_data = decimator.run(time_data)
        >>> dec_data.metadata.summary()
        {
            'file_info': None,
            'fs': [256.0, 64.0, 8.0],
            'chans': ['Ex', 'Ey', 'Hx', 'Hy'],
            'n_chans': 4,
            'n_levels': 3,
            'system': '',
            'wgs84_latitude': -999.0,
            'wgs84_longitude': -999.0,
            'easting': -999.0,
            'northing': -999.0,
            'elevation': -999.0,
            'levels_metadata': [
                {
                    'fs': 256.0,
                    'n_samples': 10000,
                    'first_time': '2020-01-01 00:00:00.000000_000000_000000_000000',
                    'last_time': '2020-01-01 00:00:39.058593_750000_000000_000000'
                },
                {
                    'fs': 64.0,
                    'n_samples': 2500,
                    'first_time': '2020-01-01 00:00:00.000000_000000_000000_000000',
                    'last_time': '2020-01-01 00:00:39.046875_000000_000000_000000'
                },
                {
                    'fs': 8.0,
                    'n_samples': 313,
                    'first_time': '2020-01-01 00:00:00.000000_000000_000000_000000',
                    'last_time': '2020-01-01 00:00:39.000000_000000_000000_000000'
                }
            ],
            'history': {
                'records': [
                    {
                        'time_local': '...',
                        'time_utc': '...',
                        'creator': {
                            'name': 'time_data_random',
                            'fs': 256,
                            'first_time': '2020-01-01 00:00:00',
                            'n_samples': 10000
                        },
                        'messages': ['Generated time data with random values'],
                        'record_type': 'process'
                    },
                    {
                        'time_local': '...',
                        'time_utc': '...',
                        'creator': {
                            'name': 'Decimator',
                            'n_levels': 4,
                            'min_samples': 256,
                            'dec_fs': [256.0, 64.0, 8.0, 2.0],
                            'dec_increments': [1, 4, 8, 4]
                        },
                        'messages': [
                            'Decimated level 0, inc. factor 1, fs 256.0',
                            'Decimated level 1, inc. factor 4, fs 64.0',
                            'Decimated level 2, inc. factor 8, fs 8.0',
                            'Completed levels [0, 1, 2] out of [0, 1, 2, 3]'
                        ],
                        'record_type': 'process'
                    }
                ]
            }
        }
        >>> for ilevel in range(0, dec_data.metadata.n_levels):
        ...     data = dec_data.get_level(ilevel)
        ...     plt.plot(dec_data.get_x(ilevel), data[0], label=f"Level{ilevel}") # doctest: +SKIP
        >>> plt.legend(loc=3) # doctest: +SKIP
        >>> plt.tight_layout() # doctest: +SKIP
        >>> plt.show() # doctest: +SKIP
    """

    def __init__(self, metadata: DecimatedMetadata, data: Dict[int, np.ndarray]):
        """Initialise decimated data"""
        self.metadata = metadata
        self.data = data

    def get_level(self, level: int) -> np.ndarray:
        """
        Get data for a decimation level

        Parameters
        ----------
        level : int
            The level

        Returns
        -------
        np.ndarary
            The data for the decimation level

        Raises
        ------
        ValueError
            If level > max_level
        """
        if level >= self.metadata.n_levels:
            raise ValueError(f"Level {level} not <= max {self.metadata.n_levels}")
        return self.data[level]

    def get_timestamps(
        self, level: int, samples: Optional[np.ndarray] = None, estimate: bool = True
    ) -> Union[np.ndarray, pd.DatetimeIndex]:
        """
        Get an array of timestamps

        Parameters
        ----------
        level : int
            The decimation level
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

        Raises
        ------
        ValueError
            If the level is not within bounds
        """
        from resistics.sampling import datetime_array, datetime_array_estimate

        if level >= self.metadata.n_levels:
            raise ValueError(f"Level {level} not <= max {self.metadata.n_levels}")
        fnc = datetime_array_estimate if estimate else datetime_array
        level_metadata = self.metadata.levels_metadata[level]
        if samples is not None:
            return fnc(level_metadata.first_time, level_metadata.fs, samples=samples)
        return fnc(
            level_metadata.first_time,
            level_metadata.fs,
            n_samples=level_metadata.n_samples,
        )

    def plot(self, max_pts: int = 10_000) -> go.Figure:
        """
        Plot the decimated data

        Parameters
        ----------
        max_pts : int, optional
            Maximum number of points to plot, by default 10_000

        Returns
        -------
        go.Figure
            Plotly Figure
        """
        from resistics.time import TimeMetadata

        if len(self.data) == 0:
            logger.error("Data is empty, no decimation levels to plot")

        fig = None
        for ilevel in range(0, self.metadata.n_levels):
            logger.info(f"Plotting decimation level {ilevel}")
            metadata_dict = self.metadata.levels_metadata[ilevel].dict()
            metadata_dict["chans"] = self.metadata.chans
            metadata_dict["chans_metadata"] = []
            time_data = TimeData(TimeMetadata(**metadata_dict), self.data[ilevel])
            if fig is None:
                fig = time_data.plot(label_prefix=f"Level {ilevel}")
            else:
                time_data.plot(fig=fig, label_prefix=f"Level {ilevel}")
        return fig

    def x_size(self, level: int) -> int:
        """
        For abstract plotting functions, return the size

        Parameters
        ----------
        level : int
            The decimation level

        Returns
        -------
        int
            The x size, equal to the number of samples
        """
        return self.metadata.levels_metadata[level].n_samples

    def get_x(
        self, level: int, samples: Optional[np.ndarray] = None
    ) -> pd.DatetimeIndex:
        """
        For plotting, get x dimension, in this case times

        Parameters
        ----------
        level : int
            The decimation level
        samples : Union[np.ndarray, None], optional
            If provided, x values (timestamps) are only returned for the
            specified samples, by default None

        Returns
        -------
        pd.DatetimeIndex
            Timestamp array
        """
        return self.get_timestamps(level, samples=samples, estimate=True)

    def to_string(self) -> str:
        """Class details as a string"""
        outstr = f"{self.type_to_string()}\n"
        data = [
            [ilevel, x.fs, x.dt, x.n_samples, x.first_time, x.last_time]
            for ilevel, x in enumerate(self.metadata.levels_metadata)
        ]
        df = pd.DataFrame(
            data=data,
            columns=["level", "fs", "dt", "n_samples", "first_time", "last_time"],
        )
        df = df.set_index("level")
        outstr += df.to_string()
        return outstr


class Decimator(ResisticsProcess):

    n_levels: int
    min_samples: int
    dec_fs: List[float]
    dec_increments: List[int]

    class Config:

        extra = "ignore"

    def run(self, time_data: TimeData) -> DecimatedData:
        """
        Decimate the TimeData

        Parameters
        ----------
        time_data : TimeData
            TimeData to decimate

        Returns
        -------
        DecimatedData
            DecimatedData instance with all the decimated data
        """
        metadata = time_data.metadata.dict()
        data = {}
        levels_metadata = []
        messages = []
        for ilevel in range(0, self.n_levels):
            factor = self.dec_increments[ilevel]
            logger.info(f"Decimating level {ilevel} with factor {factor}")
            time_data_new = self._decimate(time_data.copy(), factor)
            if time_data_new.metadata.n_samples < self.min_samples:
                logger.warning(f"n_samples < min allowed {self.min_samples}")
                break
            data[ilevel] = time_data_new.data
            time_data = time_data_new
            fs = self.dec_fs[ilevel]
            levels_metadata.append(DecimatedLevelMetadata(**time_data.metadata.dict()))
            messages.append(f"Decimated level {ilevel}, inc. factor {factor}, fs {fs}")
        completed = list(range(len(data)))
        target = list(range(self.n_levels))
        messages.append(f"Completed levels {completed} out of {target}")
        metadata = self._get_metadata(metadata, levels_metadata)
        metadata.history.add_record(self._get_record(messages))
        return DecimatedData(metadata, data)

    def _decimate(self, time_data: TimeData, factor: int) -> TimeData:
        """Decimate time data"""
        from resistics.time import Decimate

        if factor == 1:
            return time_data
        return Decimate(factor=factor).run(time_data)

    def _get_metadata(
        self,
        metadata_dict: Dict[str, Any],
        levels_metadata: List[DecimatedLevelMetadata],
    ) -> DecimatedMetadata:
        """Get the metadata for the decimated data"""
        metadata_dict["fs"] = [x.fs for x in levels_metadata]
        metadata_dict["n_levels"] = len(levels_metadata)
        metadata_dict["levels_metadata"] = levels_metadata
        return DecimatedMetadata(**metadata_dict)


class DecimatedDataWriter(ResisticsWriter):
    """Writer of resistics decimated data"""

    def run(self, dir_path: Path, dec_data: DecimatedData) -> None:
        """
        Write out DecimatedData

        Parameters
        ----------
        dir_path : Path
            The directory path to write to
        dec_data : DecimatedData
            Decimated data to write out

        Raises
        ------
        WriteError
            If unable to write to the directory
        """
        from resistics.errors import WriteError

        if not self._check_dir(dir_path):
            WriteError(dir_path, "Unable to write to directory, check logs")
        logger.info(f"Writing decimated data to {dir_path}")
        metadata_path = dir_path / "metadata.json"
        metadata = dec_data.metadata.copy()
        for ilevel in range(dec_data.metadata.n_levels):
            level_path = dir_path / f"level_{ilevel:03d}.npy"
            np.save(level_path, dec_data.get_level(ilevel))
        metadata.history.add_record(self._get_record(dir_path, type(dec_data)))
        metadata.write(metadata_path)


class DecimatedDataReader(ResisticsProcess):
    """Reader of resistics decimated data"""

    def run(self, dir_path: Path) -> DecimatedData:
        """
        Read DecimatedData

        Parameters
        ----------
        dir_path : Path
            The directory path to read from

        Returns
        -------
        DecimatedData
            The decimated data

        Raises
        ------
        ReadError
            If the directory does not exist
        """
        from resistics.errors import ReadError

        if not dir_path.exists():
            raise ReadError(dir_path, "Directory does not exist")
        logger.info(f"Reading decimated data from {dir_path}")
        metadata_path = dir_path / "metadata.json"
        metadata = DecimatedMetadata.parse_file(metadata_path)
        data = {}
        for ilevel in range(metadata.n_levels):
            level_path = dir_path / f"level_{ilevel:03d}.npy"
            data[ilevel] = np.load(level_path)
        messages = [f"Decimated data read from {dir_path}"]
        metadata.history.add_record(self._get_record(messages))
        return DecimatedData(metadata, data)
