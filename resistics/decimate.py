"""
Module for time data decimation including classes and for the following

- Definition of DecimationParameters
- Performing decimation on time data
"""
from loguru import logger
from typing import Any, Optional, Tuple, Union, Dict, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from resistics.common import ResisticsData, ResisticsProcess, ProcessHistory
from resistics.time import TimeData


class DecimationParameters(ResisticsData):
    """
    Decimation parameters

    Examples
    --------
    >>> from resistics.decimate import DecimationSetup
    >>> dec_setup = DecimationSetup(n_levels=3, per_level=2)
    >>> dec_params = dec_setup.run(128)
    >>> type(dec_params)
    <class 'resistics.decimate.DecimationParameters'>
    >>> dec_params
    <class 'resistics.decimate.DecimationParameters'>
                        0          1     fs  total_factors  incremental_factors
    Decimation level
    0                 32.0  22.627417  128.0              1                    1
    1                 16.0  11.313708   64.0              2                    2
    2                  8.0   5.656854   32.0              4                    2
    >>> dec_params[2]
    0                       8.000000
    1                       5.656854
    fs                     32.000000
    total_factors           4.000000
    incremental_factors     2.000000
    Name: 2, dtype: float64
    >>> dec_params[2,1]
    5.65685424949238
    >>> dec_params.get_total_factor(2)
    4
    >>> dec_params.get_incremental_factor(2)
    2
    """

    def __init__(
        self,
        fs: float,
        n_levels: int,
        per_level: int,
        min_samples: int,
        eval_df: pd.DataFrame,
        history: Optional[ProcessHistory] = None,
    ):
        """
        Initialise DecimationParameters

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
        history : Optional[ProcessHistory], optional
            Process history, by default None
        """
        self.fs = fs
        self.n_levels = n_levels
        self.per_level = per_level
        self.min_samples = min_samples
        self.eval_df = eval_df
        self.history = history if history is not None else ProcessHistory()

    def __getitem__(self, args: Union[int, Tuple[int, int]]) -> Union[pd.Series, float]:
        """
        Get a whole decimation level or an evaluation frequency

        Parameters
        ----------
        args : Union[int, Tuple[int, int]]
            Input arguments, can either be a single argument specifying a level,
            or two integer arguments specifying a level and an evaluation
            frequency index

        Returns
        -------
        Union[pd.Series, float]
            Series if a whole level is requested, otherwise a float when
            returning an evaluation frequency

        Raises
        ------
        ValueError
            If arguments are incorrectly specified
        """
        if isinstance(args, int):
            return self.get_level(args)
        elif isinstance(args, tuple) and len(args) == 2:
            return self.get_eval_freq(args[0], args[1])
        else:
            raise ValueError(f"Arguments {args} incorrectly specified")

    def check_level(self, level: int):
        """Check level"""
        if level < 0 or level >= self.n_levels:
            raise ValueError(f"Level {level} not 0 <= level < {self.n_levels}")

    def check_eval_idx(self, idx: int):
        """Check evaluation frequency index"""
        if idx < 0 or idx >= self.per_level:
            raise ValueError(f"Index {idx} not 0 <= index < {self.per_level}")

    def get_level(self, level: int) -> pd.Series:
        """
        Get level series

        Parameters
        ----------
        level : int
            The level

        Returns
        -------
        pd.Series
            Information about the level as a pd.Series
        """
        self.check_level(level)
        return self.eval_df.loc[level, :]

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
        return self.eval_df.loc[level, "fs"]

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
        return int(self.eval_df.loc[level, "total_factors"])

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
        return int(self.eval_df.loc[level, "incremental_factors"])

    def get_eval_freqs(self, level: int) -> pd.Series:
        """
        Get the evaluation frequencies for a level

        Parameters
        ----------
        level : int
            The level

        Returns
        -------
        pd.Series
            The evaluation frequencies
        """
        self.check_level(level)
        return self.eval_df.loc[level, list(range(self.per_level))]

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
        return self.eval_df.loc[level, idx]

    def to_string(self) -> str:
        """
        Decimation parameters as a string

        Returns
        -------
        str
            String description of decimation parameters
        """
        outstr = f"{self.type_to_string()}\n"
        outstr += self.eval_df.to_string()
        return outstr


class DecimationSetup(ResisticsProcess):
    """
    Process to calculate decimation parameters

    .. note::

        Note that the running check for DecimationSetup always returns True

    Examples
    --------
    >>> from resistics.decimate import DecimationSetup
    >>> dec_setup = DecimationSetup(n_levels=3, per_level=2)
    >>> dec_params = dec_setup.run(128)
    >>> print(dec_params)
    <class 'resistics.decimate.DecimationParameters'>
                         0          1     fs  total_factors  incremental_factors
    Decimation level
    0                 32.0  22.627417  128.0              1                    1
    1                 16.0  11.313708   64.0              2                    2
    2                  8.0   5.656854   32.0              4                    2
    """

    def __init__(
        self,
        n_levels: int = 8,
        per_level: int = 5,
        min_samples: int = 256,
        eval_freqs: Optional[np.ndarray] = None,
        div_factor: int = 2,
    ):
        """
        Get decimation paramters

        Parameters
        ----------
        n_levels : int, optional
            Number of decimation levels, by default 8
        per_level : int, optional
            Number of frequencies per level, by default 5
        min_samples : int, optional
            Number of samples to under which to quit decimating
        eval_freqs : Optional[np.ndarray], optional
            Explicit definition of evaluation frequencies as a flat array, by
            default None. Must be of size n_levels  * per_level
        div_factor : int, optional
            Minimum division factor for decimation, by default 2.
        """
        self.n_levels = n_levels
        self.per_level = per_level
        self.min_samples = min_samples
        self.eval_freqs = eval_freqs
        self.div_factor = div_factor

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
        return DecimationParameters(
            fs,
            self.n_levels,
            self.per_level,
            self.min_samples,
            self._get_decimation_parameters(fs, eval_freqs),
            ProcessHistory([self._get_process_record("Setup decimation parameters")]),
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
        eval_freqs = self.eval_freqs
        n_freqs = self.n_levels * self.per_level
        if eval_freqs is not None and eval_freqs.size != n_freqs:
            raise ValueError(f"Size eval freqs {eval_freqs.size} != n_freqs {n_freqs}")
        elif eval_freqs is not None:
            if np.any(eval_freqs > nyquist):
                raise ValueError(f"Found frequencies {eval_freqs} > nyquist {nyquist}")
        else:
            eval_freqs = get_eval_freqs(fs, n_freqs=n_freqs)
        return eval_freqs

    def _get_decimation_parameters(
        self, fs: float, eval_freqs: np.ndarray
    ) -> pd.DataFrame:
        """
        Calculate decimation parameters from evaluation frequencies

        Uses evaluation frequencies, number levels and number of evaluation
        frequencies per level to calculate fs for each decimation level and
        decimation factor.

        Parameters
        ----------
        fs : float
            Sampling frequency, Hz
        eval_freqs : np.ndarray
            Evaluation frequency as flat array

        Returns
        -------
        pd.DataFrame
            DataFrame with decimation parameters
        """
        from resistics.math import intdiv

        eval_freqs = eval_freqs.reshape(self.n_levels, self.per_level)
        params = pd.DataFrame(data=eval_freqs)
        params.index.name = "Decimation level"
        # add the decimation factors
        levels_fs = []
        levels_factors = []
        levels_increment = []
        for il in range(0, self.n_levels):
            level_fs, factor = self._get_level_params(fs, params.loc[il, 0])
            levels_fs.append(level_fs)
            levels_factors.append(factor)
            if il == 0:
                levels_increment.append(intdiv(fs, level_fs))
            else:
                prev_factor = levels_factors[-2]
                levels_increment.append(intdiv(factor, prev_factor))
        params["fs"] = levels_fs
        params["total_factors"] = levels_factors
        params["incremental_factors"] = levels_increment
        return params

    def _get_level_params(self, fs: float, f_max: float) -> Tuple[float, int]:
        """
        Get sampling frequency and decimation factor given the largest
        evaluation in the decimation level

        Parameters
        ----------
        fs : float
            Sampling frequency, Hz
        f_max : float
            Maximum evaluation frequency in the decimation level

        Returns
        -------
        Tuple[float, int]
            Sampling frequency, Hz of level and decimation factor for level
        """
        target_fs = f_max * 4
        factor = 1
        level_fs = fs
        while True:
            test_fs = level_fs / self.div_factor
            if test_fs >= target_fs:
                level_fs = test_fs
                factor *= self.div_factor
            else:
                break
        return level_fs, factor


class DecimatedData(ResisticsData):
    """
    Data class for storing decimated data

    Examples
    --------
    .. plot::
        :width: 90%

        >>> import matplotlib.pyplot as plt
        >>> from resistics.testing import time_data_random
        >>> from resistics.decimate import DecimationSetup, Decimator
        >>> time_data = time_data_random(fs=256, n_samples=10_000)
        >>> dec_setup = DecimationSetup()
        >>> dec_params = dec_setup.run(time_data.fs)
        >>> decimator = Decimator(dec_params)
        >>> decimator.check()
        True
        >>> dec_data = decimator.run(time_data)
        >>> dec_data.summary()
        ##---Begin Summary----------------------------------
        <class 'resistics.decimate.DecimatedData'>
                  fs        dt  n_samples           first_time                     last_time
        level
        0      256.0  0.003906      10000  2020-01-01 00:00:00  2020-01-01 00:00:39.05859375
        1       64.0  0.015625       2500  2020-01-01 00:00:00    2020-01-01 00:00:39.046875
        2        8.0  0.125000        313  2020-01-01 00:00:00           2020-01-01 00:00:39
        ##---End summary------------------------------------
        >>> for ilevel in range(0, dec_data.max_level + 1):
        ...     time_data = dec_data.get_level(ilevel)
        ...     plt.plot(time_data.get_x(), time_data["Hx"], label=f"Level{ilevel}") # doctest: +SKIP
        >>> plt.legend(loc=3) # doctest: +SKIP
        >>> plt.tight_layout() # doctest: +SKIP
        >>> plt.show() # doctest: +SKIP
    """

    def __init__(
        self,
        dec_params: DecimationParameters,
        chans: List[str],
        data: Dict[int, TimeData],
        history: ProcessHistory,
    ):
        """
        Decimated data

        Parameters
        ----------
        dec_params : DecimationParameters
            The decimation parameters used
        chans : List[str]
            The channels in the data
        data : Dict[int, TimeData]
            The data as a dictionary of level indices to TimeData
        history : ProcessHistory
            The process history
        """
        self.dec_params = dec_params
        self.chans = chans
        self.data = data
        self.history = history
        self.max_level = max(list(self.data.keys()))
        self.n_levels = self.max_level + 1

    def get_level(self, level: int) -> TimeData:
        """
        Get TimeData for a level

        Parameters
        ----------
        level : int
            The level

        Returns
        -------
        TimeData
            TimeData for the decimation level

        Raises
        ------
        ValueError
            If level > max_level
        """
        if level > self.max_level:
            raise ValueError(f"Level {level} not <= max {self.max_level}")
        return self.data[level]

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
        if len(self.data) == 0:
            logger.error("Data is empty, no decimation levels to plot")
        fig = self.data[0].plot(max_pts=max_pts, label_prefix="Level 0")
        for ilevel in range(1, self.max_level + 1):
            self.data[ilevel].plot(fig=fig, label_prefix=f"Level {ilevel}")
        return fig

    def to_string(self) -> str:
        """
        String detailing class info

        Returns
        -------
        str
            Class info as string
        """
        outstr = f"{self.type_to_string()}\n"
        data = [
            [ilevel, x.fs, x.dt, x.n_samples, x.first_time, x.last_time]
            for ilevel, x in self.data.items()
        ]
        df = pd.DataFrame(
            data=data,
            columns=["level", "fs", "dt", "n_samples", "first_time", "last_time"],
        )
        df = df.set_index("level")
        outstr += df.to_string()
        return outstr


class Decimator(ResisticsProcess):
    def __init__(self, dec_params: DecimationParameters):
        """
        Initialise decimator with DecimationParameters

        Parameters
        ----------
        dec_params : DecimationParameters
            Decimation parameters
        """
        self.dec_params = dec_params

    def parameters(self) -> Dict[str, Any]:
        """
        Get process parameters

        Returns
        -------
        Dict[str, Any]
            The process parameters
        """
        return {
            "fs": self.dec_params.fs,
            "n_levels": self.dec_params.n_levels,
            "per_level": self.dec_params.per_level,
            "min_samples": self.dec_params.min_samples,
        }

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
        data = {}
        chans = time_data.chans
        history = time_data.history.copy()
        messages = []
        for ilevel in range(0, self.dec_params.n_levels):
            logger.info(f"Decimating level {ilevel}")
            factor = self.dec_params.get_incremental_factor(ilevel)
            fs = self.dec_params.get_fs(ilevel)
            time_data_new = self._decimate(time_data.copy(), factor)
            if time_data_new.n_samples < self.dec_params.min_samples:
                logger.warning(f"n_samples < min allowed {self.dec_params.min_samples}")
                break
            data[ilevel] = time_data_new
            time_data = time_data_new
            messages.append(f"Decimated level {ilevel}, inc. factor {factor}, fs {fs}")
        completed = list(range(len(data)))
        target = list(range(self.dec_params.n_levels))
        messages.append(f"Completed levels {completed} out of {target}")
        history.add_record(self._get_process_record(messages))
        return DecimatedData(self.dec_params, chans, data, history)

    def _decimate(self, time_data: TimeData, factor: int) -> TimeData:
        """
        Decimate TimeData

        Parameters
        ----------
        time_data : TimeData
            TimeData to decimate
        factor : int
            Decimation factor

        Returns
        -------
        TimeData
            Decimated TimeData
        """
        from resistics.time import Decimate

        if factor == 1:
            return time_data
        dec = Decimate(factor)
        return dec.run(time_data)
