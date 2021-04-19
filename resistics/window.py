"""
Module for calculating window related data. This includes:

- Windowing utility functions such as converting from global indices to datetime
- Converting a global index array to datetime
"""
from typing import Optional, List, Tuple
from logging import getLogger
import pandas as pd

from resistics.common import ResisticsData, ResisticsProcess
from resistics.sampling import RSDateTime, RSTimeDelta
from resistics.time import TimeData
from resistics.decimate import DecimationParameters

logger = getLogger(__name__)


def window_duration(window_size: int, fs: float) -> RSTimeDelta:
    """
    Get the window duration

    Parameters
    ----------
    window_size : int
        Window size in samples
    fs : float
        Sampling frequency Hz

    Returns
    -------
    RSTimeDelta
        Duration

    Examples
    --------
    A few examples with different sampling frequencies and window sizes

    >>> from resistics.window import window_duration
    >>> duration = window_duration(512, 512)
    >>> print(duration)
    0:00:01
    >>> duration = window_duration(520, 512)
    >>> print(duration)
    0:00:01.015625
    >>> duration = window_duration(4096, 16_384)
    >>> print(duration)
    0:00:00.25
    >>> duration = window_duration(200, 0.05)
    >>> print(duration)
    1:06:40
    >>> (200 * 20) - (3600 + 6 * 60 + 40)
    0
    """
    from resistics.sampling import to_timedelta

    return to_timedelta(1 / fs) * window_size


def increment_duration(window_size: int, overlap_size: int, fs: float) -> RSTimeDelta:
    """
    Get the increment between window start times

    If the overlap_size = 0, then the time increment between windows is simply
    the window_duration. However, when there is an overlap, the increment
    between window start times has to be adjusted by the overlap size

    Parameters
    ----------
    window_size : int
        The window size in samples
    overlap_size : int
        The overlap size in samples
    fs : float
        The sample frequency Hz

    Returns
    -------
    RSTimeDelta
        The duration of the window

    Examples
    --------
    >>> from resistics.window import increment_duration
    >>> increment = increment_duration(128, 32, 128)
    >>> print(increment)
    0:00:00.75
    >>> increment = increment_duration(128*3600, 128*60, 128)
    >>> print(increment)
    0:59:00
    """
    from resistics.sampling import to_timedelta

    return to_timedelta(1 / fs) * (window_size - overlap_size)


def window_to_datetime(
    ref_time: RSDateTime, n_increments: int, increment: RSTimeDelta
) -> RSDateTime:
    """
    Convert window index to start time of window

    Parameters
    ----------
    ref_time : RSDateTime
        Reference time
    n_increments : int
        window index
    increment : RSTimeDelta
        The increment duration

    Returns
    -------
    RSDateTime
        Start time of window
    """
    return ref_time + (n_increments * increment)


def datetime_to_window(
    ref_time: RSDateTime,
    time: RSDateTime,
    increment: RSTimeDelta,
    method: str = "round",
) -> int:
    """
    Convert a datetime to a window index and start time of the window

    Parameters
    ----------
    ref_time : RSDateTime
        Reference time
    time : RSDateTime
        Datetime to convert
    increment : RSTimeDelta
        The increment duration
    method : str, optional
        Method for dealing with float results, by default "round"

    Returns
    -------
    int
        The window index relative to the reference time

    Raises
    ------
    ValueError
        If time < ref_time

    Examples
    --------
    A simple example to show the logic

    >>> from resistics.sampling import to_datetime, to_timedelta
    >>> from resistics.window import datetime_to_window, window_to_datetime
    >>> ref_time = to_datetime("2021-01-01 00:00:00")
    >>> time = to_datetime("2021-01-01 00:01:00")
    >>> increment = to_timedelta(60)
    >>> window_index = datetime_to_window(ref_time, time, increment)
    >>> window_index
    1
    >>> print(window_to_datetime(ref_time, window_index, increment))
    2021-01-01 00:01:00

    A more complex logic with window sizes, overlap sizes and sampling
    frequencies

    >>> fs = 128
    >>> window_size = 256
    >>> overlap_size = 64
    >>> ref_time = to_datetime("2021-03-15 00:00:00")
    >>> time = to_datetime("2021-04-17 18:00:00")
    >>> increment = increment_duration(window_size, overlap_size, fs)
    >>> print(increment)
    0:00:01.5
    >>> window_index = datetime_to_window(ref_time, time, increment)
    >>> window_index
    1944000
    >>> print(window_to_datetime(ref_time, window_index, increment))
    2021-04-17 18:00:00

    In this scenario, explore the use of rounding

    >>> time = to_datetime("2021-04-17 18:00:00.50")
    >>> window_index = datetime_to_window(ref_time, time, increment, method = "floor")
    >>> window_index
    1944000
    >>> print(window_to_datetime(ref_time, window_index, increment))
    2021-04-17 18:00:00
    >>> window_index = datetime_to_window(ref_time, time, increment, method = "ceil")
    >>> window_index
    1944001
    >>> print(window_to_datetime(ref_time, window_index, increment))
    2021-04-17 18:00:01.5
    >>> window_index = datetime_to_window(ref_time, time, increment, method = "round")
    >>> window_index
    1944000
    >>> print(window_to_datetime(ref_time, window_index, increment))
    2021-04-17 18:00:00
    """
    from math import floor, ceil
    from resistics.sampling import to_seconds

    if time < ref_time:
        raise ValueError(f"Time {time} < reference time {ref_time}")

    increment_days_in_seconds, increment_remaining_in_seconds = to_seconds(increment)
    increment_seconds = increment_days_in_seconds + increment_remaining_in_seconds

    delta_days_in_seconds, delta_remaining_in_seconds = to_seconds(time - ref_time)
    n_increments = delta_days_in_seconds / increment_seconds
    n_increments += delta_remaining_in_seconds / increment_remaining_in_seconds

    if n_increments.is_integer():
        n_increments = int(n_increments)
    elif method == "floor":
        n_increments = int(floor(n_increments))
    elif method == "ceil":
        n_increments = int(ceil(n_increments))
    else:
        n_increments = int(round(n_increments))
    return n_increments


def get_first_and_last_window(
    ref_time: RSDateTime, time_data: TimeData, window_size: int, overlap_size: int
) -> Tuple[int, int]:
    """
    Get first and last window for a TimeData

    Parameters
    ----------
    ref_time : RSDateTime
        The reference time
    time_data : TimeData
        The TimeData
    window_size : int
        Window size in samples
    overlap_size : int
        Overlap size in samples

    Returns
    -------
    Tuple[int, int]
        First window and last window respectively relative to the reference time

    Examples
    --------
    >>> from resistics.testing import time_data_random
    >>> from resistics.sampling import to_datetime
    >>> from resistics.window import get_first_and_last_window, window_to_datetime, increment_duration
    >>> ref_time = to_datetime("2021-01-01 00:00:00")
    >>> time_data = time_data_random(n_samples = 1000, fs=10, first_time="2021-01-01 01:00:00")
    >>> print(time_data.fs, time_data.first_time, time_data.last_time)
    10.0 2021-01-01 01:00:00 2021-01-01 01:01:39.9
    >>> window_size = 100
    >>> overlap_size = 25
    >>> first_window, last_window = get_first_and_last_window(ref_time, time_data, window_size, overlap_size)
    >>> print(first_window, last_window)
    480 492

    These window indices can be converted to start times of the windows. The
    last window is checked to make sure it does not extend past the end of the
    time data

    >>> increment = increment_duration(window_size, overlap_size, time_data.fs)
    >>> first_window_start_time = window_to_datetime(ref_time, 480, increment)
    >>> last_window_start_time = window_to_datetime(ref_time, 492, increment)
    >>> print(first_window_start_time, last_window_start_time)
    2021-01-01 01:00:00 2021-01-01 01:01:30
    >>> print(last_window_start_time + increment)
    2021-01-01 01:01:37.5
    >>> print(time_data.last_time)
    2021-01-01 01:01:39.9
    >>> time_data.last_time > last_window_start_time + increment
    True
    """
    increment = increment_duration(window_size, overlap_size, time_data.fs)
    first_window = datetime_to_window(
        ref_time, time_data.first_time, increment, method="ceil"
    )
    last_window = datetime_to_window(
        ref_time, time_data.last_time, increment, method="floor"
    )
    # adjust if there is not enough date to complete the last window
    last_window_time = window_to_datetime(ref_time, last_window, increment)
    if time_data.last_time < last_window_time + increment:
        last_window -= 1
    return first_window, last_window


def get_window_table(
    ref_time: RSDateTime, time_data: TimeData, window_size: int, overlap_size: int
) -> pd.DataFrame:
    """
    Get a DataFrame with

    Parameters
    ----------
    ref_time : RSDateTime
        Reference
    time_data : TimeData
        Time data that will be windowed
    window_size : int
        The window size
    overlap_size : int
        The overlap size

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with details about each window

    Examples
    --------
    >>> from resistics.testing import time_data_random
    >>> from resistics.sampling import to_datetime
    >>> from resistics.window import get_window_table
    >>> ref_time = to_datetime("2021-01-01 00:00:00")
    >>> time_data = time_data_random(n_samples = 1000, fs=10, first_time="2021-01-01 01:00:00")
    >>> print(time_data.fs, time_data.first_time, time_data.last_time)
    10.0 2021-01-01 01:00:00 2021-01-01 01:01:39.9
    >>> window_size = 100
    >>> overlap_size = 25
    >>> df = get_window_table(ref_time, time_data, window_size, overlap_size)
    >>> print(df.to_string())
        window  local  from_sample  to_sample            window_start              window_end
    0      480      0            0         99 2021-01-01 01:00:00.000 2021-01-01 01:00:09.900
    1      481      1           75        174 2021-01-01 01:00:07.500 2021-01-01 01:00:17.400
    2      482      2          150        249 2021-01-01 01:00:15.000 2021-01-01 01:00:24.900
    3      483      3          225        324 2021-01-01 01:00:22.500 2021-01-01 01:00:32.400
    4      484      4          300        399 2021-01-01 01:00:30.000 2021-01-01 01:00:39.900
    5      485      5          375        474 2021-01-01 01:00:37.500 2021-01-01 01:00:47.400
    6      486      6          450        549 2021-01-01 01:00:45.000 2021-01-01 01:00:54.900
    7      487      7          525        624 2021-01-01 01:00:52.500 2021-01-01 01:01:02.400
    8      488      8          600        699 2021-01-01 01:01:00.000 2021-01-01 01:01:09.900
    9      489      9          675        774 2021-01-01 01:01:07.500 2021-01-01 01:01:17.400
    10     490     10          750        849 2021-01-01 01:01:15.000 2021-01-01 01:01:24.900
    11     491     11          825        924 2021-01-01 01:01:22.500 2021-01-01 01:01:32.400
    12     492     12          900        999 2021-01-01 01:01:30.000 2021-01-01 01:01:39.900
    """
    import numpy as np
    from resistics.sampling import to_n_samples, datetime_array_estimate

    increment_size = window_size - overlap_size
    increment = increment_duration(window_size, overlap_size, time_data.fs)
    fs = time_data.fs
    first_time = time_data.first_time

    first_window, last_window = get_first_and_last_window(
        ref_time, time_data, window_size, overlap_size
    )
    first_window_time = window_to_datetime(ref_time, first_window, increment)
    n_windows = last_window - first_window + 1
    local_windows = np.arange(n_windows)
    # samples
    first_sample = to_n_samples(first_window_time - first_time, fs, method="round") - 1
    starts = datetime_array_estimate(first_window_time, fs / increment_size, n_windows)
    ends = starts + pd.Timedelta((window_size - 1) * (1 / fs), "s")
    df_dict = {
        "window": np.arange(first_window, last_window + 1),
        "local": local_windows,
        "from_sample": first_sample + (local_windows * increment_size),
        "to_sample": first_sample + window_size - 1 + (local_windows * increment_size),
        "window_start": starts,
        "window_end": ends,
    }
    return pd.DataFrame(data=df_dict)


class WindowParameters(ResisticsData):
    """
    WindowParameters

    Examples
    --------
    Generate decimation and windowing parameters for data sampled at 4096 Hz.
    Note that requesting window sizes or overlap sizes for decimation levels
    that do not exist will raise a ValueError.

    >>> from resistics.decimate import DecimationSetup
    >>> from resistics.window import WindowSetup
    >>> dec_setup = DecimationSetup(n_levels=3, per_level=3)
    >>> dec_params = dec_setup.run(4096)
    >>> dec_params
    <class 'resistics.decimate.DecimationParameters'>
                                0           1           2      fs  total_factors  incremental_factors
    Decimation level
    0                 1024.000000  724.077344  512.000000  4096.0              1                    1
    1                  362.038672  256.000000  181.019336  2048.0              2                    2
    2                  128.000000   90.509668   64.000000   512.0              8                    4
    >>> win_setup = WindowSetup()
    >>> win_params = win_setup.run(dec_params)
    >>> win_params
    <class 'resistics.window.WindowParameters'>
                    window_size  overlap_size
    Decimation level
    0                        1024           256
    1                         512           128
    2                         256            64
    >>> win_params.get_window_size(0)
    1024
    >>> win_params.get_overlap_size(0)
    256
    >>> win_params.get_overlap_size(3)
    Traceback (most recent call last):
    ...
    ValueError: Level 3 must be 0 <= level < 3
    """

    def __init__(self, n_levels: int, win_df: pd.DataFrame):
        """
        Windowing parameters per decimation level

        Parameters
        ----------
        n_levels : int
            The number of decimation levels
        win_df : pd.DataFrame
            The window and overlap size information
        """
        self.n_levels = n_levels
        self.win_df = win_df

    def check_level(self, level: int):
        """
        Check decimation level exists

        Parameters
        ----------
        level : int
            The level to check

        Raises
        ------
        ValueError
            If the decimation level is not within range
        """
        if level < 0 or level >= self.n_levels:
            raise ValueError(f"Level {level} must be 0 <= level < {self.n_levels}")

    def get_window_size(self, level: int) -> int:
        """
        Get window size

        Parameters
        ----------
        level : int
            The decimation level

        Returns
        -------
        int
            The window size
        """
        self.check_level(level)
        return self.win_df.loc[level, "window_size"]

    def get_overlap_size(self, level: int) -> int:
        """
        Get overlap size

        Parameters
        ----------
        level : int
            The decimation level

        Returns
        -------
        int
            The overlap size
        """
        self.check_level(level)
        return self.win_df.loc[level, "overlap_size"]

    def to_string(self) -> str:
        """
        Class information as a string

        Returns
        -------
        str
            Window parameters info as a string
        """
        outstr = f"{self.type_to_string()}\n"
        outstr += self.win_df.to_string()
        return outstr


class WindowSetup(ResisticsProcess):
    """
    Setup WindowParameters

    Examples
    --------
    Generate decimation and windowing parameters for data sampled at 0.05 Hz or
    20 seconds sampling period

    >>> from resistics.decimate import DecimationSetup
    >>> from resistics.window import WindowSetup
    >>> dec_setup = DecimationSetup(n_levels=3, per_level=3)
    >>> dec_params = dec_setup.run(0.05)
    >>> dec_params
    <class 'resistics.decimate.DecimationParameters'>
                            0         1         2       fs  total_factors  incremental_factors
    Decimation level
    0                 0.012500  0.008839  0.006250  0.05000              1                    1
    1                 0.004419  0.003125  0.002210  0.02500              2                    2
    2                 0.001563  0.001105  0.000781  0.00625              8                    4
    >>> win_setup = WindowSetup()
    >>> win_params = win_setup.run(dec_params)
    >>> win_params
    <class 'resistics.window.WindowParameters'>
                        window_size  overlap_size
    Decimation level
    0                         256            64
    1                         256            64
    2                         256            64

    Window parameters can also be explicitly defined

    >>> from resistics.decimate import DecimationSetup
    >>> from resistics.window import WindowSetup
    >>> dec_setup = DecimationSetup(n_levels=3, per_level=3)
    >>> dec_params = dec_setup.run(0.05)
    >>> win_setup = WindowSetup(window_sizes=[1000, 578, 104])
    >>> win_params = win_setup.run(dec_params)
    >>> win_params
    <class 'resistics.window.WindowParameters'>
                    window_size  overlap_size
    Decimation level
    0                        1000           250
    1                         578           144
    2                         104            64
    """

    def __init__(
        self,
        min_size: int = 256,
        min_overlap: int = 64,
        window_factor: int = 4,
        overlap_proportion: float = 0.25,
        window_sizes: Optional[List[int]] = None,
        overlap_sizes: Optional[List[int]] = None,
    ):
        self.min_size = min_size
        self.min_overlap = min_overlap
        self.window_factor = window_factor
        self.overlap_proportion = overlap_proportion
        self.window_sizes = window_sizes
        self.overlap_sizes = overlap_sizes

    def run(self, dec_params: DecimationParameters) -> WindowParameters:
        """
        Calculate window and overlap sizes for each decimation level based on
        decimation level sampling frequency and minimum allowable parameters

        The window and overlap sizes (number of samples) are calculated based in
        the following way:

        - window size = frequency at decimation level / window_factor
        - overlap size = window size * overlap_proportion

        This is to ensure good frequency resolution at high frequencies. At low
        sampling frequencies, this would result in very small window sizes,
        therefore, there a minimum allowable sizes for both windows and overlap
        defined by min_size and min_overlap in the initialiser. If window sizes
        or overlaps size are calculated below these respecitively, they will be
        set to the minimum values.

        Parameters
        ----------
        dec_params : DecimationParameters
            The decimation parameters

        Returns
        -------
        WindowParameters
            The window parameters, the window sizes and overlaps for each
            decimation level
        """
        if self.window_sizes is None:
            window_sizes = self._get_window_sizes(dec_params)
        else:
            window_sizes = list(self.window_sizes)

        if self.overlap_sizes is None:
            overlap_sizes = self._get_overlap_sizes(window_sizes)
        else:
            overlap_sizes = self.overlap_sizes

        win_df = pd.DataFrame(
            data={"window_size": window_sizes, "overlap_size": overlap_sizes}
        )
        win_df.index.name = "Decimation level"
        return WindowParameters(dec_params.n_levels, win_df)

    def _get_window_sizes(self, dec_params: DecimationParameters) -> List[int]:
        """
        Get the window sizes

        Parameters
        ----------
        dec_params : DecimationParameters
            Decimation parameters

        Returns
        -------
        List[int]
            Window sizes
        """
        window_sizes = []
        for ilevel in range(dec_params.n_levels):
            info = dec_params.get_level(ilevel)
            window_size = info["fs"] // self.window_factor
            if window_size < self.min_size:
                window_size = self.min_size
            window_sizes.append(int(window_size))
        return window_sizes

    def _get_overlap_sizes(self, window_sizes: List[int]) -> List[int]:
        """
        Get overlap sizes

        Parameters
        ----------
        window_sizes : List[int]
            The window sizes

        Returns
        -------
        List[int]
            The overlap sizes
        """
        overlap_sizes = []
        for window_size in window_sizes:
            overlap_size = int(window_size * self.overlap_proportion)
            if overlap_size < self.min_overlap:
                overlap_size = self.min_overlap
            overlap_sizes.append(overlap_size)
        return overlap_sizes


class WindowedData(ResisticsData):
    def __init__(self):
        pass


class Windower(ResisticsProcess):
    def __init__(self, min_windows: int = 5):
        self.min_windows = min_windows
