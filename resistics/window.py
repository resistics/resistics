"""
Module for calculating window related data. Windows can be indexed relative to
two starting indices.

- Local window index

    - Window index relative to the TimeData is called "local_win"
    - Local window indices always start at 0

- Global window index

    - The global window index is relative to the project reference time
    - The 0 index window begins at the reference time
    - This window indexing is to synchronise data across sites

The global window index is considered the default and sometimes referred to as
the window. Local windows should be explicitly referred to as local_win in
all cases.

The window module includes functionality to do the following:

- Windowing utility functions to calculate window and overlap sizes
- Functions to map windows to samples in TimeData
- Converting a global index array to datetime
"""
from loguru import logger
from typing import Optional, List, Tuple, Dict
import numpy as np
import pandas as pd

from resistics.common import ProcessHistory, ResisticsData, ResisticsProcess
from resistics.common import MetadataGroup
from resistics.sampling import RSDateTime, RSTimeDelta
from resistics.time import TimeData
from resistics.decimate import DecimationParameters, DecimatedData


def win_duration(win_size: int, fs: float) -> RSTimeDelta:
    """
    Get the window duration

    Parameters
    ----------
    win_size : int
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

    >>> from resistics.window import win_duration
    >>> duration = win_duration(512, 512)
    >>> print(duration)
    0:00:01
    >>> duration = win_duration(520, 512)
    >>> print(duration)
    0:00:01.015625
    >>> duration = win_duration(4096, 16_384)
    >>> print(duration)
    0:00:00.25
    >>> duration = win_duration(200, 0.05)
    >>> print(duration)
    1:06:40
    >>> (200 * 20) - (3600 + 6 * 60 + 40)
    0
    """
    from resistics.sampling import to_timedelta

    return to_timedelta(1 / fs) * float(win_size)


def inc_duration(win_size: int, olap_size: int, fs: float) -> RSTimeDelta:
    """
    Get the increment between window start times

    If the overlap size = 0, then the time increment between windows is simply
    the window duration. However, when there is an overlap, the increment
    between window start times has to be adjusted by the overlap size

    Parameters
    ----------
    win_size : int
        The window size in samples
    olap_size : int
        The overlap size in samples
    fs : float
        The sample frequency Hz

    Returns
    -------
    RSTimeDelta
        The duration of the window

    Examples
    --------
    >>> from resistics.window import inc_duration
    >>> increment = inc_duration(128, 32, 128)
    >>> print(increment)
    0:00:00.75
    >>> increment = inc_duration(128*3600, 128*60, 128)
    >>> print(increment)
    0:59:00
    """
    from resistics.sampling import to_timedelta

    return to_timedelta(1 / fs) * float(win_size - olap_size)


def win_to_datetime(
    ref_time: RSDateTime, global_win: int, increment: RSTimeDelta
) -> RSDateTime:
    """
    Convert reference window index to start time of window

    Parameters
    ----------
    ref_time : RSDateTime
        Reference time
    global_win : int
        Window index relative to reference time
    increment : RSTimeDelta
        The increment duration

    Returns
    -------
    RSDateTime
        Start time of window
    """
    return ref_time + (global_win * increment)


def datetime_to_win(
    ref_time: RSDateTime,
    time: RSDateTime,
    increment: RSTimeDelta,
    method: str = "round",
) -> int:
    """
    Convert a datetime to a global window index

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
        The global window index i.e. the window index relative to the reference
        time

    Raises
    ------
    ValueError
        If time < ref_time

    Examples
    --------
    A simple example to show the logic

    >>> from resistics.sampling import to_datetime, to_timedelta
    >>> from resistics.window import datetime_to_win, win_to_datetime
    >>> ref_time = to_datetime("2021-01-01 00:00:00")
    >>> time = to_datetime("2021-01-01 00:01:00")
    >>> increment = to_timedelta(60)
    >>> global_win = datetime_to_win(ref_time, time, increment)
    >>> global_win
    1
    >>> print(win_to_datetime(ref_time, global_win, increment))
    2021-01-01 00:01:00

    A more complex logic with window sizes, overlap sizes and sampling
    frequencies

    >>> fs = 128
    >>> win_size = 256
    >>> olap_size = 64
    >>> ref_time = to_datetime("2021-03-15 00:00:00")
    >>> time = to_datetime("2021-04-17 18:00:00")
    >>> increment = inc_duration(win_size, olap_size, fs)
    >>> print(increment)
    0:00:01.5
    >>> global_win = datetime_to_win(ref_time, time, increment)
    >>> global_win
    1944000
    >>> print(win_to_datetime(ref_time, global_win, increment))
    2021-04-17 18:00:00

    In this scenario, explore the use of rounding

    >>> time = to_datetime("2021-04-17 18:00:00.50")
    >>> global_win = datetime_to_win(ref_time, time, increment, method = "floor")
    >>> global_win
    1944000
    >>> print(win_to_datetime(ref_time, global_win, increment))
    2021-04-17 18:00:00
    >>> global_win = datetime_to_win(ref_time, time, increment, method = "ceil")
    >>> global_win
    1944001
    >>> print(win_to_datetime(ref_time, global_win, increment))
    2021-04-17 18:00:01.5
    >>> global_win = datetime_to_win(ref_time, time, increment, method = "round")
    >>> global_win
    1944000
    >>> print(win_to_datetime(ref_time, global_win, increment))
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


def get_first_and_last_win(
    ref_time: RSDateTime, time_data: TimeData, win_size: int, olap_size: int
) -> Tuple[int, int]:
    """
    Get first and last window for a TimeData

    Parameters
    ----------
    ref_time : RSDateTime
        The reference time
    time_data : TimeData
        The TimeData
    win_size : int
        Window size in samples
    olap_size : int
        Overlap size in samples

    Returns
    -------
    Tuple[int, int]
        First and last global windows. This is window indices relative to the
        reference time

    Examples
    --------
    >>> from resistics.testing import time_data_random
    >>> from resistics.sampling import to_datetime
    >>> from resistics.window import get_first_and_last_win, win_to_datetime, inc_duration
    >>> ref_time = to_datetime("2021-01-01 00:00:00")
    >>> time_data = time_data_random(n_samples = 1000, fs=10, first_time="2021-01-01 01:00:00")
    >>> print(time_data.fs, time_data.first_time, time_data.last_time)
    10.0 2021-01-01 01:00:00 2021-01-01 01:01:39.9
    >>> win_size = 100
    >>> olap_size = 25
    >>> first_win, last_win = get_first_and_last_win(ref_time, time_data, win_size, olap_size)
    >>> print(first_win, last_win)
    480 492

    These window indices can be converted to start times of the windows. The
    last window is checked to make sure it does not extend past the end of the
    time data

    >>> increment = inc_duration(win_size, olap_size, time_data.fs)
    >>> first_win_start_time = win_to_datetime(ref_time, 480, increment)
    >>> last_win_start_time = win_to_datetime(ref_time, 492, increment)
    >>> print(first_win_start_time, last_win_start_time)
    2021-01-01 01:00:00 2021-01-01 01:01:30
    >>> print(last_win_start_time + increment)
    2021-01-01 01:01:37.5
    >>> print(time_data.last_time)
    2021-01-01 01:01:39.9
    >>> time_data.last_time > last_win_start_time + increment
    True
    """
    increment = inc_duration(win_size, olap_size, time_data.fs)
    first_win = datetime_to_win(
        ref_time, time_data.first_time, increment, method="ceil"
    )
    last_win = datetime_to_win(ref_time, time_data.last_time, increment, method="floor")
    # adjust if there is not enough date to complete the last window
    last_win_time = win_to_datetime(ref_time, last_win, increment)
    if time_data.last_time < last_win_time + increment:
        last_win -= 1
    return first_win, last_win


def get_win_table(
    ref_time: RSDateTime, time_data: TimeData, win_size: int, olap_size: int
) -> pd.DataFrame:
    """
    Get a DataFrame with

    Parameters
    ----------
    ref_time : RSDateTime
        Reference
    time_data : TimeData
        Time data that will be windowed
    win_size : int
        The window size
    olap_size : int
        The overlap size

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with details about each window

    Examples
    --------
    .. plot::
        :width: 100%

        >>> import matplotlib.pyplot as plt
        >>> from resistics.testing import time_data_random
        >>> from resistics.sampling import to_datetime
        >>> from resistics.window import get_win_table
        >>> ref_time = to_datetime("2021-01-01 00:00:00")
        >>> time_data = time_data_random(n_samples = 1000, fs=10, first_time="2021-01-01 01:00:00")
        >>> print(time_data.fs, time_data.first_time, time_data.last_time)
        10.0 2021-01-01 01:00:00 2021-01-01 01:01:39.9
        >>> win_size = 100
        >>> olap_size = 25
        >>> df = get_win_table(ref_time, time_data, win_size, olap_size)
        >>> print(df.to_string())
            global  local  from_sample  to_sample               win_start                 win_end
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

        Plot six windows to illustrate the overlap

        >>> plt.figure(figsize=(8, 3)) # doctest: +SKIP
        >>> for idx, row in df.iterrows():
        ...     color = "red" if idx%2 == 0 else "blue"
        ...     plt.axvspan(row.loc["win_start"], row.loc["win_end"], alpha=0.5, color=color) # doctest: +SKIP
        ...     if idx > 5:
        ...         break
        >>> plt.tight_layout() # doctest: +SKIP
        >>> plt.show() # doctest: +SKIP
    """
    import numpy as np
    from resistics.sampling import to_n_samples, datetime_array_estimate

    increment_size = win_size - olap_size
    increment = inc_duration(win_size, olap_size, time_data.fs)
    fs = time_data.fs
    first_time = time_data.first_time

    first_win, last_win = get_first_and_last_win(
        ref_time, time_data, win_size, olap_size
    )
    first_win_time = win_to_datetime(ref_time, first_win, increment)
    n_wins = last_win - first_win + 1
    local_wins = np.arange(n_wins).astype(int)
    # samples
    first_sample = to_n_samples(first_win_time - first_time, fs, method="round") - 1
    starts = datetime_array_estimate(first_win_time, fs / increment_size, n_wins)
    ends = starts + pd.Timedelta((win_size - 1) * (1 / fs), "s")
    df_dict = {
        "global": np.arange(first_win, last_win + 1),
        "local": local_wins,
        "from_sample": first_sample + (local_wins * increment_size),
        "to_sample": first_sample + win_size - 1 + (local_wins * increment_size),
        "win_start": starts,
        "win_end": ends,
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
                         win_size     olap_size
    Decimation level
    0                        1024           256
    1                         512           128
    2                         256            64
    >>> win_params.get_win_size(0)
    1024
    >>> win_params.get_olap_size(0)
    256
    >>> win_params.get_olap_size(3)
    Traceback (most recent call last):
    ...
    ValueError: Level 3 must be 0 <= level < 3
    """

    def __init__(self, n_levels: int, min_n_wins: int, win_df: pd.DataFrame):
        """
        Windowing parameters per decimation level

        Parameters
        ----------
        n_levels : int
            The number of decimation levels
        min_n_wins : int
            Minimum number of windows
        win_df : pd.DataFrame
            The window and overlap size information
        """
        self.n_levels = n_levels
        self.min_n_wins = min_n_wins
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

    def get_win_size(self, level: int) -> int:
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
        return self.win_df.loc[level, "win_size"]

    def get_olap_size(self, level: int) -> int:
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
        return self.win_df.loc[level, "olap_size"]

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

    .. note::

        Note that the running check for WindowSetup always returns True

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
                         win_size     olap_size
    Decimation level
    0                         256            64
    1                         256            64
    2                         256            64

    Window parameters can also be explicitly defined

    >>> from resistics.decimate import DecimationSetup
    >>> from resistics.window import WindowSetup
    >>> dec_setup = DecimationSetup(n_levels=3, per_level=3)
    >>> dec_params = dec_setup.run(0.05)
    >>> win_setup = WindowSetup(win_sizes=[1000, 578, 104])
    >>> win_params = win_setup.run(dec_params)
    >>> win_params
    <class 'resistics.window.WindowParameters'>
                         win_size     olap_size
    Decimation level
    0                        1000           250
    1                         578           144
    2                         104            64
    """

    def __init__(
        self,
        min_size: int = 256,
        min_olap: int = 64,
        win_factor: int = 4,
        olap_proportion: float = 0.25,
        min_n_wins: int = 5,
        win_sizes: Optional[List[int]] = None,
        olap_sizes: Optional[List[int]] = None,
    ):
        """
        Initialise WindowSetup

        WindowSetup outputs the WindowParameters to use for windowing decimated
        time data.

        Parameters
        ----------
        min_size : int, optional
            Minimum window size, by default 256
        min_olap : int, optional
            Minimum overlap size, by default 64
        win_factor : int, optional
            Window factor, by default 4. Window sizes are calculated by sampling
            frequency / 4 to ensure sufficient frequency resolution. If the
            sampling frequency is small, window size will be adjusted to
            min_size
        olap_proportion : float, optional
            The proportion of the window size to use as the overlap, by default
            0.25. For example, for a window size of 128, the overlap would be
            0.25 * 128 = 32
        min_n_wins : int, optional
            The minimum number of windows needed in a decimation level, by
            default 5
        win_sizes : Optional[List[int]], optional
            Explicit define window sizes, by default None. Must have the same
            length as number of decimation levels
        olap_sizes : Optional[List[int]], optional
            Explicitly define overlap sizes, by default None. Must have the same
            length as number of decimation levels
        """
        self.min_size = min_size
        self.min_olap = min_olap
        self.win_factor = win_factor
        self.olap_proportion = olap_proportion
        self.min_n_wins = min_n_wins
        self.win_sizes = win_sizes
        self.olap_sizes = olap_sizes

    def run(self, dec_params: DecimationParameters) -> WindowParameters:
        """
        Calculate window and overlap sizes for each decimation level based on
        decimation level sampling frequency and minimum allowable parameters

        The window and overlap sizes (number of samples) are calculated based in
        the following way:

        - window size = frequency at decimation level / window factor
        - overlap size = window size * overlap proportion

        This is to ensure good frequency resolution at high frequencies. At low
        sampling frequencies, this would result in very small window sizes,
        therefore, there a minimum allowable sizes for both windows and overlap
        defined by min_size and min_olap in the initialiser. If window sizes
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
        if self.win_sizes is None:
            win_sizes = self._get_win_sizes(dec_params)
        else:
            win_sizes = list(self.win_sizes)

        if self.olap_sizes is None:
            olap_sizes = self._get_olap_sizes(win_sizes)
        else:
            olap_sizes = self.olap_sizes

        win_df = pd.DataFrame(data={"win_size": win_sizes, "olap_size": olap_sizes})
        win_df.index.name = "Decimation level"
        return WindowParameters(dec_params.n_levels, self.min_n_wins, win_df)

    def _get_win_sizes(self, dec_params: DecimationParameters) -> List[int]:
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
        win_sizes = []
        for ilevel in range(dec_params.n_levels):
            info = dec_params.get_level(ilevel)
            win_size = info["fs"] // self.win_factor
            if win_size < self.min_size:
                win_size = self.min_size
            win_sizes.append(int(win_size))
        return win_sizes

    def _get_olap_sizes(self, win_sizes: List[int]) -> List[int]:
        """
        Get overlap sizes

        Parameters
        ----------
        win_sizes : List[int]
            The window sizes

        Returns
        -------
        List[int]
            The overlap sizes
        """
        olap_sizes = []
        for win_size in win_sizes:
            olap_size = int(win_size * self.olap_proportion)
            if olap_size < self.min_olap:
                olap_size = self.min_olap
            olap_sizes.append(olap_size)
        return olap_sizes


class WindowedTimeData(ResisticsData):
    """
    WindowedTimeData for a TimeData object

    Note that the WindowedTimeData actually holds views to the TimeData data to avoid
    excess memory usage.

    Examples
    --------
    >>> from resistics.testing import time_data_linear
    >>> from resistics.sampling import to_datetime
    >>> from resistics.window import WindowerTimeData
    >>> time_data = time_data_linear(first_time="2021-01-01 01:01:00", n_samples=12)
    >>> time_data.fs
    10.0
    >>> time_data.data
    array([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.],
           [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.],
           [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.],
           [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.]])
    >>> win_size = 5
    >>> olap_size = 2
    >>> ref_time = to_datetime("2021-01-01 01:00:00")
    >>> windower = WindowerTimeData(win_size, olap_size, min_n_wins=2)
    >>> windower.check(ref_time, time_data)
    True
    >>> win_data = windower.run(time_data)
    >>> print(win_data.win_table.to_string())
           global  from_sample  to_sample               win_start                 win_end
    local
    0         200            0          4 2021-01-01 01:01:00.000 2021-01-01 01:01:00.400
    1         201            3          7 2021-01-01 01:01:00.300 2021-01-01 01:01:00.700
    2         202            6         10 2021-01-01 01:01:00.600 2021-01-01 01:01:01.000

    The first window has global index 200. Why? There is a 1 minute time
    difference between the reference time "2021-01-01 01:00:00" and the first
    time "2021-01-01 01:01:00". There is a 3 sample increment between window
    start samples (the window size - overlap size). This means there is a 0.3
    seconds increment between windows. So how many windows in the 1 minute?

    >>> 60 / 0.3
    200.0

    There are 200 windows in this 1 minute, and our data begins with the 201st
    window, which when beginning with 0-index, has index 200.
    """

    def __init__(
        self,
        metadata: MetadataGroup,
        chans: List[str],
        win_size: int,
        olap_size: int,
        win_views: np.ndarray,
        win_table: pd.DataFrame,
    ):
        """
        WindowedTimeData which can supply data for a window in TimeData

        Parameters
        ----------
        metadata : MetadataGroup
            The metadata from the time data
        chans : List[str]
            The channels in the data
        win_size : int
            Window size
        olap_size : int
            Overlap size
        win_views : np.ndarray
            The window views into the original TimeData
        win_table : pd.DataFrame
            Table outlining global window index relative to the reference time,
            the local window index, the sample ranges and time range for each
            window

        Raises
        ------
        ValueError
            If the window table is somehow malformed
        """
        self.metadata = metadata
        self.chans = chans
        self.win_size = win_size
        self.olap_size = olap_size
        self.win_views = win_views
        self.win_table = win_table.set_index("local")
        self.n_wins = len(win_table.index)
        offset = (win_table["local"] - win_table["global"]).unique()
        if len(offset) != 1:
            raise ValueError("Malformed window table, varying local to global offset")
        self.offset = offset[0]

    @property
    def fs(self) -> float:
        """
        Get sampling frequency

        Returns
        -------
        float
            The sampling frequency in Hz
        """
        return self.metadata["common", "fs"]

    def get_local(self, local_win: int) -> np.ndarray:
        """
        Get window using local index

        Parameters
        ----------
        local_win : int
            Local window index

        Returns
        -------
        np.ndarray
            Window data

        Raises
        ------
        ValueError
            If local window index is out of range
        """
        if local_win < 0 or local_win >= self.n_wins:
            raise ValueError(
                f"Local window {local_win} not 0 <= local_win < {self.n_wins}"
            )
        return self.win_views[local_win]

    def get_global(self, global_win: int) -> np.ndarray:
        """
        Get window using global index

        Parameters
        ----------
        global_win : int
            Global window index

        Returns
        -------
        np.ndarray
            Window data
        """
        return self.get_local(global_win + self.offset)

    def get_chan(self, chan: str) -> np.ndarray:
        """
        Get all the windows for a channel

        Parameters
        ----------
        chan : str
            The channel

        Returns
        -------
        np.ndarray
            The data for the channels

        Raises
        ------
        ChannelNotFoundError
            If the channel is not found in the data
        """
        from resistics.errors import ChannelNotFoundError

        if chan not in self.chans:
            raise ChannelNotFoundError(chan, self.chans)
        idx = self.chans.index(chan)
        return self.win_views[..., idx, :]

    def to_string(self) -> str:
        """
        Get WindowedTimeData info as string

        Returns
        -------
        str
            Info as string
        """
        duration = win_duration(self.win_size, self.fs)
        increment = inc_duration(self.win_size, self.olap_size, self.fs)
        outstr = f"{self.type_to_string()}\n"
        outstr += f"Number of windows: {self.n_wins}\n"
        outstr += f"Sampling frequency Hz: {self.fs}\n"
        outstr += f"Window size: {self.win_size}\n"
        outstr += f"Window duration {str(duration)}\n"
        outstr += f"Overlap size: {self.olap_size}\n"
        outstr += f"Increment duration: {str(increment)}"
        return outstr


class WindowerTimeData(ResisticsProcess):
    """
    The WindowerTimeData creates a views to the data for each window

    Rather than duplicate data for each window, the WindowerTimeData creates an array of
    views which point to the right data for the window in the original array.

    Examples
    --------
    Window time data with a window size of 5 and overlap size of 2. For the sake
    of simplicity, set the reference time to the start time of the time data.

    >>> from resistics.testing import time_data_linear
    >>> from resistics.window import WindowerTimeData
    >>> time_data = time_data_linear(n_samples=12)
    >>> time_data.data
    array([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.],
           [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.],
           [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.],
           [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.]])
    >>> win_size = 5
    >>> olap_size = 2
    >>> ref_time = time_data.first_time
    >>> str(ref_time)
    '2020-01-01 00:00:00'
    >>> windower = WindowerTimeData(win_size, olap_size)
    >>> windower.check(ref_time, time_data)
    False

    The default minimum number of windows is 5. Here we have only 3. Let's
    recreate the WindowerTimeData with a different minimum. Remember, we are running
    with a window size of 5 and overlap of 2 samples between windows.

    >>> windower = WindowerTimeData(win_size, olap_size, min_n_wins=2)
    >>> windower.check(ref_time, time_data)
    True
    >>> win_data = windower.run(time_data)
    >>> print(win_data.win_table.to_string())
               global  from_sample  to_sample           win_start                 win_end
    local
    0           0            0          4 2020-01-01 00:00:00.000 2020-01-01 00:00:00.400
    1           1            3          7 2020-01-01 00:00:00.300 2020-01-01 00:00:00.700
    2           2            6         10 2020-01-01 00:00:00.600 2020-01-01 00:00:01.000
    >>> win_data.win_views
    array([[[ 0.,  1.,  2.,  3.,  4.],
            [ 0.,  1.,  2.,  3.,  4.],
            [ 0.,  1.,  2.,  3.,  4.],
            [ 0.,  1.,  2.,  3.,  4.]],
           [[ 3.,  4.,  5.,  6.,  7.],
            [ 3.,  4.,  5.,  6.,  7.],
            [ 3.,  4.,  5.,  6.,  7.],
            [ 3.,  4.,  5.,  6.,  7.]],
           [[ 6.,  7.,  8.,  9., 10.],
            [ 6.,  7.,  8.,  9., 10.],
            [ 6.,  7.,  8.,  9., 10.],
            [ 6.,  7.,  8.,  9., 10.]]])
    """

    def __init__(self, win_size: int, olap_size: int, min_n_wins: int = 5):
        """
        Initialise TimeData windower

        Parameters
        ----------
        win_size : int
            The window size
        olap_size : int
            The overlap size
        min_n_wins : int, optional
            Minimum number of windows required, by default 5. The check to see
            how many windows can be made is done at the check stage.
        """
        self.win_size = win_size
        self.olap_size = olap_size
        self.min_n_wins = min_n_wins

    def check(
        self,
        ref_time: RSDateTime,
        time_data: TimeData,
    ) -> bool:
        """
        Check to ensure number windows is greater than min windows

        Parameters
        ----------
        ref_time : RSDateTime
            The reference time
        time_data : TimeData
            The time data

        Returns
        -------
        bool
            True if all checks complete, False is not enough windows can be made
        """
        self.ref_time = ref_time
        logger.info(f"Window size {self.win_size}, overlap size {self.olap_size}")
        self.win_table = get_win_table(
            ref_time, time_data, self.win_size, self.olap_size
        )
        n_wins = len(self.win_table.index)
        if n_wins < self.min_n_wins:
            logger.error(f"Number windows {n_wins} < minimum {self.min_n_wins}")
            return False
        logger.info(f"Time data will have {n_wins} windows")
        return True

    def run(self, time_data: TimeData) -> WindowedTimeData:
        """
        Run the windowing which gets the views into the data

        Parameters
        ----------
        time_data : TimeData
            TimeData to window

        Returns
        -------
        WindowedTimeData
            Window data for accessing windows
        """
        win_views = self._get_win_views(time_data)
        return WindowedTimeData(
            time_data.metadata,
            time_data.chans,
            self.win_size,
            self.olap_size,
            win_views,
            self.win_table,
        )

    def _get_win_views(self, time_data: TimeData) -> np.ndarray:
        """
        Get the window views

        This method uses sliding_window_view functionality in numpy.

        Parameters
        ----------
        time_data : TimeData
            TimeData to window

        Returns
        -------
        np.ndarray
            Sliding window views in an array

        Raises
        ------
        ValueError
            If win_size or olap_size is None, likely caused by not having
            called check first
        """
        from numpy.lib.stride_tricks import sliding_window_view

        if self.win_size is None or self.olap_size is None:
            raise ValueError("One or both of window/overlap is None. Run check first.")

        n_chans = time_data.n_chans
        from_sample = self.win_table.loc[0, "from_sample"]
        increment_size = self.win_size - self.olap_size

        view = np.squeeze(
            sliding_window_view(
                time_data.data,
                window_shape=(n_chans, self.win_size),
                writeable=True,
            )
        )
        return view[from_sample::increment_size]


class WindowedDecimatedData(ResisticsData):
    """
    Windows of a DecimatedData object

    Examples
    --------
    WindowedDecimatedData provides window views for all decimation levels. This
    can be useful for later spectra or statistic calculations.

    There's quite a few imports needed for this example

    >>> from resistics.sampling import to_datetime
    >>> from resistics.testing import time_data_random
    >>> from resistics.decimate import DecimationSetup, Decimator
    >>> from resistics.window import WindowSetup, WindowerDecimatedData

    Now the actual example using only defaults. Begin with decimation.

    >>> ref_time = to_datetime("2021-01-01 00:00:00")
    >>> time_data = time_data_random(first_time="2021-01-02 06:30:00", n_samples=10_000, fs=10)
    >>> dec_params = DecimationSetup().run(time_data.fs)
    >>> decimator = Decimator(dec_params)
    >>> decimator.check()
    True
    >>> dec_data = decimator.run(time_data)
    >>> print(dec_data.to_string())
    <class 'resistics.decimate.DecimatedData'>
                fs   dt  n_samples           first_time              last_time
    level
    0      10.0000  0.1      10000  2021-01-02 06:30:00  2021-01-02 06:46:39.9
    1       2.5000  0.4       2500  2021-01-02 06:30:00  2021-01-02 06:46:39.6
    2       0.3125  3.2        313  2021-01-02 06:30:00  2021-01-02 06:46:38.4

    Next, initialise the window parameters

    >>> win_params = WindowSetup().run(dec_params)
    >>> print(win_params.to_string())
    <class 'resistics.window.WindowParameters'>
                      win_size  olap_size
    Decimation level
    0                      256         64
    1                      256         64
    2                      256         64
    3                      256         64
    4                      256         64
    5                      256         64
    6                      256         64
    7                      256         64

    Perform the windowing. This actually creates views into the decimated data
    using the numpy.lib.stride_tricks.sliding_window_view function

    >>> windower = WindowerDecimatedData()
    >>> windower.check()
    True
    >>> win_data = windower.run(ref_time, win_params, dec_data)
    >>> print(win_data.to_string())
    <class 'resistics.window.WindowedDecimatedData'>
    Windowed 2 levels
    Level 0
    <class 'resistics.window.WindowedTimeData'>
    Number of windows: 51
    Sampling frequency Hz: 10.0
    Window size: 256
    Window duration 0:00:25.6
    Overlap size: 64
    Increment duration: 0:00:19.2
    Level 1
    <class 'resistics.window.WindowedTimeData'>
    Number of windows: 12
    Sampling frequency Hz: 2.5
    Window size: 256
    Window duration 0:01:42.4
    Overlap size: 64
    Increment duration: 0:01:16.8

    Each level has its own window table

    >>> wins = win_data.get_wins(1)
    >>> print(wins.win_table.to_string())
           global  from_sample  to_sample               win_start                 win_end
    local
    0        1430           60        315 2021-01-02 06:30:24.000 2021-01-02 06:32:06.000
    1        1431          252        507 2021-01-02 06:31:40.800 2021-01-02 06:33:22.800
    2        1432          444        699 2021-01-02 06:32:57.600 2021-01-02 06:34:39.600
    3        1433          636        891 2021-01-02 06:34:14.400 2021-01-02 06:35:56.400
    4        1434          828       1083 2021-01-02 06:35:31.200 2021-01-02 06:37:13.200
    5        1435         1020       1275 2021-01-02 06:36:48.000 2021-01-02 06:38:30.000
    6        1436         1212       1467 2021-01-02 06:38:04.800 2021-01-02 06:39:46.800
    7        1437         1404       1659 2021-01-02 06:39:21.600 2021-01-02 06:41:03.600
    8        1438         1596       1851 2021-01-02 06:40:38.400 2021-01-02 06:42:20.400
    9        1439         1788       2043 2021-01-02 06:41:55.200 2021-01-02 06:43:37.200
    10       1440         1980       2235 2021-01-02 06:43:12.000 2021-01-02 06:44:54.000
    11       1441         2172       2427 2021-01-02 06:44:28.800 2021-01-02 06:46:10.800
    """

    def __init__(
        self,
        chans: List[str],
        wins: Dict[int, WindowedTimeData],
        history: ProcessHistory,
    ):
        """
        Initialise

        Parameters
        ----------
        chans : List[str]
            The channels in the data
        wins : Dict[int, WindowedTimeData]
            Dictionary of decimation level to WindowedTimeData
        history : ProcessHistory
            Processing history
        """
        self.chans = chans
        self.wins = wins
        self.history = history
        self.max_level = max(list(self.wins.keys()))
        self.n_levels = self.max_level + 1

    @property
    def n_chans(self):
        return len(self.chans)

    def get_level(self, level: int) -> WindowedTimeData:
        """
        Get windows for a decimation level

        Parameters
        ----------
        level : int
            The decimation level

        Returns
        -------
        WindowedTimeData
            The windows

        Raises
        ------
        ValueError
            If decimation level is not within range
        """
        if level > self.max_level:
            raise ValueError(f"Level {level} not <= max {self.max_level}")
        return self.wins[level]

    def to_string(self) -> str:
        """
        Class information as a string

        Returns
        -------
        str
            WindowedDecimatedData information as a string
        """
        outstr = f"{self.type_to_string()}\n"
        outstr += f"Windowed {self.n_levels} levels\n"
        for ilevel in range(self.n_levels):
            outstr += f"Level {ilevel}\n"
            outstr += f"{self.wins[ilevel].to_string()}\n"
        return outstr.strip("\n")


class WindowerDecimatedData(ResisticsProcess):
    def run(
        self,
        ref_time: RSDateTime,
        win_params: WindowParameters,
        dec_data: DecimatedData,
    ) -> WindowedDecimatedData:
        """
        Perform windowing of DecimatedData

        Parameters
        ----------
        ref_time : RSDateTime
            The reference time
        win_params : WindowParameters
            The window parameters
        dec_data : DecimatedData
            The decimated data

        Returns
        -------
        WindowedDecimatedData
            Windows for decimated data
        """
        wins = {}
        messages = []
        for ilevel in range(0, dec_data.n_levels):
            time_data = dec_data.get_level(ilevel)
            win_size = win_params.get_win_size(ilevel)
            olap_size = win_params.get_olap_size(ilevel)
            logger.info(f"Windowing decimation level {ilevel}")
            logger.info(f"Window size {win_size}, overlap size {olap_size}")
            windower = WindowerTimeData(
                win_size, olap_size, min_n_wins=win_params.min_n_wins
            )
            if not windower.check(ref_time, time_data):
                continue
            windowed = windower.run(time_data)
            messages.append(f"Level {ilevel}, generated {windowed.n_wins} windows")
            messages.append(f"Window size {win_size}, olap_size {olap_size}")
            wins[ilevel] = windowed
        history = dec_data.history.copy()
        history.add_record(self._get_process_record(messages))
        return WindowedDecimatedData(dec_data.chans, wins, history)
