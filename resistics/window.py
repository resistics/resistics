"""
Module for calculating window related data. This includes

- Windowing utility functions such as converting from global indices to datetime
- Converting a global index array to datetime
"""
from typing import Optional, List
from logging import getLogger
import pandas as pd

from resistics.common import ResisticsData, ResisticsProcess
from resistics.sampling import RSTimeDelta
from resistics.decimate import DecimationParameters

logger = getLogger(__name__)


def duration(size: int, fs: float) -> RSTimeDelta:
    from resistics.sampling import to_timedelta

    return to_timedelta(1 / fs) * size


# def global_index_to_datetime(
#     gIndex: int, refTime: datetime, fs: float, windowSize: int, windowOverlap: int
# ):
#     """Global index to datetime convertor

#     Global index 0 corresponds to reference time

#     Parameters
#     ----------
#     gIndex : int
#         Globel index
#     refTime : datetime.datetime
#         Reference time
#     fs : float
#         Sampling frequency in Hz
#     windowSize : int
#         Size of windows
#     windowOverlap : int
#         Size of window overlaps

#     Returns
#     -------
#     startTime : datetime.datetime
#         Start time of global window gIndex
#     endTime : datetime.datetime
#         End time of global window gIndex
#     """
#     # global index 0 starts at refTime
#     timeOffset = 1.0 * (windowSize - windowOverlap) / fs
#     totalOffset = gIndex * timeOffset
#     startTime = refTime + timedelta(seconds=totalOffset)
#     # windowSize - 1 because inclusive of start sample
#     endTime = startTime + timedelta(seconds=1.0 * (windowSize - 1) / fs)
#     return startTime, endTime


# def gArray2datetime(
#     gArray: np.ndarray,
#     refTime: datetime,
#     fs: float,
#     windowSize: int,
#     windowOverlap: int,
# ):
#     """Global index array to datetime convertor

#     Global index 0 corresponds to reference time

#     Parameters
#     ----------
#     gArray : np.ndarray
#         Globel indices array
#     refTime : datetime.datetime
#         Reference time
#     fs : float
#         Sampling frequency in Hz
#     windowSize : int
#         Size of windows
#     windowOverlap : int
#         Size of window overlaps

#     Returns
#     -------
#     startTime : np.ndarray of datetime.datetime
#         Start times of global windows
#     endTime : np.ndarray of datetime.datetime
#         End times of global windows
#     """
#     arrSize = gArray.size
#     startTime = np.zeros(shape=(arrSize), dtype=datetime)
#     endTime = np.zeros(shape=(arrSize), dtype=datetime)
#     for i in range(0, arrSize):
#         startTime[i], endTime[i] = gIndex2datetime(
#             gArray[i], refTime, fs, windowSize, windowOverlap
#         )
#     return startTime, endTime


# def datetime2gIndex(
#     refTime: datetime, inTime: datetime, fs: float, windowSize: int, windowOverlap: int
# ):
#     """Datetime to global index convertor

#     Global index 0 corresponds to reference time. This returns the global index of the time window nearest to inTime

#     Parameters
#     ----------
#     refTime : datetime.datetime
#         Reference time
#     inTime : datetime.datetime
#         Time for which you want closest global index
#     fs : float
#         Sampling frequency in Hz
#     windowSize : int
#         Size of windows
#     windowOverlap : int
#         Size of window overlaps

#     Returns
#     -------
#     gIndex : int
#         Global window index closest to inTime
#     firstWindowTime : datetime.datetime
#         Datetime of the global window
#     """
#     # need to return the next one close
#     # calculate
#     deltaRefStart = inTime - refTime
#     winStartIncrement = (windowSize - windowOverlap) / fs
#     # calculate number of windows started before reference time
#     # and then by taking the ceiling, find the global index of the first window in the data
#     gIndex = int(math.ceil(deltaRefStart.total_seconds() / winStartIncrement))
#     # calculate start time of first global window
#     offsetSeconds = gIndex * winStartIncrement
#     # calculate the first window time
#     firstWindowTime = refTime + timedelta(seconds=offsetSeconds)
#     return gIndex, firstWindowTime


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
    def __init__(self):
        pass
