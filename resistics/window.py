"""Module for calculating window related data. Windows can be indexed relative to
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

Usually with windowing, there is a window size and windows overlap with each
other for a set number of samples. As an illustrative examples, consider a
signal sampled at 10 Hz (dt=0.1 seconds) with 24 samples. This will be windowed
using a window size of 8 samples per window and a 2 sample overlap.

```{plot}
:filename-prefix: window-overlap

>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> fs = 10
>>> n_samples = 24
>>> win_size = 8
>>> olap_size = 2
>>> times = np.arange(0, n_samples) * (1/fs)


The first window

>>> start_win1 = 0
>>> end_win1 = win_size
>>> win1_times = times[start_win1:end_win1]


The second window

>>> start_win2 = end_win1 - olap_size
>>> end_win2 = start_win2 + win_size
>>> win2_times = times[start_win2:end_win2]


The third window

>>> start_win3 = end_win2 - olap_size
>>> end_win3 = start_win3 + win_size
>>> win3_times = times[start_win3:end_win3]


The fourth window

>>> start_win4= end_win3 - olap_size
>>> end_win4 = start_win4 + win_size
>>> win4_times = times[start_win4:end_win4]


Let's look at the actual window times for each window

>>> win1_times
array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
>>> win2_times
array([0.6, 0.7, 0.8, 0.9, 1. , 1.1, 1.2, 1.3])
>>> win3_times
array([1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9])
>>> win4_times
array([1.8, 1.9, 2. , 2.1, 2.2, 2.3])


The duration and increments of windows can be calculated using provided
methods

>>> from resistics.window import win_duration, inc_duration
>>> print(win_duration(win_size, fs))
0:00:00.7
>>> print(inc_duration(win_size, olap_size, fs))
0:00:00.6


Plot the windows to give an illustration of how it works

>>> plt.plot(win1_times, np.ones_like(win1_times), "bo", label="window1") # doctest: +SKIP
>>> plt.plot(win2_times, np.ones_like(win2_times)*2, "ro", label="window2") # doctest: +SKIP
>>> plt.plot(win3_times, np.ones_like(win3_times)*3, "go", label="window3") # doctest: +SKIP
>>> plt.plot(win4_times, np.ones_like(win4_times)*4, "co", label="window4") # doctest: +SKIP
>>> plt.xlabel("Time [s]") # doctest: +SKIP
>>> plt.legend() # doctest: +SKIP
>>> plt.grid() # doctest: +SKIP
>>> plt.tight_layout() # doctest: +SKIP
>>> plt.show() # doctest: +SKIP

```
"""

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import ConfigDict, PositiveInt

from resistics.common import (
    History,
    Metadata,
    ResisticsData,
    ResisticsModel,
    ResisticsProcess,
    ResisticsWriter,
    WriteableMetadata,
    save_compressed_arrays,
)
from resistics.decimate import DecimatedData, DecimatedLevelMetadata
from resistics.errors import ProcessRunError
from resistics.sampling import HighResDateTime, RSDateTime, RSTimeDelta
from resistics.time import ChanMetadata


def win_duration(win_size: int, fs: float) -> RSTimeDelta:
    """Get the window duration

    :param win_size: Window size in samples
    :param fs: Sampling frequency Hz

    :return: Duration

    **Examples**

    A few examples with different sampling frequencies and window sizes

    ```{doctest}
    >>> from resistics.window import win_duration
    >>> duration = win_duration(512, 512)
    >>> print(duration)
    0:00:00.998046875
    >>> duration = win_duration(520, 512)
    >>> print(duration)
    0:00:01.013671875
    >>> duration = win_duration(4096, 16_384)
    >>> print(duration)
    0:00:00.24993896484375
    >>> duration = win_duration(200, 0.05)
    >>> print(duration)
    1:06:20

    ```
    """
    from resistics.sampling import to_timedelta

    return to_timedelta(1 / fs) * float(win_size - 1)


def inc_duration(win_size: int, olap_size: int, fs: float) -> RSTimeDelta:
    """Get the increment between window start times

    If the overlap size = 0, then the time increment between windows is simply
    the window duration. However, when there is an overlap, the increment
    between window start times has to be adjusted by the overlap size

    :param win_size: The window size in samples
    :param olap_size: The overlap size in samples
    :param fs: The sample frequency Hz

    :return: The duration of the window

    **Examples**

    ```{doctest}
    >>> from resistics.window import inc_duration
    >>> increment = inc_duration(128, 32, 128)
    >>> print(increment)
    0:00:00.75
    >>> increment = inc_duration(128*3600, 128*60, 128)
    >>> print(increment)
    0:59:00

    ```
    """
    from resistics.sampling import to_timedelta

    return to_timedelta(1 / fs) * float(win_size - olap_size)


def win_to_datetime(
    ref_time: RSDateTime, global_win: int, increment: RSTimeDelta
) -> RSDateTime:
    """Convert reference window index to start time of window

    :param ref_time: Reference time
    :param global_win: Window index relative to reference time
    :param increment: The increment duration

    :return: Start time of window

    **Examples**

    An example with sampling at 1 Hz, a window size of 100 and an overlap size
    of 25.

    ```{doctest}
    >>> from resistics.sampling import to_datetime
    >>> from resistics.window import inc_duration, win_to_datetime
    >>> ref_time = to_datetime("2021-01-01 00:00:00")
    >>> fs = 1
    >>> win_size = 60
    >>> olap_size = 15
    >>> increment = inc_duration(win_size, olap_size, fs)
    >>> print(increment)
    0:00:45

    ```

    The increment is the time increment between the start of time one window and
    the succeeding window.

    ```{doctest}
    >>> print(win_to_datetime(ref_time, 0, increment))
    2021-01-01 00:00:00
    >>> print(win_to_datetime(ref_time, 1, increment))
    2021-01-01 00:00:45
    >>> print(win_to_datetime(ref_time, 2, increment))
    2021-01-01 00:01:30
    >>> print(win_to_datetime(ref_time, 3, increment))
    2021-01-01 00:02:15

    ```
    """
    return ref_time + (global_win * increment)


def datetime_to_win(
    ref_time: RSDateTime,
    time: RSDateTime,
    increment: RSTimeDelta,
    method: str = "round",
) -> int:
    """Convert a datetime to a global window index

    :param ref_time: Reference time
    :param time: Datetime to convert
    :param increment: The increment duration
    :param method: Method for dealing with float results, by default "round"

    :return: The global window index i.e. the window index relative to the reference time

    :raises ValueError: If time < ref_time

    **Examples**

    A simple example to show the logic

    ```{doctest}
    >>> from resistics.sampling import to_datetime, to_timedelta
    >>> from resistics.window import datetime_to_win, win_to_datetime, inc_duration
    >>> ref_time = to_datetime("2021-01-01 00:00:00")
    >>> time = to_datetime("2021-01-01 00:01:00")
    >>> increment = to_timedelta(60)
    >>> global_win = datetime_to_win(ref_time, time, increment)
    >>> global_win
    1
    >>> print(win_to_datetime(ref_time, global_win, increment))
    2021-01-01 00:01:00

    ```

    A more complex logic with window sizes, overlap sizes and sampling
    frequencies

    ```{doctest}
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

    ```

    In this scenario, explore the use of rounding

    ```{doctest}
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

    ```

    Another example with a window duration of greater than a day

    ```{doctest}
    >>> fs = 4.8828125e-05
    >>> win_size = 64
    >>> olap_size = 16
    >>> ref_time = to_datetime("1985-07-18 01:00:20")
    >>> time = to_datetime("1985-09-22 23:00:00")
    >>> increment = inc_duration(win_size, olap_size, fs)
    >>> print(increment)
    11 days, 9:04:00
    >>> global_win = datetime_to_win(ref_time, time, increment)
    >>> global_win
    6
    >>> print(win_to_datetime(ref_time, global_win, increment))
    1985-09-24 07:24:20

    ```

    This time is greater than the time that was transformed to global window,
    1985-09-22 23:00:00. Try again, this time with the floor option.

    ```{doctest}
    >>> global_win = datetime_to_win(ref_time, time, increment, method="floor")
    >>> global_win
    5
    >>> print(win_to_datetime(ref_time, global_win, increment))
    1985-09-12 22:20:20

    ```
    """
    from math import ceil, floor

    from resistics.sampling import to_seconds

    if time < ref_time:
        raise ValueError(f"Time {time!s} < reference time {ref_time!s}")

    increment_days_in_seconds, increment_remaining_in_seconds = to_seconds(increment)
    increment_seconds = increment_days_in_seconds + increment_remaining_in_seconds

    delta_days_in_seconds, delta_remaining_in_seconds = to_seconds(time - ref_time)
    delta_total_in_seconds = delta_days_in_seconds + delta_remaining_in_seconds
    n_increments = delta_total_in_seconds / increment_seconds

    if n_increments.is_integer():
        n_increments = int(n_increments)
    elif method == "floor":
        n_increments = floor(n_increments)
    elif method == "ceil":
        n_increments = ceil(n_increments)
    else:
        n_increments = round(n_increments)
    return n_increments


def get_first_and_last_win(
    ref_time: RSDateTime,
    metadata: DecimatedLevelMetadata,
    win_size: int,
    olap_size: int,
) -> tuple[int, int]:
    """Get first and last window for a decimated data level

    ```{note}
    For the last window, on initial calculation this may be one or a
    maximum of two windows beyond the last time. The last window is adjusted
    in this function.

    Two windows may occur when the time of the last sample is in the overlap
    of the final two windows.
    ```
    :param ref_time: The reference time
    :param metadata: Metadata for the decimation level
    :param win_size: Window size in samples
    :param olap_size: Overlap size in samples

    :return: First and last global windows. This is window indices relative to the reference time

    :raises ValueError: If unable to calculate the last window correctly as this will result in
        an incorrect number of windows

    **Examples**

    Get the first and last window for the first decimation level in a decimated
    data instance.

    ```{doctest}
    >>> from resistics.testing import decimated_data_random
    >>> from resistics.sampling import to_datetime
    >>> from resistics.window import get_first_and_last_win, win_to_datetime
    >>> from resistics.window import win_duration, inc_duration
    >>> ref_time = to_datetime("2021-01-01 00:00:00")
    >>> dec_data = decimated_data_random(fs=0.1, first_time="2021-01-01 00:05:10", n_samples=100, factor=10)

    ```

    Get the metadata for decimation level 0

    ```{doctest}
    >>> level_metadata = dec_data.metadata.levels_metadata[0]
    >>> level_metadata.summary()
    {
        'fs': 10.0,
        'n_samples': 10000,
        'first_time': '2021-01-01 00:05:10.000000_000000_000000_000000',
        'last_time': '2021-01-01 00:21:49.899999_999999_977300_000000'
    }

    ```

    ```{note}
    As a point of interest, note how the last time is actually slightly
    incorrect. This is due to machine precision issues described in more
    detail here https://docs.python.org/3/tutorial/floatingpoint.html.
    Whilst there is value in using the high resolution datetime format for
    high sampling rates, there is a tradeoff. Such are the perils of
    floating point arithmetic.
    ```
    The next step is to calculate the first and last window, relative to the
    reference time

    ```{doctest}
    >>> win_size = 100
    >>> olap_size = 25
    >>> first_win, last_win = get_first_and_last_win(ref_time, level_metadata, win_size, olap_size)
    >>> print(first_win, last_win)
    42 173

    ```

    These window indices can be converted to start times of the windows. The
    last window is checked to make sure it does not extend past the end of the
    time data. First get the window duration and increments.

    ```{doctest}
    >>> duration = win_duration(win_size, level_metadata.fs)
    >>> print(duration)
    0:00:09.9
    >>> increment = inc_duration(win_size, olap_size, level_metadata.fs)
    >>> print(increment)
    0:00:07.5

    ```

    Now calculate the times of the windows

    ```{doctest}
    >>> first_win_start_time = win_to_datetime(ref_time, 42, increment)
    >>> last_win_start_time = win_to_datetime(ref_time, 173, increment)
    >>> print(first_win_start_time, last_win_start_time)
    2021-01-01 00:05:15 2021-01-01 00:21:37.5
    >>> print(last_win_start_time + duration)
    2021-01-01 00:21:47.4
    >>> print(level_metadata.last_time)
    2021-01-01 00:21:49.8999999999999773
    >>> level_metadata.last_time > last_win_start_time + increment
    True

    ```
    """
    duration = win_duration(win_size, metadata.fs)
    increment = inc_duration(win_size, olap_size, metadata.fs)
    first_win = datetime_to_win(ref_time, metadata.first_time, increment, method="ceil")
    last_win = datetime_to_win(ref_time, metadata.last_time, increment, method="floor")
    # adjust if there is not enough date to complete the last window
    last_win_time = win_to_datetime(ref_time, last_win, increment)
    for attempt in range(2):
        if metadata.last_time >= last_win_time + duration:
            break
        logger.debug(f"Adjusting last window attempt {attempt + 1}")
        last_win -= 1
        last_win_time = win_to_datetime(ref_time, last_win, increment)
    if metadata.last_time < last_win_time + duration:
        raise ValueError("Unable to correctly get the last window")
    return first_win, last_win


def get_win_starts(
    ref_time: RSDateTime,
    win_size: int,
    olap_size: int,
    fs: float,
    n_wins: int,
    index_offset: int,
) -> pd.DatetimeIndex:
    """Get window start times

    This is a useful for getting the timestamps for the windows in a dataset

    :param ref_time: The reference time
    :param win_size: The window size
    :param olap_size: The overlap size
    :param fs: The sampling frequency
    :param n_wins: The number of windows
    :param index_offset: The index offset from the reference time

    :return: The start times of the windows

    **Examples**

    ```{doctest}
    >>> import pandas as pd
    >>> from resistics.sampling import to_datetime
    >>> from resistics.window import get_win_starts
    >>> ref_time = to_datetime("2021-01-01 00:00:00")
    >>> win_size = 100
    >>> olap_size = 25
    >>> fs = 10
    >>> n_wins = 3
    >>> index_offset = 480
    >>> starts = get_win_starts(ref_time, win_size, olap_size, fs, n_wins, index_offset)
    >>> pd.Series(starts)
    0   2021-01-01 01:00:00.000
    1   2021-01-01 01:00:07.500
    2   2021-01-01 01:00:15.000
    dtype: datetime64[us]

    ```
    """
    from resistics.sampling import datetime_array_estimate

    increment = inc_duration(win_size, olap_size, fs)
    first_win_time = win_to_datetime(ref_time, index_offset, increment)
    increment_size = win_size - olap_size
    return datetime_array_estimate(first_win_time, fs / increment_size, n_wins)


def get_win_ends(
    starts: pd.DatetimeIndex,
    win_size: int,
    fs: float,
) -> pd.DatetimeIndex:
    """Get window end times

    :param starts: The start times of the windows
    :param win_size: The window size
    :param fs: The sampling frequency

    :return: The end times of the windows

    **Examples**

    ```{doctest}
    >>> import pandas as pd
    >>> from resistics.sampling import to_datetime
    >>> from resistics.window import get_win_starts, get_win_ends
    >>> ref_time = to_datetime("2021-01-01 00:00:00")
    >>> win_size = 100
    >>> olap_size = 25
    >>> fs = 10
    >>> n_wins = 3
    >>> index_offset = 480
    >>> starts = get_win_starts(ref_time, win_size, olap_size, fs, n_wins, index_offset)
    >>> pd.Series(starts)
    0   2021-01-01 01:00:00.000
    1   2021-01-01 01:00:07.500
    2   2021-01-01 01:00:15.000
    dtype: datetime64[us]
    >>> ends = get_win_ends(starts, win_size, fs)
    >>> pd.Series(ends)
    0   2021-01-01 01:00:09.900
    1   2021-01-01 01:00:17.400
    2   2021-01-01 01:00:24.900
    dtype: datetime64[ns]

    ```
    """
    return starts + pd.Timedelta((win_size - 1) * (1 / fs), "s")


def get_win_table(
    ref_time: RSDateTime,
    metadata: DecimatedLevelMetadata,
    win_size: int,
    olap_size: int,
) -> pd.DataFrame:
    """Get a DataFrame with

    :param ref_time: Reference
    :param metadata: Metadata for the decimation level
    :param win_size: The window size
    :param olap_size: The overlap size

    :return: A pandas DataFrame with details about each window

    **Examples**

    ```{plot}
    :width: 90%
    :filename-prefix: window-table

    >>> import matplotlib.pyplot as plt
    >>> from resistics.decimate import DecimatedLevelMetadata
    >>> from resistics.sampling import to_datetime, to_timedelta
    >>> from resistics.window import get_win_table
    >>> ref_time = to_datetime("2021-01-01 00:00:00")
    >>> fs = 10
    >>> n_samples = 1000
    >>> first_time = to_datetime("2021-01-01 01:00:00")
    >>> last_time = first_time + to_timedelta((n_samples-1)/fs)
    >>> metadata = DecimatedLevelMetadata(fs=10, n_samples=1000, first_time=first_time, last_time=last_time)
    >>> print(metadata.fs, metadata.first_time, metadata.last_time)
    10.0 2021-01-01 01:00:00 2021-01-01 01:01:39.9
    >>> win_size = 100
    >>> olap_size = 25
    >>> df = get_win_table(ref_time, metadata, win_size, olap_size)
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

    ```
    """
    from resistics.sampling import datetime_array_estimate, to_n_samples

    increment_size = win_size - olap_size
    increment = inc_duration(win_size, olap_size, metadata.fs)
    fs = metadata.fs
    first_time = metadata.first_time

    first_win, last_win = get_first_and_last_win(
        ref_time, metadata, win_size, olap_size
    )
    first_win_time = win_to_datetime(ref_time, first_win, increment)
    n_wins = last_win - first_win + 1
    local_wins = np.arange(n_wins).astype(int)
    # samples - use the floor here to avoid a situation where it is rounded up
    # and then there are insufficient samples
    first_sample = to_n_samples(first_win_time - first_time, fs, method="floor") - 1
    starts = datetime_array_estimate(first_win_time, fs / increment_size, n_wins)
    ends = get_win_ends(starts, win_size, fs)
    df_dict = {
        "global": np.arange(first_win, last_win + 1),
        "local": local_wins,
        "from_sample": first_sample + (local_wins * increment_size),
        "to_sample": first_sample + win_size - 1 + (local_wins * increment_size),
        "win_start": starts,
        "win_end": ends,
    }
    return pd.DataFrame(data=df_dict)


class WindowParameters(ResisticsModel):
    """Windowing parameters per decimation level

    Windowing parameters are the window and overlap size for each decimation
    level.

    :param n_levels: The number of decimation levels
    :param min_n_wins: Minimum number of windows
    :param win_sizes: The window sizes per decimation level
    :param olap_sizes: The overlap sizes per decimation level

    **Examples**

    Generate decimation and windowing parameters for data sampled at 4096 Hz.
    Note that requesting window sizes or overlap sizes for decimation levels
    that do not exist will raise a ValueError.

    ```{doctest}
    >>> from resistics.decimate import DecimationSetup
    >>> from resistics.window import WindowSetup
    >>> dec_setup = DecimationSetup(n_levels=3, per_level=3)
    >>> dec_params = dec_setup.run(4096)
    >>> dec_params.summary()
    {
        'fs': 4096.0,
        'n_levels': 3,
        'per_level': 3,
        'min_samples': 256,
        'eval_freqs': [
            1024.0,
            724.0773439350246,
            512.0,
            362.0386719675123,
            256.0,
            181.01933598375615,
            128.0,
            90.50966799187808,
            64.0
        ],
        'dec_factors': [1, 2, 8],
        'dec_increments': [1, 2, 4],
        'dec_fs': [4096.0, 2048.0, 512.0]
    }
    >>> win_params = WindowSetup().run(dec_params.n_levels, dec_params.dec_fs)
    >>> win_params.summary()
    {
        'n_levels': 3,
        'min_n_wins': 5,
        'win_sizes': [1024, 512, 128],
        'olap_sizes': [256, 128, 32]
    }
    >>> win_params.get_win_size(0)
    1024
    >>> win_params.get_olap_size(0)
    256
    >>> win_params.get_olap_size(3)
    Traceback (most recent call last):
    ...
    ValueError: Level 3 must be 0 <= level < 3

    ```
    """

    n_levels: int
    min_n_wins: int
    win_sizes: list[int]
    olap_sizes: list[int]

    def check_level(self, level: int) -> None:
        """Check that a decimation level is within range.

        :param level: Decimation level index.

        :raises ValueError: If the level is outside the configured range.
        """
        if level < 0 or level >= self.n_levels:
            raise ValueError(f"Level {level} must be 0 <= level < {self.n_levels}")

    def get_win_size(self, level: int) -> int:
        """Get the window size for a decimation level.

        :param level: Decimation level index.

        :return: Window size in samples.
        """
        self.check_level(level)
        return self.win_sizes[level]

    def get_olap_size(self, level: int) -> int:
        """Get the overlap size for a decimation level.

        :param level: Decimation level index.

        :return: Overlap size in samples.
        """
        self.check_level(level)
        return self.olap_sizes[level]


class WindowSetup(ResisticsProcess):
    """Setup WindowParameters

    WindowSetup outputs the WindowParameters to use for windowing decimated
    time data.

    Window parameters are simply the window and overlap sizes for each
    decimation level.

    :param min_size: Minimum window size, by default 128
    :param min_olap: Minimum overlap size, by default 32
    :param win_factor: Window factor, by default 4. Window sizes are calculated by sampling
        frequency / 4 to ensure sufficient frequency resolution. If the
        sampling frequency is small, window size will be adjusted to
        min_size
    :param olap_proportion: The proportion of the window size to use as the overlap, by default
        0.25. For example, for a window size of 128, the overlap would be
        0.25 * 128 = 32
    :param min_n_wins: The minimum number of windows needed in a decimation level, by
        default 5
    :param win_sizes: Explicit define window sizes, by default None. Must have the same
        length as number of decimation levels
    :param olap_sizes: Explicitly define overlap sizes, by default None. Must have the same
        length as number of decimation levels

    **Examples**

    Generate decimation and windowing parameters for data sampled at 0.05 Hz or
    20 seconds sampling period

    ```{doctest}
    >>> from resistics.decimate import DecimationSetup
    >>> from resistics.window import WindowSetup
    >>> dec_params = DecimationSetup(n_levels=3, per_level=3).run(0.05)
    >>> dec_params.summary()
    {
        'fs': 0.05,
        'n_levels': 3,
        'per_level': 3,
        'min_samples': 256,
        'eval_freqs': [
            0.0125,
            0.008838834764831844,
            0.00625,
            0.004419417382415922,
            0.003125,
            0.002209708691207961,
            0.0015625,
            0.0011048543456039805,
            0.00078125
        ],
        'dec_factors': [1, 2, 8],
        'dec_increments': [1, 2, 4],
        'dec_fs': [0.05, 0.025, 0.00625]
    }
    >>> win_params = WindowSetup().run(dec_params.n_levels, dec_params.dec_fs)
    >>> win_params.summary()
    {
        'n_levels': 3,
        'min_n_wins': 5,
        'win_sizes': [128, 128, 128],
        'olap_sizes': [32, 32, 32]
    }

    ```

    Window parameters can also be explicitly defined

    ```{doctest}
    >>> from resistics.decimate import DecimationSetup
    >>> from resistics.window import WindowSetup
    >>> dec_setup = DecimationSetup(n_levels=3, per_level=3)
    >>> dec_params = dec_setup.run(0.05)
    >>> win_setup = WindowSetup(win_sizes=[1000, 578, 104])
    >>> win_params = win_setup.run(dec_params.n_levels, dec_params.dec_fs)
    >>> win_params.summary()
    {
        'n_levels': 3,
        'min_n_wins': 5,
        'win_sizes': [1000, 578, 104],
        'olap_sizes': [250, 144, 32]
    }

    ```

    When providing explicit window and overlap sizes, the size of the overlap
    is expected to be less than the size of the window

    ```{doctest}
    >>> from resistics.decimate import DecimationSetup
    >>> from resistics.window import WindowSetup
    >>> dec_setup = DecimationSetup(n_levels=3, per_level=3)
    >>> dec_params = dec_setup.run(0.05)
    >>> win_setup = WindowSetup(win_sizes=[1000, 578, 104], olap_sizes=[1001, 600, 25])
    >>> win_params = win_setup.run(dec_params.n_levels, dec_params.dec_fs)
    Traceback (most recent call last):
    ...
    ValueError: Invalid overlaps found [ True  True False]

    ```
    """

    input_types: ClassVar[dict[str, str]] = {"dec_data": "decimated_data"}
    output_type: ClassVar[str] = "window_parameters"
    include_in_default_parameters: ClassVar[bool] = True

    min_size: int = 128
    min_olap: int = 32
    win_factor: int = 4
    olap_proportion: float = 0.25
    min_n_wins: int = 5
    win_sizes: list[int] | None = None
    olap_sizes: list[int] | None = None

    def execute(self, inputs: dict[str, Any], context: Any) -> WindowParameters:
        """Build window parameters from flow decimated data.

        :param inputs: Flow inputs containing ``dec_data``.
        :param context: Runtime context; unused by this process.

        :return: Window parameters for the available decimation levels.
        """
        del context
        dec_data = inputs["dec_data"]
        return self.run(dec_data.metadata.n_levels, dec_data.metadata.fs)

    def run(self, n_levels: int, dec_fs: list[float]) -> WindowParameters:
        """Calculate window and overlap sizes for each decimation level based on
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

        :param n_levels: The number of decimation levels
        :param dec_fs: The sampling frequencies for each decimation level

        :return: The window parameters, the window sizes and overlaps for each decimation level

        :raises ValueError: If sizes do not cover every level or an overlap exceeds its window.
        """
        if self.win_sizes is None:
            win_sizes = self._get_win_sizes(n_levels, dec_fs)
        else:
            win_sizes = list(self.win_sizes)

        if len(win_sizes) < n_levels:
            raise ValueError(f"Num. windows {len(win_sizes)} < n_levels {n_levels}")
        if len(win_sizes) > n_levels:
            # this may happen with user input windows
            # but decimated data has fewer levels
            win_sizes = win_sizes[:n_levels]

        if self.olap_sizes is None:
            olap_sizes = self._get_olap_sizes(win_sizes)
        else:
            olap_sizes = self.olap_sizes

        if len(olap_sizes) < n_levels:
            raise ValueError(f"Num. overlaps {len(olap_sizes)} < n_levels {n_levels}")
        if len(olap_sizes) > n_levels:
            # this may happen with user input windows
            # but decimated data has fewer levels
            olap_sizes = olap_sizes[:n_levels]

        invalid_olaps = np.greater(olap_sizes, win_sizes)
        if np.any(invalid_olaps):
            raise ValueError(f"Invalid overlaps found {invalid_olaps}")

        return WindowParameters(
            n_levels=n_levels,
            min_n_wins=self.min_n_wins,
            win_sizes=win_sizes,
            olap_sizes=olap_sizes,
        )

    def _get_win_sizes(self, n_levels: int, dec_fs: list[float]) -> list[int]:
        """Get the window sizes

        :param n_levels: The number of decimation levels
        :param dec_fs: The sampling frequencies for each decimation level

        :return: Window sizes
        """
        win_sizes = []
        for ilevel in range(n_levels):
            win_size = dec_fs[ilevel] // self.win_factor
            if win_size < self.min_size:
                win_size = self.min_size
            win_sizes.append(int(win_size))
        return win_sizes

    def _get_olap_sizes(self, win_sizes: list[int]) -> list[int]:
        """Get overlap sizes

        :param win_sizes: The window sizes

        :return: The overlap sizes
        """
        olap_sizes = []
        for win_size in win_sizes:
            olap_size = int(win_size * self.olap_proportion)
            if olap_size < self.min_olap:
                olap_size = self.min_olap
            olap_sizes.append(olap_size)
        return olap_sizes


class WindowedLevelMetadata(Metadata):
    """Metadata for a windowed level"""

    fs: float
    """The sampling frequency for the decimation level"""
    n_wins: int
    """The number of windows"""
    win_size: PositiveInt
    """The window size in samples"""
    olap_size: PositiveInt
    """The overlap size in samples"""
    index_offset: int
    """The global window offset for local window 0"""

    @property
    def dt(self):
        """Return the sample interval in seconds."""
        return 1 / self.fs


class WindowedMetadata(WriteableMetadata):
    """Metadata for windowed data"""

    model_config = ConfigDict(extra="ignore")

    fs: list[float]
    chans: list[str]
    n_chans: int = 0
    n_levels: int
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
    levels_metadata: list[WindowedLevelMetadata]
    ref_time: HighResDateTime
    history: History = History()


class WindowedData(ResisticsData):
    """Windows of a DecimatedData object

    The windowed data is stored in a dictionary attribute named data. This is
    a dictionary with an entry for each decimation level. The shape for a single
    decimation level is as follows:

    n_wins x n_chans x n_samples

    :param metadata: Metadata for the windowed data.
    :param data: Per-level arrays shaped as windows, channels, and samples.
    """

    def __init__(
        self,
        metadata: WindowedMetadata,
        data: dict[int, np.ndarray],
    ) -> None:
        logger.debug(f"Creating WindowedData with data type {data[0].dtype}")
        self.metadata = metadata
        self.data = data

    def get_level(self, level: int) -> np.ndarray:
        """Get windows for a decimation level

        :param level: The decimation level

        :return: The window array

        :raises ValueError: If decimation level is not within range
        """
        if level >= self.metadata.n_levels:
            raise ValueError(f"Level {level} not <= max {self.metadata.n_levels - 1}")
        return self.data[level]

    def get_local(self, level: int, local_win: int) -> np.ndarray:
        """Get window using local index

        :param level: The decimation level
        :param local_win: Local window index

        :return: Window data

        :raises ValueError: If local window index is out of range
        """
        n_wins = self.metadata.levels_metadata[level].n_wins
        if local_win < 0 or local_win >= n_wins:
            raise ValueError(f"Local window {local_win} not 0 <= local_win < {n_wins}")
        return self.get_level(level)[local_win]

    def get_global(self, level: int, global_win: int) -> np.ndarray:
        """Get window using global index

        :param level: The decimation level
        :param global_win: Global window index

        :return: Window data
        """
        index_offset = self.metadata.levels_metadata[level].index_offset
        return self.get_local(level, global_win + index_offset)

    def get_chan(self, level: int, chan: str) -> np.ndarray:
        """Get all the windows for a channel

        :param level: The decimation level
        :param chan: The channel

        :return: The data for the channels

        :raises ChannelNotFoundError: If the channel is not found in the data
        """
        from resistics.errors import ChannelNotFoundError

        if chan not in self.metadata.chans:
            raise ChannelNotFoundError(chan, self.metadata.chans)
        idx = self.metadata.chans.index(chan)
        return self.get_level(level)[..., idx, :]

    def to_string(self) -> str:
        """Render class information.

        :return: Human-readable windowed metadata.
        """
        return self.metadata.to_string()


class Windower(ResisticsProcess):
    """Windows DecimatedData

    This is the primary window making process for resistics and should be used
    when alignment of windows with a site or across sites is required.

    This method uses numpy striding to produce window views into the decimated
    data.

    **See Also**

    WindowerTarget : A windower to make a target number of windows

    **Examples**

    The Windower windows a DecimatedData object given a reference time and some
    window parameters.

    There's quite a few imports needed for this example. Begin by doing the
    imports, defining a reference time and generating random decimated data.

    ```{doctest}
    >>> from resistics.sampling import to_datetime
    >>> from resistics.testing import decimated_data_linear
    >>> from resistics.window import WindowSetup, Windower
    >>> dec_data = decimated_data_linear(fs=128)
    >>> ref_time = dec_data.metadata.first_time
    >>> print(dec_data.to_string()) # doctest: +NORMALIZE_WHITESPACE
    <class 'resistics.decimate.DecimatedData'>
               fs        dt  n_samples           first_time                        last_time
    level
    0      2048.0  0.000488      16384  2021-01-01 00:00:00  2021-01-01 00:00:07.99951171875
    1       512.0  0.001953       4096  2021-01-01 00:00:00    2021-01-01 00:00:07.998046875
    2       128.0  0.007812       1024  2021-01-01 00:00:00      2021-01-01 00:00:07.9921875

    ```

    Next, initialise the window parameters. For this example, use small windows,
    which will make inspecting them easier.

    ```{doctest}
    >>> win_params = WindowSetup(win_sizes=[16,16,16], min_olap=4).run(dec_data.metadata.n_levels, dec_data.metadata.fs)
    >>> win_params.summary()
    {
        'n_levels': 3,
        'min_n_wins': 5,
        'win_sizes': [16, 16, 16],
        'olap_sizes': [4, 4, 4]
    }

    ```

    Perform the windowing. This actually creates views into the decimated data
    using the numpy.lib.stride_tricks.sliding_window_view function. The shape
    for a data array at a decimation level is: n_wins x n_chans x win_size. The
    information about each level is also in the levels_metadata attribute of
    WindowedMetadata.

    ```{doctest}
    >>> from loguru import logger
    >>> logger.disable("resistics")
    >>> win_data = Windower().run(ref_time, win_params, dec_data)
    >>> logger.enable("resistics")
    >>> win_data.data[0].shape
    (1365, 2, 16)
    >>> for level_metadata in win_data.metadata.levels_metadata:
    ...     level_metadata.summary()
    {
        'fs': 2048.0,
        'n_wins': 1365,
        'win_size': 16,
        'olap_size': 4,
        'index_offset': 0
    }
    {
        'fs': 512.0,
        'n_wins': 341,
        'win_size': 16,
        'olap_size': 4,
        'index_offset': 0
    }
    {
        'fs': 128.0,
        'n_wins': 85,
        'win_size': 16,
        'olap_size': 4,
        'index_offset': 0
    }

    ```

    Let's look at an example of data from the first decimation level for the
    first channel. This is simply a linear set of data ranging from 0...16_383.

    ```{doctest}
    >>> dec_data.data[0][0].shape
    (16384,)
    >>> dec_data.data[0][0][[0, 1, 2, -3, -2, -1]]
    array([    0,     1,     2, 16381, 16382, 16383])

    ```

    Inspecting the first few windows shows they are as expected including the
    overlap.

    ```{doctest}
    >>> win_data.data[0][0, 0]
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])
    >>> win_data.data[0][1, 0]
    array([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27])
    >>> win_data.data[0][2, 0]
    array([24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])

    ```
    """

    input_types: ClassVar[dict[str, str]] = {
        "win_params": "window_parameters",
        "dec_data": "decimated_data",
    }
    output_type: ClassVar[str] = "windowed_data"
    runtime_requirements: ClassVar[list[str]] = ["reference_time"]
    include_in_default_parameters: ClassVar[bool] = True

    def execute(self, inputs: dict[str, Any], context: Any) -> WindowedData:
        """Window flow data using the project reference time from context.

        :param inputs: Flow inputs containing window parameters and decimated data.
        :param context: Runtime context containing the project reference time.

        :return: Windowed decimation levels aligned to the reference time.
        """
        return self.run(
            context["reference_time"], inputs["win_params"], inputs["dec_data"]
        )

    def run(
        self,
        ref_time: RSDateTime,
        win_params: WindowParameters,
        dec_data: DecimatedData,
    ) -> WindowedData:
        """Perform windowing of DecimatedData

        :param ref_time: The reference time
        :param win_params: The window parameters
        :param dec_data: The decimated data

        :return: Windows for decimated data

        :raises ProcessRunError: If the number of windows calculated in the window table does not
            match the size of the array views
        """
        metadata_dict = dec_data.metadata.model_dump()
        data = {}
        win_levels_metadata = []
        messages = []
        for ilevel in range(0, dec_data.metadata.n_levels):
            logger.info(f"Windowing decimation level {ilevel}")
            win_size = win_params.get_win_size(ilevel)
            olap_size = win_params.get_olap_size(ilevel)
            level_metadata = dec_data.metadata.levels_metadata[ilevel]
            win_table = get_win_table(ref_time, level_metadata, win_size, olap_size)
            n_wins = len(win_table.index)
            logger.info(f"{n_wins} windows, size {win_size}, overlap {olap_size}")
            messages.append(f"Level {ilevel}, generated {n_wins} windows")
            messages.append(f"Window size {win_size}, olap_size {olap_size}")

            if n_wins < win_params.min_n_wins:
                logger.debug(f"Number windows {n_wins} < min. {win_params.min_n_wins}")
                messages.append(f"Num. windows {n_wins} < min. {win_params.min_n_wins}")
                messages.append(f"Level {ilevel} incomplete, terminating windowing")
                break

            win_level_data = self._get_level_data(
                dec_data.get_level(ilevel),
                win_table,
                dec_data.metadata.n_chans,
                win_size,
                olap_size,
            )
            if win_level_data.shape[0] != n_wins:
                raise ProcessRunError(
                    self.name,
                    f"Num. windows mismatch {win_level_data.shape[0]} != {n_wins}",
                )
            win_level_metadata = self._get_level_metadata(
                level_metadata,
                win_table,
                win_size,
                olap_size,
            )
            data[ilevel] = win_level_data
            win_levels_metadata.append(win_level_metadata)
        metadata_dict["ref_time"] = ref_time
        metadata = self._get_metadata(metadata_dict, win_levels_metadata)
        metadata.history.add_record(self._get_record(messages))
        logger.info("Windowing completed")
        return WindowedData(metadata, data)

    def _get_level_data(
        self,
        data: np.ndarray,
        win_table: pd.DataFrame,
        n_chans: int,
        win_size: int,
        olap_size: int,
    ) -> np.ndarray:
        """Get window data for a decimation level

        :param data: The decimated time data for the level
        :param win_table: The window table
        :param n_chans: The number of channels
        :param win_size: The window size
        :param olap_size: The overlap size

        :return: Sliding window views in an array for the decimation level
        """
        from numpy.lib.stride_tricks import sliding_window_view

        from_sample = win_table.loc[0, "from_sample"]
        to_sample = win_table.loc[win_table.index[-1], "from_sample"]
        increment_size = win_size - olap_size
        view = np.squeeze(
            sliding_window_view(data, window_shape=(n_chans, win_size), writeable=True)
        )
        return view[from_sample : to_sample + 1 : increment_size]

    def _get_level_metadata(
        self,
        level_metadata: DecimatedLevelMetadata,
        win_table: pd.DataFrame,
        win_size: int,
        olap_size: int,
    ) -> WindowedLevelMetadata:
        """Get windowed metadata for a decimation level.

        :param level_metadata: Source decimation-level metadata.
        :param win_table: Local/global window and sample mapping.
        :param win_size: Window size in samples.
        :param olap_size: Overlap size in samples.

        :return: Metadata for the windowed level.

        :raises ValueError: If the local-to-global window offset is inconsistent.
        """
        offset = (win_table["global"] - win_table["local"]).unique()
        if len(offset) != 1:
            raise ValueError("Malformed window table, varying local to global offset")
        return WindowedLevelMetadata(
            fs=level_metadata.fs,
            n_wins=len(win_table.index),
            win_size=win_size,
            olap_size=olap_size,
            index_offset=offset[0],
        )

    def _get_metadata(
        self,
        metadata_dict: dict[str, Any],
        levels_metadata: list[WindowedLevelMetadata],
    ) -> WindowedMetadata:
        """Get aggregate metadata for windowed data.

        :param metadata_dict: Source decimated-data metadata.
        :param levels_metadata: Metadata for each completed windowed level.

        :return: Aggregate windowed-data metadata.
        """
        metadata_dict.pop("file_info")
        metadata_dict["n_levels"] = len(levels_metadata)
        metadata_dict["levels_metadata"] = levels_metadata
        return WindowedMetadata(**metadata_dict)


class WindowerTarget(Windower):
    """Windower that selects window sizes to meet a target number of windows

    The minimum window size in window parameters will be respected even if the
    generated number of windows is below the target. This is to avoid situations
    where excessively small windows sizes are selected.

    ```{warning}
    This process is primarily useful for quick processing of a single
    measurement and should not be used when any alignment of windows is
    required within a site or across sites.
    ```
    :param target: The target number of windows for each decimation level
    :param olap_proportion: The overlap proportion of the window size

    **See Also**

    Windower : The window making process to use when alignment is required
    """

    target: int = 1000
    min_size: int = 64
    olap_proportion: float = 0.25

    def run(
        self,
        ref_time: RSDateTime,
        win_params: WindowParameters,
        dec_data: DecimatedData,
    ) -> WindowedData:
        """Window every decimation level toward the configured target count.

        :param ref_time: Reference time; retained for the common windower interface.
        :param win_params: Window constraints, including the minimum window count.
        :param dec_data: Decimated data to window.

        :return: Per-level window views and their metadata.
        """
        metadata_dict = dec_data.metadata.model_dump()
        data = {}
        win_levels_metadata = []
        messages = []
        for ilevel in range(0, dec_data.metadata.n_levels):
            logger.info(f"Windowing decimation level {ilevel}")
            level_metadata = dec_data.metadata.levels_metadata[ilevel]
            win_size = self._get_win_size(level_metadata)
            olap_size = int(np.floor(self.olap_proportion * win_size))
            win_table = get_win_table(ref_time, level_metadata, win_size, olap_size)
            n_wins = len(win_table.index)
            logger.info(f"{n_wins} windows, size {win_size}, overlap {olap_size}")
            messages.append(f"Level {ilevel}, generated {n_wins} windows")
            messages.append(f"Window size {win_size}, olap_size {olap_size}")

            if n_wins < win_params.min_n_wins:
                logger.debug(f"Number windows {n_wins} < min. {win_params.min_n_wins}")
                messages.append(f"Num. windows {n_wins} < min. {win_params.min_n_wins}")
                messages.append(f"Level {ilevel} incomplete, terminating windowing")
                break

            win_level_data = self._get_level_data(
                dec_data.get_level(ilevel),
                win_table,
                dec_data.metadata.n_chans,
                win_size,
                olap_size,
            )
            win_level_metadata = self._get_level_metadata(
                level_metadata,
                win_table,
                win_size,
                olap_size,
            )
            data[ilevel] = win_level_data
            win_levels_metadata.append(win_level_metadata)
        metadata_dict["ref_time"] = metadata_dict["first_time"]
        metadata = self._get_metadata(metadata_dict, win_levels_metadata)
        metadata.history.add_record(self._get_record(messages))
        logger.info("Windowing completed")
        return WindowedData(metadata, data)

    def _get_win_size(self, level_metadata: DecimatedLevelMetadata) -> int:
        """Get window size that gives close to the target number of windows

        Windows increment by (window size - overlap size), therefore the
        follwing equation is solved,

        ```{math}
        n_{samples} / ((1 - n_{overlap})*n_{window}) = target
        ```
        Rearrangning, get,

        ```{math}
        n_{window} = n_{samples} / ((1 - n_{overlap})*target)
        ```
        :param level_metadata: The metadata for the decimation level

        :return: The window size
        """
        win_size = level_metadata.n_samples / ((1 - self.olap_proportion) * self.target)
        win_size = int(np.floor(win_size))
        if win_size < self.min_size:
            return self.min_size
        return win_size


class WindowedDataWriter(ResisticsWriter):
    """Writer of resistics windowed data"""

    def run(self, dir_path: Path, data: ResisticsData) -> None:
        """Write out WindowedData

        :param dir_path: The directory path to write to
        :param data: Windowed data to write out

        :raises TypeError: If ``data`` is not windowed data.
        :raises WriteError: If unable to write to the directory
        """
        from resistics.errors import WriteError

        if not isinstance(data, WindowedData):
            raise TypeError("WindowedDataWriter requires WindowedData")
        win_data = data
        if not self._check_dir(dir_path):
            raise WriteError(dir_path, "Unable to write to directory, check logs")
        logger.info(f"Writing windowed data to {dir_path}")
        metadata_path = dir_path / "metadata.json"
        data_path = dir_path / "data"
        save_compressed_arrays(
            data_path, {str(level): values for level, values in win_data.data.items()}
        )
        metadata = win_data.metadata.model_copy()
        metadata.history.add_record(self._get_writer_record(dir_path, type(win_data)))
        metadata.write(metadata_path)


class WindowedDataReader(ResisticsProcess):
    """Reader of resistics windowed data"""

    def run(
        self, dir_path: Path, metadata_only: bool = False
    ) -> WindowedMetadata | WindowedData:
        """Read WindowedData

        :param dir_path: The directory path to read from
        :param metadata_only: Flag for getting metadata only, by default False

        :return: The WindowedData or WindowedMetadata if metadata_only is True

        :raises ReadError: If the directory does not exist
        """
        from resistics.errors import ReadError

        if not dir_path.exists():
            raise ReadError(dir_path, "Directory does not exist")
        logger.info(f"Reading windowed data from {dir_path}")
        metadata_path = dir_path / "metadata.json"
        metadata = WindowedMetadata.model_validate_json(metadata_path.read_bytes())
        if metadata_only:
            return metadata
        data_path = dir_path / "data.npz"
        npz_file = np.load(data_path)
        data = {int(level): npz_file[level] for level in npz_file.files}
        messages = [f"Windowed data read from {dir_path}"]
        metadata.history.add_record(self._get_record(messages))
        return WindowedData(metadata, data)
