"""Module for dealing with sampling and dates including:

- Converting from samples to datetimes
- Converting from datetimes to samples
- All datetime, timedelta types are aliased as RSDateTime and RSTimeDelta
- This is to ease type hinting if the base datetime and timedelta classes change
- Currently, resistics uses attodatetime and attotimedelta from attotime
- attotime is a high precision datetime library
"""
from loguru import logger
from typing import Union, Tuple, Optional, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from attotime import attodatetime, attotimedelta

DateTimeLike = Union[str, pd.Timestamp, datetime]
TimeDeltaLike = Union[float, timedelta, pd.Timedelta]
RSDateTime = attodatetime
RSTimeDelta = attotimedelta


class HighResDateTime(RSDateTime):
    """Wrapper around RSDateTime to use for pydantic"""

    @classmethod
    def __get_validators__(cls):
        """Yield validators for RSDateTime"""
        yield cls.validate

    @classmethod
    def __modify_schema__(cls, field_schema):
        """Add to the pydantic schema"""
        field_schema.update(
            pattern="%Y-%m-%d %H:%M:%S.%f_%o_%q_%v",
            examples=["2021-01-01 00:00:00.000061_035156_250000_000000"],
        )

    @classmethod
    def validate(cls, val: Any):
        """Validator to be used by pydantic"""
        if isinstance(val, RSDateTime):
            return val
        if isinstance(val, (str, pd.Timestamp, datetime)):
            return to_datetime(val)
        raise TypeError(f"Type {type(val)} not recognised for RSDateTime")

    def __repr__(self) -> str:
        return super().__repr__()


def datetime_to_string(time: RSDateTime) -> str:
    """Convert a datetime to a string.

    Parameters
    ----------
    time : RSDateTime
        Resistics datetime

    Returns
    -------
    str
        String representation

    Examples
    --------
    >>> from resistics.sampling import to_datetime, to_timedelta, datetime_to_string
    >>> time = to_datetime("2021-01-01") + to_timedelta(1/16384)
    >>> datetime_to_string(time)
    '2021-01-01 00:00:00.000061_035156_250000_000000'
    """
    return time.strftime("%Y-%m-%d %H:%M:%S.%f_%o_%q_%v")


def datetime_from_string(time: str) -> RSDateTime:
    """Convert a string back to a datetime.

    Only a fixed format is allowed %Y-%m-%d %H:%M:%S.%f_%o_%q_%v

    Parameters
    ----------
    time : str
        time as a string

    Returns
    -------
    RSDateTime
        The resistics datetime

    Examples
    --------
    >>> from resistics.sampling import to_datetime, to_timedelta
    >>> from resistics.sampling import datetime_to_string, datetime_from_string
    >>> time = to_datetime("2021-01-01") + to_timedelta(1/16384)
    >>> time_str = datetime_to_string(time)
    >>> time_str
    '2021-01-01 00:00:00.000061_035156_250000_000000'
    >>> datetime_from_string(time_str)
    attotime.objects.attodatetime(2021, 1, 1, 0, 0, 0, 61, 35.15625)
    """
    return RSDateTime.strptime(time, "%Y-%m-%d %H:%M:%S.%f_%o_%q_%v")


def to_datetime(time: DateTimeLike) -> RSDateTime:
    """Convert a string, pd.Timestamp or datetime object to a RSDateTime.

    RSDateTime uses attodatetime which is a high precision datetime format
    helpful for high sampling frequencies.

    Parameters
    ----------
    time : DateTimeLike
        Input time as either a string, pd.Timestamp or native python datetime

    Returns
    -------
    RSDateTime
        High precision datetime object

    Examples
    --------
    >>> import pandas as pd
    >>> from resistics.sampling import to_datetime
    >>> a = "2021-01-01 00:00:00"
    >>> to_datetime(a)
    attotime.objects.attodatetime(2021, 1, 1, 0, 0, 0, 0, 0)
    >>> str(to_datetime(a))
    '2021-01-01 00:00:00'
    >>> b = pd.Timestamp(a)
    >>> str(to_datetime(b))
    '2021-01-01 00:00:00'
    >>> c = pd.Timestamp(a).to_pydatetime()
    >>> str(to_datetime(c))
    '2021-01-01 00:00:00'
    """
    if isinstance(time, str):
        try:
            return datetime_from_string(time)
        except ValueError:
            time = pd.Timestamp(time)
    elif isinstance(time, datetime):
        time = pd.Timestamp(time)
    else:
        pass
    return RSDateTime(
        time.year,
        time.month,
        time.day,
        time.hour,
        time.minute,
        time.second,
        time.microsecond,
        time.nanosecond,
    )


def to_timestamp(time: RSDateTime) -> pd.Timestamp:
    """
    Convert a RSDateTime to a pandas Timestamp

    Parameters
    ----------
    time : RSDateTime
        An RSDateTime instance

    Returns
    -------
    pd.Timestamp
        RSDateTime converted to Timestamp

    Examples
    --------
    >>> from resistics.sampling import to_datetime, to_timestamp
    >>> time = to_datetime("2021-01-01 00:30:00.345")
    >>> print(time)
    2021-01-01 00:30:00.345
    >>> to_timestamp(time)
    Timestamp('2021-01-01 00:30:00.345000')
    """
    return pd.Timestamp(time.isoformat())


def to_timedelta(delta: TimeDeltaLike) -> RSTimeDelta:
    """Get a RSTimeDelta object by providing seconds as a float or a
    pd.Timedelta.

    RSTimeDelta uses attotimedelta, a high precision timedelta object. This can
    be useful for high sampling frequencies.

    .. warning::

        At high time resolutions, there are machine precision errors that
        come into play. Therefore, if nanoseconds < 0.0001, it will be zeroed
        out

    Parameters
    ----------
    delta : TimeDeltaLike
        Timedelta as a float (assumed to be seconds), timedelta or pd.Timedelta

    Returns
    -------
    RSTimeDelta
        High precision timedelta

    Examples
    --------
    >>> import pandas as pd
    >>> from resistics.sampling import to_timedelta

    Low frequency sampling

    >>> fs = 0.0000125
    >>> to_timedelta(1/fs)
    attotime.objects.attotimedelta(0, 80000)
    >>> str(to_timedelta(1/fs))
    '22:13:20'
    >>> fs = 0.004
    >>> to_timedelta(1/fs)
    attotime.objects.attotimedelta(0, 250)
    >>> str(to_timedelta(1/fs))
    '0:04:10'
    >>> fs = 0.3125
    >>> str(to_timedelta(1/fs))
    '0:00:03.2'

    Higher frequency sampling

    >>> fs = 4096
    >>> to_timedelta(1/fs)
    attotime.objects.attotimedelta(0, 0, 244, 140.625)
    >>> str(to_timedelta(1/fs))
    '0:00:00.000244140625'
    >>> fs = 65_536
    >>> str(to_timedelta(1/fs))
    '0:00:00.0000152587890625'
    >>> fs = 524_288
    >>> str(to_timedelta(1/fs))
    '0:00:00.0000019073486328125'

    to_timedelta can also accept pandas Timedelta objects

    >>> str(to_timedelta(pd.Timedelta(1, "s")))
    '0:00:01'
    """
    from math import floor

    eps = 0.0001
    if isinstance(delta, (int, float)):
        seconds = int(floor(delta))
        delta = (delta - seconds) * 1_000_000
        microseconds = int(floor(delta))
        delta = delta - microseconds
        nanoseconds = delta * 1_000
        if nanoseconds != 0 and nanoseconds < eps:
            logger.debug(f"Setting nanoseconds {nanoseconds} to 0")
            nanoseconds = 0
        return RSTimeDelta(
            seconds=seconds, microseconds=microseconds, nanoseconds=nanoseconds
        )
    if not isinstance(delta, pd.Timedelta):
        delta = pd.Timedelta(delta)
    return RSTimeDelta(
        days=delta.days,
        seconds=delta.seconds,
        microseconds=delta.microseconds,
        nanoseconds=delta.nanoseconds,
    )


def to_seconds(delta: RSTimeDelta) -> Tuple[float, float]:
    """Convert a timedelta to seconds as a float.

    Returns a Tuple, the first value being the days in the delta converted to
    seconds, the second entry in the Tuple is the remaining amount of time
    converted to seconds.

    Parameters
    ----------
    delta : RSTimeDelta
        timedelta

    Returns
    -------
    days_in_seconds
        The days in the delta converted to seconds
    remaining_in_seconds
        The remaining amount of time in the delta converted to seconds

    Examples
    --------
    Example with a small timedelta

    >>> from resistics.sampling import to_datetime, to_timedelta, to_seconds
    >>> a = to_timedelta(1/4_096)
    >>> str(a)
    '0:00:00.000244140625'
    >>> days_in_seconds, remaining_in_seconds = to_seconds(a)
    >>> days_in_seconds
    0
    >>> remaining_in_seconds
    0.000244140625

    Example with a larger timedelta

    >>> a = to_datetime("2021-01-01 00:00:00")
    >>> b = to_datetime("2021-02-01 08:24:30")
    >>> days_in_seconds, remaining_in_seconds = to_seconds(b-a)
    >>> days_in_seconds
    2678400
    >>> remaining_in_seconds
    30270.0
    """
    days_in_seconds = delta.days * (3600 * 24)
    microseconds_in_seconds = delta.microseconds / 1_000_000
    nanoseconds_in_seconds = float(delta.nanoseconds) / 1_000_000_000
    remaining_in_seconds = (
        delta.seconds + microseconds_in_seconds + nanoseconds_in_seconds
    )
    return days_in_seconds, remaining_in_seconds


def to_n_samples(delta: RSTimeDelta, fs: float, method: str = "round") -> int:
    """Convert a timedelta to number of samples

    This method is inclusive of start and end sample.

    Parameters
    ----------
    delta : RSTimeDelta
        The timedelta
    fs : float
        The sampling frequency
    method : str
        Method to deal with floats, default is 'round'. Other options include
        'ceil' and 'floor'

    Returns
    -------
    int
        The number of samples in the timedelta

    Examples
    --------
    With sampling frequency of 4096 Hz

    >>> from resistics.sampling import to_timedelta, to_n_samples
    >>> fs = 4096
    >>> delta = to_timedelta(8*3600 + (21/fs))
    >>> str(delta)
    '8:00:00.005126953125'
    >>> to_n_samples(delta, fs=fs)
    117964822
    >>> check = (8*3600)*fs + 21
    >>> check
    117964821
    >>> check_inclusive = check + 1
    >>> check_inclusive
    117964822

    With a sampling frequency of 65536 Hz

    >>> fs = 65_536
    >>> delta = to_timedelta(2*3600 + (40_954/fs))
    >>> str(delta)
    '2:00:00.624908447265625'
    >>> to_n_samples(delta, fs=fs)
    471900155
    >>> check = 2*3600*fs + 40_954
    >>> check
    471900154
    >>> check_inclusive = check + 1
    >>> check_inclusive
    471900155
    """
    from math import floor, ceil

    days_in_seconds, remaining_in_seconds = to_seconds(delta)
    n_samples = (days_in_seconds + remaining_in_seconds) * fs
    # add 1 to be inclusive of start and end
    n_samples += 1
    if n_samples.is_integer():
        return int(n_samples)

    if method == "floor":
        return int(floor(n_samples))
    elif method == "ceil":
        return int(ceil(n_samples))
    return int(round(n_samples))


def check_sample(n_samples: int, sample: int) -> bool:
    """
    Check sample is between 0 <= from_sample < n_samples

    Parameters
    ----------
    n_samples : int
        Number of samples
    sample : int
        Sample to check

    Returns
    -------
    bool
        Return True if no errors

    Raises
    ------
    ValueError
        If sample < 0
    ValueError
        If sample > n_samples

    Examples
    --------
    >>> from resistics.sampling import check_sample
    >>> check_sample(100, 45)
    True
    >>> check_sample(100, 100)
    Traceback (most recent call last):
    ...
    ValueError: Sample 100 must be < 100
    >>> check_sample(100, -1)
    Traceback (most recent call last):
    ...
    ValueError: Sample -1 must be >= 0
    """
    if sample < 0:
        raise ValueError(f"Sample {sample} must be >= 0")
    if sample >= n_samples:
        raise ValueError(f"Sample {sample} must be < {n_samples}")
    return True


def sample_to_datetime(
    fs: float, first_time: RSDateTime, sample: int, n_samples: Optional[int] = None
) -> RSDateTime:
    """Convert a sample to a pandas Timestamp.

    Parameters
    ----------
    fs : float
        The sampling frequency
    first_time : RSDateTime
        The first time
    sample : int
        The sample
    n_samples : Optional[int], optional
        The number of samples, used for checking, by default None.
        If provided, the sample is checked to make sure it's not out of bounds.

    Returns
    -------
    RSDateTime
        The timestamp of the sample

    Raises
    ------
    ValueError
        If n_samples is provided and sample is < 0 or >= n_samples

    Examples
    --------
    >>> import pandas as pd
    >>> from resistics.sampling import to_datetime, sample_to_datetime
    >>> fs = 512
    >>> first_time = to_datetime("2021-01-02 00:00:00")
    >>> sample = 512
    >>> sample_datetime = sample_to_datetime(fs, first_time, sample)
    >>> str(sample_datetime)
    '2021-01-02 00:00:01'
    """
    if n_samples is not None and (sample < 0 or sample >= n_samples):
        raise ValueError(
            f"Sample {sample} not between 0 and number of samples {n_samples}"
        )
    return first_time + sample * to_timedelta(1 / fs)


def samples_to_datetimes(
    fs: float,
    first_time: RSDateTime,
    from_sample: int,
    to_sample: int,
) -> Tuple[RSDateTime, RSDateTime]:
    """Convert from and to samples to datetimes.

    The first sample is assumed to be 0.

    Parameters
    ----------
    fs : float
        The sampling frequency in seconds
    first_time : RSDateTime
        The time of the first sample
    from_sample : int
        The sample to read data from
    to_sample : int
        The sample to read data to

    Returns
    -------
    from_time : RSDateTime
        The timestamp to read data from
    to_time : RSDateTime
        The timestamp to read data to

    Raises
    ------
    ValueError
        If from sample is greater than or equal to to sample

    Examples
    --------
    >>> import pandas as pd
    >>> from resistics.sampling import to_datetime, samples_to_datetimes
    >>> fs = 512
    >>> first_time = to_datetime("2021-01-02 00:00:00")
    >>> from_sample = 512
    >>> to_sample = 1024
    >>> from_time, to_time = samples_to_datetimes(fs, first_time, from_sample, to_sample)
    >>> str(from_time)
    '2021-01-02 00:00:01'
    >>> str(to_time)
    '2021-01-02 00:00:02'
    """
    if from_sample >= to_sample:
        raise ValueError(f"From sample {from_sample} >= to sample {to_sample}")

    from_time = sample_to_datetime(fs, first_time, from_sample)
    to_time = sample_to_datetime(fs, first_time, to_sample)
    return from_time, to_time


def check_from_time(
    first_time: RSDateTime,
    last_time: RSDateTime,
    from_time: RSDateTime,
) -> RSDateTime:
    """Check a from time.

    - If first time <= from_time <= last_time, it will be returned unchanged.
    - If from_time < first time, then first time will be returned.
    - If from_time > last time, it will raise a ValueError.

    Parameters
    ----------
    first_time : RSDateTime
        The time of the first sample
    last_time : RSDateTime
        The time of the last sample
    from_time : RSDateTime
        Time to get the data from

    Returns
    -------
    RSDateTime
        A from time adjusted as needed given the first and last sample time

    Raises
    ------
    ValueError
        If the from time is after the time of the last sample

    Examples
    --------
    With a from time between first and last time. This should be the normal use
    case.

    >>> from resistics.sampling import to_datetime, check_from_time
    >>> first_time = to_datetime("2021-01-02 00:00:00")
    >>> last_time = to_datetime("2021-01-02 23:00:00")
    >>> from_time = to_datetime("2021-01-02 03:00:00")
    >>> from_time = check_from_time(first_time, last_time, from_time)
    >>> str(from_time)
    '2021-01-02 03:00:00'

    An alternative scenario when from time is before the time of the first
    sample

    >>> from_time = to_datetime("2021-01-01 23:00:00")
    >>> from_time = check_from_time(first_time, last_time, from_time)
    >>> str(from_time)
    '2021-01-02 00:00:00'

    An error will be raised when from time is after the time of the last sample

    >>> from_time = to_datetime("2021-01-02 23:30:00")
    >>> from_time = check_from_time(first_time, last_time, from_time)
    Traceback (most recent call last):
    ...
    ValueError: From time 2021-01-02 23:30:00 greater than time of last sample 2021-01-02 23:00:00
    """
    if from_time > last_time:
        raise ValueError(
            f"From time {str(from_time)} greater than time of last sample {str(last_time)}"
        )

    delta_first = from_time - first_time
    if delta_first.total_nanoseconds() < 0:
        return first_time
    return from_time


def check_to_time(
    first_time: RSDateTime, last_time: RSDateTime, to_time: RSDateTime
) -> RSDateTime:
    """Check a to time.

    - If  first time <= to time <= last time, it will be returned unchanged.
    - If to time > last time, then last time will be returned.
    - If to time < first time, it will raise a ValueError.

    Parameters
    ----------
    first_time : RSDateTime
        The time of the first sample
    last_time : RSDateTime
        The time of the last sample
    to_time : RSDateTime
        Time to get the data to

    Returns
    -------
    RSDateTime
        A to time adjusted as needed

    Raises
    ------
    ValueError
        If the to time is before the time of the first sample

    Examples
    --------
    With a to time between first and last time. This should be the normal use
    case.

    >>> from resistics.sampling import to_datetime, check_to_time
    >>> first_time = to_datetime("2021-01-02 00:00:00")
    >>> last_time = to_datetime("2021-01-02 23:00:00")
    >>> to_time = to_datetime("2021-01-02 20:00:00")
    >>> to_time = check_to_time(to_time, last_time, to_time)
    >>> str(to_time)
    '2021-01-02 20:00:00'

    An alternative scenario when to time is after the time of the last sample

    >>> to_time = to_datetime("2021-01-02 23:30:00")
    >>> to_time = check_to_time(first_time, last_time, to_time)
    >>> str(to_time)
    '2021-01-02 23:00:00'

    An error will be raised when to time is before the time of the first sample

    >>> to_time = to_datetime("2021-01-01 23:30:00")
    >>> to_time = check_to_time(first_time, last_time, to_time)
    Traceback (most recent call last):
    ...
    ValueError: To time 2021-01-01 23:30:00 less than time of first sample 2021-01-02 00:00:00
    """
    if to_time < first_time:
        raise ValueError(
            f"To time {str(to_time)} less than time of first sample {str(first_time)}"
        )

    delta_last = last_time - to_time
    if delta_last.total_nanoseconds() < 0:
        return last_time
    return to_time


def from_time_to_sample(
    fs: float,
    first_time: RSDateTime,
    last_time: RSDateTime,
    from_time: RSDateTime,
) -> int:
    """Get the sample for the from time.

    Parameters
    ----------
    fs : float
        Sampling frequency Hz
    first_time : RSDateTime
        Time of first sample
    last_time : RSDateTime
        Time of last sample
    from_time : RSDateTime
        From time

    Returns
    -------
    int
        The sample coincident with or after the from time

    Examples
    --------
    >>> from resistics.sampling import to_datetime, from_time_to_sample
    >>> first_time = to_datetime("2021-01-01 00:00:00")
    >>> last_time = to_datetime("2021-01-02 00:00:00")
    >>> fs = 128
    >>> fs * 60 * 60
    460800
    >>> from_time = to_datetime("2021-01-01 01:00:00")
    >>> from_time_to_sample(fs, first_time, last_time, from_time)
    460800
    >>> from_time = to_datetime("2021-01-01 01:00:00.0078125")
    >>> from_time_to_sample(fs, first_time, last_time, from_time)
    460801
    """
    from_time = check_from_time(first_time, last_time, from_time)
    delta_first = from_time - first_time
    # n_samples returns the number of samples. Need to -1 to get the sample index
    return to_n_samples(delta_first, fs, method="ceil") - 1


def to_time_to_sample(
    fs: float,
    first_time: RSDateTime,
    last_time: RSDateTime,
    to_time: RSDateTime,
) -> int:
    """Get the to time sample.

    .. warning::

        This will return the sample of the to time. In cases where this will
        be used for a range, 1 should be added to it to ensure it is included.

    Parameters
    ----------
    fs : float
        Sampling frequency Hz
    first_time : RSDateTime
        Time of first sample
    last_time : RSDateTime
        Time of last sample
    to_time : RSDateTime
        The to time

    Returns
    -------
    int
        The sample coincident with or immediately before the to time

    Examples
    --------
    >>> from resistics.sampling import to_time_to_sample
    >>> first_time = to_datetime("2021-01-01 04:00:00")
    >>> last_time = to_datetime("2021-01-01 13:00:00")
    >>> fs = 4096
    >>> fs * 60 * 60
    14745600
    >>> to_time = to_datetime("2021-01-01 05:00:00")
    >>> to_time_to_sample(fs, first_time, last_time, to_time)
    14745600
    >>> fs * 70 * 60
    17203200
    >>> to_time = to_datetime("2021-01-01 05:10:00")
    >>> to_time_to_sample(fs, first_time, last_time, to_time)
    17203200
    """
    to_time = check_to_time(first_time, last_time, to_time)
    delta_first = to_time - first_time
    # n_samples returns the number of samples. Need to -1 to get the sample index
    return to_n_samples(delta_first, fs, method="floor") - 1


def datetimes_to_samples(
    fs: float,
    first_time: RSDateTime,
    last_time: RSDateTime,
    from_time: RSDateTime,
    to_time: RSDateTime,
) -> Tuple[int, int]:
    """Convert from and to time to samples.

    .. warning::

        If using these samples in ranging, the from sample can be left unchanged
        but one should be added to the to sample to ensure it is included.

    .. note::

        If from_time is not a sample timestamp, the next sample is taken
        If to_time is not a sample timestamp, the previous sample is taken

    Parameters
    ----------
    fs : float
        The sampling frequency in Hz
    first_time : RSDateTime
        The time of the first sample
    last_time : RSDateTime
        The time of the last sample
    from_time : RSDateTime
        A from time
    to_time : RSDateTime
        A to time

    Returns
    -------
    from_sample : int
        Sample to read data from
    to_sample : int
        Sample to read data to

    Examples
    --------
    >>> from resistics.sampling import to_datetime, datetimes_to_samples
    >>> first_time = to_datetime("2021-01-01 04:00:00")
    >>> last_time = to_datetime("2021-01-01 05:30:00")
    >>> from_time = to_datetime("2021-01-01 05:00:00")
    >>> to_time = to_datetime("2021-01-01 05:10:00")
    >>> fs = 16_384
    >>> fs * 60 * 60
    58982400
    >>> fs * 70 * 60
    68812800
    >>> from_sample, to_sample = datetimes_to_samples(fs, first_time, last_time, from_time, to_time)
    >>> from_sample
    58982400
    >>> to_sample
    68812800
    """
    from_sample = from_time_to_sample(fs, first_time, last_time, from_time)
    to_sample = to_time_to_sample(fs, first_time, last_time, to_time)
    return from_sample, to_sample


def datetime_array(
    first_time: RSDateTime,
    fs: float,
    n_samples: Optional[int] = None,
    samples: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Get a datetime array in high resolution.

    This will return a high resolution datetime array. This method is more
    computationally demanding than a pandas date_range. As a result, in cases
    where exact datetimes are not required, it is suggested to use
    datetime_array_estimate instead.

    Parameters
    ----------
    first_time : RSDateTime
        The first time
    fs : float
        The sampling frequency
    n_samples : Optional[int], optional
        The number of samples, by default None
    samples : Optional[np.ndarray], optional
        The samples for which to return a datetime, by default None

    Returns
    -------
    np.ndarray
        Numpy array of RSDateTimes

    Raises
    ------
    ValueError
        If both n_samples and samples is None

    Examples
    --------
    This examples shows the value of using higher resolution datetimes, however
    this is computationally more expensive.

    >>> import pandas as pd
    >>> from resistics.sampling import to_datetime, datetime_array
    >>> first_time = to_datetime("2021-01-01 00:00:00")
    >>> fs = 4096
    >>> n_samples = 100
    >>> arr = datetime_array(first_time, fs, n_samples=n_samples)
    >>> str(arr[-1])
    '2021-01-01 00:00:00.024169921875'
    >>> pdarr = pd.date_range(start="2021-01-01 00:00:00", freq=pd.Timedelta(1/4096, "s"), periods=n_samples)
    >>> pdarr[-1]
    Timestamp('2021-01-01 00:00:00.024169959', freq='244141N')
    """
    if n_samples is not None:
        return first_time + np.arange(n_samples) * to_timedelta(1 / fs)
    if samples is None:
        raise ValueError("One of n_samples or samples must be provided")
    return first_time + samples * to_timedelta(1 / fs)


def datetime_array_estimate(
    first_time: Union[RSDateTime, datetime, str, pd.Timestamp],
    fs: float,
    n_samples: Optional[int] = None,
    samples: Optional[np.ndarray] = None,
) -> pd.DatetimeIndex:
    """Estimate datetime array with lower precision but much faster
    performance.

    Parameters
    ----------
    first_time : Union[RSDateTime, datetime, str, pd.Timestamp]
        The first time
    fs : float
        The sampling frequency
    n_samples : Optional[int], optional
        The number of samples, by default None
    samples : Optional[np.ndarray], optional
        An array of samples to return datetimes for, by default None

    Returns
    -------
    pd.DatetimeIndex
        A pandas DatetimeIndex

    Raises
    ------
    ValueError
        If both n_samples and samples are None

    Examples
    --------
    >>> import pandas as pd
    >>> from resistics.sampling import to_datetime, datetime_array_estimate
    >>> first_time = to_datetime("2021-01-01 00:00:00")
    >>> fs = 128
    >>> n_samples = 1_000
    >>> arr = datetime_array_estimate(first_time, fs, n_samples=n_samples)
    >>> print(f"{arr[0]} - {arr[-1]}")
    2021-01-01 00:00:00 - 2021-01-01 00:00:07.804687500
    """
    if isinstance(first_time, RSDateTime):
        first_time = pd.to_datetime(first_time.isoformat())

    dt = pd.Timedelta(1 / fs, "s")
    if n_samples is not None:
        return pd.date_range(start=first_time, freq=dt, periods=n_samples)
    if samples is None:
        raise ValueError("One of n_samples or samples must be provided")
    return pd.to_datetime(pd.to_datetime(first_time) + samples * dt)
