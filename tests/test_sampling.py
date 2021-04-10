import pytest
from typing import Tuple, Union
from datetime import datetime, timedelta
from resistics.sampling import RSDateTime, RSTimeDelta, to_datetime, to_timedelta 
import pandas as pd


@pytest.mark.parametrize(
    "time, expected",
    [
        ("2021-01-02 00:00:00", RSDateTime(2021, 1, 2)),
        (pd.Timestamp("2021-01-02 00:00:00"), RSDateTime(2021, 1, 2)),
        (datetime(2021, 1, 2), RSDateTime(2021, 1, 2))
    ],
)
def test_to_datetime(time: Union[str, pd.Timestamp, datetime], expected):
    """Test converting to datetime"""
    assert to_datetime(time) == expected


@pytest.mark.parametrize(
    "delta, expected",
    [
        (1/4096, RSTimeDelta(seconds=1/4096)),
        (pd.Timedelta(0.1, "s"), RSTimeDelta(microseconds=100_000)),
        (timedelta(milliseconds=100), RSTimeDelta(microseconds=100_000))
    ],
)
def test_to_timedelta(delta: Union[float, timedelta, pd.Timedelta], expected: RSTimeDelta):
    """Test converting to timedelta"""
    assert to_timedelta(delta) == expected


@pytest.mark.parametrize(
    "fs, first_time, sample, expected",
    [
        (
            512,
            to_datetime("2021-01-02 00:00:00"),
            512,
            to_datetime("2021-01-02 00:00:01"),
        ),
        (
            16_384,
            to_datetime("2021-01-01 00:00:00"),
            20_000,
            RSDateTime(2021, 1, 1, 0, 0, 1, 220703, 125),
        ),        
    ],
)
def test_sample_to_datetimes(
    fs: float,
    first_time: RSDateTime,
    sample: int,
    expected: RSDateTime,
) -> None:
    """Test converting sample to datetimes"""
    from resistics.sampling import sample_to_datetime

    assert expected == sample_to_datetime(fs, first_time, sample)


@pytest.mark.parametrize(
    "fs, first_time, from_sample, to_sample, expected_from, expected_to",
    [
        (
            4096,
            to_datetime("2021-01-02 00:00:00"),
            10_010,
            25_999,
            RSDateTime(2021, 1, 2, 0, 0, 2, 443847, 656.25),
            RSDateTime(2021, 1, 2, 0, 0, 6, 347412, 109.375),
        ),
    ],
)
def test_samples_to_datetimes(
    fs: float,
    first_time: RSDateTime,
    from_sample: int,
    to_sample: int,
    expected_from: RSDateTime,
    expected_to: RSDateTime,
) -> None:
    """Test converting samples to datetimes"""
    from resistics.sampling import samples_to_datetimes

    from_time, to_time = samples_to_datetimes(fs, first_time, from_sample, to_sample)
    assert from_time == expected_from
    assert to_time == expected_to


@pytest.mark.parametrize(
    "first_time, last_time, from_time, expected, raises",
    [
        (
            to_datetime("2021-01-02 00:00:00"),
            to_datetime("2021-01-02 23:00:00"),
            to_datetime("2021-01-01 23:00:00"),
            to_datetime("2021-01-02 00:00:00"),
            False,
        ),
        (
            to_datetime("2021-01-02 00:00:00"),
            to_datetime("2021-01-02 23:00:00"),
            to_datetime("2021-01-02 03:00:00"),
            to_datetime("2021-01-02 03:00:00"),
            False,
        ),
        (
            to_datetime("2021-01-02 00:00:00"),
            to_datetime("2021-01-02 23:00:00"),
            to_datetime("2021-01-02 23:30:00"),
            to_datetime("2021-01-02 00:00:00"),
            True,
        ),
    ],
)
def test_check_from_time(
    first_time: RSDateTime,
    last_time: RSDateTime,
    from_time: RSDateTime,
    expected: RSDateTime,
    raises: bool,
) -> None:
    """Test adjusting from time"""
    from resistics.sampling import check_from_time

    if raises:
        with pytest.raises(ValueError):
            check_from_time(first_time, last_time, from_time)
    else:
        from_time = check_from_time(first_time, last_time, from_time)
        assert from_time == expected


@pytest.mark.parametrize(
    "first_time, last_time, to_time, expected, raises",
    [
        (
            to_datetime("2021-01-02 00:00:00"),
            to_datetime("2021-01-02 23:00:00"),
            to_datetime("2021-01-02 23:30:00"),
            to_datetime("2021-01-02 23:00:00"),
            False,
        ),
        (
            to_datetime("2021-01-02 00:00:00"),
            to_datetime("2021-01-02 23:00:00"),
            to_datetime("2021-01-02 03:00:00"),
            to_datetime("2021-01-02 03:00:00"),
            False,
        ),
        (
            to_datetime("2021-01-02 00:00:00"),
            to_datetime("2021-01-02 23:00:00"),
            to_datetime("2021-01-01 23:30:00"),
            to_datetime("2021-01-02 00:00:00"),
            True,
        ),
    ],
)
def test_check_to_time(
    first_time: RSDateTime,
    last_time: RSDateTime,
    to_time: RSDateTime,
    expected: RSDateTime,
    raises: bool,
) -> None:
    """Check adjusting to time"""
    from resistics.sampling import check_to_time

    if raises:
        with pytest.raises(ValueError):
            check_to_time(first_time, last_time, to_time)
    else:
        to_time = check_to_time(first_time, last_time, to_time)
        assert to_time == expected


@pytest.mark.parametrize(
    "fs, first_time, last_time, from_time, expected",
    [
        (
            512,
            to_datetime("2021-01-02 00:00:00"),
            to_datetime("2021-01-02 23:00:00"),
            to_datetime("2021-01-02 00:00:00"),
            0,
        ),
        (
            128,
            to_datetime("2021-01-02 00:00:00"),
            to_datetime("2021-01-03 01:00:00"),
            to_datetime("2021-01-02 00:00:00"),
            0,
        ),
        (
            128,
            to_datetime("2021-01-02 00:00:00"),
            to_datetime("2021-01-03 01:00:00"),
            to_datetime("2021-01-02 01:01:02"),
            468_736,
        ),
        (
            0.5,
            to_datetime("2021-01-02 00:00:00"),
            to_datetime("2021-01-03 01:00:00"),
            to_datetime("2021-01-02 01:01:01"),
            1_831,
        ),
        (
            16_384,
            to_datetime("2021-01-02 00:00:00"),
            to_datetime("2021-01-02 00:00:00") + 400_000*to_timedelta(1/16_384),            
            to_datetime("2021-01-02 00:00:00") + 193_435*to_timedelta(1/16_384),
            193_435,
        ),
    ],
)
def test_from_time_to_sample(
    fs: float,
    first_time: RSDateTime,
    last_time: RSDateTime,
    from_time: RSDateTime,
    expected: int,
) -> None:
    """Test converting datetimes to samples"""
    from resistics.sampling import from_time_to_sample

    assert expected == from_time_to_sample(fs, first_time, last_time, from_time)


@pytest.mark.parametrize(
    "fs, first_time, last_time, to_time, expected",
    [
        (
            512,
            to_datetime("2021-01-02 00:00:00"),
            to_datetime("2021-01-02 23:00:00"),
            to_datetime("2021-01-02 23:00:00"),
            42_393_600,
        ),
        (
            128,
            to_datetime("2021-01-02 00:00:00"),
            to_datetime("2021-01-03 01:00:00"),
            to_datetime("2021-01-03 01:00:00"),
            11_520_000,
        ),
        (
            128,
            to_datetime("2021-01-02 00:00:00"),
            to_datetime("2021-01-03 01:00:00"),
            to_datetime("2021-01-02 02:22:31"),
            1_094_528,
        ),
        (
            0.5,
            to_datetime("2021-01-02 00:00:00"),
            to_datetime("2021-01-03 01:00:00"),
            to_datetime("2021-01-02 02:22:31"),
            4_275,
        ),
        (
            16_384,
            to_datetime("2021-01-02 00:00:00"),
            to_datetime("2021-01-02 00:00:00") + 400_000*to_timedelta(1/16_384),            
            to_datetime("2021-01-02 00:00:00") + 374_653*to_timedelta(1/16_384),
            374_653,
        ),
    ],
)
def test_to_time_to_sample(
    fs: float,
    first_time: RSDateTime,
    last_time: RSDateTime,
    to_time: RSDateTime,
    expected: int,
) -> None:
    """Test converting datetimes to samples"""
    from resistics.sampling import to_time_to_sample

    assert expected == to_time_to_sample(fs, first_time, last_time, to_time)


@pytest.mark.parametrize(
    "fs, first_time, last_time, from_time, to_time, expected",
    [
        (
            512,
            to_datetime("2021-01-02 00:00:00"),
            to_datetime("2021-01-02 23:00:00"),
            to_datetime("2021-01-02 00:00:00"),
            to_datetime("2021-01-02 23:00:00"),
            (0, 42_393_600),
        ),
        (
            128,
            to_datetime("2021-01-02 00:00:00"),
            to_datetime("2021-01-03 01:00:00"),
            to_datetime("2021-01-02 00:00:00"),
            to_datetime("2021-01-03 01:00:00"),
            (0, 11_520_000),
        ),
        (
            128,
            to_datetime("2021-01-02 00:00:00"),
            to_datetime("2021-01-03 01:00:00"),
            to_datetime("2021-01-02 01:01:02"),
            to_datetime("2021-01-02 02:22:31"),
            (468_736, 1_094_528),
        ),
        (
            0.5,
            to_datetime("2021-01-02 00:00:00"),
            to_datetime("2021-01-03 01:00:00"),
            to_datetime("2021-01-02 01:01:01"),
            to_datetime("2021-01-02 02:22:31"),
            (1_831, 4_275),
        ),
    ],
)
def test_datetimes_to_samples(
    fs: float,
    first_time: RSDateTime,
    last_time: RSDateTime,
    from_time: RSDateTime,
    to_time: RSDateTime,
    expected: Tuple[int, int],
) -> None:
    """Test converting datetimes to samples"""
    from resistics.sampling import datetimes_to_samples

    from_sample, to_sample = datetimes_to_samples(
        fs, first_time, last_time, from_time, to_time
    )
    assert (from_sample, to_sample) == expected
