"""Tests for resistics.common"""
from typing import List
import pytest
import pandas as pd

from resistics.sampling import datetime_to_string, datetime_from_string

Timestamp = pd.Timestamp
Timedelta = pd.Timedelta
rstime = datetime_from_string("2021-01-01 00:33:24.543443_457873_000000_000000")
rstime_str = datetime_to_string(rstime)


def test_dir_formats() -> None:
    """Test data directory formats"""
    from resistics.common import data_dir_names

    assert data_dir_names() == ["meas", "run", "phnx", "lemi"]


def test_electric_chans() -> None:
    """Test recognised electric channels"""
    from resistics.common import electric_chans

    assert electric_chans() == ["Ex", "Ey", "E1", "E2", "E3", "E4"]


@pytest.mark.parametrize(
    "chan, expected",
    [("Ex", True), ("Ey", True), ("Hx", False), ("Hy", False), ("Hz", False)],
)
def test_is_electric(chan: str, expected: bool) -> None:
    """Test flagging whether a channel is electric"""
    from resistics.common import is_electric

    assert is_electric(chan) == expected


def test_magnetic_chans() -> None:
    """Test recognised magnetic channels"""
    from resistics.common import magnetic_chans

    assert magnetic_chans() == ["Hx", "Hy", "Hz", "Bx", "By", "Bz"]


@pytest.mark.parametrize(
    "chan, expected",
    [("Ex", False), ("Ey", False), ("Hx", True), ("Hy", True), ("Hz", True)],
)
def test_is_magnetic(chan: str, expected: bool) -> None:
    """Test flagging magnetic channels"""
    from resistics.common import is_magnetic

    assert is_magnetic(chan) == expected


@pytest.mark.parametrize(
    "chan, expected",
    [("Ex", "Ex"), ("Bx", "Hx"), ("By", "Hy"), ("Bz", "Hz"), ("Cx", "Cx")],
)
def test_to_resistics_chan(chan: str, expected: bool) -> None:
    """Test to_resistics_chan"""
    from resistics.common import to_resistics_chan

    assert to_resistics_chan(chan) == expected


@pytest.mark.parametrize(
    "chan, chan_list, expect_raise",
    [("Ex", ["Ex", "Hy"], False), ("Ex", ["Hy", "Hy"], True)],
)
def test_check_chan(chan: str, chan_list: List[str], expect_raise: bool) -> None:
    """Test check_chan"""
    from resistics.errors import ChannelNotFoundError
    from resistics.common import check_chan

    if expect_raise:
        with pytest.raises(ChannelNotFoundError):
            check_chan(chan, chan_list)
    else:
        assert check_chan(chan, chan_list)


@pytest.mark.parametrize("fs, expected", [(0.0005, "0_000500"), (128.0, "128_000000")])
def test_strformat_fs(fs: float, expected: str) -> None:
    """Test formatting sampling frequency for output"""
    from resistics.common import fs_to_string

    assert fs_to_string(fs) == expected


@pytest.mark.parametrize(
    "data, sep, precision, scientific, expected",
    [
        ([1, 2, 3, 4, 5], "\t", 8, False, "1\t2\t3\t4\t5"),
        ([1, 2, 3, 4, 5], ", ", 8, True, "1, 2, 3, 4, 5"),
        ([1.0, 2.1, 3.2, 4, 5], " ", 2, False, "1.00 2.10 3.20 4.00 5.00"),
        ([121.8, 3195.2, 0.00414], " ", 2, True, "1.22e+02 3.20e+03 4.14e-03"),
    ],
)
def test_array_to_string(
    data: List[float], sep: str, precision: int, scientific: bool, expected: str
) -> None:
    """Test array to string formatting"""
    from resistics.common import array_to_string
    import numpy as np

    data = np.array(data)
    assert array_to_string(data, sep, precision, scientific) == expected


@pytest.mark.parametrize(
    "lst, expected",
    [
        ([1, 2, 4, 6, "mixed"], "1, 2, 4, 6, mixed"),
        ([1, 2, 4, 6], "1, 2, 4, 6"),
        (["hello", 2, 3.16, 6], "hello, 2, 3.16, 6"),
    ],
)
def test_list_to_string(lst: List, expected: str) -> None:
    from resistics.common import list_to_string

    assert list_to_string(lst) == expected


@pytest.mark.parametrize(
    "lst, expected",
    [
        (
            [1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21, 24, 26, 35, 40, 45],
            "1-4:1,6-12:2,15-24:3,26,35-45:5",
        ),
    ],
)
def test_list_to_ranges(lst: List, expected: str) -> None:
    from resistics.common import list_to_ranges

    testlist = [1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21, 24, 26, 35, 40, 45]
    assert list_to_ranges(testlist) == "1-4:1,6-12:2,15-24:3,26,35-45:5"
