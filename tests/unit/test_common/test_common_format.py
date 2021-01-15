"""Test resistics.common.format"""
from typing import List
import pytest


@pytest.mark.parametrize(
    "ns, expected", [(False, "%Y-%m-%d %H:%M:%S"), (True, "%Y-%m-%d %H:%M:%S.%f")]
)
def test_datetime_format(ns: bool, expected: str) -> None:
    """Test getting the datetime formatting style"""
    from resistics.common.format import datetime_format

    assert datetime_format(ns) == expected


@pytest.mark.parametrize("fs, expected", [(0.0005, "0_000500"), (128.0, "128_000000")])
def test_strformat_fs(fs: float, expected: str) -> None:
    """Test formatting sampling frequency for output"""
    from resistics.common.format import strformat_fs

    assert strformat_fs(fs) == expected


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
    data: List, sep: str, precision: int, scientific: bool, expected: str
) -> None:
    """Test array to string formatting"""
    from resistics.common.format import array_to_string
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
    from resistics.common.format import list_to_string

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
    from resistics.common.format import list_to_ranges

    testlist = [1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21, 24, 26, 35, 40, 45]
    assert list_to_ranges(testlist) == "1-4:1,6-12:2,15-24:3,26,35-45:5"
