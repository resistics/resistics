"""Tests for resistics.common"""
from typing import List, Callable
from pathlib import Path
import pytest
import pandas as pd

from resistics.errors import NotFileError, NotDirectoryError
from resistics.sampling import datetime_to_string, datetime_from_string

Timestamp = pd.Timestamp
Timedelta = pd.Timedelta
rstime = datetime_from_string("2021-01-01 00:33:24.543443_457873_000000_000000")
rstime_str = datetime_to_string(rstime)


def mock_return_true(*args):
    """A mock function to return True"""
    return True


def mock_return_false(*args):
    """A mock function to return False"""
    return False


def test_get_version():
    """Test getting the resistics version"""
    import resistics as rs
    from resistics.common import get_version

    assert get_version() == rs.__version__


@pytest.mark.parametrize(
    "fnc_exists, fnc_is_file, expected",
    [
        (mock_return_true, mock_return_true, True),
        (mock_return_true, mock_return_false, False),
        (mock_return_false, mock_return_true, False),
        (mock_return_false, mock_return_false, False),
    ],
)
def test_is_file(
    monkeypatch, fnc_exists: Callable, fnc_is_file: Callable, expected: bool
):
    """Test checking if a path is a file"""
    from resistics.common import is_file

    monkeypatch.setattr(Path, "exists", fnc_exists)
    monkeypatch.setattr(Path, "is_file", fnc_is_file)
    assert is_file(Path("test")) == expected


@pytest.mark.parametrize(
    "fnc_exists, fnc_is_file, file_not_found, not_file_error",
    [
        (mock_return_true, mock_return_true, False, False),
        (mock_return_true, mock_return_false, False, True),
        (mock_return_false, mock_return_true, True, False),
        (mock_return_false, mock_return_false, True, False),
    ],
)
def test_assert_file(
    monkeypatch, fnc_exists, fnc_is_file, file_not_found, not_file_error
):
    """Test assert_file raises errors if Path is not a file"""
    from resistics.common import assert_file

    monkeypatch.setattr(Path, "exists", fnc_exists)
    monkeypatch.setattr(Path, "is_file", fnc_is_file)
    if file_not_found:
        with pytest.raises(FileNotFoundError):
            assert_file(Path("test"))
        return
    if not_file_error:
        with pytest.raises(NotFileError):
            assert_file(Path("test"))
        return
    assert assert_file(Path("test")) is None


@pytest.mark.parametrize(
    "fnc_exists, fnc_is_dir, expected",
    [
        (mock_return_true, mock_return_true, True),
        (mock_return_true, mock_return_false, False),
        (mock_return_false, mock_return_true, False),
        (mock_return_false, mock_return_false, False),
    ],
)
def test_is_dir(
    monkeypatch, fnc_exists: Callable, fnc_is_dir: Callable, expected: bool
):
    """Test checking if a path is a directory"""
    from resistics.common import is_dir

    monkeypatch.setattr(Path, "exists", fnc_exists)
    monkeypatch.setattr(Path, "is_dir", fnc_is_dir)
    assert is_dir(Path("test")) == expected


@pytest.mark.parametrize(
    "fnc_exists, fnc_is_dir, file_not_found, not_directory_error",
    [
        (mock_return_true, mock_return_true, False, False),
        (mock_return_true, mock_return_false, False, True),
        (mock_return_false, mock_return_true, True, False),
        (mock_return_false, mock_return_false, True, False),
    ],
)
def test_assert_dir(
    monkeypatch, fnc_exists, fnc_is_dir, file_not_found, not_directory_error
):
    """Test assert_dir raises errors if Path is not a directory"""
    from resistics.common import assert_dir

    monkeypatch.setattr(Path, "exists", fnc_exists)
    monkeypatch.setattr(Path, "is_dir", fnc_is_dir)
    if file_not_found:
        with pytest.raises(FileNotFoundError):
            assert_dir(Path("test"))
        return
    if not_directory_error:
        with pytest.raises(NotDirectoryError):
            assert_dir(Path("test"))
        return
    assert assert_dir(Path("test")) is None


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


def test_resistics_process():
    """Test initialising a resistics process"""
    from resistics.common import ResisticsProcess
    from resistics.decimate import DecimationSetup

    process = {
        "name": "DecimationSetup",
        "n_levels": 8,
        "per_level": 5,
        "min_samples": 256,
        "div_factor": 2,
        "eval_freqs": None,
    }
    assert ResisticsProcess.validate(process) == DecimationSetup(**process)


def test_resistics_process_errors():
    """Test errors when initialising a resistics process"""
    from resistics.common import ResisticsProcess

    process = {
        "n_levels": 8,
        "per_level": 5,
        "min_samples": 256,
        "div_factor": 2,
        "eval_freqs": None,
    }
    with pytest.raises(KeyError):
        ResisticsProcess.validate(process)
    with pytest.raises(ValueError):
        ResisticsProcess.validate(5)
    process["name"] = "Unknown"
    with pytest.raises(ValueError):
        ResisticsProcess.validate(process)
