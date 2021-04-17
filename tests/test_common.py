"""Tests for resistics.common"""
from typing import Dict, List, Any, Type
import pytest
import pandas as pd

from resistics.sampling import RSDateTime
from resistics.sampling import datetime_to_string, datetime_from_string

Timestamp = pd.Timestamp
Timedelta = pd.Timedelta
rstime = datetime_from_string("2021-01-01 00:33:24.543443_457873_000000_000000")
rstime_str = datetime_to_string(rstime)


@pytest.mark.parametrize(
    "input, expected",
    [
        ("Ex", "Ex"),
        (5, 5),
        (4.3, 4.3),
        (True, "True"),
        (False, "False"),
        (rstime, rstime_str),
        (Timestamp("2021-01-01"), Timestamp("2021-01-01").isoformat()),
        (Timedelta(1 / 128), Timedelta(1 / 128).total_seconds()),
        (["1", "2", "3", "4"], "1, 2, 3, 4"),
    ],
)
def test_serialize(input: Any, expected: Any) -> None:
    """Test serialize functions"""
    from resistics.common import serialize

    assert serialize(input) == expected


@pytest.mark.parametrize(
    "input, expected_type, expected",
    [
        ("Ex", str, "Ex"),
        (5, int, 5),
        (4.3, float, 4.3),
        ("True", bool, True),
        ("False", bool, False),
        (rstime_str, RSDateTime, rstime),
        (Timestamp("2021-01-01").isoformat(), Timestamp, Timestamp("2021-01-01")),
        (Timedelta(1 / 128).total_seconds(), Timedelta, Timedelta(1 / 128)),
        ("1, 2, 3, 4", list, ["1", "2", "3", "4"]),
    ],
)
def test_deserialize(input: Any, expected_type: Type[Any], expected: Any) -> None:
    """Test deserialize"""
    from resistics.common import deserialize

    assert deserialize(input, expected_type) == expected


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
    "lines, expected",
    [
        (["3 4 5 6", "4 5 6 7"], [[3, 4, 5, 6], [4, 5, 6, 7]]),
    ],
)
def test_lines_to_array(lines: List[str], expected) -> None:
    """Test array to string formatting"""
    import numpy as np
    from resistics.common import lines_to_array

    expected = np.array(expected)
    np.testing.assert_equal(lines_to_array(lines), expected)


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


@pytest.mark.parametrize(
    "in_val, format_type, expected, raises",
    [
        (5, str, "5", False),
        ("2020-01-01", pd.Timestamp, pd.Timestamp("2020-01-01"), False),
        ("asdafd", int, None, True),
    ],
)
def test_format_value(
    in_val: Any, format_type: Type[Any], expected: Any, raises: bool
) -> None:
    from resistics.common import format_value

    if raises:
        with pytest.raises(TypeError):
            format_value(in_val, format_type)
    else:
        assert expected == format_value(in_val, format_type)


@pytest.mark.parametrize(
    "input_dict, specifications, expected, raises",
    [
        (
            {"a": "12", "b": "something", "c": "-2.3"},
            {
                "a": {"type": int, "default": None},
                "c": {"type": float, "default": None},
            },
            {"a": 12, "b": "something", "c": -2.3},
            False,
        ),
        (
            {"a": "12", "b": "something", "c": "-2.3"},
            {
                "a": {"type": int, "default": None},
                "d": {"type": str, "default": "sensor"},
            },
            {"a": 12, "b": "something", "c": "-2.3", "d": "sensor"},
            False,
        ),
        (
            {"a": "12", "b": "something", "c": "-2.3"},
            {"a": {"type": int, "default": None}, "d": {"type": str, "default": None}},
            {"a": 12, "b": "something", "c": "-2.3"},
            True,
        ),
    ],
)
def test_format_dict(
    input_dict: Dict[str, Any],
    specifications: Dict[str, Dict[str, Any]],
    expected: Dict[str, Any],
    raises: bool,
):
    from resistics.common import format_dict

    if raises:
        with pytest.raises(KeyError):
            format_dict(input_dict, specifications)
    else:
        assert expected == format_dict(input_dict, specifications)


@pytest.mark.parametrize(
    "input_dict, specs, expected, raises",
    [
        (
            {"a": "12", "b": "something", "c": "-2.3"},
            {
                "a": {"type": int, "default": None},
                "c": {"type": float, "default": None},
            },
            {"a": 12, "b": "something", "c": -2.3, "describes": "unknown"},
            False,
        ),
        (
            {"a": "12", "b": "something", "c": "-2.3", "describes": "ex"},
            {
                "a": {"type": int, "default": None},
                "d": {"type": str, "default": "sensor"},
            },
            {"a": 12, "b": "something", "c": "-2.3", "d": "sensor", "describes": "ex"},
            False,
        ),
        (
            {"a": "12", "b": "something", "c": "-2.3", "describes": "something"},
            {"a": {"type": int, "default": None}, "d": {"type": str, "default": None}},
            {"a": 12, "b": "something", "c": "-2.3", "describes": "something"},
            True,
        ),
    ],
)
def test_metadata(
    input_dict: Dict[str, Any],
    specs: Dict[str, Dict[str, Any]],
    expected: Dict[str, Any],
    raises: bool,
):
    """Test Metadata"""
    import pandas as pd
    from resistics.common import Metadata

    if raises:
        with pytest.raises(KeyError):
            metadata = Metadata(input_dict, specs)
        return

    metadata = Metadata(input_dict, specs)
    assert set(metadata.keys()) == set(expected.keys())
    for key in metadata:
        assert metadata[key] == expected[key]
    metadata_series = metadata.to_series().sort_index()
    expected_series = pd.Series(data=expected).sort_index()
    pd.testing.assert_series_equal(metadata_series, expected_series)
    assert metadata.to_dict() == expected


def test_metadata_from_specs():
    """Test making a metadata from specifications"""
    from datetime import datetime
    import pandas as pd
    from resistics.common import metadata_from_specs

    specs = {
        "describes": {"default": "example", "type": str},
        "a": {"default": 10, "type": int},
        "b": {"default": "hello", "type": str},
        "c": {"default": -7.3, "type": float},
        "d": {"default": pd.to_datetime("2020-01-01 00:00:00"), "type": datetime},
    }
    # use defaults only
    metadata = metadata_from_specs(specs)
    for key in metadata:
        assert metadata[key] == specs[key]["default"]
    # again but with a manually specified val
    overwrite = {"b": "bye bye"}
    metadata = metadata_from_specs(specs, overwrite)
    for key in metadata:
        expected = overwrite[key] if key in overwrite else specs[key]["default"]
        assert metadata[key] == expected


def test_MetadataGroup():
    """Test a MetadataGroup"""
    from resistics.common import MetadataGroup

    grp1 = {"common": {"a": 5, "b": "12", "c": 7.3}}
    specs1 = {
        "a": {"type": int, "default": 0},
        "b": {"type": str, "default": "hello"},
        "f": {"type": float, "default": -3.0},
        "describes": {"type": str, "default": "grp1"},
    }
    grp2 = {
        "Ex": {"x": "5", "y": 12, "z": "something"},
        "C2": {"x": "3", "y": -7, "z": "other"},
    }
    specs2 = {
        "u": {"type": str, "default": "Null"},
        "x": {"type": str, "default": "str"},
        "y": {"type": str, "default": int},
        "describes": {"type": str, "default": "grp2"},
    }

    metadata_grp = MetadataGroup(grp1, specs1)
    metadata_grp.add_entries(grp2, specs2)
    # check the values
    for entry, entry_metadata in grp1.items():
        keys = list(entry_metadata.keys())
        values = [
            entry_metadata[key] if key in entry_metadata else specs1[key]["default"]
            for key in keys
        ]
        for key, value in zip(keys, values):
            assert metadata_grp[entry, key] == value

    for entry, entry_metadata in grp2.items():
        keys = list(entry_metadata.keys())
        values = [
            entry_metadata[key] if key in entry_metadata else specs2[key]["default"]
            for key in keys
        ]
        for key, value in zip(keys, values):
            assert metadata_grp[entry, key] == value


def test_metadata_group_from_specs():
    """Test making a metadata group from specifications"""
    from resistics.common import metadata_group_from_specs

    specs1 = {
        "a": {"type": int, "default": 0},
        "b": {"type": str, "default": "hello"},
        "f": {"type": float, "default": -3.0},
    }
    specs2 = {
        "u": {"type": str, "default": "Null"},
        "x": {"type": float, "default": -1032.2},
        "y": {"type": bool, "default": True},
        "describes": {"type": str, "default": "spec2"},
    }
    grp_specs = {"common": specs1, "Ex": specs2, "C2": specs2}
    override = {
        "common": {"a": 5, "f": 7.3},
        "Ex": {"u": "5", "y": False},
        "C2": {"u": "3", "x": -7.0},
    }

    # defaults only
    expected = {
        "common": {"a": 0, "b": "hello", "f": -3.0, "describes": "unknown"},
        "Ex": {"u": "Null", "x": -1032.2, "y": True, "describes": "spec2"},
        "C2": {"u": "Null", "x": -1032.2, "y": True, "describes": "spec2"},
    }
    metadata_grp = metadata_group_from_specs(grp_specs)
    assert metadata_grp.to_dict() == expected

    # with specifications
    expected = {
        "common": {"a": 5, "b": "hello", "f": 7.3, "describes": "unknown"},
        "Ex": {"u": "5", "x": -1032.2, "y": False, "describes": "spec2"},
        "C2": {"u": "3", "x": -7.0, "y": True, "describes": "spec2"},
    }
    metadata_grp = metadata_group_from_specs(grp_specs, override)
    assert metadata_grp.to_dict() == expected
