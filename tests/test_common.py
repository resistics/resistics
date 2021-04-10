"""Tests for resistics.common"""
from typing import Dict, List, Any, Type
import pytest
import pandas as pd


def test_dir_formats() -> None:
    from resistics.common import data_dir_names

    assert data_dir_names() == ["meas", "run", "phnx", "lemi"]


def test_electric_chans() -> None:
    from resistics.common import electric_chans

    assert electric_chans() == ["Ex", "Ey", "E1", "E2", "E3", "E4"]


@pytest.mark.parametrize(
    "chan, expected",
    [("Ex", True), ("Ey", True), ("Hx", False), ("Hy", False), ("Hz", False)],
)
def test_is_electric(chan: str, expected: bool) -> None:
    from resistics.common import is_electric

    assert is_electric(chan) == expected


def test_magnetic_chans() -> None:
    from resistics.common import magnetic_chans

    assert magnetic_chans() == ["Hx", "Hy", "Hz", "Bx", "By", "Bz"]


@pytest.mark.parametrize(
    "chan, expected",
    [("Ex", False), ("Ey", False), ("Hx", True), ("Hy", True), ("Hz", True)],
)
def test_is_magnetic(chan: str, expected: bool) -> None:
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
    in_val: Any, format_type: Type, expected: Any, raises: bool
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
    "headers_dict, specs, expected, raises",
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
def test_Headers(
    headers_dict: Dict[str, Any],
    specs: Dict[str, Dict[str, Any]],
    expected: Dict[str, Any],
    raises: bool,
):
    """Test Headers"""
    import pandas as pd
    from resistics.common import Headers

    if raises:
        with pytest.raises(KeyError):
            headers = Headers(headers_dict, specs)
    else:
        headers = Headers(headers_dict, specs)
        assert set(headers.keys()) == set(expected.keys())
        for header in headers:
            assert headers[header] == expected[header]
        pd.testing.assert_series_equal(headers.to_series(), pd.Series(data=expected))
        assert headers.to_dict() == expected


def test_template_headers():
    """Test making a template header"""
    from datetime import datetime
    import pandas as pd
    from resistics.common import template_headers

    specs = {
        "a": {"default": 10, "type": int},
        "b": {"default": "hello", "type": str},
        "c": {"default": -7.3, "type": float},
        "d": {"default": pd.to_datetime("2020-01-01 00:00:00"), "type": datetime},
    }
    # use defaults only
    template = template_headers(specs)
    for header in template:
        assert template[header] == specs[header]["default"]
    # again but with a manually specified val
    overwrite = {"b": "bye bye"}
    template = template_headers(specs, overwrite)
    for header in template:
        if header in overwrite:
            assert template[header] == overwrite[header]
        else:
            assert template[header] == specs[header]["default"]


@pytest.mark.parametrize(
    "dataset_headers, chan_headers, dataset_specs, chan_specs, expected_dataset, expected_chan",
    [
        (
            {"a": "5", "b": 12, "c": "7.3"},
            {
                "Ex": {"a": "5", "b": 12, "c": "something"},
                "C2": {"a": "3", "b": -7, "c": "other"},
            },
            {
                "a": {"type": int, "default": None},
                "b": {"type": str, "default": None},
                "c": {"type": float, "default": None},
                "d": {"type": int, "default": 12},
            },
            {
                "a": {"type": int, "default": 0},
                "b": {"type": str, "default": "hello"},
                "f": {"type": float, "default": -3.0},
            },
            {"a": 5, "b": "12", "c": 7.3, "d": 12},
            {
                "Ex": {"a": 5, "b": "12", "c": "something", "f": -3.0},
                "C2": {"a": 3, "b": "-7", "c": "other", "f": -3.0},
            },
        ),
    ],
)
def test_DatasetHeader(
    dataset_headers: Dict[str, Any],
    chan_headers: Dict[str, Any],
    dataset_specs: Dict[str, Dict[str, Any]],
    chan_specs: Dict[str, Dict[str, Any]],
    expected_dataset: Dict[str, Any],
    expected_chan: Dict[str, Any],
):
    """Test dataset headers"""
    from resistics.common import DatasetHeaders

    headers = DatasetHeaders(dataset_headers, chan_headers, dataset_specs, chan_specs)
    for header in headers.dataset_keys():
        assert headers[header] == expected_dataset[header]
    for chan in headers.chans:
        for header in headers.chan_keys(chan):
            assert headers[chan, header] == expected_chan[chan][header]


def test_template_dataset_headers():
    """Test making a template dataset header"""
    from datetime import datetime
    import pandas as pd
    from resistics.common import template_dataset_headers

    dataset_specs = {
        "a": {"default": 10, "type": int},
        "b": {"default": "hello", "type": str},
        "c": {"default": -7.3, "type": float},
        "d": {"default": pd.to_datetime("2020-01-01 00:00:00"), "type": datetime},
    }
    dataset_vals = {"a": 3}
    chan_specs = {
        "f": {"default": 10, "type": int},
        "g": {"default": "hello", "type": str},
        "h": {"default": -7.3, "type": float},
        "i": {"default": pd.to_datetime("2020-01-01 00:00:00"), "type": datetime},
    }
    chan_vals = {"Ex": {"h": -12}, "Ey": {"g": "something"}}

    # use defaults only
    template = template_dataset_headers(["Ex", "Ey"], dataset_specs, chan_specs)
    for header in template.dataset_keys():
        assert template[header] == dataset_specs[header]["default"]
    for chan in template.chans:
        for header in template.chan_keys(chan):
            assert template[chan, header] == chan_specs[header]["default"]
    # again but with a manually specified val
    template = template_dataset_headers(
        ["Ex", "Ey"], dataset_specs, chan_specs, dataset_vals, chan_vals
    )
    for header in template.dataset_keys():
        if header in dataset_vals:
            assert template[header] == dataset_vals[header]
        else:
            assert template[header] == dataset_specs[header]["default"]
    for chan in template.chans:
        for header in template.chan_keys(chan):
            if chan in chan_vals and header in chan_vals[chan]:
                assert template[chan, header] == chan_vals[chan][header]
            else:
                assert template[chan, header] == chan_specs[header]["default"]