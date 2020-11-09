"""Tests for resistics.common.checks"""
from typing import Dict
import pytest


@pytest.mark.parametrize(
    "default, keywords, expected",
    [
        ({"k1": 1, "k2": 2}, {"k2": 3}, {"k1": 1, "k2": 3}),
        ({"k1": "l", "k2": 2}, {"k1": "n"}, {"k1": "n", "k2": 2}),
        ({"k1": "l", "k2": 2}, {"k3": "n"}, {"k1": "l", "k2": 2}),
    ],
)
def test_parse_keywords(default: Dict, keywords: Dict, expected: Dict) -> None:
    from resistics.common.checks import parse_keywords

    parsed = parse_keywords(default, keywords)
    assert parsed.keys() == default.keys()
    for kw, val in expected.items():
        assert val == parsed[kw]


def test_electric_chans() -> None:
    from resistics.common.checks import electric_chans

    assert electric_chans() == ["Ex", "Ey", "E1", "E2", "E3", "E4"]


@pytest.mark.parametrize(
    "chan, expected",
    [("Ex", True), ("Ey", True), ("Hx", False), ("Hy", False), ("Hz", False)],
)
def test_is_electric(chan: str, expected: bool) -> None:
    from resistics.common.checks import is_electric

    assert is_electric(chan) == expected


def test_magChannelsList() -> None:
    from resistics.common.checks import magnetic_chans

    assert magnetic_chans() == ["Hx", "Hy", "Hz", "Bx", "By", "Bz"]


@pytest.mark.parametrize(
    "chan, expected",
    [("Ex", False), ("Ey", False), ("Hx", True), ("Hy", True), ("Hz", True)],
)
def test_isMagnetic(chan: str, expected: bool) -> None:
    from resistics.common.checks import is_magnetic

    assert is_magnetic(chan) == expected


@pytest.mark.parametrize(
    "chan, expected",
    [("Ex", "Ex"), ("Bx", "Hx"), ("By", "Hy"), ("Bz", "Hz"), ("Cx", "Cx")],
)
def test_to_resistics_chan(chan: str, expected: bool) -> None:
    from resistics.common.checks import to_resistics_chan

    assert to_resistics_chan(chan) == expected
