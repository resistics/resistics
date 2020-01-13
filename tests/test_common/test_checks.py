"""Tests for resistics.common.checks"""


def test_parseKeywords() -> None:
    from resistics.common.checks import parseKeywords

    default = {"test1": 1, "test2": 2}
    keywords = {"test1": 3}
    parsed = parseKeywords(default, keywords)
    assert parsed["test1"] == 3
    assert parsed["test2"] == 2


def test_elecChannelsList() -> None:
    from resistics.common.checks import elecChannelsList

    assert elecChannelsList() == ["Ex", "Ey", "E1", "E2", "E3", "E4"]


def test_isElectric() -> None:
    from resistics.common.checks import isElectric

    assert isElectric("Ex")
    assert isElectric("Ey")
    assert not isElectric("Hx")
    assert not isElectric("Hy")
    assert not isElectric("Hz")


def test_magChannelsList() -> None:
    from resistics.common.checks import magChannelsList

    assert magChannelsList() == ["Hx", "Hy", "Hz", "Bx", "By", "Bz"]


def test_isMagnetic() -> None:
    from resistics.common.checks import isMagnetic

    assert not isMagnetic("Ex")
    assert not isMagnetic("Ey")
    assert isMagnetic("Hx")
    assert isMagnetic("Hy")
    assert isMagnetic("Hz")


def consistentChans() -> None:
    from resistics.common.checks import consistentChans

    assert consistentChans("Bx") == "Hx"
    assert consistentChans("By") == "Hy"
    assert consistentChans("Bz") == "Hz"
    assert consistentChans("Ex") == "Ex"
    assert consistentChans("Ey") == "Ey"
