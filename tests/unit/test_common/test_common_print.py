"""Test resistics.common.print"""


def test_generalPrint(capfd) -> None:
    from resistics.common.print import generalPrint

    generalPrint("test", "test")
    captured = capfd.readouterr()
    out = captured.out[9:]
    assert out == "test: test\n"


def test_breakPrint(capfd) -> None:
    from resistics.common.print import breakPrint

    breakPrint()
    captured = capfd.readouterr()
    test = "---------------------------------------------------\n\n"
    test += "---------------------------------------------------\n"
    assert test == captured.out


# def test_blockPrint(capfd) -> None:
#     from resistics.common.print import blockPrint

#     generalPrint(pre, "####################")
#     generalPrint(pre, "{} INFO BEGIN".format(pre.upper()))
#     generalPrint(pre, "####################")
#     if isinstance(text, str):
#         generalPrint(pre, text)
#     else:
#         for t in text:
#             generalPrint(pre, t)
#     generalPrint(pre, "####################")
#     generalPrint(pre, "{} INFO END".format(pre.upper()))
#     generalPrint(pre, "####################")


def test_warningPrint(capfd) -> None:
    from resistics.common.print import warningPrint

    warningPrint("test", "test")
    captured = capfd.readouterr()
    out = captured.out[9:]
    assert out == "TEST: test\n"


def test_errorPrint(capfd) -> None:
    from resistics.common.print import errorPrint
    import pytest

    errorPrint("test", "test")
    captured = capfd.readouterr()
    assert captured.out[:5] == "ERROR"
    assert captured.out[16:] == "TEST: test\n"

    errorPrint("test", "test", quitrun=False)
    captured = capfd.readouterr()
    assert captured.out[:5] == "ERROR"
    assert captured.out[16:] == "TEST: test\n"

    with pytest.raises(SystemExit):
        errorPrint("test", "test", quitrun=True)
        captured = capfd.readouterr()
        assert captured.out[:5] == "ERROR"
        assert captured.out[16:] == "TEST: test\n"


def test_breakComment() -> None:
    from resistics.common.print import breakComment

    assert breakComment() == "---------------------------------------------------"


def test_arrayToString() -> None:
    from resistics.common.print import arrayToString
    import numpy as np

    testarray = np.array([1.453, 54.3465, 534.4554654, 6])
    assert (
        arrayToString(testarray) == "1.45300000, 54.34650000, 534.45546540, 6.00000000"
    )
    assert arrayToString(testarray, decimals=4) == "1.4530, 54.3465, 534.4555, 6.0000"
    assert (
        arrayToString(testarray, tabs=True, decimals=2) == "1.45\t54.35\t534.46\t6.00"
    )


def test_arrayToStringSci() -> None:
    from resistics.common.print import arrayToStringSci
    import numpy as np

    testarray = np.array(
        [1.4354, 22345.234665, 0.000432341, 5.4325454, 753454.4834, 0.0010004239]
    )
    assert (
        "1.435400e+00, 2.234523e+04, 4.323410e-04, 5.432545e+00, 7.534545e+05, 1.000424e-03"
        == arrayToStringSci(testarray)
    )


def test_arrayToStringInt() -> None:
    from resistics.common.print import arrayToStringInt
    import numpy as np

    testarray = np.array([1, 2, 4, 5, 7, 10])
    assert "1, 2, 4, 5, 7, 10" == arrayToStringInt(testarray)


def test_listToString() -> None:
    from resistics.common.print import listToString

    testlist = [1, 2, 4, 6, "mixed"]
    assert listToString(testlist) == "1, 2, 4, 6, mixed"


def test_list2rangesFormatter() -> None:
    from resistics.common.print import list2rangesFormatter

    start = 0
    end = 10
    step = 2
    assert "0-10:2" == list2rangesFormatter(start, end, step)


def test_list2ranges() -> None:
    from resistics.common.print import list2ranges

    testlist = [1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21, 24, 26, 35, 40, 45]
    assert list2ranges(testlist) == "1-4:1,6-12:2,15-24:3,26,35-45:5"
