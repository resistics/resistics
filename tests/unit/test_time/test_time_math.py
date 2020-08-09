"""Test math module of resistics, time subpackage"""


def test_math_polarityReversal():
    """Test polarityReversal"""
    from resistics.time.data import TimeData
    from resistics.time.math import polarityReversal
    from datetime import datetime
    import numpy as np

    data = {
        "Ex": np.array([4, 0, 5, 0, 0, 2, 4, 3], dtype=float),
        "Hy": np.array([2, 9, 7, 6, 2, 0, 0, 5], dtype=float),
    }
    sampleFreq = 128
    startTime = "2020-01-01 13:00:00.000"
    stopTime = "2020-01-01 13:00:00.055"
    timeData = TimeData(sampleFreq, startTime, stopTime, data)

    # try different consecutive zeros
    reversal = {"Ex": True, "Hy": False}
    timeDataProc = polarityReversal(timeData, reversal)
    assert timeDataProc.sampleFreq == 128
    np.testing.assert_equal(timeDataProc["Ex"], -1 * data["Ex"])
    np.testing.assert_equal(timeDataProc["Hy"], data["Hy"])
    assert timeDataProc.startTime == datetime.strptime(
        startTime, "%Y-%m-%d %H:%M:%S.%f"
    )
    assert timeDataProc.stopTime == datetime.strptime(stopTime, "%Y-%m-%d %H:%M:%S.%f")
    assert timeDataProc.comments == [
        "Polarity reversal with parameters: {}".format(reversal)
    ]


def test_math_scale():
    """Test polarityReversal"""
    from resistics.time.data import TimeData
    from resistics.time.math import scale
    from datetime import datetime
    import numpy as np

    data = {
        "Ex": np.array([4, 0, 5, 0, 0, 2, 4, 3], dtype=float),
        "Hy": np.array([2, 9, 7, 6, 2, 0, 0, 5], dtype=float),
    }
    sampleFreq = 128
    startTime = "2020-01-01 13:00:00.000"
    stopTime = "2020-01-01 13:00:00.055"
    timeData = TimeData(sampleFreq, startTime, stopTime, data)

    # scale data
    scalars = {"Ex": -5, "Hy": 12}
    timeDataProc = scale(timeData, scalars)
    assert timeDataProc.sampleFreq == 128
    np.testing.assert_equal(timeDataProc["Ex"], -5 * data["Ex"])
    np.testing.assert_equal(timeDataProc["Hy"], 12 * data["Hy"])
    assert timeDataProc.startTime == datetime.strptime(
        startTime, "%Y-%m-%d %H:%M:%S.%f"
    )
    assert timeDataProc.stopTime == datetime.strptime(stopTime, "%Y-%m-%d %H:%M:%S.%f")
    assert timeDataProc.comments == [
        "Time data scaled with scalars: {}".format(scalars)
    ]


def test_math_normalise():
    """Test normalisation of TimeData"""
    from resistics.time.data import TimeData
    from resistics.time.math import normalise
    from datetime import datetime
    import numpy as np

    data = {
        "Ex": np.array([4, 0, 3, 0], dtype=float),
        "Hy": np.array([0, 5, 0, 12], dtype=float),
    }
    sampleFreq = 128
    startTime = "2020-01-01 13:00:00.000"
    stopTime = "2020-01-01 13:00:00.020"
    timeData = TimeData(sampleFreq, startTime, stopTime, data)

    # normalise data
    timeDataProc = normalise(timeData)
    assert timeDataProc.sampleFreq == 128
    np.testing.assert_equal(timeDataProc["Ex"], np.array([0.8, 0, 0.6, 0]))
    np.testing.assert_equal(timeDataProc["Hy"], np.array([0, 5 / 13, 0, 12 / 13]))
    assert timeDataProc.startTime == datetime.strptime(
        startTime, "%Y-%m-%d %H:%M:%S.%f"
    )
    assert timeDataProc.stopTime == datetime.strptime(stopTime, "%Y-%m-%d %H:%M:%S.%f")
    assert timeDataProc.comments == ["Data channels normalised"]
