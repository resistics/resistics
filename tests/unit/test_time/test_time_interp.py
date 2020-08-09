"""Test interpolation of TimeData"""


def test_interp_interpolateToSeconds():
    """Test interpolation to second"""
    from datetime import datetime
    import numpy as np
    from resistics.time.data import TimeData
    from resistics.time.interp import interpolateToSecond

    sampleFreq = 10
    startTime = "2020-01-01 12:00:00.950"
    stopTime = "2020-01-01 12:00:09.850"
    data = {
        "Ex": np.arange(0, 100) - 0.5,
        "Ey": np.arange(0, 100) - 0.5,
        "Hx": np.arange(0, 100) - 0.5,
        "Hy": np.arange(0, 100) - 0.5,
        "Hz": np.arange(0, 100) - 0.5,
    }
    comments = ["This is a test"]
    timeData = TimeData(sampleFreq, startTime, stopTime, data, comments)
    np.testing.assert_almost_equal(data["Ex"][0], -0.5)
    np.testing.assert_almost_equal(data["Ex"][-1], 98.5)
    # do the interpolation
    timeDataInterp = interpolateToSecond(timeData)
    assert timeDataInterp.numSamples == 99
    np.testing.assert_equal(
        np.round(timeDataInterp["Ex"]).astype(int), np.arange(0, 99)
    )
    assert timeDataInterp.startTime == datetime(2020, 1, 1, 12, 0, 1)
    assert timeDataInterp.stopTime == datetime(2020, 1, 1, 12, 0, 10, 800000)
    assert timeDataInterp.comments == comments + [
        "Time data interpolated to nearest second. New start time 2020-01-01 12:00:01, new end time 2020-01-01 12:00:10.800000, new number of samples 99"
    ]


def test_interp_fillGap():
    """Test gap filling between two sections where recording has been interrupted"""
    from datetime import datetime
    import numpy as np
    from resistics.time.data import TimeData
    from resistics.time.interp import fillGap

    sampleFreq = 1
    startTime1 = "2020-01-01 12:00:01.000"
    stopTime1 = "2020-01-01 12:00:05.000"
    data1 = {
        "Ex": np.arange(0, 5),
        "Ey": np.arange(0, 5),
        "Hx": np.arange(0, 5),
        "Hy": np.arange(0, 5),
        "Hz": np.arange(0, 5),
    }
    timeData1 = TimeData(sampleFreq, startTime1, stopTime1, data1, "TimeData1")

    sampleFreq = 1
    startTime2 = "2020-01-01 12:00:10.000"
    stopTime2 = "2020-01-01 12:00:19.000"
    data2 = {
        "Ex": np.arange(9, 20),
        "Ey": np.arange(9, 20),
        "Hx": np.arange(9, 20),
        "Hy": np.arange(9, 20),
        "Hz": np.arange(9, 20),
    }
    timeData2 = TimeData(sampleFreq, startTime2, stopTime2, data2, "TimeData2")

    timeDataFilled = fillGap(timeData1, timeData2)
    np.testing.assert_equal(timeDataFilled["Ex"], np.arange(0, 20))
    np.testing.assert_equal(timeDataFilled["Ey"], np.arange(0, 20))
    np.testing.assert_equal(timeDataFilled["Hx"], np.arange(0, 20))
    np.testing.assert_equal(timeDataFilled["Hy"], np.arange(0, 20))
    np.testing.assert_equal(timeDataFilled["Hz"], np.arange(0, 20))
    assert timeDataFilled.startTime == datetime.strptime(
        startTime1, "%Y-%m-%d %H:%M:%S.%f"
    )
    assert timeDataFilled.stopTime == datetime.strptime(
        stopTime2, "%Y-%m-%d %H:%M:%S.%f"
    )
    comment = ["-----------------------------", "TimeData1 comments"]
    comment += ["TimeData1"]
    comment += ["-----------------------------", "TimeData2 comments"]
    comment += ["TimeData2"]
    comment += ["-----------------------------"]
    comment += ["Gap filled from 2020-01-01 12:00:06 to 2020-01-01 12:00:09"]
    assert timeDataFilled.comments == comment

