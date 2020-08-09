"""Test clean module of resistics, time subpackage"""


def test_clean_removeZeros():
    """Test removeZeros"""
    from resistics.time.data import TimeData
    from resistics.time.clean import removeZeros
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
    assert timeData.sampleFreq == sampleFreq
    assert timeData.startTime == datetime.strptime(startTime, "%Y-%m-%d %H:%M:%S.%f")
    assert timeData.stopTime == datetime.strptime(stopTime, "%Y-%m-%d %H:%M:%S.%f")
    assert timeData.numSamples == 8
    assert timeData.numChans == 2
    assert timeData.period == 1 / 128
    assert timeData.nyquist == 64
    assert timeData.getComments() == []
    for chan in timeData:
        np.testing.assert_equal(timeData[chan], data[chan])
    timeDataProc = removeZeros(timeData)
    for chan in timeData:
        np.testing.assert_equal(timeDataProc[chan], data[chan])
    assert timeDataProc.comments == [
        "Sections of 20 consecutive zeros have been interpolated"
    ]
    # now change conzeros
    timeDataProc = removeZeros(timeData, 1)
    np.testing.assert_equal(timeDataProc["Ex"], np.array([4, 4.5, 5, 4, 3, 2, 4, 3]))
    np.testing.assert_equal(timeDataProc["Hy"], np.array([2, 9, 7, 6, 2, 3, 4, 5]))
    assert timeDataProc.comments == [
        "Sections of 1 consecutive zeros have been interpolated"
    ]


def test_clean_removeNans():
    """Test removeNans"""
    from resistics.time.data import TimeData
    from resistics.time.clean import removeNans
    from datetime import datetime
    import numpy as np

    data = {
        "Ex": np.array([4, np.nan, 5, np.nan, np.nan, 2, 4, 3], dtype=float),
        "Hy": np.array([2, 9, 7, 6, 2, np.nan, np.nan, 5], dtype=float),
    }
    sampleFreq = 128
    startTime = "2020-01-01 13:00:00.000"
    stopTime = "2020-01-01 13:00:00.055"
    timeData = TimeData(sampleFreq, startTime, stopTime, data)
    assert timeData.sampleFreq == sampleFreq
    assert timeData.startTime == datetime.strptime(startTime, "%Y-%m-%d %H:%M:%S.%f")
    assert timeData.stopTime == datetime.strptime(stopTime, "%Y-%m-%d %H:%M:%S.%f")
    assert timeData.numSamples == 8
    assert timeData.numChans == 2
    assert timeData.period == 1 / 128
    assert timeData.nyquist == 64
    assert timeData.getComments() == []
    for chan in timeData:
        np.testing.assert_equal(timeData[chan], data[chan])
    # remove nans
    timeDataProc = removeNans(timeData)
    np.testing.assert_equal(timeDataProc["Ex"], np.array([4, 4.5, 5, 4, 3, 2, 4, 3]))
    np.testing.assert_equal(timeDataProc["Hy"], np.array([2, 9, 7, 6, 2, 3, 4, 5]))
    assert timeDataProc.comments == ["NaN values in data have been interpolated"]
