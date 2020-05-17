"""
Tests for time data objects
"""

def test_time_data() -> None:
    """Test the time data object"""
    from resistics.time.data import TimeData
    from datetime import datetime
    import numpy as np

    sampleFreq = 128
    startTime = "2020-01-01 12:00:00.000"
    stopTime = "2020-01-01 12:01:00.000"
    data = {
        "Ex": np.random.randint(0, 100, size=128*60),
        "Ey": np.random.randint(0, 100, size=128*60),
        "Hx": np.random.randint(0, 100, size=128*60),
        "Hy": np.random.randint(0, 100, size=128*60),
        "Hz": np.random.randint(0, 100, size=128*60),
    }
    comments = ["This is a test"]
    timeData = TimeData(sampleFreq, startTime, stopTime, data, comments)
    assert timeData.sampleFreq == sampleFreq
    assert timeData.period == 1/128
    assert timeData.numSamples == 128*60
    assert timeData.numChans == 5
    assert timeData.startTime == datetime.strptime(startTime, "%Y-%m-%d %H:%M:%S.%f")
    assert timeData.stopTime == datetime.strptime(stopTime, "%Y-%m-%d %H:%M:%S.%f")
    assert timeData.comments == ["This is a test"]
    for chan in data:
        np.testing.assert_equal(timeData[chan], data[chan])
    timeData.addComment("Here is a new comment")
    assert timeData.comments == ["This is a test", "Here is a new comment"]


def test_timedata_getset():
    """Test the get set methods of spectrum data"""
    from resistics.time.data import TimeData
    from datetime import datetime
    import numpy as np

    sampleFreq = 128
    startTime = "2020-01-01 12:00:00.000"
    stopTime = "2020-01-01 12:01:00.000"
    data = {
        "Ex": np.random.randint(0, 100, size=128*60),
        "Ey": np.random.randint(0, 100, size=128*60),
        "Hx": np.random.randint(0, 100, size=128*60),
        "Hy": np.random.randint(0, 100, size=128*60),
        "Hz": np.random.randint(0, 100, size=128*60),
    }
    comments = ["This is a test"]
    timeData = TimeData(sampleFreq, startTime, stopTime, data, comments)
    for chan in timeData.chans:
        np.testing.assert_equal(timeData[chan], data[chan])
    # now set all chans to Ex
    for chan in timeData.chans:
        if chan == "Ex":
            continue
        timeData[chan] = data["Ex"]
    for chan in timeData.chans:
        np.testing.assert_equal(timeData[chan], data["Ex"])