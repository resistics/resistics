"""
Tests for time data objects
"""


def test_timedata() -> None:
    """Test the time data object"""
    from resistics.time.data import TimeData
    from datetime import datetime
    import numpy as np

    sampleFreq = 128
    startTime = "2020-01-01 12:00:00.000"
    stopTime = "2020-01-01 12:01:00.000"
    data = {
        "Ex": np.random.randint(0, 100, size=(128 * 60) + 1),
        "Ey": np.random.randint(0, 100, size=(128 * 60) + 1),
        "Hx": np.random.randint(0, 100, size=(128 * 60) + 1),
        "Hy": np.random.randint(0, 100, size=(128 * 60) + 1),
        "Hz": np.random.randint(0, 100, size=(128 * 60) + 1),
    }
    comments = ["This is a test"]
    timeData = TimeData(sampleFreq, startTime, stopTime, data, comments)
    assert timeData.sampleFreq == sampleFreq
    assert timeData.period == 1 / 128
    assert timeData.numSamples == 128 * 60 + 1
    assert timeData.numChans == 5
    assert timeData.startTime == datetime.strptime(startTime, "%Y-%m-%d %H:%M:%S.%f")
    assert timeData.stopTime == datetime.strptime(stopTime, "%Y-%m-%d %H:%M:%S.%f")
    assert timeData.comments == ["This is a test"]
    # check accessors
    for chan in data:
        np.testing.assert_equal(timeData[chan], data[chan])
        np.testing.assert_equal(timeData.getChannel(chan), data[chan])
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
        "Ex": np.random.randint(0, 100, size=(128 * 60) + 1),
        "Ey": np.random.randint(0, 100, size=(128 * 60) + 1),
        "Hx": np.random.randint(0, 100, size=(128 * 60) + 1),
        "Hy": np.random.randint(0, 100, size=(128 * 60) + 1),
        "Hz": np.random.randint(0, 100, size=(128 * 60) + 1),
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
        np.testing.assert_equal(timeData.getChannel(chan), data["Ex"])
    # set using setChannel
    for chan in timeData.chans:
        if chan == "Ex":
            continue
        timeData.setChannel(chan, data["Hx"])
    for chan in timeData.chans:
        np.testing.assert_equal(timeData[chan], data["Hx"])
        np.testing.assert_equal(timeData.getChannel(chan), data["Hx"])


def test_timedata_iter():
    """Test iteration of time data"""
    from resistics.time.data import TimeData
    from datetime import datetime
    import numpy as np

    sampleFreq = 128
    startTime = "2020-01-01 12:00:00.000"
    stopTime = "2020-01-01 12:01:00.000"
    data = {
        "Ex": np.random.randint(0, 100, size=(128 * 60) + 1),
        "Ey": np.random.randint(0, 100, size=(128 * 60) + 1),
        "Hx": np.random.randint(0, 100, size=(128 * 60) + 1),
        "Hy": np.random.randint(0, 100, size=(128 * 60) + 1),
        "Hz": np.random.randint(0, 100, size=(128 * 60) + 1),
    }
    comments = ["This is a test"]
    timeData = TimeData(sampleFreq, startTime, stopTime, data, comments)
    timeIter = iter(timeData)
    assert next(timeIter) == "Ex"
    assert next(timeIter) == "Ey"
    assert next(timeIter) == "Hx"
    assert next(timeIter) == "Hy"
    assert next(timeIter) == "Hz"
    for idx, chan in enumerate(timeData):
        assert chan == timeData.chans[idx]
