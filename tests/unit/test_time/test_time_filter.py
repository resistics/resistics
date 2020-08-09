"""Test the filter module of resistics.time subpackage"""


def get_impulse():
    """Get an impulse signal for testing filters"""
    import numpy as np
    from resistics.time.data import TimeData

    sampleFreq = 128
    startTime = "2020-01-01 12:00:00.000"
    stopTime = "2020-01-01 12:01:00.000"
    data = {
        "Ex": np.zeros(shape=(128 * 60) + 1),
        "Hy": np.zeros(shape=(128 * 60) + 1),
    }
    # put an impulse in
    data["Ex"][128 * 30] = 1
    data["Hy"][128 * 30] = 1
    comments = ["This is a test"]
    return TimeData(sampleFreq, startTime, stopTime, data, comments)


def test_filter_lowPass():
    """Test low pass filtering"""
    from resistics.time.filter import lowPass

    timeData = get_impulse()
    timeDataProc = lowPass(timeData, 30)
    print(timeDataProc.data)
    assert timeDataProc.sampleFreq == timeData.sampleFreq
    assert timeDataProc.startTime == timeData.startTime
    assert timeDataProc.stopTime == timeData.stopTime
    assert timeDataProc.numSamples == timeData.numSamples
    assert timeDataProc.comments == timeData.comments + [
        "Low pass filter applied with cutoff 30 Hz"
    ]


def test_filter_highPass():
    """Test low pass filtering"""
    from resistics.time.filter import highPass

    timeData = get_impulse()
    timeDataProc = highPass(timeData, 30)
    print(timeDataProc.data)
    assert timeDataProc.sampleFreq == timeData.sampleFreq
    assert timeDataProc.startTime == timeData.startTime
    assert timeDataProc.stopTime == timeData.stopTime
    assert timeDataProc.numSamples == timeData.numSamples
    assert timeDataProc.comments == timeData.comments + [
        "High pass filter applied with cutoff 30 Hz"
    ]


def test_filter_bandPass():
    """Test low pass filtering"""
    from resistics.time.filter import bandPass

    timeData = get_impulse()
    timeDataProc = bandPass(timeData, 20, 40)
    print(timeDataProc.data)
    assert timeDataProc.sampleFreq == timeData.sampleFreq
    assert timeDataProc.startTime == timeData.startTime
    assert timeDataProc.stopTime == timeData.stopTime
    assert timeDataProc.numSamples == timeData.numSamples
    assert timeDataProc.comments == timeData.comments + [
        "Band pass filter applied with cutoffs 20 Hz and 40 Hz"
    ]


def test_filter_notch():
    """Test low pass filtering"""
    from resistics.time.filter import notchFilter

    timeData = get_impulse()
    timeDataProc = notchFilter(timeData, 30, 10)
    print(timeDataProc.data)
    assert timeDataProc.sampleFreq == timeData.sampleFreq
    assert timeDataProc.startTime == timeData.startTime
    assert timeDataProc.stopTime == timeData.stopTime
    assert timeDataProc.numSamples == timeData.numSamples
    assert timeDataProc.comments == timeData.comments + [
        "Notch filter applied at 30 Hz with band 10 Hz"
    ]


def test_filter_resample_odd_samples():
    """Test resampling"""
    from resistics.time.data import TimeData
    from resistics.time.filter import resample
    from datetime import datetime, timedelta
    import numpy as np

    sampleFreq = 128
    numSamples = sampleFreq * 60 + 1
    startTime = datetime.strptime("2020-01-01 12:00:00.000", "%Y-%m-%d %H:%M:%S.%f")
    stopTime = startTime + timedelta(seconds=((numSamples - 1) * (1 / sampleFreq)))
    data = {
        "Ex": np.random.randint(0, 100, size=(numSamples)),
        "Hy": np.random.randint(0, 100, size=(numSamples)),
    }
    comments = ["This is a test"]
    timeData = TimeData(sampleFreq, startTime, stopTime, data, comments)
    timeDataProc = resample(timeData, 64)
    assert timeDataProc.sampleFreq == 64
    assert timeDataProc.numSamples == numSamples // 2 + 1
    assert timeDataProc.startTime == timeData.startTime
    assert timeDataProc.stopTime == timeData.stopTime
    assert timeDataProc.comments == timeData.comments + [
        "Time data resampled from 128.000000 Hz to 64.000000 Hz"
    ]


def test_filter_resample_even_samples():
    """Test resampling"""
    from resistics.time.data import TimeData
    from resistics.time.filter import resample
    from datetime import datetime, timedelta
    import numpy as np

    sampleFreq = 100
    numSamples = sampleFreq * 60
    startTime = datetime.strptime("2020-01-01 12:00:00.000", "%Y-%m-%d %H:%M:%S.%f")
    stopTime = startTime + timedelta(seconds=((numSamples - 1) * (1 / sampleFreq)))
    data = {
        "Ex": np.random.randint(0, 100, size=(numSamples)),
        "Hy": np.random.randint(0, 100, size=(numSamples)),
    }
    comments = ["This is a test"]
    timeData = TimeData(sampleFreq, startTime, stopTime, data, comments)
    timeDataProc = resample(timeData, 50)
    assert timeDataProc.sampleFreq == 50
    assert timeDataProc.numSamples == numSamples // 2
    assert timeDataProc.startTime == timeData.startTime
    assert timeDataProc.stopTime == timeData.stopTime - timedelta(
        seconds=(1 / sampleFreq)
    )
    assert timeDataProc.comments == timeData.comments + [
        "Time data resampled from 100.000000 Hz to 50.000000 Hz"
    ]


def test_filter_downsample():
    """Test downsampling"""
    from resistics.time.data import TimeData
    from resistics.time.filter import downsample
    from datetime import datetime, timedelta
    import numpy as np

    sampleFreq = 128
    numSamples = sampleFreq * 60 + 1
    startTime = datetime.strptime("2020-01-01 12:00:00.000", "%Y-%m-%d %H:%M:%S.%f")
    stopTime = startTime + timedelta(seconds=((numSamples - 1) * (1 / sampleFreq)))
    data = {
        "Ex": np.random.randint(0, 100, size=(numSamples)),
        "Hy": np.random.randint(0, 100, size=(numSamples)),
    }
    comments = ["This is a test"]
    timeData = TimeData(sampleFreq, startTime, stopTime, data, comments)
    timeDataProc = downsample(timeData, 8)
    assert timeDataProc.sampleFreq == 16
    assert timeDataProc.numSamples == numSamples // 8 + 1
    assert timeDataProc.startTime == timeData.startTime
    assert timeDataProc.stopTime == timeData.stopTime
    assert timeDataProc.comments == timeData.comments + [
        "Time data decimated from 128.000000 Hz to 16.000000 Hz, new start time 2020-01-01 12:00:00, new end time 2020-01-01 12:01:00"
    ]
