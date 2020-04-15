def test_spectradata():
    """Test spectra data"""
    from resistics.spectra.data import SpectrumData
    from resistics.common.format import datetimeFormat
    import numpy as np
    from datetime import datetime

    startTime = "2020-01-01 00:00:00.000000"
    stopTime = "2020-01-01 00:00:00.062500"
    data = {}
    data["Ex"] = np.array([1 + 3j, 2 + 5j, 7 + 6j, 3 + 2j])
    data["Hy"] = np.array([2 + 9j, 9 + 1j, 8 + 8j, 6 + 2j])
    specData = SpectrumData(8, 4, 128, startTime, stopTime, data)
    assert specData.windowSize == 8
    assert specData.dataSize == 4
    assert specData.startTime == datetime.strptime(startTime, datetimeFormat(ns=True))
    assert specData.stopTime == datetime.strptime(stopTime, datetimeFormat(ns=True))
    assert specData.sampleFreq == 128
    assert specData.numChans == 2
    assert specData.chans == ["Ex", "Hy"]
    assert specData.getComments() == []
    specData.addComment("This is a comment")
    assert specData.getComments() == ["This is a comment"]
    np.testing.assert_almost_equal(specData.data["Ex"], data["Ex"])
    np.testing.assert_almost_equal(specData.data["Hy"], data["Hy"])
    assert specData.nyquist == 64
    np.testing.assert_almost_equal(specData.freqArray, [0, (64 / 3), (64 / 3) * 2, 64])


def test_powerdata():
    "Test PowerData"
    from resistics.spectra.data import PowerData
    import numpy as np

    data = {}
    data["Ex-Hx"] = np.array([0 + 4j, 1 + 3j, 6 + 6j, 7 + 3j])
    data["Ex-Hy"] = np.array([0 + 1j, 5 + 2j, 5 + 4j, 2 + 3j])
    apower = PowerData("cross", 4096, data)
    # make sure that it matches autopower
    assert apower.dataSize == 4
    assert apower.sampleFreq == 4096
    assert apower.numPowers == 2
    assert apower.powers == ["Ex-Hx", "Ex-Hy"]
    assert np.array_equal(apower.data["Ex-Hx"], data["Ex-Hx"])
    assert np.array_equal(apower.data["Ex-Hy"], data["Ex-Hy"])
    assert apower.nyquist == 2048
    assert np.array_equal(apower.freqArray, [0, 2048 / 3, 2048 * 2 / 3, 2048])


def test_powerdata_smooth():
    """Test PowerData smooth"""
    from resistics.spectra.data import PowerData
    from resistics.common.smooth import smooth1d
    import numpy as np

    data = {}
    data["Ex-Hx"] = np.array(
        [
            0 + 4j,
            -1 + 3j,
            6 + 6j,
            7 + 3j,
            -6 - 1j,
            5 + 5j,
            6 + 2j,
            1 - 3j,
            -7 + 8j,
            1 - 9j,
            3 + 4j,
        ]
    )
    data["Ex-Hy"] = np.array(
        [
            0 + 1j,
            5 + 2j,
            5 - 4j,
            2 + 3j,
            4 + 3j,
            1 - 6j,
            -2 + 4j,
            -2 + 7j,
            4 + 4j,
            3 - 2j,
            6 + 3j,
        ]
    )
    apower = PowerData("cross", 128, data)
    # make sure that it matches autopower
    assert apower.dataSize == 11
    assert apower.sampleFreq == 128
    assert apower.numPowers == 2
    assert apower.powers == ["Ex-Hx", "Ex-Hy"]
    assert np.array_equal(apower.data["Ex-Hx"], data["Ex-Hx"])
    assert np.array_equal(apower.data["Ex-Hy"], data["Ex-Hy"])
    assert apower.nyquist == 64
    assert np.array_equal(apower.freqArray, np.linspace(0, 64, 11))
    smooth = apower.smooth(3, "boxcar")
    for power in smooth.data:
        print(power, smooth.data[power])

    assert False



def test_powerdata_interpolate():
    """Test PowerData interpolate"""
    from resistics.spectra.data import PowerData
    import numpy as np

    data = {}
    data["Ex-Hx"] = np.array(
        [
            0 + 4j,
            -1 + 3j,
            6 + 6j,
            7 + 3j,
            -6 - 1j,
            5 + 5j,
            6 + 2j,
            1 - 3j,
            -7 + 8j,
            1 - 9j,
            3 + 4j,
        ]
    )
    data["Ex-Hy"] = np.array(
        [
            0 + 1j,
            5 + 2j,
            5 - 4j,
            2 + 3j,
            4 + 3j,
            1 - 6j,
            -2 + 4j,
            -2 + 7j,
            4 + 4j,
            3 - 2j,
            6 + 3j,
        ]
    )
    apower = PowerData("cross", 128, data)
    # make sure that it matches autopower
    assert apower.dataSize == 11
    assert apower.sampleFreq == 128
    assert apower.numPowers == 2
    assert apower.powers == ["Ex-Hx", "Ex-Hy"]
    assert np.array_equal(apower.data["Ex-Hx"], data["Ex-Hx"])
    assert np.array_equal(apower.data["Ex-Hy"], data["Ex-Hy"])
    assert apower.nyquist == 64
    assert np.array_equal(apower.freqArray, np.linspace(0, 64, 11))
    interpolated = apower.interpolate([5, 10, 15, 20])
    for power in interpolated.data:
        print(power, interpolated.data[power])
    assert False
