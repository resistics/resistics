"""Test resistics.spectra.calculator module"""


def test_spectra_calculator_nopre():
    """Test Fourier transfrom"""
    from resistics.common.format import datetimeFormat
    from resistics.time.data import TimeData
    from resistics.spectra.calculator import SpectrumCalculator
    import numpy as np
    from datetime import datetime

    # intialise some time data
    sampleFreq = 128
    startTime = "2020-01-01 00:00:00.000000"
    stopTime = "2020-01-01 00:00:00.062500"
    data = {}
    # test with impulse on zero and impulse shifted to give a phase
    data["Ex"] = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    data["Hy"] = np.array([0, 1, 0, 0, 0, 0, 0, 0])
    timeData = TimeData(sampleFreq, startTime, stopTime, data)
    specCalc = SpectrumCalculator(128, 8)
    specCalc.detrend = False
    specCalc.applywindow = False
    specData = specCalc.calcFourierCoeff(timeData)
    assert np.absolute(specData.nyquist - 64) < 0.000001
    assert specData.windowSize == 8
    assert specData.dataSize == 5
    assert specData.numChans == 2
    assert sorted(specData.chans) == sorted(["Ex", "Hy"])
    assert specData.startTime == datetime.strptime(startTime, datetimeFormat(ns=True))
    assert specData.stopTime == datetime.strptime(stopTime, datetimeFormat(ns=True))
    np.testing.assert_array_almost_equal(specData.freqArray, [0, 16, 32, 48, 64])
    np.testing.assert_array_almost_equal(
        specData.data["Ex"],
        [1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j],
    )
    np.testing.assert_array_almost_equal(
        specData.data["Hy"],
        [1 + 0j, 0.707107 - 0.707107j, 0 - 1j, -0.707107 - 0.707107j, -1 + 0j,],
    )


def test_spectra_calculator_detrend():
    """Test Fourier transfrom with linear detrend applied"""
    from resistics.common.format import datetimeFormat
    from resistics.time.data import TimeData
    from resistics.spectra.calculator import SpectrumCalculator
    import numpy as np
    from datetime import datetime

    # intialise some time data
    sampleFreq = 128
    startTime = "2020-01-01 00:00:00.000000"
    stopTime = "2020-01-01 00:00:00.062500"
    data = {}
    # test with impulse on zero and impulse shifted to give a phase
    data["Ex"] = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    data["Hy"] = np.array([0, 1, 0, 0, 0, 0, 0, 0])
    timeData = TimeData(sampleFreq, startTime, stopTime, data)
    specCalc = SpectrumCalculator(128, 8)
    specCalc.detrend = True
    specCalc.applywindow = False
    specData = specCalc.calcFourierCoeff(timeData)
    assert np.absolute(specData.nyquist - 64) < 0.000001
    assert specData.windowSize == 8
    assert specData.dataSize == 5
    assert specData.numChans == 2
    assert sorted(specData.chans) == sorted(["Ex", "Hy"])
    assert specData.startTime == datetime.strptime(startTime, datetimeFormat(ns=True))
    assert specData.stopTime == datetime.strptime(stopTime, datetimeFormat(ns=True))
    np.testing.assert_array_almost_equal(specData.freqArray, [0, 16, 32, 48, 64])
    np.testing.assert_array_almost_equal(
        specData.data["Ex"],
        [1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j],
    )
    np.testing.assert_array_almost_equal(
        specData.data["Hy"],
        [1 + 0j, 0.707107 - 0.707107j, 0 - 1j, -0.707107 - 0.707107j, -1 + 0j,],
    )


def test_spectra_calculator_window():
    """Test Fourier transfrom with linear detrend applied"""
    from resistics.common.format import datetimeFormat
    from resistics.time.data import TimeData
    from resistics.spectra.calculator import SpectrumCalculator
    import numpy as np
    from datetime import datetime

    # intialise some time data
    sampleFreq = 128
    startTime = "2020-01-01 00:00:00.000000"
    stopTime = "2020-01-01 00:00:00.062500"
    data = {}
    # test with impulse on zero and impulse shifted to give a phase
    data["Ex"] = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    data["Hy"] = np.array([0, 1, 0, 0, 0, 0, 0, 0])
    timeData = TimeData(sampleFreq, startTime, stopTime, data)
    specCalc = SpectrumCalculator(128, 8)
    specCalc.detrend = False
    specCalc.applywindow = True
    specData = specCalc.calcFourierCoeff(timeData)
    assert np.absolute(specData.nyquist - 64) < 0.000001
    assert specData.windowSize == 8
    assert specData.dataSize == 5
    assert specData.numChans == 2
    assert sorted(specData.chans) == sorted(["Ex", "Hy"])
    assert specData.startTime == datetime.strptime(startTime, datetimeFormat(ns=True))
    assert specData.stopTime == datetime.strptime(stopTime, datetimeFormat(ns=True))
    np.testing.assert_array_almost_equal(specData.freqArray, [0, 16, 32, 48, 64])
    np.testing.assert_array_almost_equal(
        specData.data["Ex"],
        [1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j],
    )
    np.testing.assert_array_almost_equal(
        specData.data["Hy"],
        [1 + 0j, 0.707107 - 0.707107j, 0 - 1j, -0.707107 - 0.707107j, -1 + 0j,],
    )


def test_spectra_crosspowers():
    """Test crosspowers"""
    from resistics.spectra.data import SpectrumData
    from resistics.spectra.calculator import crosspowers
    import numpy as np

    data = {}
    data["Ex"] = np.array([1 + 3j, 2 + 5j, 7 + 6j, 3 + 2j])
    data["Ey"] = np.array([1 + 3j, 2 + 5j, 7 + 6j, 3 + 2j])
    data["Hx"] = np.array([2 + 9j, 9 + 1j, 8 + 8j, 6 + 2j])
    data["Hy"] = np.array([2 + 9j, 9 + 1j, 8 + 8j, 6 + 2j])
    specData = SpectrumData(
        8, 4, 128, "2020-01-01 00:00:00.000000", "2020-01-01 00:00:00.062500", data
    )
    # make sure that it matches autopower
    xpowers = crosspowers(specData)
    assert xpowers.dataSize == specData.dataSize
    assert xpowers.sampleFreq == specData.sampleFreq
    assert xpowers.numPowers == 16
    assert sorted(xpowers.powers) == sorted(
        [
            ["Ex","Ex"],
            ["Ex","Ey"],
            ["Ex","Hx"],
            ["Ex","Hy"],
            ["Ey","Ex"],
            ["Ey","Ey"],
            ["Ey","Hx"],
            ["Ey","Hy"],
            ["Hx","Ex"],
            ["Hx","Ey"],
            ["Hx","Hx"],
            ["Hx","Hy"],
            ["Hy","Ex"],
            ["Hy","Ey"],
            ["Hy","Hx"],
            ["Hy","Hy"],
        ]
    )
    # assert on the data elements
    for chan1 in specData.chans:
        for chan2 in specData.chans:
            cross1 = xpowers.getPower(chan1, chan2)
            cross2 = data[chan1]*np.conjugate(data[chan2])
            np.testing.assert_equal(cross1, cross2)