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


def test_spectra_autopower():
    """Test autopower function"""
    from resistics.spectra.data import SpectrumData
    from resistics.spectra.calculator import autopower
    import numpy as np

    # create spectrum data object
    data = {}
    data["Ex"] = np.array([1 + 3j, 2 + 5j, 7 + 6j, 3 + 2j])
    data["Hy"] = np.array([2 + 9j, 9 + 1j, 8 + 8j, 6 + 2j])
    specData = SpectrumData(
        8, 4, 128, "2020-01-01 00:00:00.000000", "2020-01-01 00:00:00.062500", data
    )
    apower = autopower(specData)
    assert apower.dataSize == specData.dataSize
    assert apower.sampleFreq == specData.sampleFreq
    assert apower.numPowers == 4
    assert sorted(apower.powers) == sorted(["Ex-Ex", "Ex-Hy", "Hy-Ex", "Hy-Hy"])
    # assert on the data elements
    assert np.array_equal(
        apower.data["Ex-Ex"], [10.0 + 0.0j, 29.0 + 0.0j, 85.0 + 0.0j, 13.0 + 0.0j]
    )
    assert np.array_equal(
        apower.data["Ex-Hy"], [29.0 - 3.0j, 23.0 + 43.0j, 104.0 - 8.0j, 22.0 + 6.0j]
    )
    assert np.array_equal(
        apower.data["Hy-Ex"], [29.0 + 3.0j, 23.0 - 43.0j, 104.0 + 8.0j, 22.0 - 6.0j]
    )
    assert np.array_equal(
        apower.data["Hy-Hy"], [85.0 + 0.0j, 82.0 + 0.0j, 128.0 + 0.0j, 40.0 + 0.0j]
    )


def test_spectra_crosspower_auto():
    """Test crosspowers"""
    from resistics.spectra.data import SpectrumData
    from resistics.spectra.calculator import crosspower
    import numpy as np

    data = {}
    data["Ex"] = np.array([1 + 3j, 2 + 5j, 7 + 6j, 3 + 2j])
    data["Hy"] = np.array([2 + 9j, 9 + 1j, 8 + 8j, 6 + 2j])
    specData1 = SpectrumData(
        8, 4, 128, "2020-01-01 00:00:00.000000", "2020-01-01 00:00:00.062500", data
    )

    data = {}
    data["Ex"] = np.array([1 + 3j, 2 + 5j, 7 + 6j, 3 + 2j])
    data["Hy"] = np.array([2 + 9j, 9 + 1j, 8 + 8j, 6 + 2j])
    specData2 = SpectrumData(
        8, 4, 128, "2020-01-01 00:00:00.000000", "2020-01-01 00:00:00.062500", data
    )
    # make sure that it matches autopower
    apower = crosspower(specData1, specData2)
    assert apower.dataSize == specData1.dataSize
    assert apower.sampleFreq == specData1.sampleFreq
    assert apower.numPowers == 4
    assert sorted(apower.powers) == sorted(["Ex-Ex", "Ex-Hy", "Hy-Ex", "Hy-Hy"])
    # assert on the data elements
    assert np.array_equal(
        apower.data["Ex-Ex"], [10.0 + 0.0j, 29.0 + 0.0j, 85.0 + 0.0j, 13.0 + 0.0j]
    )
    assert np.array_equal(
        apower.data["Ex-Hy"], [29.0 - 3.0j, 23.0 + 43.0j, 104.0 - 8.0j, 22.0 + 6.0j]
    )
    assert np.array_equal(
        apower.data["Hy-Ex"], [29.0 + 3.0j, 23.0 - 43.0j, 104.0 + 8.0j, 22.0 - 6.0j]
    )
    assert np.array_equal(
        apower.data["Hy-Hy"], [85.0 + 0.0j, 82.0 + 0.0j, 128.0 + 0.0j, 40.0 + 0.0j]
    )


def test_spectra_crosspower_cross():
    """Test crosspowers"""
    from resistics.spectra.data import SpectrumData
    from resistics.spectra.calculator import crosspower
    import numpy as np

    data = {}
    data["Ex"] = np.array([1 + 3j, 2 + 5j, 7 + 6j, 3 + 2j])
    data["Hy"] = np.array([2 + 9j, 9 + 1j, 8 + 8j, 6 + 2j])
    specData1 = SpectrumData(
        8, 4, 128, "2020-01-01 00:00:00.000000", "2020-01-01 00:00:00.062500", data
    )

    data = {}
    data["Ey"] = np.array([0 + 4j, 1 + 3j, 6 + 6j, 7 + 3j])
    data["Hx"] = np.array([0 + 1j, 5 + 2j, 5 + 4j, 2 + 3j])
    specData2 = SpectrumData(
        8, 4, 128, "2020-01-01 00:00:00.000000", "2020-01-01 00:00:00.062500", data
    )
    # make sure that it matches autopower
    apower = crosspower(specData1, specData2)
    assert apower.dataSize == specData1.dataSize
    assert apower.sampleFreq == specData1.sampleFreq
    assert apower.numPowers == 4
    assert sorted(apower.powers) == sorted(["Ex-Ey", "Ex-Hx", "Hy-Ey", "Hy-Hx"])
    # assert on the data elements
    assert np.array_equal(
        apower.data["Ex-Ey"], [12.0 - 4.0j, 17.0 - 1.0j, 78.0 - 6.0j, 27.0 + 5.0j]
    )
    assert np.array_equal(
        apower.data["Ex-Hx"], [3.0 - 1.0j, 20.0 + 21.0j, 59.0 + 2.0j, 12.0 - 5.0j]
    )
    assert np.array_equal(
        apower.data["Hy-Ey"], [36.0 - 8.0j, 12.0 - 26.0j, 96.0 + 0.0j, 48.0 - 4.0j]
    )
    assert np.array_equal(
        apower.data["Hy-Hx"], [9.0 - 2.0j, 47.0 - 13.0j, 72.0 + 8.0j, 18.0 - 14.0j]
    )
