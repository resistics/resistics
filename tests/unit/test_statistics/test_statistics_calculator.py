def get_spectrum_data():
    """Get a dummy spectrum dataset"""
    from resistics.spectra.data import SpectrumData
    import numpy as np

    # add some data
    startTime = "2020-01-01 00:00:00.000000"
    stopTime = "2020-01-01 00:00:00.062500"
    data = {}
    data["Ex"] = np.array([1 + 3j, -2 + 5j, 7 - 6j, 3 + 2j, 4 + 8j])
    data["Ey"] = np.array([12 - 4j, -6 + 2j, 2 + 6j, -4 - 2j, -6 - 6j])
    data["Hx"] = np.array([-3 + 3j, -11 + 7j, 4 - 1j, 1 + 9j, 2 + 2j])
    data["Hy"] = np.array([2 + 9j, 9 + 1j, 8 + 8j, 6 + 2j, 5 + 2j])
    specData = SpectrumData(8, 5, 128, startTime, stopTime, data)
    evalfreq = np.array([24, 40])
    return specData, evalfreq


def test_statistics_calculator() -> None:
    """Test initialising the statistics calculator"""
    from resistics.statistics.calculator import StatisticCalculator
    import numpy as np

    calculator = StatisticCalculator()
    assert calculator.inChans == ["Hx", "Hy"]
    assert calculator.inSize == 2
    assert calculator.outChans == ["Ex", "Ey"]
    assert calculator.outSize == 2
    assert calculator.specChans == ["Hx", "Hy", "Ex", "Ey"]
    assert calculator.remoteChans == ["Hx", "Hy"]
    assert calculator.psdChans == ["Ex", "Ey", "Hx", "Hy"]
    assert calculator.cohPairs == [
        ["Ex", "Hx"],
        ["Ex", "Hy"],
        ["Ey", "Hx"],
        ["Ey", "Hy"],
    ]
    assert calculator.polDirs == [["Ex", "Ey"], ["Hx", "Hy"]]
    specData, evalfreq = get_spectrum_data()
    calculator.winLen = 1
    assert calculator.winLen == 1
    calculator.setSpectra(specData.freqArray, specData, evalfreq)
    # expected results
    powerDict = {
        "Hx-Hx": [18.0 + 0.0j, 170.0 + 0.0j, 17.0 + 0.0j, 82.0 + 0.0j, 8.0 + 0.0j],
        "Hx-Hy": [21.0 + 33.0j, -92.0 + 74.0j, 24.0 - 40.0j, 24.0 + 52.0j, 14.0 + 6.0j],
        "Hx-Ex": [6.0 + 12.0j, 57.0 + 41.0j, 34.0 + 17.0j, 21.0 + 25.0j, 24.0 - 8.0j],
        "Hx-Ey": [
            -48.0 + 24.0j,
            80.0 - 20.0j,
            2.0 - 26.0j,
            -22.0 - 34.0j,
            -24.0 + 0.0j,
        ],
        "Hy-Hx": [21.0 - 33.0j, -92.0 - 74.0j, 24.0 + 40.0j, 24.0 - 52.0j, 14.0 - 6.0j],
        "Hy-Hy": [85.0 + 0.0j, 82.0 + 0.0j, 128.0 + 0.0j, 40.0 + 0.0j, 29.0 + 0.0j],
        "Hy-Ex": [29.0 + 3.0j, -13.0 - 47.0j, 8.0 + 104.0j, 22.0 - 6.0j, 36.0 - 32.0j],
        "Hy-Ey": [
            -12.0 + 116.0j,
            -52.0 - 24.0j,
            64.0 - 32.0j,
            -28.0 + 4.0j,
            -42.0 + 18.0j,
        ],
        "Ex-Hx": [6.0 - 12.0j, 57.0 - 41.0j, 34.0 - 17.0j, 21.0 - 25.0j, 24.0 + 8.0j],
        "Ex-Hy": [29.0 - 3.0j, -13.0 + 47.0j, 8.0 - 104.0j, 22.0 + 6.0j, 36.0 + 32.0j],
        "Ex-Ex": [10.0 + 0.0j, 29.0 + 0.0j, 85.0 + 0.0j, 13.0 + 0.0j, 80.0 + 0.0j],
        "Ex-Ey": [
            0.0 + 40.0j,
            22.0 - 26.0j,
            -22.0 - 54.0j,
            -16.0 - 2.0j,
            -72.0 - 24.0j,
        ],
        "Ey-Hx": [
            -48.0 - 24.0j,
            80.0 + 20.0j,
            2.0 + 26.0j,
            -22.0 + 34.0j,
            -24.0 - 0.0j,
        ],
        "Ey-Hy": [
            -12.0 - 116.0j,
            -52.0 + 24.0j,
            64.0 + 32.0j,
            -28.0 - 4.0j,
            -42.0 - 18.0j,
        ],
        "Ey-Ex": [
            0.0 - 40.0j,
            22.0 + 26.0j,
            -22.0 + 54.0j,
            -16.0 + 2.0j,
            -72.0 + 24.0j,
        ],
        "Ey-Ey": [160.0 + 0.0j, 40.0 + 0.0j, 40.0 + 0.0j, 20.0 + 0.0j, 72.0 + 0.0j],
    }
    evalDict = {
        "Hx-Hx": np.array([93.5 + 0.0j, 49.5 + 0.0j]),
        "Hx-Hy": np.array([-34.0 + 17.0j, 24.0 + 6.0j]),
        "Hx-Ex": np.array([45.5 + 29.0j, 27.5 + 21.0j]),
        "Hx-Ey": np.array([41.0 - 23.0j, -10.0 - 30.0j]),
        "Hy-Hx": np.array([-34.0 - 17.0j, 24.0 - 6.0j]),
        "Hy-Hy": np.array([105.0 + 0.0j, 84.0 + 0.0j]),
        "Hy-Ex": np.array([-2.5 + 28.5j, 15.0 + 49.0j]),
        "Hy-Ey": np.array([6.0 - 28.0j, 18.0 - 14.0j]),
        "Ex-Hx": np.array([45.5 - 29.0j, 27.5 - 21.0j]),
        "Ex-Hy": np.array([-2.5 - 28.5j, 15.0 - 49.0j]),
        "Ex-Ex": np.array([57.0 + 0.0j, 49.0 + 0.0j]),
        "Ex-Ey": np.array([0.0 - 40.0j, -19.0 - 28.0j]),
        "Ey-Hx": np.array([41.0 + 23.0j, -10.0 + 30.0j]),
        "Ey-Hy": np.array([6.0 + 28.0j, 18.0 + 14.0j]),
        "Ey-Ex": np.array([0.0 + 40.0j, -19.0 + 28.0j]),
        "Ey-Ey": np.array([40.0 + 0.0j, 30.0 + 0.0j]),
    }
    # check the autopower data
    for key in powerDict:
        splitkey = key.split("-")
        chan1 = splitkey[0]
        chan2 = splitkey[1]
        np.testing.assert_almost_equal(
            calculator.xpowers.getPower(chan1, chan2), np.array(powerDict[key])
        )
        np.testing.assert_almost_equal(
            calculator.xpowersEval.getPower(chan1, chan2), evalDict[key]
        )


def test_statistics_calculator_absval():
    """Test absolute value calculator"""
    from resistics.statistics.calculator import StatisticCalculator
    import numpy as np

    specData, evalfreq = get_spectrum_data()
    calculator = StatisticCalculator()
    calculator.winLen = 1
    assert calculator.winLen == 1
    calculator.setSpectra(specData.freqArray, specData, evalfreq)
    statData = calculator.getDataForStatName("absvalEqn")
    testData = {
        24: {
            "absExHx": 53.956000593075835,
            "absEyHx": 47.01063709417264,
            "absHxHx": 93.5,
            "absHyHx": 38.01315561749642,
            "absExHy": 28.609439001839934,
            "absEyHy": 28.635642126552707,
            "absHxHy": 38.01315561749642,
            "absHyHy": 105.0,
            "absExEx": 57.0,
            "absEyEx": 40.0,
            "absHxEx": 53.956000593075835,
            "absHyEx": 28.609439001839934,
            "absExEy": 40.0,
            "absEyEy": 40.0,
            "absHxEy": 47.01063709417264,
            "absHyEy": 28.635642126552707,
        },
        40: {
            "absExHx": 34.60130055359191,
            "absEyHx": 31.622776601683793,
            "absHxHx": 49.5,
            "absHyHx": 24.73863375370596,
            "absExHy": 51.24451190127583,
            "absEyHy": 22.80350850198276,
            "absHxHy": 24.73863375370596,
            "absHyHy": 84.0,
            "absExEx": 49.0,
            "absEyEx": 33.83784863137726,
            "absHxEx": 34.60130055359191,
            "absHyEx": 51.24451190127583,
            "absExEy": 33.83784863137726,
            "absEyEy": 30.0,
            "absHxEy": 31.622776601683793,
            "absHyEy": 22.80350850198276,
        },
    }
    for efreq in evalfreq:
        for key, val in statData[efreq].items():
            np.testing.assert_almost_equal(val, testData[efreq][key])


def test_statistics_calculator_coherence():
    """Test absolute value calculator"""
    from resistics.statistics.calculator import StatisticCalculator
    import numpy as np

    specData, evalfreq = get_spectrum_data()
    calculator = StatisticCalculator()
    calculator.winLen = 1
    assert calculator.winLen == 1
    calculator.setSpectra(specData.freqArray, specData, evalfreq)
    statData = calculator.getDataForStatName("coherence")
    testData = {
        24: {
            "cohExHx": 0.5462519936204147,
            "cohExHy": 0.13675856307435255,
            "cohEyHx": 0.590909090909091,
            "cohEyHy": 0.19523809523809524,
        },
        40: {
            "cohExHx": 0.49360956503813647,
            "cohExHy": 0.6379980563654033,
            "cohEyHx": 0.6734006734006734,
            "cohEyHy": 0.20634920634920634,
        },
    }
    for efreq in evalfreq:
        for key, val in statData[efreq].items():
            np.testing.assert_almost_equal(val, testData[efreq][key])


def test_statistics_calculator_partial_coherence():
    """Test absolute value calculator"""
    from resistics.statistics.calculator import StatisticCalculator
    import numpy as np

    specData, evalfreq = get_spectrum_data()
    calculator = StatisticCalculator()
    calculator.winLen = 1
    assert calculator.winLen == 1
    calculator.setSpectra(specData.freqArray, specData, evalfreq)
    statData = calculator.getDataForStatName("partialCoherence")
    testData = {
        24: {
            "cohExHx": 0.5462519936204147,
            "cohExHy": 0.13675856307435255,
            "cohEyHx": 0.590909090909091,
            "cohEyHy": 0.19523809523809524,
            "bivarEx": (1 + 0j),
            "parExHx": (1 + 0j),
            "parExHy": (1 + 0j),
            "bivarEy": (1 + 4.4408920985006264e-17j),
            "parEyHx": (1 + 5.518268288077701e-17j),
            "parEyHy": (0.9999999999999999 + 1.085551401855709e-16j),
        },
        40: {
            "cohExHx": 0.49360956503813647,
            "cohExHy": 0.6379980563654033,
            "cohEyHx": 0.6734006734006734,
            "cohEyHy": 0.20634920634920634,
            "bivarEx": (1 + 0j),
            "parExHx": (1 + 0j),
            "parExHy": (1 + 0j),
            "bivarEy": (1 + 2.960594732333751e-17j),
            "parEyHx": (0.9999999999999999 + 3.7303493627405256e-17j),
            "parEyHy": (0.9999999999999999 + 9.064913768073444e-17j),
        },
    }
    for efreq in evalfreq:
        for key, val in statData[efreq].items():
            np.testing.assert_almost_equal(val, testData[efreq][key])


def test_statistics_calculator_power_spectral_density():
    """Test absolute value calculator"""
    from resistics.statistics.calculator import StatisticCalculator
    import numpy as np

    specData, evalfreq = get_spectrum_data()
    calculator = StatisticCalculator()
    calculator.winLen = 1
    assert calculator.winLen == 1
    calculator.setSpectra(specData.freqArray, specData, evalfreq)
    statData = calculator.getDataForStatName("powerSpectralDensity")
    testData = {
        24: {"psdEx": 912.0, "psdEy": 640.0, "psdHx": 1496.0, "psdHy": 1680.0},
        40: {"psdEx": 784.0, "psdEy": 480.0, "psdHx": 792.0, "psdHy": 1344.0},
    }
    for efreq in evalfreq:
        for key, val in statData[efreq].items():
            np.testing.assert_almost_equal(val, testData[efreq][key])


def test_statistics_calculator_polarisation_direction():
    """Test absolute value calculator"""
    from resistics.statistics.calculator import StatisticCalculator
    import numpy as np

    specData, evalfreq = get_spectrum_data()
    calculator = StatisticCalculator()
    calculator.winLen = 1
    assert calculator.winLen == 1
    calculator.setSpectra(specData.freqArray, specData, evalfreq)
    statData = calculator.getDataForStatName("polarisationDirection")
    testData = {
        24: {"polExEy": 0.0, "polHxHy": 80.4010969314582},
        40: {"polExEy": -63.43494882292201, "polHxHy": -54.293308599397115},
    }
    for efreq in evalfreq:
        for key, val in statData[efreq].items():
            np.testing.assert_almost_equal(val, testData[efreq][key])


def test_statistics_calculator_res_phase():
    """Test absolute value calculator"""
    from resistics.statistics.calculator import StatisticCalculator
    import numpy as np

    specData, evalfreq = get_spectrum_data()
    calculator = StatisticCalculator()
    calculator.winLen = 1
    assert calculator.winLen == 1
    calculator.setSpectra(specData.freqArray, specData, evalfreq)
    statData = calculator.getDataForStatName("resPhase")
    testData = {
        24: {
            "ExHxRes": 0.0051423310440927615,
            "ExHxPhase": -38.08089717250079,
            "ExHxReal": 0.6183338309943266,
            "ExHxImag": -0.484502836667662,
            "ExHyRes": 0.002406937394247041,
            "ExHyPhase": -79.49669804710025,
            "ExHyReal": 0.09796954314720807,
            "ExHyImag": -0.5284263959390865,
            "EyHxRes": 0.003364188314919875,
            "EyHxPhase": 40.70059399014801,
            "EyHxReal": 0.48169602866527317,
            "EyHxImag": 0.4143326366079426,
            "EyHyRes": 0.001522842639593909,
            "EyHyPhase": 49.044485574181074,
            "EyHyReal": 0.2802030456852794,
            "EyHyImag": 0.3228426395939085,
        },
        40: {
            "ExHxRes": 0.0021009588268471532,
            "ExHxPhase": -12.512585801455565,
            "ExHxReal": 0.6328257191201355,
            "ExHxImag": -0.14043993231810512,
            "ExHyRes": 0.0017318809926677931,
            "ExHyPhase": -91.1943471837543,
            "ExHyReal": -0.012267343485617588,
            "ExHyImag": -0.5884094754653127,
            "EyHxRes": 0.002820078962210943,
            "EyHxPhase": 120.6095367512591,
            "EyHxReal": -0.3824027072758038,
            "EyHxImag": 0.6463620981387479,
            "EyHyRes": 0.0006838691483361542,
            "EyHyPhase": 1.4419233716812918,
            "EyHyReal": 0.36971235194585467,
            "EyHyImag": 0.009306260575296085,
        },
    }
    for efreq in evalfreq:
        for key, val in statData[efreq].items():
            np.testing.assert_almost_equal(val, testData[efreq][key])


def test_statistics_calculator_transfer_function():
    """Test absolute value calculator"""
    from resistics.statistics.calculator import StatisticCalculator
    import numpy as np

    specData, evalfreq = get_spectrum_data()
    calculator = StatisticCalculator()
    calculator.winLen = 1
    assert calculator.winLen == 1
    calculator.setSpectra(specData.freqArray, specData, evalfreq)
    statData = calculator.getDataForStatName("transferFunction")
    testData = {
        24: {
            "ExHxRes": 0.0051423310440927615,
            "ExHxPhase": -38.08089717250079,
            "ExHxReal": 0.6183338309943266,
            "ExHxImag": -0.484502836667662,
            "ExHyRes": 0.002406937394247041,
            "ExHyPhase": -79.49669804710025,
            "ExHyReal": 0.09796954314720807,
            "ExHyImag": -0.5284263959390865,
            "EyHxRes": 0.003364188314919875,
            "EyHxPhase": 40.70059399014801,
            "EyHxReal": 0.48169602866527317,
            "EyHxImag": 0.4143326366079426,
            "EyHyRes": 0.001522842639593909,
            "EyHyPhase": 49.044485574181074,
            "EyHyReal": 0.2802030456852794,
            "EyHyImag": 0.3228426395939085,
        },
        40: {
            "ExHxRes": 0.0021009588268471532,
            "ExHxPhase": -12.512585801455565,
            "ExHxReal": 0.6328257191201355,
            "ExHxImag": -0.14043993231810512,
            "ExHyRes": 0.0017318809926677931,
            "ExHyPhase": -91.1943471837543,
            "ExHyReal": -0.012267343485617588,
            "ExHyImag": -0.5884094754653127,
            "EyHxRes": 0.002820078962210943,
            "EyHxPhase": 120.6095367512591,
            "EyHxReal": -0.3824027072758038,
            "EyHxImag": 0.6463620981387479,
            "EyHyRes": 0.0006838691483361542,
            "EyHyPhase": 1.4419233716812918,
            "EyHyReal": 0.36971235194585467,
            "EyHyImag": 0.009306260575296085,
        },
    }
    for efreq in evalfreq:
        for key, val in statData[efreq].items():
            np.testing.assert_almost_equal(val, testData[efreq][key])
