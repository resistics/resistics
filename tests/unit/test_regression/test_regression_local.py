class mock_decimation_parameters:
    """Mock decimation parameters"""

    def __init__(self):
        import numpy as np

        self.numLevels = 1
        self.evalfreq = np.array([24, 40])

    def getEvalFrequenciesForLevel(self, declevel):
        return self.evalfreq


class mock_window_parameters:
    """Mock window parameters"""

    def __init__(self):
        return


class mock_spectra_data:
    """Mock spectra data"""

    def __init__(self, data):
        import numpy as np

        self.data = {}
        for chan in data:
            self.data[chan] = np.array(data[chan])
        self.chans = list(self.data.keys())
        self.windowSize = 8
        self.dataSize = len(self.data[self.chans[0]])
        self.freqArray = np.array([0, 16, 32, 48, 64])
        self.sampleFreq = 128
        self.startTime = "2020-01-01 00:00:00.00"
        self.stopTime = "2020-01-01 01:00:00.00"


class mock_spectra_io:
    """Mock spectra io"""

    def __init__(self, data, windows):
        self.data = []
        for d in data:
            self.data.append(mock_spectra_data(d))
        self.windows = windows

    def readBinaryBatchGlobal(self, globalIndices):
        return self.data, self.windows

    def closeFile(self):
        return


class mock_window_selector:
    """Mock window selector"""

    def __init__(self):
        self.decParams = mock_decimation_parameters()
        self.winParams = mock_window_parameters()
        self.sites = ["site1", "site2"]
        self.specdir = "test"

    def getDataSize(self, declevel):
        return 5

    def getNumSharedWindows(self, declevel):
        return 3

    def getUnmaskedWindowsLevel(self, declevel):
        return set([1, 2, 3])

    def getWindowsForFreq(self, declevel, eIdx):
        return set([1, 2, 3])

    def getSpecReaderBatches(self, declevel):
        data1 = [
            {
                "Ex": [3, 4, 5, 6, 7],
                "Ey": [3, 4, 5, 6, 7],
                "Hx": [5, 4, 2, 3, 5],
                "Hy": [5, 6, 7, 8, 6],
            },
            {
                "Ex": [3, 4, 5, 6, 7],
                "Ey": [3, 4, 5, 6, 7],
                "Hx": [5, 4, 2, 3, 5],
                "Hy": [5, 6, 7, 8, 6],
            },
            {
                "Ex": [3, 4, 5, 6, 7],
                "Ey": [3, 4, 5, 6, 7],
                "Hx": [5, 4, 2, 3, 5],
                "Hy": [5, 6, 7, 8, 6],
            },
        ]
        data2 = [
            {
                "Ex": [3, 4, 5, 6, 7],
                "Ey": [3, 4, 5, 6, 7],
                "Hx": [5, 4, 2, 3, 5],
                "Hy": [5, 6, 7, 8, 6],
            },
            {
                "Ex": [3, 4, 5, 6, 7],
                "Ey": [3, 4, 5, 6, 7],
                "Hx": [5, 4, 2, 3, 5],
                "Hy": [5, 6, 7, 8, 6],
            },
            {
                "Ex": [3, 4, 5, 6, 7],
                "Ey": [3, 4, 5, 6, 7],
                "Hx": [5, 4, 2, 3, 5],
                "Hy": [5, 6, 7, 8, 6],
            },
        ]
        windows = [1, 2, 3]
        site1_spec = mock_spectra_io(data1, windows)
        site2_spec = mock_spectra_io(data2, windows)
        return [{"globalrange": [1, 3], "site1": site1_spec, "site2": site2_spec}]


def mock_local_regressor_writeResult(a, b, c, d, e, f):
    return


def test_local_regressor_setCores():
    """Test local regressor setCores"""
    from resistics.regression.local import LocalRegressor

    selector = mock_window_selector()
    regressor = LocalRegressor(selector, "test")
    assert regressor.ncores == 0
    regressor.setCores(4)
    assert regressor.ncores == 4
    regressor.setCores(11)
    assert regressor.ncores == 11


def test_local_regressor_setSmooth():
    """Test local regressor setSmooth"""
    from resistics.regression.local import LocalRegressor

    selector = mock_window_selector()
    regressor = LocalRegressor(selector, "test")
    assert regressor.smoothFunc == "hann"
    assert regressor.smoothLen is None
    regressor.setSmooth("hann", 7)
    assert regressor.smoothFunc == "hann"
    assert regressor.smoothLen == 7
    regressor.setSmooth("parzen", 8)
    assert regressor.smoothFunc == "parzen"
    assert regressor.smoothLen == 8
    assert regressor.getSmoothLen(65) == 9


def test_local_regressor_setMethod():
    """Test local regressor setMethod"""
    from resistics.regression.local import LocalRegressor

    selector = mock_window_selector()
    regressor = LocalRegressor(selector, "test")
    assert regressor.method == "cm"
    assert regressor.intercept == False
    assert regressor.stack == False
    regressor.setMethod("ols", intercept=False, stack=False)
    assert regressor.method == "ols"
    assert regressor.intercept == False
    assert regressor.stack == False
    regressor.setMethod("cm", intercept=True, stack=False)
    assert regressor.method == "cm"
    assert regressor.intercept == True
    assert regressor.stack == False
    regressor.setMethod("mm", intercept=False, stack=True)
    assert regressor.method == "mm"
    assert regressor.intercept == False
    assert regressor.stack == True


def test_local_regressor_setInput():
    """Test local regressor setInput"""
    from resistics.regression.local import LocalRegressor

    selector = mock_window_selector()
    regressor = LocalRegressor(selector, "test")
    regressor.setInput("site1", ["Hx", "Hy"])
    assert regressor.inSite == "site1"
    assert regressor.inChannels == ["Hx", "Hy"]
    assert regressor.inSize == 2
    assert regressor.inCross == ["Hx", "Hy"]
    regressor.setInput("site2", ["Ex", "Hy", "Hx"], inCross=["Ex", "Hy"])
    assert regressor.inSite == "site2"
    assert regressor.inChannels == ["Ex", "Hy", "Hx"]
    assert regressor.inSize == 3
    assert regressor.inCross == ["Ex", "Hy"]


def test_local_regressor_setOutput():
    """Test local regressor setOutput"""
    from resistics.regression.local import LocalRegressor

    selector = mock_window_selector()
    regressor = LocalRegressor(selector, "test")
    regressor.setOutput("site2", ["Ex", "Hy", "Hx"])
    assert regressor.outSite == "site2"
    assert regressor.outChannels == ["Ex", "Hy", "Hx"]
    assert regressor.outSize == 3
    assert regressor.outCross == ["Ex", "Hy", "Hx"]
    regressor.setOutput("site1", ["Hx", "Hy"], outCross=["Ex", "Hy"])
    assert regressor.outSite == "site1"
    assert regressor.outChannels == ["Hx", "Hy"]
    assert regressor.outSize == 2
    assert regressor.outCross == ["Ex", "Hy"]


def test_local_regressor_getSmoothLen(monkeypatch):
    """Test local regressor process"""
    from resistics.regression.local import LocalRegressor
    import numpy as np

    selector = mock_window_selector()
    regressor = LocalRegressor(selector, "test")
    regressor.setInput("site1", ["Hx", "Hy"])
    regressor.setOutput("site1", ["Ex", "Ey"])
    regressor.setSmooth("hann", 1)
    assert regressor.smoothLen == 1
    regressor.setSmooth("hann", 12)
    assert regressor.getSmoothLen(65) == 13


def test_local_regressor_ols(monkeypatch):
    """Test local regressor process"""
    from resistics.regression.local import LocalRegressor
    import numpy as np

    # patch out the writeTF function
    monkeypatch.setattr(LocalRegressor, "writeResult", mock_local_regressor_writeResult)

    selector = mock_window_selector()
    regressor = LocalRegressor(selector, "test")
    regressor.setInput("site1", ["Hx", "Hy"], inCross=["Hx", "Hy"])
    regressor.setOutput("site1", ["Ex", "Ey"], outCross=["Ex", "Ey"])
    regressor.setMethod("ols", intercept=False, stack=False)
    regressor.setSmooth("hann", 1)
    regressor.process()
    # expected
    expected_evalfreq = np.array([24, 40])
    expected_impedances = [
        np.array([[-0.125 + 0.0j, 0.75 + 0.0j], [-0.125 + 0.0j, 0.75 + 0.0j]]),
        np.array([[0.4 + 0.0j, 0.6 + 0.0j], [0.4 + 0.0j, 0.6 + 0.0j]]),
    ]
    expected_variances = [
        np.array([[1.25354931e-28, 2.56307567e-29], [1.25354931e-28, 2.56307567e-29]]),
        np.array([[2.42964025e-27, 2.76113237e-28], [2.42964025e-27, 2.76113237e-28]]),
    ]
    np.testing.assert_equal(regressor.evalFreq, expected_evalfreq)
    np.testing.assert_almost_equal(regressor.parameters[0], expected_impedances[0])
    np.testing.assert_almost_equal(regressor.parameters[1], expected_impedances[1])
    np.testing.assert_almost_equal(regressor.variances[0], expected_variances[0])
    np.testing.assert_almost_equal(regressor.variances[1], expected_variances[1])


def test_local_regressor_mm(monkeypatch):
    """Test local regressor process"""
    from resistics.regression.local import LocalRegressor
    import numpy as np

    # patch out the writeTF function
    monkeypatch.setattr(LocalRegressor, "writeResult", mock_local_regressor_writeResult)

    selector = mock_window_selector()
    regressor = LocalRegressor(selector, "test")
    regressor.setInput("site1", ["Hx", "Hy"], inCross=["Hx", "Hy"])
    regressor.setOutput("site1", ["Ex", "Ey"], outCross=["Ex", "Ey"])
    regressor.setMethod("mm", intercept=False, stack=False)
    regressor.setSmooth("hann", 1)
    regressor.process()
    # expected
    expected_evalfreq = np.array([24, 40])
    expected_impedances = [
        np.array([[-0.125 + 0.0j, 0.75 + 0.0j], [-0.125 + 0.0j, 0.75 + 0.0j]]),
        np.array([[0.4 + 0.0j, 0.6 + 0.0j], [0.4 + 0.0j, 0.6 + 0.0j]]),
    ]
    expected_variances = [
        np.array([[1.25354931e-28, 2.56307567e-29], [1.25354931e-28, 2.56307567e-29]]),
        np.array([[2.42964025e-27, 2.76113237e-28], [2.42964025e-27, 2.76113237e-28]]),
    ]
    np.testing.assert_equal(regressor.evalFreq, expected_evalfreq)
    np.testing.assert_almost_equal(regressor.parameters[0], expected_impedances[0])
    np.testing.assert_almost_equal(regressor.parameters[1], expected_impedances[1])
    np.testing.assert_almost_equal(regressor.variances[0], expected_variances[0])
    np.testing.assert_almost_equal(regressor.variances[1], expected_variances[1])


def test_local_regressor_cm(monkeypatch):
    """Test local regressor process"""
    from resistics.regression.local import LocalRegressor
    import numpy as np

    # patch out the writeTF function
    monkeypatch.setattr(LocalRegressor, "writeResult", mock_local_regressor_writeResult)

    selector = mock_window_selector()
    regressor = LocalRegressor(selector, "test")
    regressor.setInput("site1", ["Hx", "Hy"], inCross=["Hx", "Hy"])
    regressor.setOutput("site1", ["Ex", "Ey"], outCross=["Ex", "Ey"])
    regressor.setMethod("cm", intercept=False, stack=False)
    regressor.setSmooth("hann", 1)
    regressor.process()
    # expected
    expected_evalfreq = np.array([24, 40])
    expected_impedances = [
        np.array([[-0.125 + 0.0j, 0.75 + 0.0j], [-0.125 + 0.0j, 0.75 + 0.0j]]),
        np.array([[0.4 + 0.0j, 0.6 + 0.0j], [0.4 + 0.0j, 0.6 + 0.0j]]),
    ]
    expected_variances = [
        np.array([[1.25354931e-28, 2.56307567e-29], [1.25354931e-28, 2.56307567e-29]]),
        np.array([[2.42964025e-27, 2.76113237e-28], [2.42964025e-27, 2.76113237e-28]]),
    ]
    np.testing.assert_equal(regressor.evalFreq, expected_evalfreq)
    np.testing.assert_almost_equal(regressor.parameters[0], expected_impedances[0])
    np.testing.assert_almost_equal(regressor.parameters[1], expected_impedances[1])
    np.testing.assert_almost_equal(regressor.variances[0], expected_variances[0])
    np.testing.assert_almost_equal(regressor.variances[1], expected_variances[1])


def test_local_regressor_ols_intercept(monkeypatch):
    """Test local regressor process"""
    from resistics.regression.local import LocalRegressor
    import numpy as np

    # patch out the writeTF function
    monkeypatch.setattr(LocalRegressor, "writeResult", mock_local_regressor_writeResult)

    selector = mock_window_selector()
    regressor = LocalRegressor(selector, "test")
    regressor.setInput("site1", ["Hx", "Hy"], inCross=["Hx", "Hy"])
    regressor.setOutput("site1", ["Ex", "Ey"], outCross=["Ex", "Ey"])
    regressor.setMethod("ols", intercept=True, stack=False)
    regressor.setSmooth("hann", 1)
    regressor.process()
    # expected
    expected_evalfreq = np.array([24, 40])
    expected_impedances = [
        np.array([[-0.125 + 0.0j, 0.75 + 0.0j], [-0.125 + 0.0j, 0.75 + 0.0j]]),
        np.array(
            [
                [0.22439024 + 0.0j, 0.65853659 + 0.0j],
                [0.22439024 + 0.0j, 0.65853659 + 0.0j],
            ]
        ),
    ]
    expected_variances = [
        np.array([[1.06708149e-28, 1.64277231e-29], [1.06708149e-28, 1.64277231e-29]]),
        np.array([[8.58602488e-15, 9.54002764e-16], [8.58602488e-15, 9.54002764e-16]]),
    ]
    np.testing.assert_equal(regressor.evalFreq, expected_evalfreq)
    np.testing.assert_almost_equal(regressor.parameters[0], expected_impedances[0])
    np.testing.assert_almost_equal(regressor.parameters[1], expected_impedances[1])
    np.testing.assert_almost_equal(regressor.variances[0], expected_variances[0])
    np.testing.assert_almost_equal(regressor.variances[1], expected_variances[1])


def test_local_regressor_mm_intercept(monkeypatch):
    """Test local regressor process"""
    from resistics.regression.local import LocalRegressor
    import numpy as np

    # patch out the writeTF function
    monkeypatch.setattr(LocalRegressor, "writeResult", mock_local_regressor_writeResult)

    selector = mock_window_selector()
    regressor = LocalRegressor(selector, "test")
    regressor.setInput("site1", ["Hx", "Hy"], inCross=["Hx", "Hy"])
    regressor.setOutput("site1", ["Ex", "Ey"], outCross=["Ex", "Ey"])
    regressor.setMethod("mm", intercept=True, stack=False)
    regressor.setSmooth("hann", 1)
    regressor.process()
    # expected
    expected_evalfreq = np.array([24, 40])
    expected_impedances = [
        np.array([[-0.125 + 0.0j, 0.75 + 0.0j], [-0.125 + 0.0j, 0.75 + 0.0j]]),
        np.array([[0.4 + 0.0j, 0.6 + 0.0j], [0.4 + 0.0j, 0.6 + 0.0j],]),
    ]
    expected_variances = [
        np.array([[5.01419722e-28, 1.02523027e-28], [5.01419722e-28, 1.02523027e-28]]),
        np.array([[1.55496976e-25, 1.76712472e-26], [1.55496976e-25, 1.76712472e-26]]),
    ]
    np.testing.assert_equal(regressor.evalFreq, expected_evalfreq)
    np.testing.assert_almost_equal(regressor.parameters[0], expected_impedances[0])
    np.testing.assert_almost_equal(regressor.parameters[1], expected_impedances[1])
    np.testing.assert_almost_equal(regressor.variances[0], expected_variances[0])
    np.testing.assert_almost_equal(regressor.variances[1], expected_variances[1])


def test_local_regressor_cm_intercept(monkeypatch):
    """Test local regressor process"""
    from resistics.regression.local import LocalRegressor
    import numpy as np

    # patch out the writeTF function
    monkeypatch.setattr(LocalRegressor, "writeResult", mock_local_regressor_writeResult)

    selector = mock_window_selector()
    regressor = LocalRegressor(selector, "test")
    regressor.setInput("site1", ["Hx", "Hy"], inCross=["Hx", "Hy"])
    regressor.setOutput("site1", ["Ex", "Ey"], outCross=["Ex", "Ey"])
    regressor.setMethod("cm", intercept=True, stack=False)
    regressor.setSmooth("hann", 1)
    regressor.process()
    # expected
    expected_evalfreq = np.array([24, 40])
    expected_impedances = [
        np.array([[-0.125 + 0.0j, 0.75 + 0.0j], [-0.125 + 0.0j, 0.75 + 0.0j]]),
        np.array([[0.4 + 0.0j, 0.6 + 0.0j], [0.4 + 0.0j, 0.6 + 0.0j]]),
    ]
    expected_variances = [
        np.array([[5.01419722e-28, 1.02523027e-28], [5.01419722e-28, 1.02523027e-28]]),
        np.array([[1.55496976e-25, 1.76712472e-26], [1.55496976e-25, 1.76712472e-26]]),
    ]
    np.testing.assert_equal(regressor.evalFreq, expected_evalfreq)
    np.testing.assert_almost_equal(regressor.parameters[0], expected_impedances[0])
    np.testing.assert_almost_equal(regressor.parameters[1], expected_impedances[1])
    np.testing.assert_almost_equal(regressor.variances[0], expected_variances[0])
    np.testing.assert_almost_equal(regressor.variances[1], expected_variances[1])
