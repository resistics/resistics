"""Test single site and intersite regression with no remote reference data"""


def mock_local_regressor_writeResult(a, b, c, d, e, f):
    return


def test_local_regressor_setCores():
    """Test local regressor setCores"""
    from resistics.regression.local import LocalRegressor
    from mocks import mock_window_selector

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
    from mocks import mock_window_selector

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
    from mocks import mock_window_selector

    selector = mock_window_selector()
    regressor = LocalRegressor(selector, "test")
    assert regressor.method == "cm"
    assert regressor.intercept == False
    regressor.setMethod("ols", intercept=False)
    assert regressor.method == "ols"
    assert regressor.intercept == False
    regressor.setMethod("cm", intercept=True)
    assert regressor.method == "cm"
    assert regressor.intercept == True
    regressor.setMethod("mm", intercept=False)
    assert regressor.method == "mm"
    assert regressor.intercept == False


def test_local_regressor_setInput():
    """Test local regressor setInput"""
    from resistics.regression.local import LocalRegressor
    from mocks import mock_window_selector

    selector = mock_window_selector()
    regressor = LocalRegressor(selector, "test")
    regressor.setInput("local", ["Hx", "Hy"])
    assert regressor.inSite == "local"
    assert regressor.inChannels == ["Hx", "Hy"]
    assert regressor.inSize == 2
    assert regressor.inCross == ["Hx", "Hy"]
    regressor.setInput("inter", ["Ex", "Hy", "Hx"], inCross=["Ex", "Hy"])
    assert regressor.inSite == "inter"
    assert regressor.inChannels == ["Ex", "Hy", "Hx"]
    assert regressor.inSize == 3
    assert regressor.inCross == ["Ex", "Hy"]


def test_local_regressor_setOutput():
    """Test local regressor setOutput"""
    from resistics.regression.local import LocalRegressor
    from mocks import mock_window_selector

    selector = mock_window_selector()
    regressor = LocalRegressor(selector, "test")
    regressor.setOutput("inter", ["Ex", "Hy", "Hx"])
    assert regressor.outSite == "inter"
    assert regressor.outChannels == ["Ex", "Hy", "Hx"]
    assert regressor.outSize == 3
    assert regressor.outCross == ["Ex", "Hy", "Hx"]
    regressor.setOutput("local", ["Hx", "Hy"], outCross=["Ex", "Hy"])
    assert regressor.outSite == "local"
    assert regressor.outChannels == ["Hx", "Hy"]
    assert regressor.outSize == 2
    assert regressor.outCross == ["Ex", "Hy"]


def test_local_regressor_getSmoothLen(monkeypatch):
    """Test local regressor process"""
    from resistics.regression.local import LocalRegressor
    from mocks import mock_window_selector
    import numpy as np

    selector = mock_window_selector()
    from mocks import mock_window_selector

    regressor = LocalRegressor(selector, "test")
    regressor.setInput("local", ["Hx", "Hy"])
    regressor.setOutput("local", ["Ex", "Ey"])
    regressor.setSmooth("hann", 1)
    assert regressor.smoothLen == 1
    regressor.setSmooth("hann", 12)
    assert regressor.getSmoothLen(65) == 13


def test_local_regressor_ols(monkeypatch):
    """Test local regressor process using standard parameters"""
    from resistics.regression.local import LocalRegressor
    from mocks import mock_window_selector
    import numpy as np

    # patch out the writeTF function
    monkeypatch.setattr(LocalRegressor, "writeResult", mock_local_regressor_writeResult)

    selector = mock_window_selector()
    regressor = LocalRegressor(selector, "test")
    regressor.setInput("local", ["Hx", "Hy"])
    regressor.setOutput("local", ["Ex", "Ey"])
    regressor.setMethod("ols", intercept=False)
    regressor.setSmooth("hann", 1)
    assert regressor.inCross == ["Hx", "Hy"]
    assert regressor.outCross == ["Ex", "Ey"]    
    regressor.process()
    # expected
    expected_evalfreq = np.array([24, 40])
    expected_impedances = [
        np.array([[3 + 0.0j, 5 + 0.0j], [2 + 0.0j, 7 + 0.0j]]),
        np.array([[3 + 0.0j, 5 + 0.0j], [2 + 0.0j, 7 + 0.0j]]),
    ]
    expected_variances = [
        np.array([[2.08722609e-28, 1.34197332e-28], [2.31914010e-29, 1.49108146e-29]]),
        np.array([[4.59299270e-29, 6.12727229e-29], [1.34977745e-28, 1.80066778e-28]]),
    ]
    np.testing.assert_equal(regressor.evalFreq, expected_evalfreq)
    np.testing.assert_almost_equal(regressor.parameters[0], expected_impedances[0])
    np.testing.assert_almost_equal(regressor.parameters[1], expected_impedances[1])
    np.testing.assert_almost_equal(regressor.variances[0], expected_variances[0])
    np.testing.assert_almost_equal(regressor.variances[1], expected_variances[1])


def test_local_regressor_mm(monkeypatch):
    """Test local regressor process using standard parameters"""
    from resistics.regression.local import LocalRegressor
    from mocks import mock_window_selector
    import numpy as np

    # patch out the writeTF function
    monkeypatch.setattr(LocalRegressor, "writeResult", mock_local_regressor_writeResult)

    selector = mock_window_selector()
    regressor = LocalRegressor(selector, "test")
    regressor.setInput("local", ["Hx", "Hy"])
    regressor.setOutput("local", ["Ex", "Ey"])
    regressor.setMethod("mm", intercept=False)
    regressor.setSmooth("hann", 1)
    assert regressor.inCross == ["Hx", "Hy"]
    assert regressor.outCross == ["Ex", "Ey"]    
    regressor.process()
    # expected
    expected_evalfreq = np.array([24, 40])
    expected_impedances = [
        np.array([[3 + 0.0j, 5 + 0.0j], [2 + 0.0j, 7 + 0.0j]]),
        np.array([[3 + 0.0j, 5 + 0.0j], [2 + 0.0j, 7 + 0.0j]]),
    ]
    expected_variances = [
        np.array([[2.08722609e-28, 1.34197332e-28], [2.31914010e-29, 1.49108146e-29]]),
        np.array([[4.59299270e-29, 6.12727229e-29], [1.34977745e-28, 1.80066778e-28]]),
    ]
    np.testing.assert_equal(regressor.evalFreq, expected_evalfreq)
    np.testing.assert_almost_equal(regressor.parameters[0], expected_impedances[0])
    np.testing.assert_almost_equal(regressor.parameters[1], expected_impedances[1])
    np.testing.assert_almost_equal(regressor.variances[0], expected_variances[0])
    np.testing.assert_almost_equal(regressor.variances[1], expected_variances[1])


def test_local_regressor_cm(monkeypatch):
    """Test local regressor process using standard parameters"""
    from resistics.regression.local import LocalRegressor
    from mocks import mock_window_selector
    import numpy as np

    # patch out the writeTF function
    monkeypatch.setattr(LocalRegressor, "writeResult", mock_local_regressor_writeResult)

    selector = mock_window_selector()
    regressor = LocalRegressor(selector, "test")
    regressor.setInput("local", ["Hx", "Hy"])
    regressor.setOutput("local", ["Ex", "Ey"])
    regressor.setMethod("cm", intercept=False)
    regressor.setSmooth("hann", 1)
    assert regressor.inCross == ["Hx", "Hy"]
    assert regressor.outCross == ["Ex", "Ey"]
    regressor.process()
    # expected
    expected_evalfreq = np.array([24, 40])
    expected_impedances = [
        np.array([[3 + 0.0j, 5 + 0.0j], [2 + 0.0j, 7 + 0.0j]]),
        np.array([[3 + 0.0j, 5 + 0.0j], [2 + 0.0j, 7 + 0.0j]]),
    ]
    expected_variances = [
        np.array([[2.08722609e-28, 1.34197332e-28], [2.31914010e-29, 1.49108146e-29]]),
        np.array([[4.59299270e-29, 6.12727229e-29], [1.34977745e-28, 1.80066778e-28]]),
    ]
    np.testing.assert_equal(regressor.evalFreq, expected_evalfreq)
    np.testing.assert_almost_equal(regressor.parameters[0], expected_impedances[0])
    np.testing.assert_almost_equal(regressor.parameters[1], expected_impedances[1])
    np.testing.assert_almost_equal(regressor.variances[0], expected_variances[0])
    np.testing.assert_almost_equal(regressor.variances[1], expected_variances[1])


def test_local_regressor_ols_noise(monkeypatch):
    """Test local regressor process with addition of noise"""
    from resistics.regression.local import LocalRegressor
    from mocks import mock_window_selector
    import numpy as np

    # patch out the writeTF function
    monkeypatch.setattr(LocalRegressor, "writeResult", mock_local_regressor_writeResult)

    selector = mock_window_selector(localnoise=True)
    regressor = LocalRegressor(selector, "test")
    regressor.setInput("local", ["Hx", "Hy"], inCross=["Hx", "Hy"])
    regressor.setOutput("local", ["Ex", "Ey"], outCross=["Ex", "Ey"])
    regressor.setMethod("ols", intercept=False)
    regressor.setSmooth("hann", 1)
    regressor.process()
    # expected
    expected_evalfreq = np.array([24, 40])
    expected_impedances = [
        np.array(
            [
                [4.28572653 + 0.0j, 3.39668529 + 0.0j],
                [0.95079059 + 0.0j, 7.70464292 + 0.0j],
            ]
        ),
        np.array(
            [
                [4.16918831 + 0.0j, 3.4135369 + 0.0j],
                [-3.12128344 + 0.0j, 12.22876995 + 0.0j],
            ]
        ),
    ]
    expected_variances = [
        np.array([[0.00041475, 0.00026418], [0.1143862, 0.07286062]]),
        np.array([[0.00432538, 0.00572858], [2.26643718, 3.00169024]]),
    ]
    np.testing.assert_equal(regressor.evalFreq, expected_evalfreq)
    np.testing.assert_almost_equal(regressor.parameters[0], expected_impedances[0])
    np.testing.assert_almost_equal(regressor.parameters[1], expected_impedances[1])
    np.testing.assert_almost_equal(regressor.variances[0], expected_variances[0])
    np.testing.assert_almost_equal(regressor.variances[1], expected_variances[1])


def test_local_regressor_mm_noise(monkeypatch):
    """Test local regressor process with addition of noise"""
    from resistics.regression.local import LocalRegressor
    from mocks import mock_window_selector
    import numpy as np

    # patch out the writeTF function
    monkeypatch.setattr(LocalRegressor, "writeResult", mock_local_regressor_writeResult)

    selector = mock_window_selector(localnoise=True)
    regressor = LocalRegressor(selector, "test")
    regressor.setInput("local", ["Hx", "Hy"], inCross=["Hx", "Hy"])
    regressor.setOutput("local", ["Ex", "Ey"], outCross=["Ex", "Ey"])
    regressor.setMethod("mm", intercept=False)
    regressor.setSmooth("hann", 1)
    regressor.process()
    # expected
    expected_evalfreq = np.array([24, 40])
    expected_impedances = [
        np.array(
            [
                [4.30696333 + 0.0j, 3.38002297 + 0.0j],
                [2.5217861 + 0.0j, 6.44672487 + 0.0j],
            ]
        ),
        np.array(
            [
                [4.23654503 + 0.0j, 3.33331255 + 0.0j],
                [-8.19366818 + 0.0j, 18.06316472 + 0.0j],
            ]
        ),
    ]
    expected_variances = [
        np.array([[1.72623564e-04, 1.09956096e-04], [1.55939006e-01, 9.93285271e-02]]),
        np.array([[0.00691985, 0.00916471], [0.03841539, 0.0508777]]),
    ]
    np.testing.assert_equal(regressor.evalFreq, expected_evalfreq)
    np.testing.assert_almost_equal(regressor.parameters[0], expected_impedances[0])
    np.testing.assert_almost_equal(regressor.parameters[1], expected_impedances[1])
    np.testing.assert_almost_equal(regressor.variances[0], expected_variances[0])
    np.testing.assert_almost_equal(regressor.variances[1], expected_variances[1])


def test_local_regressor_cm_noise(monkeypatch):
    """Test local regressor process with addition of noise"""
    from resistics.regression.local import LocalRegressor
    from mocks import mock_window_selector
    import numpy as np

    # patch out the writeTF function
    monkeypatch.setattr(LocalRegressor, "writeResult", mock_local_regressor_writeResult)

    selector = mock_window_selector(localnoise=True)
    regressor = LocalRegressor(selector, "test")
    regressor.setInput("local", ["Hx", "Hy"], inCross=["Hx", "Hy"])
    regressor.setOutput("local", ["Ex", "Ey"], outCross=["Ex", "Ey"])
    regressor.setMethod("cm", intercept=False)
    regressor.setSmooth("hann", 1)
    regressor.process()
    # expected
    expected_evalfreq = np.array([24, 40])
    expected_impedances = [
        np.array(
            [
                [4.0781224 + 0.0j, 3.567369 + 0.0j],
                [1.85376383 + 0.0j, 6.98454182 + 0.0j],
            ]
        ),
        np.array(
            [
                [3.79687613 + 0.0j, 3.83744749 + 0.0j],
                [4.64021435 + 0.0j, 3.41744201 + 0.0j],
            ]
        ),
    ]
    expected_variances = [
        np.array([[0.04024341, 0.02563386], [0.0075834, 0.0048304]]),
        np.array([[0.00760705, 0.01007484], [2.48409977, 3.28996458]]),
    ]
    np.testing.assert_equal(regressor.evalFreq, expected_evalfreq)
    np.testing.assert_almost_equal(regressor.parameters[0], expected_impedances[0])
    np.testing.assert_almost_equal(regressor.parameters[1], expected_impedances[1])
    np.testing.assert_almost_equal(regressor.variances[0], expected_variances[0])
    np.testing.assert_almost_equal(regressor.variances[1], expected_variances[1])


def test_local_regressor_ols_intercept(monkeypatch):
    """Test local regressor process with addition of intercept term"""
    from resistics.regression.local import LocalRegressor
    from mocks import mock_window_selector
    import numpy as np

    # patch out the writeTF function
    monkeypatch.setattr(LocalRegressor, "writeResult", mock_local_regressor_writeResult)

    selector = mock_window_selector(intercept=True)
    regressor = LocalRegressor(selector, "test")
    regressor.setInput("local", ["Hx", "Hy"], inCross=["Hx", "Hy"])
    regressor.setOutput("local", ["Ex", "Ey"], outCross=["Ex", "Ey"])
    regressor.setMethod("ols", intercept=True)
    regressor.setSmooth("hann", 1)
    regressor.process()
    # expected
    expected_evalfreq = np.array([24, 40])
    expected_impedances = [
        np.array(
            [
                [3.0 + 0.0j, 5.0 + 0.0j, 5.0 + 0.0j],
                [2.0 + 0.0j, 7.0 + 0.0j, -9.0 + 0.0j],
            ]
        ),
        np.array(
            [
                [3.0 + 0.0j, 5.0 + 0.0j, 5.0 + 0.0j],
                [2.0 + 0.0j, 7.0 + 0.0j, -9.0 + 0.0j],
            ]
        ),
    ]
    expected_variances = [
        np.array(
            [
                [1.21088688e-23, 7.25740911e-24, 3.34802360e-23],
                [3.95687739e-23, 2.37154094e-23, 1.09405091e-22],
            ]
        ),
        np.array(
            [
                [2.48945809e-27, 3.32569517e-27, 2.95106525e-26],
                [1.00812766e-27, 1.34676912e-27, 1.19505948e-26],
            ]
        ),
    ]
    np.testing.assert_equal(regressor.evalFreq, expected_evalfreq)
    np.testing.assert_almost_equal(regressor.parameters[0], expected_impedances[0])
    np.testing.assert_almost_equal(regressor.parameters[1], expected_impedances[1])
    np.testing.assert_almost_equal(regressor.variances[0], expected_variances[0])
    np.testing.assert_almost_equal(regressor.variances[1], expected_variances[1])


def test_local_regressor_mm_intercept(monkeypatch):
    """Test local regressor process with addition of intercept term"""
    from resistics.regression.local import LocalRegressor
    from mocks import mock_window_selector
    import numpy as np

    # patch out the writeTF function
    monkeypatch.setattr(LocalRegressor, "writeResult", mock_local_regressor_writeResult)

    selector = mock_window_selector(intercept=True)
    regressor = LocalRegressor(selector, "test")
    regressor.setInput("local", ["Hx", "Hy"], inCross=["Hx", "Hy"])
    regressor.setOutput("local", ["Ex", "Ey"], outCross=["Ex", "Ey"])
    regressor.setMethod("mm", intercept=True)
    regressor.setSmooth("hann", 1)
    regressor.process()
    # expected
    expected_evalfreq = np.array([24, 40])
    expected_impedances = [
        np.array(
            [
                [3.0 + 0.0j, 5.0 + 0.0j, 5.0 + 0.0j],
                [2.0 + 0.0j, 7.0 + 0.0j, -9.0 + 0.0j],
            ]
        ),
        np.array(
            [
                [3.0 + 0.0j, 5.0 + 0.0j, 5.0 + 0.0j],
                [2.0 + 0.0j, 7.0 + 0.0j, -9.0 + 0.0j],
            ]
        ),
    ]
    expected_variances = [
        np.array(
            [
                [1.21088688e-23, 7.25740911e-24, 3.34802360e-23],
                [3.95687739e-23, 2.37154094e-23, 1.09405091e-22],
            ]
        ),
        np.array(
            [
                [2.48945809e-27, 3.32569517e-27, 2.95106525e-26],
                [1.00812766e-27, 1.34676912e-27, 1.19505948e-26],
            ]
        ),
    ]
    print(regressor.parameters)
    print(regressor.variances)
    np.testing.assert_equal(regressor.evalFreq, expected_evalfreq)
    np.testing.assert_almost_equal(regressor.parameters[0], expected_impedances[0])
    np.testing.assert_almost_equal(regressor.parameters[1], expected_impedances[1])
    np.testing.assert_almost_equal(regressor.variances[0], expected_variances[0])
    np.testing.assert_almost_equal(regressor.variances[1], expected_variances[1])


def test_local_regressor_cm_intercept(monkeypatch):
    """Test local regressor process with addition of intercept term"""
    from resistics.regression.local import LocalRegressor
    from mocks import mock_window_selector
    import numpy as np

    # patch out the writeTF function
    monkeypatch.setattr(LocalRegressor, "writeResult", mock_local_regressor_writeResult)

    selector = mock_window_selector(intercept=True)
    regressor = LocalRegressor(selector, "test")
    regressor.setInput("local", ["Hx", "Hy"], inCross=["Hx", "Hy"])
    regressor.setOutput("local", ["Ex", "Ey"], outCross=["Ex", "Ey"])
    regressor.setMethod("cm", intercept=True)
    regressor.setSmooth("hann", 1)
    regressor.process()
    # expected
    expected_evalfreq = np.array([24, 40])
    expected_impedances = [
        np.array(
            [
                [3.0 + 0.0j, 5.0 + 0.0j, 5.0 + 0.0j],
                [2.0 + 0.0j, 7.0 + 0.0j, -9.0 + 0.0j],
            ]
        ),
        np.array(
            [
                [3.0 + 0.0j, 5.0 + 0.0j, 5.0 + 0.0j],
                [2.0 + 0.0j, 7.0 + 0.0j, -9.0 + 0.0j],
            ]
        ),
    ]
    expected_variances = [
        np.array(
            [
                [1.21088688e-23, 7.25740911e-24, 3.34802360e-23],
                [3.95687739e-23, 2.37154094e-23, 1.09405091e-22],
            ]
        ),
        np.array(
            [
                [2.48945809e-27, 3.32569517e-27, 2.95106525e-26],
                [1.00812766e-27, 1.34676912e-27, 1.19505948e-26],
            ]
        ),
    ]
    np.testing.assert_equal(regressor.evalFreq, expected_evalfreq)
    np.testing.assert_almost_equal(regressor.parameters[0], expected_impedances[0])
    np.testing.assert_almost_equal(regressor.parameters[1], expected_impedances[1])
    np.testing.assert_almost_equal(regressor.variances[0], expected_variances[0])
    np.testing.assert_almost_equal(regressor.variances[1], expected_variances[1])


def test_local_regressor_ols_tipper(monkeypatch):
    """Test local regressor process for different combination of inputs and ouputs (tipper)"""
    from resistics.regression.local import LocalRegressor
    from mocks import mock_window_selector
    import numpy as np

    # patch out the writeTF function
    monkeypatch.setattr(LocalRegressor, "writeResult", mock_local_regressor_writeResult)

    selector = mock_window_selector()
    regressor = LocalRegressor(selector, "test")
    regressor.setInput("local", ["Hx", "Hy"], inCross=["Hx", "Hy"])
    regressor.setOutput("local", ["Hz"], outCross=None)
    regressor.setMethod("ols", intercept=False)
    regressor.setSmooth("hann", 1)
    regressor.process()
    # expected
    expected_evalfreq = np.array([24, 40])
    expected_impedances = [
        np.array([[2.0 + 0.0j, 8.0 + 0.0j]]),
        np.array([[2.0 + 0.0j, 8.0 + 0.0j]]),
    ]
    expected_variances = [
        np.array([[3.41232327e-27, 2.19175834e-27]]),
        np.array([[5.18941937e-29, 6.91709078e-29]]),
    ]
    np.testing.assert_equal(regressor.evalFreq, expected_evalfreq)
    np.testing.assert_almost_equal(regressor.parameters[0], expected_impedances[0])
    np.testing.assert_almost_equal(regressor.parameters[1], expected_impedances[1])
    np.testing.assert_almost_equal(regressor.variances[0], expected_variances[0])
    np.testing.assert_almost_equal(regressor.variances[1], expected_variances[1])


def test_local_regressor_mm_tipper(monkeypatch):
    """Test local regressor process for different combination of inputs and ouputs (tipper)"""
    from resistics.regression.local import LocalRegressor
    from mocks import mock_window_selector
    import numpy as np

    # patch out the writeTF function
    monkeypatch.setattr(LocalRegressor, "writeResult", mock_local_regressor_writeResult)

    selector = mock_window_selector()
    regressor = LocalRegressor(selector, "test")
    regressor.setInput("local", ["Hx", "Hy"], inCross=["Hx", "Hy"])
    regressor.setOutput("local", ["Hz"], outCross=None)
    regressor.setMethod("mm", intercept=False)
    regressor.setSmooth("hann", 1)
    regressor.process()
    # expected
    expected_evalfreq = np.array([24, 40])
    expected_impedances = [
        np.array([[2.0 + 0.0j, 8.0 + 0.0j]]),
        np.array([[2.0 + 0.0j, 8.0 + 0.0j]]),
    ]
    expected_variances = [
        np.array([[3.41232327e-27, 2.19175834e-27]]),
        np.array([[5.18941937e-29, 6.91709078e-29]]),
    ]
    np.testing.assert_equal(regressor.evalFreq, expected_evalfreq)
    np.testing.assert_almost_equal(regressor.parameters[0], expected_impedances[0])
    np.testing.assert_almost_equal(regressor.parameters[1], expected_impedances[1])
    np.testing.assert_almost_equal(regressor.variances[0], expected_variances[0])
    np.testing.assert_almost_equal(regressor.variances[1], expected_variances[1])


def test_local_regressor_cm_tipper(monkeypatch):
    """Test local regressor process for different combination of inputs and ouputs (tipper)"""
    from resistics.regression.local import LocalRegressor
    from mocks import mock_window_selector
    import numpy as np

    # patch out the writeTF function
    monkeypatch.setattr(LocalRegressor, "writeResult", mock_local_regressor_writeResult)

    selector = mock_window_selector()
    regressor = LocalRegressor(selector, "test")
    regressor.setInput("local", ["Hx", "Hy"], inCross=["Hx", "Hy"])
    regressor.setOutput("local", ["Hz"], outCross=None)
    regressor.setMethod("cm", intercept=False)
    regressor.setSmooth("hann", 1)
    regressor.process()
    # expected
    expected_evalfreq = np.array([24, 40])
    expected_impedances = [
        np.array([[2.0 + 0.0j, 8.0 + 0.0j]]),
        np.array([[2.0 + 0.0j, 8.0 + 0.0j]]),
    ]
    expected_variances = [
        np.array([[3.41232327e-27, 2.19175834e-27]]),
        np.array([[5.18941937e-29, 6.91709078e-29]]),
    ]
    np.testing.assert_equal(regressor.evalFreq, expected_evalfreq)
    np.testing.assert_almost_equal(regressor.parameters[0], expected_impedances[0])
    np.testing.assert_almost_equal(regressor.parameters[1], expected_impedances[1])
    np.testing.assert_almost_equal(regressor.variances[0], expected_variances[0])
    np.testing.assert_almost_equal(regressor.variances[1], expected_variances[1])


def test_local_regressor_ols_intersite(monkeypatch):
    """Test local regressor process for intersite processing"""
    from resistics.regression.local import LocalRegressor
    from mocks import mock_window_selector
    import numpy as np

    # patch out the writeTF function
    monkeypatch.setattr(LocalRegressor, "writeResult", mock_local_regressor_writeResult)

    selector = mock_window_selector()
    regressor = LocalRegressor(selector, "test")
    regressor.setInput("local", ["Hx", "Hy"], inCross=["Hx", "Hy"])
    regressor.setOutput("inter", ["Ex", "Ey"], outCross=["Ex", "Ey"])
    regressor.setMethod("ols", intercept=False)
    regressor.setSmooth("hann", 1)
    regressor.process()
    # expected
    expected_evalfreq = np.array([24, 40])
    expected_impedances = [
        np.array([[6.0 + 0.0j, 8.0 + 0.0j], [4.0 + 0.0j, 1.0 + 0.0j]]),
        np.array([[6.0 + 0.0j, 8.0 + 0.0j], [4.0 + 0.0j, 1.0 + 0.0j]]),
    ]
    expected_variances = [
        np.array([[9.14649740e-29, 5.89343794e-29], [1.42914022e-30, 9.20849678e-31]]),
        np.array([[1.38145310e-28, 1.84606779e-28], [3.45363276e-29, 4.61516947e-29]]),
    ]
    np.testing.assert_equal(regressor.evalFreq, expected_evalfreq)
    np.testing.assert_almost_equal(regressor.parameters[0], expected_impedances[0])
    np.testing.assert_almost_equal(regressor.parameters[1], expected_impedances[1])
    np.testing.assert_almost_equal(regressor.variances[0], expected_variances[0])
    np.testing.assert_almost_equal(regressor.variances[1], expected_variances[1])


def test_local_regressor_mm_intersite(monkeypatch):
    """Test local regressor process for intersite processing"""
    from resistics.regression.local import LocalRegressor
    from mocks import mock_window_selector
    import numpy as np

    # patch out the writeTF function
    monkeypatch.setattr(LocalRegressor, "writeResult", mock_local_regressor_writeResult)

    selector = mock_window_selector()
    regressor = LocalRegressor(selector, "test")
    regressor.setInput("local", ["Hx", "Hy"], inCross=["Hx", "Hy"])
    regressor.setOutput("inter", ["Ex", "Ey"], outCross=["Ex", "Ey"])
    regressor.setMethod("mm", intercept=False)
    regressor.setSmooth("hann", 1)
    regressor.process()
    # expected
    expected_evalfreq = np.array([24, 40])
    expected_impedances = [
        np.array([[6.0 + 0.0j, 8.0 + 0.0j], [4.0 + 0.0j, 1.0 + 0.0j]]),
        np.array([[6.0 + 0.0j, 8.0 + 0.0j], [4.0 + 0.0j, 1.0 + 0.0j]]),
    ]
    expected_variances = [
        np.array([[9.14649740e-29, 5.89343794e-29], [1.42914022e-30, 9.20849678e-31]]),
        np.array([[1.38145310e-28, 1.84606779e-28], [3.45363276e-29, 4.61516947e-29]]),
    ]
    np.testing.assert_equal(regressor.evalFreq, expected_evalfreq)
    np.testing.assert_almost_equal(regressor.parameters[0], expected_impedances[0])
    np.testing.assert_almost_equal(regressor.parameters[1], expected_impedances[1])
    np.testing.assert_almost_equal(regressor.variances[0], expected_variances[0])
    np.testing.assert_almost_equal(regressor.variances[1], expected_variances[1])


def test_local_regressor_cm_intersite(monkeypatch):
    """Test local regressor process for intersite processing"""
    from resistics.regression.local import LocalRegressor
    from mocks import mock_window_selector
    import numpy as np

    # patch out the writeTF function
    monkeypatch.setattr(LocalRegressor, "writeResult", mock_local_regressor_writeResult)

    selector = mock_window_selector()
    regressor = LocalRegressor(selector, "test")
    regressor.setInput("local", ["Hx", "Hy"], inCross=["Hx", "Hy"])
    regressor.setOutput("inter", ["Ex", "Ey"], outCross=["Ex", "Ey"])
    regressor.setMethod("cm", intercept=False)
    regressor.setSmooth("hann", 1)
    regressor.process()
    # expected
    expected_evalfreq = np.array([24, 40])
    expected_impedances = [
        np.array([[6.0 + 0.0j, 8.0 + 0.0j], [4.0 + 0.0j, 1.0 + 0.0j]]),
        np.array([[6.0 + 0.0j, 8.0 + 0.0j], [4.0 + 0.0j, 1.0 + 0.0j]]),
    ]
    expected_variances = [
        np.array([[9.14649740e-29, 5.89343794e-29], [1.42914022e-30, 9.20849678e-31]]),
        np.array([[1.38145310e-28, 1.84606779e-28], [3.45363276e-29, 4.61516947e-29]]),
    ]
    np.testing.assert_equal(regressor.evalFreq, expected_evalfreq)
    np.testing.assert_almost_equal(regressor.parameters[0], expected_impedances[0])
    np.testing.assert_almost_equal(regressor.parameters[1], expected_impedances[1])
    np.testing.assert_almost_equal(regressor.variances[0], expected_variances[0])
    np.testing.assert_almost_equal(regressor.variances[1], expected_variances[1])
