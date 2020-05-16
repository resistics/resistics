"""Test remote reference regression"""


def mock_remote_regressor_writeResult(a, b, c, d, e, f):
    return


def test_remote_regressor_setCores():
    """Test remote regressor setCores"""
    from resistics.regression.remote import RemoteRegressor
    from mocks import mock_window_selector

    selector = mock_window_selector()
    regressor = RemoteRegressor(selector, "test")
    assert regressor.ncores == 0
    regressor.setCores(4)
    assert regressor.ncores == 4
    regressor.setCores(11)
    assert regressor.ncores == 11


def test_remote_regressor_setSmooth():
    """Test remote regressor setSmooth"""
    from resistics.regression.remote import RemoteRegressor
    from mocks import mock_window_selector

    selector = mock_window_selector()
    regressor = RemoteRegressor(selector, "test")
    assert regressor.smoothFunc == "hann"
    assert regressor.smoothLen is None
    regressor.setSmooth("hann", 7)
    assert regressor.smoothFunc == "hann"
    assert regressor.smoothLen == 7
    regressor.setSmooth("parzen", 8)
    assert regressor.smoothFunc == "parzen"
    assert regressor.smoothLen == 8
    assert regressor.getSmoothLen(65) == 9


def test_remote_regressor_setMethod():
    """Test remote regressor setMethod"""
    from resistics.regression.remote import RemoteRegressor
    from mocks import mock_window_selector

    selector = mock_window_selector()
    regressor = RemoteRegressor(selector, "test")
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


def test_remote_regressor_setInput():
    """Test remote regressor setInput"""
    from resistics.regression.remote import RemoteRegressor
    from mocks import mock_window_selector

    selector = mock_window_selector()
    regressor = RemoteRegressor(selector, "test")
    regressor.setInput("site1", ["Hx", "Hy"])
    assert regressor.inSite == "site1"
    assert regressor.inChannels == ["Hx", "Hy"]
    assert regressor.inSize == 2
    assert regressor.inCross == []
    regressor.setInput("site2", ["Ex", "Hy", "Hx"], inCross=["Ex", "Hy"])
    assert regressor.inSite == "site2"
    assert regressor.inChannels == ["Ex", "Hy", "Hx"]
    assert regressor.inSize == 3
    assert regressor.inCross == ["Ex", "Hy"]


def test_remote_regressor_setOutput():
    """Test remote regressor setOutput"""
    from resistics.regression.remote import RemoteRegressor
    from mocks import mock_window_selector

    selector = mock_window_selector()
    regressor = RemoteRegressor(selector, "test")
    regressor.setOutput("site2", ["Ex", "Hy", "Hx"])
    assert regressor.outSite == "site2"
    assert regressor.outChannels == ["Ex", "Hy", "Hx"]
    assert regressor.outSize == 3
    assert regressor.outCross == []
    regressor.setOutput("site1", ["Hx", "Hy"], outCross=["Ex", "Hy"])
    assert regressor.outSite == "site1"
    assert regressor.outChannels == ["Hx", "Hy"]
    assert regressor.outSize == 2
    assert regressor.outCross == ["Ex", "Hy"]


def test_remote_regressor_setRemote():
    """Test remote regressor setRemote"""
    from resistics.regression.remote import RemoteRegressor
    from mocks import mock_window_selector

    selector = mock_window_selector()
    regressor = RemoteRegressor(selector, "test")
    assert regressor.remoteCross == ["Hx", "Hy"]
    regressor.setRemote("remote1", ["Ex", "Hy", "Hx"])
    assert regressor.remoteSite == "remote1"
    assert regressor.remoteCross == ["Ex", "Hy", "Hx"]
    assert regressor.remoteSize == 3
    regressor.setRemote("remote2", ["Hx"])
    assert regressor.remoteSite == "remote2"
    assert regressor.remoteCross == ["Hx"]
    assert regressor.remoteSize == 1


def test_remote_regressor_getSmoothLen(monkeypatch):
    """Test remote regressor process"""
    from resistics.regression.remote import RemoteRegressor
    from mocks import mock_window_selector
    import numpy as np

    selector = mock_window_selector()
    regressor = RemoteRegressor(selector, "test")
    regressor.setInput("site1", ["Hx", "Hy"])
    regressor.setOutput("site1", ["Ex", "Ey"])
    regressor.setRemote("remote1", ["Hx", "Hy"])
    regressor.setSmooth("hann", 1)
    assert regressor.smoothLen == 1
    regressor.setSmooth("hann", 12)
    assert regressor.getSmoothLen(65) == 13


def test_remote_regressor_ols(monkeypatch):
    """Test remote regressor process"""
    from resistics.regression.remote import RemoteRegressor
    from mocks import mock_window_selector
    import numpy as np

    # patch out the writeTF function
    monkeypatch.setattr(
        RemoteRegressor, "writeResult", mock_remote_regressor_writeResult
    )

    selector = mock_window_selector()
    regressor = RemoteRegressor(selector, "test")
    regressor.setInput("local", ["Hx", "Hy"])
    regressor.setOutput("local", ["Ex", "Ey"])
    regressor.setRemote("remote", ["Hx", "Hy"])
    regressor.setMethod("ols", intercept=False)
    regressor.setSmooth("hann", 1)
    assert regressor.inCross == []
    assert regressor.outCross == []
    regressor.process()
    # expected
    expected_evalfreq = np.array([24, 40])
    expected_impedances = [
        np.array([[3.0 + 0.0j, 5.0 + 0.0j], [2.0 + 0.0j, 7.0 + 0.0j]]),
        np.array([[3.0 + 0.0j, 5.0 + 0.0j], [2.0 + 0.0j, 7.0 + 0.0j]]),
    ]
    expected_variances = [
        np.array([[2.78469341e-28, 1.79044909e-28], [1.64240080e-27, 1.05599956e-27]]),
        np.array([[1.47463485e-29, 1.96730235e-29], [5.89853939e-29, 7.86920938e-29]]),
    ]
    np.testing.assert_equal(regressor.evalFreq, expected_evalfreq)
    np.testing.assert_almost_equal(regressor.parameters[0], expected_impedances[0])
    np.testing.assert_almost_equal(regressor.parameters[1], expected_impedances[1])
    np.testing.assert_almost_equal(regressor.variances[0], expected_variances[0])
    np.testing.assert_almost_equal(regressor.variances[1], expected_variances[1])


def test_remote_regressor_mm(monkeypatch):
    """Test remote regressor process"""
    from resistics.regression.remote import RemoteRegressor
    from mocks import mock_window_selector
    import numpy as np

    # patch out the writeTF function
    monkeypatch.setattr(
        RemoteRegressor, "writeResult", mock_remote_regressor_writeResult
    )

    selector = mock_window_selector()
    regressor = RemoteRegressor(selector, "test")
    regressor.setInput("local", ["Hx", "Hy"])
    regressor.setOutput("local", ["Ex", "Ey"])
    regressor.setRemote("remote", ["Hx", "Hy"])
    regressor.setMethod("mm", intercept=False)
    regressor.setSmooth("hann", 1)
    assert regressor.inCross == []
    assert regressor.outCross == []
    regressor.process()
    # expected
    expected_evalfreq = np.array([24, 40])
    expected_impedances = [
        np.array([[3.0 + 0.0j, 5.0 + 0.0j], [2.0 + 0.0j, 7.0 + 0.0j]]),
        np.array([[3.0 + 0.0j, 5.0 + 0.0j], [2.0 + 0.0j, 7.0 + 0.0j]]),
    ]
    expected_variances = [
        np.array([[2.78469341e-28, 1.79044909e-28], [1.64240080e-27, 1.05599956e-27]]),
        np.array([[1.47463485e-29, 1.96730235e-29], [5.89853939e-29, 7.86920938e-29]]),
    ]
    np.testing.assert_equal(regressor.evalFreq, expected_evalfreq)
    np.testing.assert_almost_equal(regressor.parameters[0], expected_impedances[0])
    np.testing.assert_almost_equal(regressor.parameters[1], expected_impedances[1])
    np.testing.assert_almost_equal(regressor.variances[0], expected_variances[0])
    np.testing.assert_almost_equal(regressor.variances[1], expected_variances[1])


def test_remote_regressor_cm(monkeypatch):
    """Test remote regressor process"""
    from resistics.regression.remote import RemoteRegressor
    from mocks import mock_window_selector
    import numpy as np

    # patch out the writeTF function
    monkeypatch.setattr(
        RemoteRegressor, "writeResult", mock_remote_regressor_writeResult
    )

    selector = mock_window_selector()
    regressor = RemoteRegressor(selector, "test")
    regressor.setInput("local", ["Hx", "Hy"])
    regressor.setOutput("local", ["Ex", "Ey"])
    regressor.setRemote("remote", ["Hx", "Hy"])
    regressor.setMethod("cm", intercept=False)
    regressor.setSmooth("hann", 1)
    assert regressor.inCross == []
    assert regressor.outCross == []
    regressor.process()
    # expected
    expected_evalfreq = np.array([24, 40])
    expected_impedances = [
        np.array([[3.0 + 0.0j, 5.0 + 0.0j], [2.0 + 0.0j, 7.0 + 0.0j]]),
        np.array([[3.0 + 0.0j, 5.0 + 0.0j], [2.0 + 0.0j, 7.0 + 0.0j]]),
    ]
    expected_variances = [
        np.array([[2.78469341e-28, 1.79044909e-28], [1.64240080e-27, 1.05599956e-27]]),
        np.array([[1.47463485e-29, 1.96730235e-29], [5.89853939e-29, 7.86920938e-29]]),
    ]
    np.testing.assert_equal(regressor.evalFreq, expected_evalfreq)
    np.testing.assert_almost_equal(regressor.parameters[0], expected_impedances[0])
    np.testing.assert_almost_equal(regressor.parameters[1], expected_impedances[1])
    np.testing.assert_almost_equal(regressor.variances[0], expected_variances[0])
    np.testing.assert_almost_equal(regressor.variances[1], expected_variances[1])


def test_local_regressor_ols_noise(monkeypatch):
    """Test local regressor process with some noise added to the localsite"""
    from resistics.regression.remote import RemoteRegressor
    from mocks import mock_window_selector
    import numpy as np

    # patch out the writeTF function
    monkeypatch.setattr(
        RemoteRegressor, "writeResult", mock_remote_regressor_writeResult
    )

    selector = mock_window_selector(localnoise=True)
    regressor = RemoteRegressor(selector, "test")
    regressor.setInput("local", ["Hx", "Hy"])
    regressor.setOutput("local", ["Ex", "Ey"])
    regressor.setRemote("remote", ["Hx", "Hy"])
    regressor.setMethod("ols", intercept=False)
    regressor.setSmooth("hann", 1)
    assert regressor.inCross == []
    assert regressor.outCross == []
    regressor.process()
    # expected
    expected_evalfreq = np.array([24, 40])
    expected_impedances = [
        np.array(
            [
                [4.17149691 + 0.0j, 3.48769594 + 0.0j],
                [1.48229546 + 0.0j, 7.28114671 + 0.0j],
            ]
        ),
        np.array(
            [
                [4.08086708 + 0.0j, 3.51508884 + 0.0j],
                [-1.80159334 + 0.0j, 10.70916092 + 0.0j],
            ]
        ),
    ]
    expected_variances = [
        np.array([[0.01284865, 0.00819992], [0.68188147, 0.43517191]]),
        np.array([[0.03504211, 0.04651421], [18.06330838, 23.97687941]]),
    ]
    np.testing.assert_equal(regressor.evalFreq, expected_evalfreq)
    np.testing.assert_almost_equal(regressor.parameters[0], expected_impedances[0])
    np.testing.assert_almost_equal(regressor.parameters[1], expected_impedances[1])
    np.testing.assert_almost_equal(regressor.variances[0], expected_variances[0])
    np.testing.assert_almost_equal(regressor.variances[1], expected_variances[1])


def test_local_regressor_mm_noise(monkeypatch):
    """Test local regressor process with some noise added to the localsite"""
    from resistics.regression.remote import RemoteRegressor
    from mocks import mock_window_selector
    import numpy as np

    # patch out the writeTF function
    monkeypatch.setattr(
        RemoteRegressor, "writeResult", mock_remote_regressor_writeResult
    )

    selector = mock_window_selector(localnoise=True)
    regressor = RemoteRegressor(selector, "test")
    regressor.setInput("local", ["Hx", "Hy"])
    regressor.setOutput("local", ["Ex", "Ey"])
    regressor.setRemote("remote", ["Hx", "Hy"])
    regressor.setMethod("mm", intercept=False)
    regressor.setSmooth("hann", 1)
    assert regressor.inCross == []
    assert regressor.outCross == []
    regressor.process()
    # expected
    expected_evalfreq = np.array([24, 40])
    expected_impedances = [
        np.array(
            [
                [4.28467732 + 0.0j, 3.39812143 + 0.0j],
                [2.33069538 + 0.0j, 6.59175726 + 0.0j],
            ]
        ),
        np.array(
            [
                [3.77407493 + 0.0j, 3.86441015 + 0.0j],
                [1.21575149 + 0.0j, 7.35576611 + 0.0j],
            ]
        ),
    ]
    expected_variances = [
        np.array([[5.31246449e-04, 3.39037709e-04], [3.13860843e00, 2.00303760e00]]),
        np.array([[0.11543121, 0.15322112], [47.59264394, 63.17353725]]),
    ]
    np.testing.assert_equal(regressor.evalFreq, expected_evalfreq)
    np.testing.assert_almost_equal(regressor.parameters[0], expected_impedances[0])
    np.testing.assert_almost_equal(regressor.parameters[1], expected_impedances[1])
    np.testing.assert_almost_equal(regressor.variances[0], expected_variances[0])
    np.testing.assert_almost_equal(regressor.variances[1], expected_variances[1])


def test_local_regressor_cm_noise(monkeypatch):
    """Test local regressor process with some noise added to the localsite"""
    from resistics.regression.remote import RemoteRegressor
    from mocks import mock_window_selector
    import numpy as np

    # patch out the writeTF function
    monkeypatch.setattr(
        RemoteRegressor, "writeResult", mock_remote_regressor_writeResult
    )

    selector = mock_window_selector(localnoise=True)
    regressor = RemoteRegressor(selector, "test")
    regressor.setInput("local", ["Hx", "Hy"])
    regressor.setOutput("local", ["Ex", "Ey"])
    regressor.setRemote("remote", ["Hx", "Hy"])
    regressor.setMethod("cm", intercept=False)
    regressor.setSmooth("hann", 1)
    assert regressor.inCross == []
    assert regressor.outCross == []
    regressor.process()
    # expected
    expected_evalfreq = np.array([24, 40])
    expected_impedances = [
        np.array(
            [
                [4.32416997 + 0.0j, 3.36294245 + 0.0j],
                [1.06908193 + 0.0j, 7.60932995 + 0.0j],
            ]
        ),
        np.array(
            [
                [4.10868381 + 0.0j, 3.465675 + 0.0j],
                [1.78789942 + 0.0j, 7.24148737 + 0.0j],
            ]
        ),
    ]
    expected_variances = [
        np.array([[7.01571556e-04, 4.47737982e-04], [4.71858833e-01, 3.01136955e-01]]),
        np.array([[0.03930329, 0.05217041], [0.02580595, 0.03425431]]),
    ]
    np.testing.assert_equal(regressor.evalFreq, expected_evalfreq)
    np.testing.assert_almost_equal(regressor.parameters[0], expected_impedances[0])
    np.testing.assert_almost_equal(regressor.parameters[1], expected_impedances[1])
    np.testing.assert_almost_equal(regressor.variances[0], expected_variances[0])
    np.testing.assert_almost_equal(regressor.variances[1], expected_variances[1])


def test_remote_regressor_ols_intersite(monkeypatch):
    """Test remote regressor process with a different site for the ouput"""
    from resistics.regression.remote import RemoteRegressor
    from mocks import mock_window_selector
    import numpy as np

    # patch out the writeTF function
    monkeypatch.setattr(
        RemoteRegressor, "writeResult", mock_remote_regressor_writeResult
    )

    selector = mock_window_selector()
    regressor = RemoteRegressor(selector, "test")
    regressor.setInput("local", ["Hx", "Hy"])
    regressor.setOutput("inter", ["Ex", "Ey"])
    regressor.setRemote("remote", ["Hx", "Hy"])
    regressor.setMethod("ols", intercept=False)
    regressor.setSmooth("hann", 1)
    assert regressor.inCross == []
    assert regressor.outCross == []
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


def test_remote_regressor_mm_intersite(monkeypatch):
    """Test remote regressor process with a different site for the ouput"""
    from resistics.regression.remote import RemoteRegressor
    from mocks import mock_window_selector
    import numpy as np

    # patch out the writeTF function
    monkeypatch.setattr(
        RemoteRegressor, "writeResult", mock_remote_regressor_writeResult
    )

    selector = mock_window_selector()
    regressor = RemoteRegressor(selector, "test")
    regressor.setInput("local", ["Hx", "Hy"])
    regressor.setOutput("inter", ["Ex", "Ey"])
    regressor.setRemote("remote", ["Hx", "Hy"])
    regressor.setMethod("mm", intercept=False)
    regressor.setSmooth("hann", 1)
    assert regressor.inCross == []
    assert regressor.outCross == []
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


def test_remote_regressor_cm_intersite(monkeypatch):
    """Test remote regressor process with a different site for the ouput"""
    from resistics.regression.remote import RemoteRegressor
    from mocks import mock_window_selector
    import numpy as np

    # patch out the writeTF function
    monkeypatch.setattr(
        RemoteRegressor, "writeResult", mock_remote_regressor_writeResult
    )

    selector = mock_window_selector()
    regressor = RemoteRegressor(selector, "test")
    regressor.setInput("local", ["Hx", "Hy"])
    regressor.setOutput("inter", ["Ex", "Ey"])
    regressor.setRemote("remote", ["Hx", "Hy"])
    regressor.setMethod("cm", intercept=False)
    regressor.setSmooth("hann", 1)
    assert regressor.inCross == []
    assert regressor.outCross == []
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


def test_remote_regressor_ols_intercept(monkeypatch):
    """Test remote regressor process with an intercept term"""
    from resistics.regression.remote import RemoteRegressor
    from mocks import mock_window_selector
    import numpy as np

    # patch out the writeTF function
    monkeypatch.setattr(
        RemoteRegressor, "writeResult", mock_remote_regressor_writeResult
    )

    selector = mock_window_selector(intercept=True)
    regressor = RemoteRegressor(selector, "test")
    regressor.setInput("local", ["Hx", "Hy"])
    regressor.setOutput("local", ["Ex", "Ey"])
    regressor.setRemote("remote", ["Hx", "Hy"])
    regressor.setMethod("ols", intercept=True)
    regressor.setSmooth("hann", 1)
    assert regressor.inCross == []
    assert regressor.outCross == []
    assert regressor.remoteCross == ["Hx", "Hy"]
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


def test_remote_regressor_mm_intercept(monkeypatch):
    """Test remote regressor process with an intercept term"""
    from resistics.regression.remote import RemoteRegressor
    from mocks import mock_window_selector
    import numpy as np

    # patch out the writeTF function
    monkeypatch.setattr(
        RemoteRegressor, "writeResult", mock_remote_regressor_writeResult
    )

    selector = mock_window_selector(intercept=True)
    regressor = RemoteRegressor(selector, "test")
    regressor.setInput("local", ["Hx", "Hy"])
    regressor.setOutput("local", ["Ex", "Ey"])
    regressor.setRemote("remote", ["Hx", "Hy"])
    regressor.setMethod("mm", intercept=True)
    regressor.setSmooth("hann", 1)
    assert regressor.inCross == []
    assert regressor.outCross == []
    assert regressor.remoteCross == ["Hx", "Hy"]
    regressor.process()
    # expected
    expected_evalfreq = np.array([24, 40])
    expected_impedances = [
        np.array(
            [
                [5.1887448 + 0.0j, 3.23769067 + 0.0j, 1.72606573 + 0.0j],
                [-0.50578302 + 0.0j, 9.02951015 + 0.0j, -5.3106386 + 0.0j],
            ]
        ),
        np.array(
            [
                [4.16876826 + 0.0j, 3.93359941 + 0.0j, 0.1393923 + 0.0j],
                [2.56283624 + 0.0j, 5.80703278 + 0.0j, 0.57298163 + 0.0j],
            ]
        ),
    ]
    expected_variances = [
        np.array(
            [[0.13001168, 0.07693714, 0.4976025], [0.2015383, 0.11926451, 0.77136119],]
        ),
        np.array(
            [
                [4.28780289, 6.03441972, 48.5332116],
                [8.09793129, 11.39658644, 91.65967348],
            ]
        ),
    ]
    np.testing.assert_equal(regressor.evalFreq, expected_evalfreq)
    np.testing.assert_almost_equal(regressor.parameters[0], expected_impedances[0])
    np.testing.assert_almost_equal(regressor.parameters[1], expected_impedances[1])
    np.testing.assert_almost_equal(regressor.variances[0], expected_variances[0])
    np.testing.assert_almost_equal(regressor.variances[1], expected_variances[1])


def test_remote_regressor_cm_intercept(monkeypatch):
    """Test remote regressor process with an intercept term"""
    from resistics.regression.remote import RemoteRegressor
    from mocks import mock_window_selector
    import numpy as np

    # patch out the writeTF function
    monkeypatch.setattr(
        RemoteRegressor, "writeResult", mock_remote_regressor_writeResult
    )

    selector = mock_window_selector(intercept=True)
    regressor = RemoteRegressor(selector, "test")
    regressor.setInput("local", ["Hx", "Hy"])
    regressor.setOutput("local", ["Ex", "Ey"])
    regressor.setRemote("remote", ["Hx", "Hy"])
    regressor.setMethod("cm", intercept=True)
    regressor.setSmooth("hann", 1)
    assert regressor.inCross == []
    assert regressor.outCross == []
    assert regressor.remoteCross == ["Hx", "Hy"]
    regressor.process()
    # expected
    expected_evalfreq = np.array([24, 40])
    expected_impedances = [
        np.array(
            [
                [4.5708818 + 0.0j, 3.83094409 + 0.0j, 0.61631404 + 0.0j],
                [-0.34507489 + 0.0j, 8.7452132 + 0.0j, -2.45585909 + 0.0j],
            ]
        ),
        np.array(
            [
                [3.78599245 + 0.0j, 4.36870089 + 0.0j, 0.04394717 + 0.0j],
                [0.71686719 + 0.0j, 8.0305959 + 0.0j, -0.9092429 + 0.0j],
            ]
        ),
    ]
    expected_variances = [
        np.array(
            [
                [7.10053809, 4.20189219, 27.17637004],
                [15.82405112, 9.36421381, 60.56446191],
            ]
        ),
        np.array(
            [
                [3.28799791, 4.62734876, 37.21651907],
                [8.76270122, 12.33214734, 99.1841377],
            ]
        ),
    ]
    np.testing.assert_equal(regressor.evalFreq, expected_evalfreq)
    np.testing.assert_almost_equal(regressor.parameters[0], expected_impedances[0])
    np.testing.assert_almost_equal(regressor.parameters[1], expected_impedances[1])
    np.testing.assert_almost_equal(regressor.variances[0], expected_variances[0])
    np.testing.assert_almost_equal(regressor.variances[1], expected_variances[1])
