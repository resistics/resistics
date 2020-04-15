def test_regression_data():
    """Test regression data"""
    from test_regression_robust import example_data
    from resistics.regression.data import RegressionData
    import numpy as np

    obs, A = example_data(noise=False, outliers=False, intercept=False)
    regdata = RegressionData(A, obs)
    regdata.printInfo()
    np.testing.assert_equal(regdata.A, A)
    np.testing.assert_equal(regdata.y, obs)

    # initialise with more
    regdata = RegressionData(
        A,
        obs,
        params=np.ones(shape=(2)),
        resids=np.zeros(shape=(2)),
        weights=np.ones(shape=(2)),
        scale=4.7,
        method="cm",
    )
    np.testing.assert_equal(regdata.A, A)
    np.testing.assert_equal(regdata.y, obs)
    np.testing.assert_equal(regdata.params, np.ones(shape=(2)))
    np.testing.assert_equal(regdata.resids, np.zeros(shape=(2)))
    np.testing.assert_equal(regdata.weights, np.ones(shape=(2)))
    np.testing.assert_equal(regdata.scale, 4.7)
    assert regdata.method == "cm"
    assert regdata.rms == 0

    # trying updating the model parameters
    regdata.setModelParameters(
        np.ones(shape=(2)) * 8, resids=np.array([3, 4]), weights=np.array([4, 6])
    )
    np.testing.assert_equal(regdata.A, A)
    np.testing.assert_equal(regdata.y, obs)
    np.testing.assert_equal(regdata.params, np.ones(shape=(2)) * 8)
    np.testing.assert_equal(regdata.resids, np.array([3, 4]))
    np.testing.assert_equal(regdata.weights, np.array([4, 6]))
    np.testing.assert_equal(regdata.scale, 4.7)
    assert regdata.method == "cm"
    assert regdata.rms == 5
