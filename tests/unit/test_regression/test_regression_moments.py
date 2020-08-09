def test_regression_moments_mad() -> None:
    """Test the MAD - median deviation from median"""
    from resistics.regression.moments import mad, getScale
    import numpy as np
    import scipy.stats as stats
    from statsmodels.robust.scale import mad as sm_mad

    np.random.seed(12345)
    fat_tails = stats.t(6).rvs(40)
    smout = sm_mad(np.absolute(fat_tails))
    madout = mad(fat_tails)
    scaleout = getScale(fat_tails, "mad")
    np.testing.assert_almost_equal(smout, madout)
    np.testing.assert_almost_equal(smout, scaleout)

def test_regression_moments_mad0() -> None:
    """This is MAD with zeros removed and using a 0 location"""
    from resistics.regression.moments import mad0, getScale
    import numpy as np
    import scipy.stats as stats

    data = np.array([0, 1, 2, 3, 1, 3, 4, 0, 4, 5])
    expected = 3/stats.norm.ppf(3/4.)
    assert mad0(data) == expected
    assert getScale(data, "mad0") == expected
    assert getScale(data) == expected
    data = np.array([0, -1, 2, -3, 1, 3, -4, 0, 4, 5])
    expected = 3/stats.norm.ppf(3/4.)
    assert mad0(data) == (3/stats.norm.ppf(3/4.))
    assert getScale(data, "mad0") == expected
    assert getScale(data) == expected