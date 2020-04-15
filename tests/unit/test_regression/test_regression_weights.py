def test_bisquare() -> None:
    """The weights for bisquare"""
    from resistics.regression.weights import bisquare, getWeights
    import statsmodels.api as sm
    import numpy as np

    data = np.array([1, 2, 3, 4, 5, 6])
    weights = bisquare(data)
    getweights = getWeights(data, "bisquare")
    bisq = sm.robust.norms.TukeyBiweight()
    smweights = bisq.weights(data)
    np.testing.assert_equal(weights, smweights)
    np.testing.assert_equal(getweights, smweights)


def test_huber() -> None:
    """The weights for Huber"""
    from resistics.regression.weights import huber, getWeights
    import statsmodels.api as sm
    import numpy as np

    data = np.array([1, 2, 3, 4, 5, 6])
    weights = huber(data)
    getweights = getWeights(data, "huber")
    hub = sm.robust.norms.HuberT()
    smweights = hub.weights(data)
    np.testing.assert_equal(weights, smweights)
    np.testing.assert_equal(getweights, smweights)


def test_hampel() -> None:
    """The weights for Hampel"""
    from resistics.regression.weights import hampel, getWeights
    import statsmodels.api as sm
    import numpy as np

    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    weights = hampel(data)
    getweights = getWeights(data, "hampel")
    hamp = sm.robust.norms.Hampel()
    smweights = hamp.weights(data)
    np.testing.assert_equal(weights, smweights)
    np.testing.assert_equal(getweights, smweights)


def test_trimmedMean() -> None:
    """The weights for trimmed mean"""
    from resistics.regression.weights import trimmedMean, getWeights
    import statsmodels.api as sm
    import numpy as np

    data = np.array([1, 2, 3, 4, 5, 6])
    weights = trimmedMean(data)
    getweights = getWeights(data, "trimmedMean")
    tmean = sm.robust.norms.TrimmedMean()
    smweights = tmean.weights(data)
    np.testing.assert_equal(weights, smweights)
    np.testing.assert_equal(getweights, smweights)


def test_andrewsWave() -> None:
    """The weights for Andrews wave"""
    from resistics.regression.weights import andrewsWave, getWeights
    import statsmodels.api as sm
    import numpy as np

    data = np.array([1, 2, 3, 4, 5, 6])
    weights = andrewsWave(data)
    getweights = getWeights(data, "andrewsWave")
    awave = sm.robust.norms.AndrewWave()
    smweights = awave.weights(data)
    np.testing.assert_equal(weights, smweights)
    np.testing.assert_equal(getweights, smweights)


def test_leastSquares() -> None:
    """The weights for least squares solution"""
    from resistics.regression.weights import leastSquares, getWeights
    import numpy as np

    data = np.array([1, 2, 3, 4, 5, 6])
    weights = leastSquares(data)
    getweights = getWeights(data, "leastsquares")
    np.testing.assert_equal(weights, np.ones(data.size))
    np.testing.assert_equal(getweights, weights)