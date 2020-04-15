def example_data(noise: bool = False, outliers: bool = False, intercept: bool = False):
    """Example data for testing linear regression

    Method returns the observations and predictors. The regressors/parameters are 2.5 and 4, which need to be estimated through linear regression.
    
    Parameters
    ----------
    noise : bool
        Boolean flag for adding noise
    outliers : bool
        Boolean flag for adding larger outliers
    intercept : bool
        Flag for adding an intercept

    Returns
    -------
    obs : np.ndarray
        The observations
    A : np.ndarray
        The predictors
    """
    import numpy as np

    # seed
    np.random.seed(1)
    a1 = np.arange(50)
    a2 = np.arange(-25, 25, 1)
    # create a linear function of this
    obs = (2.5 * a1) + (4 * a2)

    if intercept:
        # add a constant term
        obs = obs + 10
    if noise:
        # add noise to observations
        mean = 0
        std = 3
        obs = obs + np.random.normal(mean, std, obs.size)
    if outliers:
        # add few outliers to observations
        for ii in range(0, 5):
            index = np.random.randint(0, obs.size)
            obs[index] = np.random.randint(std * 4, std * 20)

    A = np.transpose(np.vstack((a1, a2)))
    return obs, A


def test_olsModel() -> None:
    """Test ordinary least squares"""
    from resistics.regression.robust import olsModel
    import numpy as np
    import statsmodels.api as sm

    # no noise and no intercept
    obs, A = example_data(noise=False, outliers=False, intercept=False)
    soln = olsModel(A, obs)
    mod = sm.OLS(obs, A)
    res = mod.fit()
    np.testing.assert_almost_equal(soln.params, res.params)
    np.testing.assert_almost_equal(soln.params, np.array([2.5, 4]))

    # noise and no intercept
    obs, A = example_data(noise=True, outliers=True, intercept=False)
    soln = olsModel(A, obs)
    mod = sm.OLS(obs, A)
    res = mod.fit()
    np.testing.assert_almost_equal(soln.params, res.params)

    # intercept and no noise
    obs, A = example_data(noise=False, outliers=False, intercept=True)
    soln = olsModel(A, obs, intercept=True)
    Aintercept = np.hstack((np.ones(shape=(A.shape[0], 1)), A))
    mod = sm.OLS(obs, Aintercept)
    res = mod.fit()
    np.testing.assert_almost_equal(soln.params, res.params.astype("complex"))

    # with noise and outliers
    obs, A = example_data(noise=True, outliers=True, intercept=True)
    soln = olsModel(A, obs, intercept=True)
    Aintercept = np.hstack((np.ones(shape=(A.shape[0], 1)), A))
    mod = sm.OLS(obs, Aintercept)
    res = mod.fit()
    np.testing.assert_almost_equal(soln.params, res.params.astype("complex"))


def test_mestimateModel():
    """Test mestimateModel
    
    The resistics mestimate is slightly different to the statsmodels mestimate. Therefore, very close are not expected.
    """
    from resistics.regression.robust import mestimateModel
    import numpy as np
    import statsmodels.api as sm

    obs, A = example_data(noise=True, outliers=True, intercept=False)
    soln = mestimateModel(A, obs, weights="huber")
    mod = sm.RLM(obs, A, M=sm.robust.norms.HuberT())
    res = mod.fit()
    np.testing.assert_almost_equal(soln.params, [2.52082274 + 0.0j, 3.95876367 + 0.0j])
    np.testing.assert_almost_equal(soln.params, res.params, decimal=2)

    # intercept and no noise
    obs, A = example_data(noise=False, outliers=False, intercept=True)
    soln = mestimateModel(A, obs, weights="huber", intercept=True)
    print(soln)
    Aintercept = np.hstack((np.ones(shape=(A.shape[0], 1)), A))
    mod = sm.RLM(obs, Aintercept, M=sm.robust.norms.HuberT())
    res = mod.fit()
    np.testing.assert_almost_equal(soln.params, res.params, decimal=5)

    # with noise and outliers
    obs, A = example_data(noise=True, outliers=True, intercept=True)
    soln = mestimateModel(A, obs, weights="huber", intercept=True)
    Aintercept = np.hstack((np.ones(shape=(A.shape[0], 1)), A))
    mod = sm.RLM(obs, Aintercept, M=sm.robust.norms.HuberT())
    res = mod.fit()
    np.testing.assert_almost_equal(
        soln.params, [-0.02582017 + 0.0j, 2.91786737 + 0.0j, 3.56337161 + 0.0j]
    )
    np.testing.assert_almost_equal(soln.params, res.params, decimal=2)


def test_mmestimateModel():
    """Test mmestimateModel
    
    This is not expected to give the same values as statsmodels, but should be close. It is a two stage mestimate.
    """
    from resistics.regression.robust import mmestimateModel
    import numpy as np
    import statsmodels.api as sm

    obs, A = example_data(noise=True, outliers=True, intercept=False)
    soln = mmestimateModel(A, obs, weights="huber")
    mod = sm.RLM(obs, A, M=sm.robust.norms.HuberT())
    res = mod.fit()
    np.testing.assert_almost_equal(soln.params, [2.50586291 + 0.0j, 4.01456129 + 0.0j])
    np.testing.assert_almost_equal(soln.params, res.params, decimal=1)

    # intercept and no noise
    obs, A = example_data(noise=False, outliers=False, intercept=True)
    soln = mmestimateModel(A, obs, weights="huber", intercept=True)
    Aintercept = np.hstack((np.ones(shape=(A.shape[0], 1)), A))
    mod = sm.RLM(obs, Aintercept, M=sm.robust.norms.HuberT())
    res = mod.fit()
    np.testing.assert_almost_equal(soln.params, res.params, decimal=5)

    # with noise and outliers
    obs, A = example_data(noise=True, outliers=True, intercept=True)
    soln = mmestimateModel(A, obs, weights="huber", intercept=True)
    Aintercept = np.hstack((np.ones(shape=(A.shape[0], 1)), A))
    mod = sm.RLM(obs, Aintercept, M=sm.robust.norms.HuberT())
    res = mod.fit()
    np.testing.assert_almost_equal(
        soln.params, [-0.02842478 + 0.0j, 2.90523962 + 0.0j, 3.61585903 + 0.0j]
    )
    np.testing.assert_almost_equal(soln.params, res.params, decimal=1)


def test_chaterjeeMachler():
    """Test chaterjeeMachler robust regression"""
    from resistics.regression.robust import chatterjeeMachler
    import numpy as np
    import statsmodels.api as sm

    obs, A = example_data(noise=True, outliers=True, intercept=False)
    soln = chatterjeeMachler(A, obs)
    np.testing.assert_almost_equal(
        soln.params, [2.50767299 + 0.0j, 3.88405236 + 0.0j]
    )

    # intercept and no noise
    obs, A = example_data(noise=False, outliers=False, intercept=True)
    soln = chatterjeeMachler(A, obs, intercept=True)
    np.testing.assert_almost_equal(
        soln.params, [-0.02791069 + 0.0j, 2.90111643 + 0.0j, 3.59888357 + 0.0j]
    )

    # with noise and outliers
    obs, A = example_data(noise=True, outliers=True, intercept=True)
    soln = chatterjeeMachler(A, obs, intercept=True)
    np.testing.assert_almost_equal(
        soln.params, [-0.02543469 + 0.0j, 2.91432684 + 0.0j, 3.55019399 + 0.0j]
    )


def test_applyWeights() -> None:
    """Test application of weights to arrays"""
    from resistics.regression.robust import applyWeights
    import numpy as np

    # test values
    y = np.array([0, 1, 4, 6, 7, 3])
    A = np.array([[5, 3], [7, 2], [3, 1], [4, 9], [6, 3], [2, 1],])
    weights = np.array([0.09, 0.01, 0.49, 1, 0.25, 0.36])
    # expected values
    yexpected = np.array([0, 0.1, 2.8, 6, 3.5, 1.8])
    Aexpected = np.array(
        [[1.5, 0.9], [0.7, 0.2], [2.1, 0.7], [4.0, 9.0], [3.0, 1.5], [1.2, 0.6],],
        dtype="complex",
    )
    Anew, ynew = applyWeights(A, y, weights)
    np.testing.assert_almost_equal(Anew, Aexpected)
    np.testing.assert_almost_equal(ynew, yexpected)


def test_hermitianTranspose() -> None:
    """Test the hermitian transpose"""
    from resistics.regression.robust import hermitianTranspose
    import numpy as np

    data = [
        [3 + 2j, 4 + 6j, -3 + 5j],
        [6 - 2j, 3 - 1j, -7 - 4j],
        [5 + 1j, -5 - 5j, -6 - 3j],
    ]
    expected = [
        [3 - 2j, 6 + 2j, 5 - 1j],
        [4 - 6j, 3 + 1j, -5 + 5j],
        [-3 - 5j, -7 + 4j, -6 + 3j],
    ]
    np.testing.assert_equal(hermitianTranspose(data), np.array(expected))


def test_regression_robust_maxiter() -> None:
    """Test the maximum iterations for robust regression"""
    from resistics.regression.robust import maxiter

    assert maxiter() >= 100


# def test_regression_robust_mestimate():
#     from resistics.regression.robust import sampleMedian, sampleMAD

#     modelMean = 0
#     modelStd = 5
#     x = np.arange(1000)
#     y = np.random.normal(modelMean, modelStd, x.size)
#     ones = np.ones(shape=(x.size))
#     # add large outliers
#     numOutliers = 450
#     for ii in range(0, numOutliers):
#         index = np.random.randint(0, x.size)
#         y[index] = np.random.randint(std * 4, std * 20)
#     # compute mean
#     mean = np.average(y)
#     std = np.std(y)
#     # compute mad
#     med = sampleMedian(y)
#     mad = sampleMAD(y)
#     # mestimates
#     mestLocation, mestScale = mestimate(y)


# def test_mestimateModel():
#     breakPrint()
#     generalPrint("basic_Robust",
#                  "Running test function: test_mestimateModel")
#     # let's generate some data
#     x = np.arange(1000)
#     y = np.arange(-50, 50, 0.1)
#     # create a linear function of this
#     z = 2.5 * x + y
#     # let's add some noise
#     mean = 0
#     std = 3
#     noise = np.random.normal(0, 3, x.size)
#     # print noise.shape
#     z = z + noise
#     # now add some outliers
#     numOutliers = 80
#     for i in range(0, numOutliers):
#         index = np.random.randint(0, x.size)
#         z[index] = np.random.randint(std * 4, std * 20)

#     A = np.transpose(np.vstack((x, y)))
#     # now try and do a robust regression
#     components = mestimateModel(A, z)
#     print(components)
#     # plt.figure()
#     # plt.plot()


# def testRobustRegression():
#     breakPrint()
#     generalPrint("basic_Robust",
#                  "Running test function: testRobustRegression")
#     # random seed
#     np.random.seed(0)
#     # the function
#     x = np.arange(150)
#     y = 12 + 0.5 * x
#     # noise
#     mean = 0
#     std = 3
#     noise = np.random.normal(mean, 3 * std, x.size)
#     # add noise
#     yNoise = y + noise
#     # now add some outliers
#     numOutliers = 30
#     for i in range(0, numOutliers):
#         index = np.random.randint(0, x.size)
#         yNoise[index] = yNoise[index] + np.random.randint(-1000, 1000)

#     # now add some outliers
#     xNoise = np.array(x)
#     numOutliers = 30
#     for i in range(0, numOutliers):
#         index = np.random.randint(0, x.size)
#         xNoise[index] = x[index] + np.random.randint(-5000, 5000)
#     xNoise = xNoise.reshape((x.size, 1))

#     # lets use m estimate
#     paramsM, residsM, scaleM, weightsM = mestimateModel(
#         xNoise, yNoise, intercept=True)
#     # lets use mm estimate
#     paramsMM, residsMM, scaleMM, weightsMM = mmestimateModel(
#         xNoise, yNoise, intercept=True)
#     # lets test chatterjee machler
#     paramsCM, residsCM, weightsCM = chatterjeeMachler(
#         xNoise, yNoise, intercept=True)
#     # lets test chatterjee machler mod
#     paramsModCM, residsModCM, weightsModCM = chatterjeeMachlerMod(
#         xNoise, yNoise, intercept=True)

#     # let's plot Pdiag
#     # plt.figure()
#     # n, bins, patches = plt.hist(
#     #     Pdiag, 50, normed=0, facecolor='green', alpha=0.75)

#     # try and predict
#     yM = paramsM[0] + paramsM[1] * x
#     yMM = paramsMM[0] + paramsMM[1] * x
#     yCM = paramsCM[0] + paramsCM[1] * x
#     yCM_mod = paramsModCM[0] + paramsModCM[1] * x

#     plt.figure()
#     plt.scatter(x, y, marker="s", color="black")
#     plt.scatter(xNoise, yNoise)
#     plt.plot(x, yM)
#     plt.plot(x, yMM)
#     plt.plot(x, yCM)
#     plt.plot(x, yCM_mod)
#     plt.legend([
#         "M estimate", "MM estimate", "chatterjeeMachler",
#         "chatterjeeMachlerMod"
#     ],
#                loc="lower left")
#     plt.show()


# def testRobustRegression2D():
#     breakPrint()
#     generalPrint("basic_Robust",
#                  "Running test function: testRobustRegression2D")
#     # random seed
#     np.random.seed(0)
#     numPts = 300
#     # the function
#     x1 = np.arange(numPts, dtype="float")
#     x2 = 10 * np.cos(2.0 * np.pi * 10 * x1 / np.max(x1))
#     y = 12 + 0.5 * x1 + 3 * x2
#     # noise
#     mean = 0
#     std = 3
#     noise = np.random.normal(mean, 3 * std, numPts)
#     # add noise
#     yNoise = y + noise
#     # now add some outliers
#     numOutliers = 140
#     for i in range(0, numOutliers):
#         index = np.random.randint(0, numPts)
#         yNoise[index] = yNoise[index] + np.random.randint(-100, 100)

#     # now add some outliers
#     x1Noise = np.array(x1)
#     x2Noise = np.array(x2)
#     numOutliers = 5
#     for i in range(0, numOutliers):
#         index = np.random.randint(0, numPts)
#         x1Noise[index] = x1[index] + np.random.randint(-500, 500)
#         index = np.random.randint(0, numPts)
#         x2Noise[index] = x2[index] + np.random.randint(-500, 500)

#     x1Noise = x1Noise.reshape((x1.size, 1))
#     x2Noise = x2Noise.reshape((x2.size, 1))
#     X = np.hstack((x1Noise, x2Noise))

#     # lets use m estimate
#     paramsM, residsM, scaleM, weightsM = mestimateModel(
#         X, yNoise, intercept=True)
#     # lets use mm estimate
#     paramsMM, residsMM, scaleMM, weightsMM = mmestimateModel(
#         X, yNoise, intercept=True)
#     # lets test chatterjee machler
#     paramsCM, residsCM, weightsCM = chatterjeeMachler(
#         X, yNoise, intercept=True)
#     # lets test chatterjee machler mod
#     paramsModCM, residsModCM, weightsModCM = chatterjeeMachlerMod(
#         X, yNoise, intercept=True)
#     # lets test chatterjee machler hadi
#     paramsCMHadi, residsCMHadi, weightsCMHadi = chatterjeeMachlerHadi(
#         X, yNoise, intercept=True)

#     # try and predict
#     yM = paramsM[0] + paramsM[1] * x1 + paramsM[2] * x2
#     yMM = paramsMM[0] + paramsMM[1] * x1 + paramsMM[2] * x2
#     yCM = paramsCM[0] + paramsCM[1] * x1 + paramsCM[2] * x2
#     yCM_mod = paramsModCM[0] + paramsModCM[1] * x1 + paramsModCM[2] * x2
#     yCM_Hadi = paramsCMHadi[0] + paramsCMHadi[1] * x1 + paramsCMHadi[2] * x2

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(x1, x2, y, marker="s", color="black")
#     ax.scatter(x1Noise, x2Noise, yNoise, marker="*", s=50, color="goldenrod")
#     # plt.plot(x1, x2, zs=yM)
#     plt.plot(x1, x2, zs=yMM)
#     # plt.plot(x1, x2, zs=yCM)
#     plt.plot(x1, x2, zs=yCM_mod)
#     # plt.plot(x1, x2, zs=yCM_Hadi)
#     # plt.legend(["M estimate", "MM estimate", "chatterjeeMachler", "chatterjeeMachlerMod", "chatterjeeMachlerHadi"], loc="lower left")
#     plt.legend(["MM estimate", "chatterjeeMachlerMod"], loc="lower left")
#     plt.show()


# test_mestimate()
# test_mestimateModel()
# testRobustRegression()
# testRobustRegression2D()


# def plot_weights(support, weights, xlabels, xticks):
#     #fig = plt.figure(figsize=(12,8))
#     ax = fig.add_subplot(111)
#     #ax.plot(support, weights_func(support))
#     ax.plot(support, weights)
#     ax.set_xticks(xticks)
#     ax.set_xticklabels(xlabels, fontsize=16)
#     ax.set_ylim(-.1, 1.1)
#     return ax

# # get the data
# prestige = sm.datasets.get_rdataset("Duncan", "car", cache=True).data
# print(prestige.head(10))

# # getting the variables for me
# obs = np.array(prestige.prestige)
# predictor = np.empty(shape=(obs.size,2))
# predictor[:,0] = prestige.income
# predictor[:,1] = prestige.education

# # plot if wanted
# fig = plt.figure(figsize=(12,12))
# ax1 = fig.add_subplot(211, xlabel='Income', ylabel='Prestige')
# ax1.scatter(prestige.income, prestige.prestige)
# xy_outlier = prestige.ix['minister'][['income','prestige']]
# ax1.annotate('Minister', xy_outlier, xy_outlier+1, fontsize=16)
# ax2 = fig.add_subplot(212, xlabel='Education',
#                            ylabel='Prestige')
# ax2.scatter(prestige.education, prestige.prestige)
# # plt.show()

# ols_model = ols('prestige ~ income + education', prestige).fit()
# print(ols_model.summary())

# print("######################")
# print("Built in OLS")
# print("######################")
# # now get the robust estimate using huber
# params, resids, squareResid, rank, s = olsModel(predictor, obs, intercept=True)
# print(params)

# print("######################")
# print("Checking M-estimate")
# print("######################")
# rlm_model = rlm('prestige ~ income + education', prestige, M=sm.robust.norms.HuberT(t=1.345)).fit()
# print(rlm_model.summary())
# print("######################")
# print("Built in M-estimate")
# print("######################")
# params, resids, scale, weights = mestimateModel(predictor, obs, weights="huber", intercept=True)
# print(params)
