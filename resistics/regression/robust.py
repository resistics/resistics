"""
The source for these functions is Robust Statisitics, Huber, 2009
in general, linear regression is# have observations y and predictors A
y is multiple observations/response
x are the independent variables and is unknown
and y is a linear function of x => y = Ax
y = nobs
A = nobs * nregressors
x = nregressors
"""

import numpy as np
import numpy.linalg as linalg
import scipy.stats as stats
from typing import List, Dict, Tuple

from resistics.common.checks import parseKeywords


def mmestimateModel(A: np.ndarray, y: np.ndarray, **kwargs):
    r"""2 stage M estimate

    Solves for :math:`x` where,

    .. math::        
        y = Ax .

    Parameters
    ----------
    A : np.ndarray
        Predictors, size nobs*nregressors
    y : np.ndarray
        Observations, size nobs
    initial : Dict
        Initial solution with parameters, scale and residuals
    scale : optional
        A scale estimate
    intercept : bool, optional
        True or False for adding an intercept term
    
    Returns
    -------
    params : np.ndarray
        Values in x
    resids : np.ndarray
        Residuals = y - Ax
    scale : float
        Robust measure of variance
    weights : np.ndarray
        Weights used in robust regression   
    """
    options = parseKeywords(defaultDictionary(), kwargs, printkw=False)
    intercept = options["intercept"]
    # this uses an initial mestimate with huber to give a measure of scale
    # and then a second with bisquare or hampel weights
    if "initial" in kwargs:
        if "scale" not in kwargs["initial"]:
            kwargs["initial"]["scale"] = sampleMAD0(kwargs["initial"]["resids"])
        params, resids, scale, weights = mestimateModel(
            A, y, weights="huber", initial=kwargs["initial"], intercept=intercept
        )
        # now do another, but with a different weighting function
        kwargs["initial"]["scale"] = scale
        # kwargs["initial"]["params"] = params # put the new solution in, because simply then doing bisquare, which has zero weights, might mess things up
        # kwargs["initial"]["resids"] = resids
        params2, resids2, scale2, weights2 = mestimateModel(
            A, y, weights="bisquare", initial=kwargs["initial"], intercept=intercept
        )
    else:
        params, resids, scale, weights = mestimateModel(
            A, y, weights="huber", intercept=intercept
        )
        # now do another, but with a different weighting function
        params2, resids2, scale2, weights2 = mestimateModel(
            A, y, weights="bisquare", scale=scale, intercept=intercept
        )

    return params2, resids2, scale2, weights2


def mestimateModel(A: np.ndarray, y: np.ndarray, **kwargs) -> Tuple:
    r"""Mestimate robust least squares

    Solves for :math:`x` where,

    .. math::        
        y = Ax .

    Good method for dependent outliers (in :math:`y`). Not robust against independent outliers (leverage points)

    Parameters
    ----------
    A : np.ndarray
        Predictors, size nobs*nregressors
    y : np.ndarray
        Observations, size nobs
    initial : 
    scale : optional
        A scale estimate
    intercept : bool, optional
        True or False for adding an intercept term

    Returns
    -------
    params : np.ndarray
        Values in x
    resids : np.ndarray
        Residuals = y - Ax
    scale : float
        Robust measure of variance
    weights : np.ndarray
        Weights used in robust regression    
    """
    options = parseKeywords(defaultDictionary(), kwargs, printkw=False)
    # calculate the leverage
    n = A.shape[0]
    p = A.shape[1]
    pnRatio = 1.0 * p / n

    # calculate the projection matrix
    q, r = linalg.qr(A)
    Pdiag = np.empty(shape=(n), dtype="float")
    for i in range(0, n):
        Pdiag[i] = np.absolute(np.sum(q[i, :] * np.conjugate(q[i, :]))).real
    del q, r
    Pdiag = Pdiag / np.max(Pdiag)
    leverageScale = sampleMAD0(Pdiag)
    leverageWeights = getRobustLocationWeights(
        Pdiag / leverageScale, "huber"
    )  # this should nowhere be equal to zero because of the previous line

    if options["intercept"] == True:
        # add column of ones for constant term
        A = np.hstack((np.ones(shape=(A.shape[0], 1), dtype="complex"), A))

    # see whether to do an initial OLS model or whether one is provided
    if options["initial"]:
        params, resids, scale = initialFromDict(options["initial"])
    else:
        params, resids, squareResid, rank, s = olsModel(A, y)
        scale = sampleMAD0(resids)

    # if an initial model was not provided but an initial scale was, replace the one here
    if options["scale"]:
        scale = options["scale"]

    # standardised residuals and weights
    weights = (
        getRobustLocationWeights(resids / scale, options["weights"]) * leverageWeights
    )

    # iteratively weighted least squares
    iteration = 0
    while iteration < options["maxiter"]:
        # do the weighted least-squares
        Anew, ynew = weightLS(A, y, weights)
        paramsNew, squareResidNew, rankNew, sNew = linalg.lstsq(Anew, ynew, rcond=None)
        residsNew = y - np.dot(A, paramsNew)
        # check residsNew to make sure not all zeros (i.e. will happen in undetermined or equally determined system)
        if np.sum(np.absolute(residsNew)) < eps():
            # then return everything here
            return paramsNew, residsNew, scale, weights
        scale = sampleMAD0(residsNew)
        # standardise and calculate weights
        weightsNew = (
            getRobustLocationWeights(residsNew / scale, options["weights"])
            * leverageWeights
        )
        # increment iteration and save weightsNew
        iteration = iteration + 1
        weights = weightsNew
        params = paramsNew

        # check to see whether the change is smaller than the tolerance
        # use the R method of checking change in residuals (can check change in params)
        changeResids = linalg.norm(residsNew - resids) / linalg.norm(residsNew)
        if changeResids < eps():
            # update residuals
            resids = residsNew
            break
        # update residuals
        resids = residsNew
    return params, resids, scale, weights


def olsModel(A, y, **kwargs) -> Tuple:
    r"""Ordinary least squares

    Solves for :math:`x` where,

    .. math::       
        y = Ax .

    Parameters
    ----------
    A : np.ndarray
        Predictors, size nobs*nregressors
    y : np.ndarray
        Observations, size nobs
    intercept : bool, optional
        True or False for adding an intercept term

    Returns
    -------
    params : np.ndarray
        Least squares solution
    resids : np.ndarray
        Residuals
    squareResid : np.ndarray
        Square residuals
    rank : int
        Rank of matrix A
    s : np.ndarray
        Singular values of A
    """
    options = parseKeywords(defaultDictionary(), kwargs, printkw=False)
    if options["intercept"]:
        # add a constant term for the intercept
        A = np.hstack((np.ones(shape=(A.shape[0], 1), dtype="complex"), A))
    params, squareResid, rank, s = linalg.lstsq(A, y, rcond=None)
    resids = y - np.dot(A, params)
    return params, resids, squareResid, rank, s


def chatterjeeMachler(A: np.ndarray, y: np.ndarray, **kwargs) -> Tuple:
    r"""Robust bounded influence solver
    
    Solves for :math:`x` where,

    .. math::  
        y = Ax .

    Being a bounded influence operator, should be robust against both outliers in dependent and independent variables.

    Parameters
    ----------
    A : np.ndarray
        Predictors, size nobs*nregressors
    y : np.ndarray
        Observations, size nobs
    intercept : bool, optional
        True or False for adding an intercept term

    Returns
    -------
    params : np.ndarray
        Values in x
    resids : np.ndarray
        Residuals = y - Ax
    weights : np.ndarray
        Weights used in robust regression     
    """
    options = parseKeywords(defaultDictionary(), kwargs, printkw=False)
    # generalPrint("S-Estimate", "Using weight function = {}".format(weightFnc))
    if options["intercept"] == True:
        # add column of ones for constant term
        A = np.hstack((np.ones(shape=(A.shape[0], 1), dtype="complex"), A))

    # now calculate p and n
    n = A.shape[0]
    p = A.shape[1]
    pnRatio = 1.0 * p / n

    # calculate the projection matrix
    q, r = linalg.qr(A)
    Pdiag = np.empty(shape=(n), dtype="float")
    for i in range(0, n):
        Pdiag[i] = np.absolute(np.sum(q[i, :] * np.conjugate(q[i, :]))).real
    del q, r
    # and save an array for later
    Pdiag = Pdiag / np.max(Pdiag)
    weightsNom = np.power(1.0 - Pdiag, 2)

    # weights for the first iteration
    tmp = np.ones(shape=(n), dtype="float") * pnRatio
    tmp = np.maximum(Pdiag, tmp)
    weights = np.reciprocal(tmp)

    # iteratively weighted least squares
    iteration = 0
    while iteration < options["maxiter"]:
        # do the weighted least-squares
        Anew, ynew = weightLS(A, y, weights)
        paramsNew, squareResidNew, rankNew, sNew = linalg.lstsq(Anew, ynew, rcond=None)
        residsNew = y - np.dot(A, paramsNew)
        # check residsNew to make sure not all zeros (i.e. will happen in undetermined or equally determined system)
        if np.sum(np.absolute(residsNew)) < eps():
            # return everything here
            return paramsNew, residsNew, weights
        residsAbs = np.absolute(residsNew)
        residsMedian = np.median(residsAbs)
        # now compute the new weights
        weightsDenom = np.maximum(
            residsAbs, np.ones(shape=(n), dtype="float") * residsMedian
        )
        weightsNew = weightsNom / weightsDenom

        # increment iteration
        iteration = iteration + 1
        weights = weightsNew
        params = paramsNew

        if iteration > 1:
            # check to see whether the change is smaller than the tolerance
            changeResids = linalg.norm(residsNew - resids) / linalg.norm(residsNew)
            if changeResids < eps():
                # update resids
                resids = residsNew
                break
        # update resids
        resids = residsNew
    return params, resids, weights


def chatterjeeMachlerMod(A, y, **kwargs):
    # using the weights in chaterjeeMachler means that min resids val in median(resids)
    # instead, use M estimate weights with a modified residual which includes a measure of leverage
    # for this, use residuals / (1-p)^2
    # I wonder if this will have a divide by zero bug

    # now calculate p and n
    n = A.shape[0]
    p = A.shape[1]
    pnRatio = 1.0 * p / n

    # calculate the projection matrix
    q, r = linalg.qr(A)
    Pdiag = np.empty(shape=(n), dtype="float")
    for i in range(0, n):
        Pdiag[i] = np.absolute(np.sum(q[i, :] * np.conjugate(q[i, :]))).real
    del q, r
    Pdiag = Pdiag / (np.max(Pdiag) + 0.0000000001)
    locP = np.median(Pdiag)
    scaleP = sampleMAD(Pdiag)
    # bound = locP + 6*scaleP
    bound = locP + 6 * scaleP
    indices = np.where(Pdiag > bound)
    Pdiag[indices] = 0.99999
    leverageMeas = np.power(1.0 - Pdiag, 2)

    # weights for the first iteration
    # this is purely based on the leverage
    tmp = np.ones(shape=(n), dtype="float") * pnRatio
    tmp = np.maximum(Pdiag, tmp)
    weights = np.reciprocal(tmp)

    # get options
    options = parseKeywords(defaultDictionary(), kwargs, printkw=False)
    # generalPrint("S-Estimate", "Using weight function = {}".format(weightFnc))
    if options["intercept"] == True:
        # add column of ones for constant term
        A = np.hstack((np.ones(shape=(A.shape[0], 1), dtype="complex"), A))

    # iteratively weighted least squares
    iteration = 0
    while iteration < options["maxiter"]:
        # do the weighted least-squares
        Anew, ynew = weightLS(A, y, weights)
        paramsNew, squareResidNew, rankNew, sNew = linalg.lstsq(Anew, ynew, rcond=None)
        residsNew = y - np.dot(A, paramsNew)
        # check residsNew to make sure not all zeros (i.e. will happen in undetermined or equally determined system)
        if np.sum(np.absolute(residsNew)) < eps():
            # then return everything here
            return paramsNew, residsNew, weights
        residsNew = residsNew / leverageMeas
        scale = sampleMAD0(residsNew)

        # standardise and calculate weights
        residsNew = residsNew / scale
        weightsNew = getRobustLocationWeights(residsNew, "huber")
        # increment iteration
        iteration = iteration + 1
        weights = weightsNew
        params = paramsNew

        if iteration > 1:
            # check to see whether the change is smaller than the tolerance
            changeResids = linalg.norm(residsNew - resids) / linalg.norm(residsNew)
            if changeResids < eps():
                # update resids
                resids = residsNew
                break
        # update resids
        resids = residsNew

    # now do the same again, but with a different function
    # do the least squares solution
    params, resids, squareResid, rank, s = olsModel(A, y)
    resids = resids / leverageMeas
    resids = resids / scale
    weights = getRobustLocationWeights(resids, "trimmedMean")
    # iteratively weighted least squares
    iteration = 0
    while iteration < options["maxiter"]:
        # do the weighted least-squares
        Anew, ynew = weightLS(A, y, weights)
        paramsNew, squareResidNew, rankNew, sNew = linalg.lstsq(Anew, ynew, rcond=None)
        residsNew = y - np.dot(A, paramsNew)
        # check residsNew to make sure not all zeros (i.e. will happen in undetermined or equally determined system)
        if np.sum(np.absolute(residsNew)) < eps():
            # then return everything here
            return paramsNew, residsNew, weights

        residsNew = residsNew / leverageMeas
        scale = sampleMAD0(residsNew)

        # standardise and calculate weights
        residsNew = residsNew / scale
        weightsNew = getRobustLocationWeights(residsNew, options["weights"])
        # increment iteration
        iteration = iteration + 1
        weights = weightsNew
        params = paramsNew

        # check to see whether the change is smaller than the tolerance
        changeResids = linalg.norm(residsNew - resids) / linalg.norm(residsNew)
        if changeResids < eps():
            # update resids
            resids = residsNew
            break
        # update resids
        resids = residsNew

    # at the end, return the components
    return params, resids, weights


def chatterjeeMachlerHadi(X, y, **kwargs):
    r"""Regression based on Hadi distances



    # Another regression method based on Hadi distances
    # implemented from the paper A Re-Weighted Least Squares Method for Robust Regression Estimation
    # Billor, Hadi    
    """
    # basic info
    options = parseKeywords(defaultDictionary(), kwargs, printkw=False)

    # for the distances, will use absX - do this before adding intercept term
    # a column of all ones will cause problems with non full rank covariance matrices
    absX = np.absolute(X)

    # now calculate p and n
    n = absX.shape[0]
    p = absX.shape[1]

    # we treat the X matrix as a multivariate matrix with n observations and p variables
    # first need to find a basic subset free of outliers
    correctionFactor = 1 + (1.0 * (p + 1) / (n - p)) + (2.0 / (n - 1 - 3 * p))
    chi = stats.chi2(p, 0)
    alpha = 0.05
    chi2bound = correctionFactor * chi.pdf(alpha / n)
    # calculate h, this is the size of the firt basic subset
    # note that this is the value h, the index of the hth element is h-1
    h = int(1.0 * (n + p + 1) / 2)  # here, only want the integer part of this
    # need to get the coordinatewise medians - this is the median of the columns
    medians = np.median(absX)
    # now compute the matrix to help calculate the distance
    A = np.zeros(shape=(p, p))
    for i in range(0, n):
        tmp = absX[i, :] - medians
        A += np.outer(tmp, tmp)
    A = 1.0 / (n - 1) * A

    # now calculate initial distances
    dInit = calculateDistCMH(n, absX, medians, A)

    # now get the h smallest values of d
    sortOrder = np.argsort(dInit)
    indices = sortOrder[0:h]
    means = np.average(absX[indices, :])
    covariance = np.cov(
        absX[indices], rowvar=False
    )  # observations in rows, columns are variables
    dH = calculateDistCMH(n, absX, means, covariance)

    # rearrange into n observations into order and partition into two initial subsets
    # one subset p+1, the n-p-1
    sortOrder = np.argsort(dH)
    indicesBasic = sortOrder[: p + 1]
    # there is a rank issue here, but ignore for now - natural observations will presumably be full rank
    means = np.average(absX[indicesBasic, :])
    covariance = np.cov(absX[indicesBasic], rowvar=False)
    dist = calculateDistCMH(n, absX, means, covariance)

    # create the basic subset
    r = p + 2
    increment = (h - r) / 100
    if increment < 1:
        increment = 1  # here, limiting to 100 iterations of this
    while r <= h:
        sortOrder = np.argsort(dist)
        indices = sortOrder[:r]  # indices start from zero, hence the - 1
        means = np.average(absX[indices])
        covariance = np.cov(absX[indices], rowvar=False)
        dist = calculateDistCMH(n, absX, means, covariance)
        if h - r > 0 and h - r < increment:
            r = h
        else:
            r += increment

    # now the second part = add more points and exclude outliers to basic set
    # all distances above r+1 = outliers
    # r = p + 1
    # increment = (n - 1 - r)/100
    while r < n:
        sortOrder = np.argsort(dist)
        dist2 = np.power(dist, 2)
        if dist2[sortOrder[r]] > chi2bound:
            break  # then leave, everything else is an outlier - it would be good if this could be saved somehow
        # otherwise, continue adding points
        sortOrder = np.argsort(dist)
        indices = sortOrder[:r]
        means = np.average(absX[indices])
        covariance = np.cov(absX[indices], rowvar=False)
        dist = calculateDistCMH(n, absX, means, covariance)
        if n - 1 - r > 0 and n - 1 - r < increment:
            r = n - 1
        else:
            r += increment

    # now with the Hadi distances calculated, can proceed to do the robust regression
    # normalise and manipulate Hadi distances
    dist = dist / np.max(dist)
    # for the median, use the basic subset
    # indicesBasic = sortOrder[:r]
    # distMedian = np.median(dist[indicesBasic]) # I am using on indicesBasic
    distMedian = np.median(dist)  # the paper suggests using the median of the complete
    tmp = np.maximum(dist, np.ones(shape=(n)) * distMedian)
    dist = np.reciprocal(tmp)
    dist2 = np.power(dist, 2)
    dist = dist2 / np.sum(dist2)

    # calculate first set of weights - this is simply dist
    weights = dist

    # now add the additional constant intercept column if required
    if options["intercept"] == True:
        # add column of ones for constant term
        X = np.hstack((np.ones(shape=(X.shape[0], 1), dtype="complex"), X))

    n = X.shape[0]
    p = X.shape[1]

    # iteratively weighted least squares
    iteration = 0
    while iteration < options["maxiter"]:
        # do the weighted least-squares
        Anew, ynew = weightLS(X, y, weights)
        paramsNew, squareResidNew, rankNew, sNew = linalg.lstsq(Anew, ynew, rcond=None)
        residsNew = y - np.dot(X, paramsNew)
        # check residsNew to make sure not all zeros (i.e. will happen in undetermined or equally determined system)
        if np.sum(np.absolute(residsNew)) < eps():
            # then return everything here
            return paramsNew, residsNew, weights

        residsAbs = np.absolute(residsNew)
        residsSquare = np.power(residsAbs, 2)
        residsNew = residsSquare / np.sum(residsSquare)
        residsMedian = np.median(residsAbs)

        # calculate the new weights
        tmpDenom = np.maximum(
            residsNew, np.ones(shape=(n), dtype="float") * residsMedian
        )
        tmp = (1 - dist) / tmpDenom
        weightsNew = np.power(tmp, 2) / np.sum(np.power(tmp, 2))

        # increment iteration
        iteration = iteration + 1
        weights = weightsNew
        params = paramsNew

        if iteration > 1:
            # check to see whether the change is smaller than the tolerance
            changeResids = linalg.norm(residsNew - resids) / linalg.norm(residsNew)
            if changeResids < eps():
                # update resids
                resids = residsNew
                break
        # update resids
        resids = residsNew

    # at the end, return the components
    return params, resids, weights


def calculateDistCMH(n, x, mean, covariance):
    inv = np.linalg.inv(covariance)
    dist = np.empty(shape=(n), dtype="float")
    for i in range(0, n):
        tmp = x[i, :] - mean
        dist[i] = np.sqrt(np.dot(tmp, np.dot(inv, tmp)))
    return dist


def weightLS(A: np.ndarray, y: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray]:
    r"""Transform A and y using the weights to perform a weighted least squares

    .. math::
        \sqrt{weights} y = \sqrt{weights} A x ,
    
    is equivalent to,
    
    .. math::     
        A^H weights y = A^H weights A x ,
    
    where :math:`A^H` is the hermitian transpose.

    In this method, both y and A are multipled by the square root of the weights and then returned.

    Parameters
    ----------
    y : np.ndarray
        Observations
    A : np.ndarray
        Regressors

    Returns
    ----------
    y : np.ndarray
        Observations multipled by the square root of the weights
    A : np.ndarray
        Regressors multipled by the square root of the weights
    """
    ynew = np.sqrt(weights) * y
    Anew = np.empty(shape=A.shape, dtype="complex")
    for col in range(0, A.shape[1]):
        Anew[:, col] = np.sqrt(weights) * A[:, col]
    return Anew, ynew


def hermitianTranspose(mat: np.ndarray) -> np.ndarray:
    """Hermitian transpose (transpose and complex conjugation)
    
    Parameters
    ----------
    np.ndarray
        Vector, matrix to Hermitian transpose
    
    Returns
    -------
    np.ndarray
        Hermitian transpose
    """
    return np.conjugate(np.transpose(mat))


def initialFromDict(initDict: Dict) -> Tuple:
    """Returns initial model from provided initial model dictionary
    
    Helps for two stage robust regression.

    Parameters
    ----------
    Dict
        Initial model to use for robust regression with the parameters, residuals and scale estimate
    
    Returns
    -------
    parameters : np.ndarray
        
    resids : np.ndarray
        The residuals
    scale : float
        Initial estimate of scale
    """
    return initDict["params"], initDict["resids"], initDict["scale"]


def defaultDictionary() -> Dict:
    """Robust regression defaults
    
    Returns
    -------
    Dict
        Default regression options
    """
    outDict = {}
    outDict["weights"] = "bisquare"
    outDict["maxiter"] = maxIter()
    outDict["initial"] = False
    outDict["scale"] = False
    outDict["intercept"] = False
    return outDict


def getRobustLocationWeights(r: np.ndarray, weight: str) -> np.ndarray:
    """Robust weighting schemes
    
    Parameters
    ----------
    r : np.ndarray
        Residuals
    weight : str
        The type of weighting to use

    Returns
    -------
    weights : np.ndarray
        The robust weights
    """
    # the second argument, k, is a tuning constant
    if weight == "huber":
        k = 1.345
        # k = 0.5
        return huberLocationWeights(r, k)
    elif weight == "hampel":
        k = 8
        return hampelLocationWeights(r, k)
    elif weight == "trimmedMean":
        k = 2
        return trimmedMeanLocationWeights(r, k)
    elif weight == "andrewsWave":
        k = 1.339
        return andrewsWaveLocationWeights(r, k)
    elif weight == "leastsq":
        return leastSquaresLocationWeights(r)
    else:
        # use bisquare weights
        k = 4.685
        # k = 1.0
        return bisquareLocationWeights(r, k)


def huberLocationWeights(r: np.ndarray, k: float) -> np.ndarray:
    """Huber location weights
    
    Parameters
    ----------
    r : np.ndarray
        Residuals
    k : float
        Tuning parameter

    Returns
    -------
    weights : np.ndarray
        The robust weights
    """
    weights = np.ones(shape=r.size, dtype="complex")
    for idx, val in enumerate(np.absolute(r)):
        if val > k:
            # relying on numpy doing the right thing when dividing by zero
            weights[idx] = k / val
    return weights.real


def bisquareLocationWeights(r: np.ndarray, k: float) -> np.ndarray:
    """Bisquare location weights
    
    Parameters
    ----------
    r : np.ndarray
        Residuals
    k : float
        Tuning parameter

    Returns
    -------
    weights : np.ndarray
        The robust weights
    """
    ones = np.ones(shape=(r.size), dtype="complex")
    threshR = np.minimum(ones, np.absolute(r / k))
    # threshR = np.maximum(-1*ones, threshR)
    return np.power((1 - np.power(threshR, 2)), 2).real


def hampelLocationWeights(r: np.ndarray, k: float) -> np.ndarray:
    """Hampel location weights
    
    Parameters
    ----------
    r : np.ndarray
        Residuals
    k : float
        Tuning parameter

    Returns
    -------
    weights : np.ndarray
        The robust weights
    """
    a = k / 4
    b = k / 2
    weights = np.ones(shape=r.size, dtype="complex")
    for idx, val in enumerate(np.absolute(r)):
        if val > a and val <= b:
            weights[idx] = a / val
        if val > b and val <= k:
            weights[idx] = a * (k - val) / (val * (k - b))
        if val > k:
            weights[idx] = 0
    return weights.real


def trimmedMeanLocationWeights(r: np.ndarray, k: float) -> np.ndarray:
    """Trimmed mean location weights
    
    Parameters
    ----------
    r : np.ndarray
        Residuals
    k : float
        Tuning parameter

    Returns
    -------
    weights : np.ndarray
        The robust weights
    """
    weights = np.zeros(shape=r.size, dtype="complex")
    indices = np.where(np.absolute(r) <= k)
    weights[indices] = 1
    return weights.real


def andrewsWaveLocationWeights(r: np.ndarray, k: float) -> np.ndarray:
    """Andrews Wave location weights
    
    Parameters
    ----------
    r : np.ndarray
        Residuals
    k : float
        Tuning parameter

    Returns
    -------
    weights : np.ndarray
        The robust weights
    """
    weights = np.zeros(shape=r.size, dtype="complex")
    testVal = k * np.pi
    for idx, val in enumerate(np.absolute(r)):
        if val < testVal:
            weights[idx] = np.sin(val / k) / (val / k)
    return weights.real


def leastSquaresLocationWeights(r: np.ndarray):
    """Least squares weights, which are all equal to 1

    Parameters
    ----------
    r : np.ndarray
        Residuals

    Returns
    -------
    weights : np.ndarray
        The robust weights
    """
    return np.ones(shape=(r.size), dtype="complex")


def sampleMedian(data):
    """Calculate the median of an array

    Mean is not a robust estimator of locations as it can be broken by a single outlying value. The median is a more robust choice.
    
    Parameters
    ----------
    np.ndarray
        Data for which to calculate median
    
    Returns
    -------
    float
        The median
    """
    return np.median(data)


def sampleMAD(data):
    """Median absolute deviation

    The standard deviation is not robust against outliers, hence use the MAD.
    
    Parameters
    ----------
    np.ndarray
        Data for which to calculate MAD
    
    Returns
    -------
    float
        The MAD    
    """
    absData = np.absolute(data)
    mad = sampleMedian(np.absolute(absData - sampleMedian(absData)))
    return mad / 0.67448975019608171


def sampleMAD0(data):
    """Median absolute deviation using an estimate of the location as 0

    When the location estimate is zero (rather than the median), the MAD essentially reduces to a median. This should be over non zero data. Useful for calculating variance of residuals.

    Parameters
    ----------
    np.ndarray
        Data for which to calculate MAD. This is often residuals when using 0 as an estimate of location. 
    
    Returns
    -------
    float
        The MAD using zero as an esimate of location   
    """
    absData = np.absolute(data)
    inputIndices = np.where(absData != 0.0)
    mad = sampleMedian(absData[inputIndices])
    # mad = sampleMedian(np.absolute(data))
    return mad / 0.67448975019608171


def eps() -> float:
    """Small number
    
    Returns
    -------
    float
        A small number for quitting robust regression
    """
    return 0.0001


def maxIter() -> int:
    """Maximum number of iterations
    
    Returns
    -------
    int
        The maximum number of iterations
    """
    return 100
