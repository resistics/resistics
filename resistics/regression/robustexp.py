"""
These are experimental robust regression methods.
For linear regression have:
    
    - observations y
    - predictors A
    - unknown parameters (or regressors) x

In the MT world, observations (y) are the electric field measurements and predictors (A) are the input magnetic field measurements. The unknown parameters x are the components of the transfer function.

The shape of the various arrays are:

    - shape(y) = nobs
    - shape(A) = nobs * nregressors
    - shape(x) = nregressors
"""
import numpy as np
from typing import List, Dict, Tuple

from resistics.common.checks import parseKeywords


def chatterjeeMachlerMod(A, y, **kwargs):
    # using the weights in chaterjeeMachler means that min resids val in median(resids)
    # instead, use M estimate weights with a modified residual which includes a measure of leverage
    # for this, use residuals / (1-p)^2
    # I wonder if this will have a divide by zero bug
    from resistics.common.math import eps
    from resistics.regression.moments import getLocation, getScale
    from resistics.regression.weights import getWeights
    from resistics.regression.robust import defaultOptions, applyWeights, olsModel
    import numpy.linalg as linalg

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
    locP = getLocation(Pdiag, "median")
    scaleP = getScale(Pdiag, "mad")
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
    options = parseKeywords(defaultOptions(), kwargs, printkw=False)
    # generalPrint("S-Estimate", "Using weight function = {}".format(weightFnc))
    if options["intercept"] == True:
        # add column of ones for constant term
        A = np.hstack((np.ones(shape=(A.shape[0], 1), dtype="complex"), A))

    # iteratively weighted least squares
    iteration = 0
    while iteration < options["maxiter"]:
        # do the weighted least-squares
        Anew, ynew = applyWeights(A, y, weights)
        paramsNew, squareResidNew, rankNew, sNew = linalg.lstsq(Anew, ynew, rcond=None)
        residsNew = y - np.dot(A, paramsNew)
        # check residsNew to make sure not all zeros (i.e. will happen in undetermined or equally determined system)
        if np.sum(np.absolute(residsNew)) < eps():
            # then return everything here
            return paramsNew, residsNew, weights
        residsNew = residsNew / leverageMeas
        scale = getScale(residsNew, "mad0")

        # standardise and calculate weights
        residsNew = residsNew / scale
        weightsNew = getWeights(residsNew, "huber")
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
    weights = getWeights(resids, "trimmedMean")
    # iteratively weighted least squares
    iteration = 0
    while iteration < options["maxiter"]:
        # do the weighted least-squares
        Anew, ynew = applyWeights(A, y, weights)
        paramsNew, squareResidNew, rankNew, sNew = linalg.lstsq(Anew, ynew, rcond=None)
        residsNew = y - np.dot(A, paramsNew)
        # check residsNew to make sure not all zeros (i.e. will happen in undetermined or equally determined system)
        if np.sum(np.absolute(residsNew)) < eps():
            # then return everything here
            return paramsNew, residsNew, weights

        residsNew = residsNew / leverageMeas
        scale = getScale(residsNew, "mad0")

        # standardise and calculate weights
        residsNew = residsNew / scale
        weightsNew = getWeights(residsNew, options["weights"])
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
    
    Another regression method based on Hadi distances
    implemented from the paper A Re-Weighted Least Squares Method for Robust Regression Estimation
    Billor, Hadi    
    """
    import scipy.stats as stats
    from resistics.common.math import eps
    from resistics.regression.robust import defaultOptions, applyWeights
    import numpy.linalg as linalg

    # basic info
    options = parseKeywords(defaultOptions(), kwargs, printkw=False)

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
        Anew, ynew = applyWeights(X, y, weights)
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
    """Calculate the distance for the Chatterjee Machler Hadi"""
    import numpy.linalg as linalg

    inv = linalg.inv(covariance)
    dist = np.empty(shape=(n), dtype="float")
    for i in range(0, n):
        tmp = x[i, :] - mean
        dist[i] = np.sqrt(np.dot(tmp, np.dot(inv, tmp)))
    return dist
