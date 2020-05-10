"""
The source for these functions is Robust Statisitics, Huber, 2009.
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
from typing import List, Dict, Tuple, Any

from resistics.common.checks import parseKeywords


def defaultOptions() -> Dict:
    """Robust regression defaults
    
    Returns
    -------
    Dict
        Default regression options
    """
    outDict = {}
    outDict["weights"] = "bisquare"
    outDict["maxiter"] = maxiter()
    outDict["initial"] = False
    outDict["scale"] = False
    outDict["intercept"] = False
    return outDict


def initialModel(initial: Dict) -> Tuple:
    """Returns initial model from provided initial model dictionary
    
    This helps for two stage robust regression, where the model from the first stage becomes the initial model for the second stage.

    Parameters
    ----------
    Dict
        Initial model to use for robust regression with the parameters, residuals and scale estimate
    
    Returns
    -------
    parameters : np.ndarray
        The parameters array
    resids : np.ndarray
        The residuals
    scale : float
        Initial estimate of scale
    """
    return initial["params"], initial["resids"], initial["scale"]


def olsModel(A, y, **kwargs) -> Dict[str, Any]:
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
    RegressionData
        RegressionData instance with the parameters and residuals
    """
    from resistics.regression.data import RegressionData
    import numpy.linalg as linalg

    options = parseKeywords(defaultOptions(), kwargs, printkw=False)
    if options["intercept"]:
        # add a constant term for the intercept
        A = np.hstack((np.ones(shape=(A.shape[0], 1), dtype="complex"), A))
    params, _squareResid, _rank, _s = linalg.lstsq(A, y, rcond=None)
    resids = y - np.dot(A, params)
    return RegressionData(A, y, params=params, resids=resids)


def mestimateModel(A: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
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
    initial : Dict
        Initial model parameters and scale
    scale : optional
        A scale estimate
    intercept : bool, optional
        True or False for adding an intercept term
    weights : str, optional
        The weights to use

    Returns
    -------
    RegressionData
        RegressionData instance with the parameters, residuals, weights and scale
    """
    from resistics.common.math import eps
    from resistics.regression.moments import getLocation, getScale
    from resistics.regression.weights import getWeights
    from resistics.regression.data import RegressionData
    import numpy.linalg as linalg

    options = parseKeywords(defaultOptions(), kwargs, printkw=False)
    # calculate the leverage
    n = A.shape[0]
    p = A.shape[1]
    # calculate the projection matrix
    q, r = linalg.qr(A)
    Pdiag = np.empty(shape=(n), dtype="float")
    for ii in range(0, n):
        Pdiag[ii] = np.absolute(np.sum(q[ii, :] * np.conjugate(q[ii, :]))).real
    Pdiag = Pdiag / np.max(Pdiag)
    leverageScale = getScale(Pdiag, "mad0")
    leverageWeights = getWeights(Pdiag / leverageScale, "huber")

    if options["intercept"] == True:
        # add column of ones for constant term
        A = np.hstack((np.ones(shape=(A.shape[0], 1), dtype="complex"), A))

    # see whether to do an initial OLS model or whether one is provided
    if options["initial"]:
        params, resids, scale = initialModel(options["initial"])
    else:
        soln = olsModel(A, y)
        resids = soln.resids
        scale = getScale(resids, "mad0")

    # if an initial model was not provided but an initial scale was, replace the one here
    if options["scale"]:
        scale = options["scale"]

    # standardised residuals and weights
    weights = getWeights(resids / scale, options["weights"]) * leverageWeights
    # iteratively weighted least squares
    iteration = 0
    while iteration < options["maxiter"]:
        # do the weighted least-squares
        Anew, ynew = applyWeights(A, y, weights)
        paramsNew, _squareResidNew, _rankNew, _sNew = linalg.lstsq(
            Anew, ynew, rcond=None
        )
        residsNew = y - np.dot(A, paramsNew)

        if np.sum(np.absolute(residsNew)) < eps():
            return RegressionData(
                A, y, params=paramsNew, resids=residsNew, scale=scale, weights=weights
            )

        # standardise and calculate weights
        scale = getScale(residsNew, "mad0")
        weightsNew = getWeights(residsNew / scale, options["weights"]) * leverageWeights
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

    return RegressionData(
        A, y, params=params, resids=resids, scale=scale, weights=weights
    )


def mmestimateModel(A: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
    r"""Two stage M estimate

    The two stage M estimate uses an initial mestimate with huber weights to give a measure of scale. A second M estimate is then performed using the calculated measure of scale. The second stage M estimate uses bisquare weights unless otherwise specified.

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
    RegressionData
        RegressionData instance with the parameters, residuals, weights and scale
    """
    from resistics.regression.moments import getScale
    import numpy.linalg as linalg

    options = parseKeywords(defaultOptions(), kwargs, printkw=False)
    intercept = options["intercept"]

    if "initial" in kwargs:
        # an initial solution is provided
        if "scale" not in kwargs["initial"]:
            kwargs["initial"]["scale"] = getScale(kwargs["initial"]["resids"], "mad0")
        soln1 = mestimateModel(
            A, y, weights="huber", initial=kwargs["initial"], intercept=intercept
        )
        # update the scale in the initial solution and perform another mestimate
        kwargs["initial"]["scale"] = soln1.scale
        # now do another, but with a different weighting function
        soln2 = mestimateModel(
            A, y, weights="bisquare", initial=kwargs["initial"], intercept=intercept
        )
    else:
        # no initial solution, calculate one
        soln1 = mestimateModel(A, y, weights="huber", intercept=intercept)
        # now do another, but with a different weighting function
        soln2 = mestimateModel(
            A, y, weights="bisquare", scale=soln1.scale, intercept=intercept
        )

    return soln2


def chatterjeeMachler(A: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
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
    Dict[str, Any]
        Dictionary with keys params, resids, resids2 and weights corresponding to parameters (solution), residuals, squared residuals and weights used in the weighted least squares respectively.    
    """
    from resistics.common.math import eps
    from resistics.regression.data import RegressionData
    import numpy.linalg as linalg

    options = parseKeywords(defaultOptions(), kwargs, printkw=False)
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
        Anew, ynew = applyWeights(A, y, weights)
        paramsNew, _squareResidNew, _rankNew, _sNew = linalg.lstsq(
            Anew, ynew, rcond=None
        )
        residsNew = y - np.dot(A, paramsNew)

        if np.sum(np.absolute(residsNew)) < eps():
            return RegressionData(
                A, y, params=paramsNew, resids=residsNew, weights=weights
            )

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

    return RegressionData(A, y, params=params, resids=resids, weights=weights)


def applyWeights(
    A: np.ndarray, y: np.ndarray, weights: np.ndarray
) -> Tuple[np.ndarray]:
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


def maxiter() -> int:
    """Maximum number of iterations
    
    Returns
    -------
    int
        The maximum number of iterations
    """
    return 100
