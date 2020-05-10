import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List, Union

from resistics.common.base import ResisticsBase


class RegressionData(ResisticsBase):
    """Class for holding regression data
    
    Attributes
    ----------
    A : np.ndarray
        The predictors
    y : np.ndarray
        The observations
    params : np.ndarray, None, optional
        The model solution
    resids : np.ndarray, None, optional
        The residuals
    weights : np.ndarray, None, optional
        The weights
    scale : float
        The estimate of scale
    method : str
        The method of solution

    Methods
    -------
    __init__(A, y, params=None, resids=None, weights=None, scale=None, method=None)
        Intialise with the predictors and observations
    setModelParameters(params, resids=None, weights=None, scale=None, method=None)
        Add the model parameters
    printList()
        Class information as a list of strings        
    """

    def __init__(
        self,
        A: np.ndarray,
        y: np.ndarray,
        params: Union[np.ndarray, None] = None,
        resids: Union[np.ndarray, None] = None,
        weights: Union[np.ndarray, None] = None,
        scale: Union[float, None] = None,
        method: Union[str, None] = None,
    ) -> None:
        """Initialise with predictors and observations
        
        Paramters
        ---------
        A : np.ndarray
            The predictors
        y : np.ndarray
            The observations
        params : np.ndarray, None, optional
            The model solution
        resids : np.ndarray, None, optional
            The residuals
        weights : np.ndarray, None, optional
            The weights
        scale : float
            The estimate of scale
        method : str
            The method of solution
        """
        self.A: np.ndarray = A
        self.y: np.ndarray = y
        self.params: Union[np.ndarray, None] = params
        self.resids: Union[np.ndarray, None] = resids
        self.weights: Union[np.ndarray, None] = weights
        self.scale: Union[float, None] = scale
        self.method: Union[str, None] = method
        # calculate sizes
        self.nobservations = self.y.size
        self.npredictors = self.A.shape[1]

    @property
    def rms(self) -> float:
        """Get the frequency array of the spectra data

        Returns
        -------
        float
            The Root Mean Square error
        """
        if self.resids is None:
            return None
        return np.sqrt(np.sum(np.power(self.resids, 2)))
    
    @property
    def gaussianVariances(self) -> np.ndarray:
        """Calculate gaussian parametric variances for uncertainty estimation
        
        Returns
        -------
        np.ndarray
            The variances
        """
        from resistics.regression.robust import hermitianTranspose
        from resistics.regression.moments import getScale

        if self.params is None:
            return None
        resids = self.resids
        if resids is None:
            resids = self.y - np.dot(self.A, self.params)
        scale = self.scale
        if scale is None:
            scale = getScale(resids, "mad0")

        # calculate residual variance (standard deviation squared)
        varianceResid = scale * scale
        
        # calculate predictor variance - this is a pxp square matrix
        # should the weights be incorporated?
        # variancePredict = np.dot(hermitianTranspose(self.A), weights*self.A) 
        variancePredict = np.dot(hermitianTranspose(self.A), self.A)
        variancePredict = np.linalg.inv(variancePredict)  

        # calculate output uncertainty 
        varianceParams = 1.91472 * varianceResid * variancePredict
        # take the diagonal elements - this should be a real number
        varianceParams = np.diag(varianceParams).real 
        return varianceParams 

    def setModelParameters(
        self,
        params: np.ndarray,
        resids: Union[np.ndarray, None] = None,
        weights: Union[np.ndarray, None] = None,
        scale: Union[float, None] = None,
        method: Union[str, None] = None,
    ) -> None:
        """Set the model parameters
        
        Paramters
        ---------
        params : np.ndarray
            The model parameters
        resids : np.ndarray, None, optional
            The residuals
        weights : np.ndarray, None, optional
            The weights
        scale : float
            The estimate of scale
        method : str
            The method of solution
        """
        self.params = params
        if resids is not None:
            self.resids = resids
        if weights is not None:
            self.weights = weights
        if scale is not None:
            self.scale = scale
        if method is not None:
            self.method = method

    def printList(self):
        """Class information as a list of strings

        Returns
        -------
        out : List[str]
            List of strings with information
        """
        textLst = []
        textLst.append("Number of observations = {:d}".format(self.nobservations))
        textLst.append("Number of predictors = {:d}".format(self.npredictors))
        textLst.append("Model parameters = {}".format(self.params))
        if self.resids is not None:
            textLst.append("RMS = {:.6f}".format(self.rms))
        return textLst
