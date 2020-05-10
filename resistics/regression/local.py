import os
import random
import numpy as np
import scipy.interpolate as interp
from typing import List, Dict, Tuple, Union

from resistics.common.base import ResisticsBase
from resistics.spectra.data import PowerData
from resistics.window.selector import WindowSelector
from resistics.transfunc.data import TransferFunctionData
from resistics.transfunc.io import TransferFunctionWriter
from resistics.common.io import checkAndMakeDir, fileFormatSampleFreq
from resistics.regression.robust import (
    olsModel,
    chatterjeeMachler,
    mmestimateModel,
)


class LocalRegressor(ResisticsBase):
    """Performs single site (or intersite) transfer function calculations 

    By default, the LocalRegression is setup to calculate the impedance tensor using Hx, Hy as input channels and Ex, Ey as output channels. To calculate the Tipper, the appropriate input and output channels have to be set.

    Attributes
    ----------
    winSelector : WindowSelector
        A WindowSelector instance which will be used to help read in the correct data
    decParams : DecimationParameters
        A DecimationParameters instance defining the decimation scheme
    winParams : WindowParameters
        A WindowParameters instance with the windowing parameters
    outpath : str
        The path to write the transfer function data out to
    ncores : int
        The number of cores to use
    method : str
        The solution method
    inSite : str
        The input site
    inChannels : List[str]
        The input channels
    inSize : int 
        The number of input channels
    inCross : List[str] 
        The channels to use from the input site to calculate crosspowers
    outSite: str = "dummy"
        The output site
    outChannels : List[str] 
        The output channels
    outSize : int
        The number of output channels
    outCross : List[str]
        The channels to use from the output site to calculate crosspowers 
    intercept : bool
        Flag for adding an intercept term
    mmOptions : Union[Dict, None]
        Options from MM estimates
    cmOptions : Union[Dict, None]
        Options for Chaterjee Machler solutions
    olsOptions : Union[Dict, None]
        Options for ordinary least squares
    smoothFunc : str
        The window function to use
    smoothLen : Union[int, None]
        The smooth length to use. If None, will calculate automatically.
    evalFreq : List
        The calculated evaluation frequencies
    parameters : List
        The parameters calculated for each evaluation frequency
    variances : List
        The variances calculated for each evaluation frequency
    postpend : str
        The postpend string to add to the transfer function output file

    Methods
    -------
    __init__(proj, winSelector, outpath)
        Initialise with a window selector and the path to write the result out to
    defaultInCross()
        The default cross channels to use from the input site
    defaultOutCross()
        The default cross channels to use from the output site
    setCores(ncores)
        Set the number of cores to use
    setMethod(method)
        Set the solution method to use
    setSmooth(smoothFunc, smoothLen)
        The smoothing window function and length in number of samples
    setInput(inSite, inChannels, inCross)
        Set the input site, the input site channels and input site cross power channels
    setOutput(outSite, outChannels, outCross)
        Set the output site, the output site channels and output site cross power channels
    process()
        Process the spectra to calculate the transfer function
    processLevel(declevel)
        Process a single decimation level
    processBatches(declevel, unmaskedWindows)
        Process the batches for a decimation level
    processBatch(batch, batchedWindows, evalFreq, smoothLen)
        Process a single batch of spectra data
    processFrequency(declevel, eIdx, crosspowerData, global2local)
        Process the crosspower data for an individual evaluation frequency
    getSmoothLen(dataSize)
        Get the window smooth length given a the number of frequency samples
    checkForBadValues(crosspowerData)
        Check the spectral data for bad values that might cause an error
    getCrossSize()
        Get the number of channels to calculate cross spectra for   
    prepareLinearEqn(crosspowerData, eIdx)
        Prepare regressors and observations for an evaluation frequency from crosspowers data
    solve(numWindows, obs, reg)
        Solve the linear problem
    olsSolve(numWindows, obs, reg)
        Solve the linear problem using ordinary least squares
    mmSolve(numWindows, obs, reg)
        Solve the linear problem using MM estimates
    cmSolve(numWindows, obs, reg)
        Solve the linear problem using the Chaterjee Machler method
    expSolve(numWindows, obs, reg)   
        Experimental solve for testing new methods. Not currently implemented.
    writeResult(specdir, postpend, freq, data, variances, **kwargs)
        Write out the transfer function
    printList()
        Class status returned as list of strings
    """

    def __init__(self, winSelector: WindowSelector, outpath: str) -> None:
        """Intialise the LocalRegressor

        Parameters
        ----------
        winSelector : WindowSelector
            A window selector instance
        outpath : str
            The path to write the transfer function data to
        """
        self.winSelector: WindowSelector = winSelector
        self.decParams = winSelector.decParams
        self.winParams = winSelector.winParams
        self.outpath: str = outpath
        # configuration parameters
        self.ncores = 0
        self.method: str = "cm"
        # input site options
        self.inSite: str = "dummy"
        self.inChannels: List[str] = ["Hx", "Hy"]
        self.inSize: int = len(self.inChannels)
        self.inCross: List[str] = self.defaultInCross(self.inChannels)
        # output site options
        self.outSite: str = "dummy"
        self.outChannels: List[str] = ["Ex", "Ey"]
        self.outSize: int = len(self.outChannels)
        self.outCross: List[str] = self.defaultOutCross(self.outChannels)
        # solution options
        self.intercept: bool = False
        self.stack: bool = False
        self.mmOptions: Union[None, Dict] = None
        self.cmOptions: Union[None, Dict] = None
        self.olsOptions: Union[None, Dict] = None
        # smoothing options
        self.smoothFunc: str = "hann"
        self.smoothLen: Union[int, None] = None
        # attributes to store transfer function data
        self.evalFreq: List[float] = []
        self.parameters: List[np.ndarray] = []
        self.variances: List[np.ndarray] = []
        # output filename
        self.postpend: str = ""

    def defaultInCross(self, inChannels: List[str]) -> List[str]:
        """Returning the default cross channels for the input site
        
        Parameters
        ----------
        inChannels : List[str]
            The input channels
        
        Returns
        -------
        inCross : List[str]
            The cross channels for the input site
        """
        return inChannels

    def defaultOutCross(self, outChannels: List[str]) -> List[str]:
        """Returning the default cross channels for the output site
        
        Parameters
        ----------
        outChannels : List[str]
            The output channels
        
        Returns
        -------
        outCross : List[str]
            The cross channels for the output site
        """
        return outChannels

    def setCores(self, ncores: int = 0) -> None:
        """Set the number of cores to use
        
        Parameters
        ----------
        ncores : int
            The number of cores to use. Best to leave some cores spare.
        """
        self.ncores = ncores

    def setMethod(
        self, method: str, intercept: bool = False, stack: bool = False, **kwargs
    ) -> None:
        """Set the processing method
        
        Parameters
        ----------
        method : str
            The processing method to use
        intercept : bool
            Whether to add an intercept term or not. Default False.
        stack : bool
            Whether to stack the data or not. Default False.
        """
        self.method = method
        self.intercept = intercept
        self.stack = stack
        # particular solver options
        if "mm" in kwargs:
            self.mmOptions = kwargs["mm"]
        if "cm" in kwargs:
            self.cmOptions = kwargs["cm"]
        if "ols" in kwargs:
            self.cmOptions = kwargs["ols"]

    def setSmooth(self, smoothFunc: str, smoothLen: int) -> None:
        """Set the smoothing parameters
        
        Parameters
        ----------
        smoothFunc : str
            The smoothing function
        smoothLen : int
            The smoothing length. Set to 0 to automatically calculate a smoothing length from the data size
        """
        self.smoothFunc = smoothFunc
        self.smoothLen = smoothLen

    def setInput(
        self, inSite: str, inChannels: List[str], inCross: Union[List[str], None] = None
    ) -> None:
        """Set information about input site and channels
    
        Parameters
        ----------
        inSite : str
            Site to use for input channel data
        inChannels : List[str]
            Channels to use as the input in the linear system
        inCross : List[str]
            The channels from the input site to use for cross spectra
        """
        self.inSite = inSite
        self.inChannels = inChannels
        self.inSize = len(inChannels)
        self.inCross = inCross
        if self.inCross is None:
            self.inCross = self.defaultInCross(self.inChannels)

    def setOutput(
        self,
        outSite: str,
        outChannels: List[str],
        outCross: Union[List[str], None] = None,
    ) -> None:
        """Set information about output site and channels
    
        Parameters
        ----------
        outSite : str
            Site to use for output channel data
        outChannels : List[str]
            Channels to use as the output in the linear system
        outCross : List[str]
            The channels from the output site to use for cross spectra
        """
        self.outSite = outSite
        self.outChannels = outChannels
        self.outSize = len(outChannels)
        self.outCross = outCross
        if self.outCross is None:
            self.outCross = self.defaultOutCross(self.outChannels)

    def process(self) -> None:
        """Process spectra data

        Process each decimation level and write out the transfer function.

        Parameters
        ----------
        ncores : int
            The number of cores to run the cross spectra calculations on. Default 0, single core  
        """
        numLevels: int = self.decParams.numLevels
        for declevel in range(0, numLevels):
            self.printBreak()
            self.printText("Processing decimation level {}".format(declevel))
            self.processLevel(declevel)
        self.printBreak()
        # check to make sure some data was processed
        if len(self.evalFreq) == 0:
            self.printWarning(
                "No data was found at any decimation level for insite {}, outsite {} and specdir {}".format(
                    self.inSite, self.outSite, self.winSelector.specdir
                )
            )
            return

        self.writeResult(
            self.winSelector.specdir,
            self.postpend,
            self.evalFreq,
            self.parameters,
            self.variances,
        )

    def processLevel(self, declevel: int) -> None:
        """Process a decimation level
        
        The processing sequence for each decimation level is as below:

        1. Get shared (unmasked) windows for all relevant sites (inSite and outSite)
        2. For shared unmasked windows
            
            - Calculate out the cross-power spectra.
            - Interpolate calculated cross-power data to the evaluation frequencies for the decimation level.
        
        3. For each evaluation frequency
            
            - Do the robust processing to calculate the transfer function at that evaluation frequency.

        The spectral power data is smoothed as this tends to improve results. The smoothing can be changed by setting the smoothing parameters. This method is still subject to change in the future as it is an area of active work

        Parameters
        ----------
        declevel : int
            The decimation level    
        """
        numWindows = self.winSelector.getNumSharedWindows(declevel)
        unmaskedWindows = self.winSelector.getUnmaskedWindowsLevel(declevel)
        numUnmasked = len(unmaskedWindows)
        self.printText(
            "Total shared windows for decimation level = {}".format(numWindows)
        )
        self.printText(
            "Total unmasked windows for decimation level = {}".format(numUnmasked)
        )
        if numUnmasked == 0:
            self.printText(
                "No unmasked windows found at this decimation level ({:d}), continuing to next level".format(
                    declevel
                )
            )
            return None

        # process batches and collect cross spectra data
        self.printText("{} windows will be processed".format(numUnmasked))
        crosspowerData, global2local = self.processBatches(declevel, unmaskedWindows)

        # process the individual evaluation frequencies
        evalFreq = self.decParams.getEvalFrequenciesForLevel(declevel)
        for eIdx in range(0, len(evalFreq)):
            self.printText(
                "Processing evaluation frequency = {:.6f} [Hz], period = {:.6f} [s]".format(
                    evalFreq[eIdx], 1 / evalFreq[eIdx]
                )
            )
            self.processFrequency(declevel, eIdx, crosspowerData, global2local)

    def processBatches(
        self, declevel: int, unmaskedWindows: set
    ) -> Tuple[np.ndarray, Dict[int, int]]:
        """Process the spectra batches
        
        Parameters
        ----------
        declevel : int
            The decimation level
        unmaskedWindows : set
            The unmasked windows for the decimation level
        
        Returns
        -------
        crosspowerData : np.ndarray[PowerData]
            A numpy array of crosspower data
        global2local : Dict[int, int]
            Global to local window map
        """
        evalFreq = self.decParams.getEvalFrequenciesForLevel(declevel)
        numUnmasked = len(unmaskedWindows)
        dataSize = self.winSelector.getDataSize(declevel)
        smoothLen = self.getSmoothLen(dataSize)
        crosspowerData: np.ndarray = np.empty(shape=(numUnmasked), dtype=PowerData)
        # global to local map to help choose windows for each evaluation frequency
        localWin: int = 0
        global2local: Dict = {}
        # process spectral batches
        spectraBatches = self.winSelector.getSpecReaderBatches(declevel)
        numBatches = len(spectraBatches)
        for batchIdx, batch in enumerate(spectraBatches):
            # find the unmasked batched windows and add the data
            batchedWindows = unmaskedWindows.intersection(
                set(range(batch["globalrange"][0], batch["globalrange"][1] + 1))
            )
            self.printText(
                "Processing batch {:d} of {:d}: Global window range {:d} to {:d}, {:.3%} of data".format(
                    batchIdx + 1,
                    numBatches,
                    batch["globalrange"][0],
                    batch["globalrange"][1],
                    len(batchedWindows) / numUnmasked,
                )
            )
            if len(batchedWindows) == 0:
                self.printText(
                    "Batch does not contribute any windows. Continuing to next batch."
                )
                continue
            # process the batch, returns the crosspowers and updated and correctly ordered batchedWindows
            crosspowers, batchedWindows = self.processBatch(
                batch, batchedWindows, evalFreq, smoothLen
            )
            for batchWin, globalWin in enumerate(batchedWindows):
                crosspowerData[localWin] = crosspowers[batchWin]
                # local to global map and increment local window
                global2local[globalWin] = localWin
                localWin += 1

        # close spectra files for decimation level
        for batch in spectraBatches:
            for site in self.winSelector.sites:
                batch[site].closeFile()

        return crosspowerData, global2local

    def processBatch(
        self, batch: Dict, batchedWindows: set, evalFreq: np.ndarray, smoothLen: int
    ) -> Tuple[List[PowerData], np.ndarray]:
        """Process a single batch
        
        Parameters
        ----------
        batch : Dict
            The batch information (about the spectra readers, the global window range etc)
        batchedWindws : set
            The windows in the batch. This is a set, so unsorted. A sorted version with correct order gets returned by the function
        evalFreq : np.ndarray
            The evaluation frequencies
        smoothLen : int
            The smoothing length for the crosspower data

        Returns
        -------
        crosspowers : List[PowerData]
            List of PowerData with the crosspowers
        batchedWindows : np.ndarray
            The correctly ordered windows for the batch
        """
        from resistics.regression.crosspowers import localCrosspowers

        # collect spectrum data and set batchedWindows to ensure proper order
        inReader = batch[self.inSite]
        inData, batchedWindows = inReader.readBinaryBatchGlobal(
            globalIndices=batchedWindows
        )
        if self.outSite != self.inSite:
            outReader = batch[self.outSite]
            outData, _gIndicesOut = outReader.readBinaryBatchGlobal(
                globalIndices=batchedWindows
            )
        else:
            outData = inData

        self.printText("Calculating batch crosspowers...")
        crosspowers: List[PowerData] = localCrosspowers(
            self.ncores,
            inData,
            self.inChannels,
            self.inCross,
            outData,
            self.outChannels,
            self.outCross,
            smoothLen,
            self.smoothFunc,
            evalFreq,
        )
        self.printText("Calculation of crosspowers for batch complete")
        return crosspowers, batchedWindows

    def processFrequency(
        self,
        declevel: int,
        eIdx: int,
        crosspowerData: np.ndarray,
        global2local: Dict[int, int],
    ) -> None:
        """Process an evaluation frequency
        
        Parameters
        ----------
        declevel : int
            The decimation level
        eIdx : int
            The evaluation frequency index
        crosspowerData : np.ndarray[PowerData]
            A numpy array of PowerData
        global2local : Dict[int, int]
            The global to local window map        
        """
        # get the constrained windows for the evaluation frequency
        evalFreqWindows = self.winSelector.getWindowsForFreq(declevel, eIdx)
        if len(evalFreqWindows) == 0:
            self.printText("No windows found - possibly due to masking")
            return None
        localWinIndices = []
        for iWin in evalFreqWindows:
            localWinIndices.append(global2local[iWin])

        self.printText("{:d} windows will be solved for".format(len(localWinIndices)))
        # restrict processing to data that meets constraints for this evaluation frequency
        numSolveWindows, obs, reg = self.prepareLinearEqn(
            crosspowerData[localWinIndices], eIdx
        )
        outputParams, outputVars = self.solve(numSolveWindows, obs, reg)
        # save evaluation frequency and calculated transfer function and variances
        evalFreq = self.decParams.getEvalFrequenciesForLevel(declevel)
        self.evalFreq.append(evalFreq[eIdx])
        self.parameters.append(outputParams)
        self.variances.append(outputVars)

    def getSmoothLen(self, datasize):
        """Window smoothing length

        Power spectra data is smoothed. This returns the size of the smoothing window.
    
        Parameters
        ----------
        datasize : int
            The number of frequency points

        Returns
        -------
        smoothLen : int
            Smoothing size
        """
        if (self.smoothLen is not None) and (self.smoothLen) >= 1:
            if self.smoothLen % 2 == 0:
                self.printWarning(
                    "Smoothing window should be odd, changed from {:d} to {:d}".format(
                        self.smoothLen, self.smoothLen + 1
                    )
                )
                self.smoothLen += 1
            return self.smoothLen

        # calculate based on datasize
        winSmooth = datasize * 1.0 / 16.0
        if winSmooth < 3:
            return 3
        # otherwise round to nearest odd number
        winSmooth = np.ceil(winSmooth) // 2
        return int(winSmooth * 2 + 1)

    def checkForBadValues(self, crosspowerData: np.ndarray):
        """Check data for bad values and remove
        
        Parameters
        ----------
        crosspowerData : np.ndarray[PowerData] 
            numpy array of PowerData

        Returns
        -------
        numGoodWindows : int
            The number of good windows
        goodData : np.ndarray[PowerData]
            The cross-spectra data with bad windows removed
        """
        numWindows = crosspowerData.size
        finiteArray = np.ones(shape=(numWindows))
        for iWin, winPower in enumerate(crosspowerData):
            if not winPower.isFinite():
                finiteArray[iWin] = 0
        numGoodWindows = sum(finiteArray)
        if numGoodWindows == numWindows:
            return numWindows, crosspowerData
        self.printWarning(
            "Bad data found...number of windows reduced from {} to {}".format(
                numWindows, numGoodWindows
            )
        )
        goodWindowIndices = np.where(finiteArray == 1)
        return numGoodWindows, crosspowerData[goodWindowIndices]

    def getCrossSize(self):
        """This essentially returns the number of equations per window

        The total number of cross channels that have been used in calculating crosspowers

        Returns
        -------
        crossSize : int
            The total number of cross channels used for crosspowers
        """
        return len(self.inCross) + len(self.outCross)

    def prepareLinearEqn(
        self, crosspowerData: np.ndarray, eIdx: int
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        r"""Prepare data as a linear equation for the robust regression

        This prepares the data for the following type of solution,

        .. math::
            y = Ax,

        where :math:`y` is the observations, :math:`A` is the regressors and :math:`x` is the unknown. 

        The number of observations is number of windows * number of cross-power channels
        The shapes of the arrays are as follows:
        
            - y is (number of output channels, number of observations)
            - A is (number of output channels, number of observations, number of input channels)
            - x is (number of output channels, number of input channels)

        Consider the impedance tensor,

        .. math::
            :nowrap:

            \begin{eqnarray}
            E_x & = & Z_{xx} H_x + Z_{xy} H_y \\
            E_y & = & Z_{yx} H_x + Z_{yy} H_y 
            \end{eqnarray}  

        Here, there are two input channels, :math:`H_x`, :math:`H_y` and two output channels :math:`E_x` and :math:`E_y`. In total, there are four components of the unknown impedance tensor, :math:`Z_{xx}`, :math:`Z_{xy}`, :math:`Z_{yx}`, :math:`Z_{yy}` (number of input channels * number of output channels). The number of observations is the number of windows multiplied by the number of channels used for cross-power spectra.     

        Parameters
        ----------
        crosspowerData : np.ndarray
            Cross-power spectral data at evaluation frequencies
        eIdex : int
            The evaluation frequency index (for the decimation level)

        Returns
        -------
        numWindows : int
            The number of windows included in the regression (after bad value removal)
        obs : np.ndarray
            Observations array
        reg : np.ndarray 
            Regressors array
        """
        numWindows = crosspowerData.size
        # remove windows with bad values
        numWindows, crosspowerData = self.checkForBadValues(crosspowerData)
        # calculate out the number of equations per window per output variable
        numInCross = len(self.inCross)
        crossSize = self.getCrossSize()
        # add In and Out to the required channels
        inChans = [inChan + "In" for inChan in self.inChannels]
        inCross = [inChan + "In" for inChan in self.inCross]
        outChans = [outChan + "Out" for outChan in self.outChannels]
        outCross = [outChan + "Out" for outChan in self.outCross]
        # construct our arrays
        obs = np.empty(shape=(self.outSize, crossSize * numWindows), dtype="complex")
        reg = np.empty(
            shape=(self.outSize, crossSize * numWindows, self.inSize), dtype="complex"
        )
        for iWin, winPower in enumerate(crosspowerData):
            iOffset = iWin * crossSize
            for iOut, outChan in enumerate(outChans):
                # add in all the observations and regressors for this window
                # the cross channels from the input data
                for iC, xC in enumerate(inCross):
                    obs[iOut, iOffset + iC] = winPower.getPower(outChan, xC, eIdx)
                    for iIn, inChan in enumerate(inChans):
                        reg[iOut, iOffset + iC, iIn] = winPower.getPower(
                            inChan, xC, eIdx
                        )
                # the cross channels from the output data
                for iC, xC in enumerate(outCross):
                    iC += numInCross
                    obs[iOut, iOffset + iC] = winPower.getPower(outChan, xC, eIdx)
                    for iIn, inChan in enumerate(inChans):
                        reg[iOut, iOffset + iC, iIn] = winPower.getPower(
                            inChan, xC, eIdx
                        )
        return numWindows, obs, reg

    def solve(
        self, numWindows: int, obs: np.ndarray, reg: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve the linear equation

        Parameters
        ----------
        numWindows : int
            The number of windows
        obs : np.ndarray
            The observations
        reg : np.ndarray
            The regressors

        Returns
        -------
        outputParams : np.ndarray
            The solution to the regression problem
        outputVariances : np.ndarray
            The variances for the solution   
        """
        if self.method == "ols":
            return self.olsSolve(numWindows, obs, reg)
        elif self.method == "mm":
            return self.mmSolve(numWindows, obs, reg)
        elif self.method == "cm":
            return self.cmSolve(numWindows, obs, reg)
        elif self.method == "exp":
            return self.expSolve(numWindows, obs, reg)
        else:
            return self.cmSolve(numWindows, obs, reg)

    def olsSolve(
        self, numWindows: int, obs: np.ndarray, reg: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Ordinary least squares regression processing

        Perform ordinary least regression processing using observations and regressors for a single evaluation frequency. 

        Parameters
        ----------
        numWindows : int
            The number of windows
        obs : np.ndarray
            The observations
        reg : np.ndarray
            The regressors

        Returns
        -------
        outputParams : np.ndarray
            The solution to the regression problem
        outputVariances : np.ndarray
            The variances for the solution       
        """
        # create array for parameters
        outputParams = np.empty(shape=(self.outSize, self.inSize), dtype="complex")
        outputVars = np.empty(shape=(self.outSize, self.inSize), dtype="float")
        # solve
        for i in range(0, self.outSize):
            observation = obs[i, :]
            predictors = reg[i, :, :]
            # save the output
            soln = olsModel(predictors, observation, intercept=self.intercept)
            params = soln.params
            variances = soln.gaussianVariances
            if self.intercept:
                params = params[1:]
                variances = variances[1:]
            outputParams[i] = params
            outputVars[i] = variances
        return outputParams, outputVars

    def mmSolve(
        self, numWindows: int, obs: np.ndarray, reg: np.ndarray
    ) -> Tuple[np.ndarray]:
        """Robust regression processing

        Perform robust regression processing using mmestimates. Initially does an mm solution, stacks the data given the weights and does another mm solution on the stacked data. 

        Parameters
        ----------
        numWindows : int
            The number of windows
        obs : np.ndarray
            The observations
        reg : np.ndarray
            The regressors

        Returns
        -------
        outputParams : np.ndarray
            The solution to the regression problem
        outputVars : np.ndarray
            The variances for the solution   
        """
        from resistics.regression.data import RegressionData

        crossSize = self.getCrossSize()
        # create array for output
        outputParams = np.empty(shape=(self.outSize, self.inSize), dtype="complex")
        outputVars = np.empty(shape=(self.outSize, self.inSize), dtype="float")
        # solve
        for ii in range(0, self.outSize):
            observation = obs[ii, :]
            predictors = reg[ii, :, :]
            # initial solution
            soln1 = mmestimateModel(predictors, observation, intercept=self.intercept)
            observation2 = np.zeros(shape=(crossSize), dtype="complex")
            predictors2 = np.zeros(shape=(crossSize, self.inSize), dtype="complex")
            for iChan in range(0, crossSize):
                # now need to have my indexing array
                indexArray = np.arange(iChan, numWindows * crossSize, crossSize)
                weightsLim = soln1.weights[indexArray]
                # stack observations
                observation2[iChan] = (
                    np.sum(obs[ii, indexArray] * weightsLim) / numWindows
                )
                # stack regressors
                for j in range(0, self.inSize):
                    predictors2[iChan, j] = (
                        np.sum(reg[ii, indexArray, j] * weightsLim) / numWindows
                    )
            # second solution on stacked data
            soln2 = mmestimateModel(predictors2, observation2, intercept=self.intercept)
            params = soln2.params
            # calulcate variances with respect to full equations, not stacked equations
            # use soln1 A and y to make sure intecept is incorporated
            variances = RegressionData(soln1.A, soln1.y, params).gaussianVariances
            if self.intercept:
                params = soln2.params[1:]
                variances = variances[1:]
            outputParams[ii] = params
            outputVars[ii] = variances
        return outputParams, outputVars

    def cmSolve(
        self, numWindows: int, obs: np.ndarray, reg: np.ndarray
    ) -> Tuple[np.ndarray]:
        """Robust regression processing

        Perform robust regression processing using Chaterjee Machler. Initially does an Chaterjee Machler solution, stacks the data given the weights and does another Chaterjee Machler solution on the stacked data.  

        Parameters
        ----------
        numWindows : int
            The number of windows
        obs : np.ndarray
            The observations
        reg : np.ndarray
            The regressors

        Returns
        -------
        outputParams : np.ndarray
            The solution to the regression problem
        outputVars : np.ndarray
            The variances for the solution   
        """
        from resistics.regression.data import RegressionData

        crossSize = self.getCrossSize()
        # create array for output
        outputParams = np.empty(shape=(self.outSize, self.inSize), dtype="complex")
        outputVars = np.empty(shape=(self.outSize, self.inSize), dtype="float")
        # solve
        for ii in range(0, self.outSize):
            observation = obs[ii, :]
            predictors = reg[ii, :, :]
            # save the output
            soln1 = chatterjeeMachler(predictors, observation, intercept=self.intercept)
            observation2 = np.zeros(shape=(crossSize), dtype="complex")
            predictors2 = np.zeros(shape=(crossSize, self.inSize), dtype="complex")
            for iChan in range(0, crossSize):
                # now need to have my indexing array
                indexArray = np.arange(iChan, numWindows * crossSize, crossSize)
                weightsLim = soln1.weights[indexArray]
                # stack observations
                observation2[iChan] = (
                    np.sum(obs[ii, indexArray] * weightsLim) / numWindows
                )
                # stack regressors
                for j in range(0, self.inSize):
                    predictors2[iChan, j] = (
                        np.sum(reg[ii, indexArray, j] * weightsLim) / numWindows
                    )
            # second solution on stacked data
            soln2 = chatterjeeMachler(
                predictors2, observation2, intercept=self.intercept
            )
            params = soln2.params
            # calulcate variances with respect to full equations, not stacked equations
            # use soln1 A and y to make sure intecept is incorporated
            variances = RegressionData(soln1.A, soln1.y, params).gaussianVariances
            if self.intercept:
                params = soln2.params[1:]
                variances = variances[1:]
            outputParams[ii] = params
            outputVars[ii] = variances
        return outputParams, outputVars

    def expSolve(
        self, numWindows: int, obs: np.ndarray, reg: np.ndarray
    ) -> Tuple[np.ndarray]:
        """Robust regression processing

        Perform robust regression processing using observations and regressors for a single evaluation frequency. 

        Parameters
        ----------
        numWindows : int
            The number of windows
        obs : np.ndarray
            The observations
        reg : np.ndarray
            The regressors

        Returns
        -------
        output : np.ndarray
            The solution to the regression problem
        varOutput : np.ndarray
            The variance
        """
        raise NotImplementedError

    def writeResult(
        self,
        specdir: str,
        postpend: str,
        freq,
        data: np.ndarray,
        variances: np.ndarray,
        **kwargs
    ):
        """Write the transfer function file

        Parameters
        ----------
        specdir : str
            The spectra data being used for the transfer function estimate
        postpend : str
            The optional postpend to the transfer function file
        data : np.ndarray
            The transfer function estimates
        variances : np.ndarray
            The transfer function variances
        remotesite : str, optional
            Optionally, if there is a remote site
        remotechans : List[str], optional
            Optionally add the remote channels if there is a remote site
        """
        # path for writing out to
        sampleFreqStr = fileFormatSampleFreq(self.decParams.sampleFreq)
        if postpend == "":
            filename = "{}_fs{:s}_{}".format(self.outSite, sampleFreqStr, specdir)
        else:
            filename = "{}_fs{:s}_{}_{}".format(
                self.outSite, sampleFreqStr, specdir, postpend
            )
        datapath = os.path.join(self.outpath, sampleFreqStr)
        checkAndMakeDir(datapath)
        outfile = os.path.join(datapath, filename)
        # now construct the transferFunctionData object
        numFreq = len(freq)
        dataDict = {}
        varDict = {}
        for i in range(0, self.outSize):
            for j in range(0, self.inSize):
                key = "{}{}".format(self.outChannels[i], self.inChannels[j])
                dataArray = np.empty(shape=(numFreq), dtype="complex")
                varArray = np.empty(shape=(len(freq)), dtype="float")
                for ifreq in range(0, numFreq):
                    dataArray[ifreq] = data[ifreq][i, j]
                    varArray[ifreq] = variances[ifreq][i, j]
                dataDict[key] = dataArray
                varDict[key] = varArray
        tfData = TransferFunctionData(freq, dataDict, varDict)
        # now make the writer and write out
        tfWriter = TransferFunctionWriter(outfile, tfData)
        tfWriter.setHeaders(
            sampleFreq=self.decParams.sampleFreq,
            insite=self.inSite,
            inchans=self.inChannels,
            outsite=self.outSite,
            outchans=self.outChannels,
        )
        if "remotesite" in kwargs:
            tfWriter.addHeader("remotesite", kwargs["remotesite"])
        if "remotechans" in kwargs:
            tfWriter.addHeader("remotechans", kwargs["remotechans"])
        tfWriter.write()

    def printList(self) -> List[str]:
        """Class information as a list of strings

        Returns
        -------
        list
            List of strings with information
        """
        textLst = []
        textLst.append("Input Site = {}".format(self.inSite))
        textLst.append("Input Site Channels = {}".format(self.inChannels))
        textLst.append("Input Site Cross Channels = {}".format(self.inCross))
        textLst.append("Output Site = {}".format(self.outSite))
        textLst.append("Output Site Channels = {}".format(self.outChannels))
        textLst.append("Output Site Cross Channels = {}".format(self.outCross))
        textLst.append("Sample frequency = {:.3f}".format(self.decParams.sampleFreq))
        return textLst
