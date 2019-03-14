import os
import numpy as np
import scipy.signal as signal
import scipy.interpolate as interp
from typing import List

# import from package
from resistics.calculators.processorSingleSite import ProcessorSingleSite
from resistics.utilities.utilsSmooth import smooth1d
from resistics.utilities.utilsRobust import (
    chatterjeeMachler,
    sampleMAD0,
    hermitianTranspose,
    olsModel,
    mmestimateModel,
)


class ProcessorRemoteReference(ProcessorSingleSite):
    """Perform remote reference processing of magnetotelluric data

    Remote reference processing uses a different equation for spectral power than the single site processor. The equation that the processor uses is:

    Attributes
    ----------
    winSelector : WindowSelect
        A window selector object which defines which windows to use in the linear model
    decParams : DecimationParameters
        DecimationParameters object with information about the decimation scheme
    winParams : WindowParameters
        WindowParameters object with information about the windowing
    outpath : str 
        Location to put the calculated transfer functions (Edi files)
    inSite : str 
        The site to use for the input channels 
    inChannels: List[str] (["Hx", "Hy"])
        List of hannels to use as input channels for the linear system
    inSize : int 
        Number of input channels
    outSite : str 
        The site to use for the output channels
    outChannels : List[str] (["Ex", "Ey"])
        List of channels to use as output channels for the linear system
    outSize : int
        Number of output channels
    allChannels : List[str] 
        inChannels and outChannels combined into a single list
    crossChannels : List[str] 
        
    intercept : bool (default False)
        Flag for including an intercept (static) term in the linear system
    method : str (options, "ols", "cm") 
        String for describing what solution method to use
    win : str (default hanning)
        Window function to use in robust solution
    winSmooth : int (default -1)
        The size of the window smoother. If -1, this will be autocalculated based on data size
    postpend : str (default "")
        String to postpend to the output filename to help file management
    evalFreq : List[float] or np.ndarray
        The evaluation frequencies
    impedances : List

    variances : List

    Methods
    -------
    __init__(proj, maskData)
        Initialise with a Project instance and MaskData instance


    printList()
        Class status returned as list of strings
    """

    def __init__(self, winSelector, outpath: str):
        # required
        self.winSelector = winSelector
        self.decParams = winSelector.decParams
        self.winParams = winSelector.winParams
        self.outpath: str = outpath

        # default parameters for user options
        self.inSite: str = ""
        self.inChannels: List[str] = []
        self.outSite: str = ""
        self.outChannels: List[str] = []
        self.remoteSite: str = ""
        self.remoteChannels: List[str] = []
        # smoothing options
        self.win: str = "hanning"
        self.winSmooth: int = -1
        # intercept options
        self.intercept: bool = False
        # output filename
        self.postpend: str = ""
        # evaluation frequency data
        self.evalFreq = []
        self.impedances = []
        self.variances = []

    def setRemote(self, remoteSite: str, remoteChannels: List[str]) -> None:
        """Set information about input site and channels
    
        Parameters
        ----------
        remoteSite : str
            Site to use for input channel data
        remoteChannels : List[str]
            Channels to use as the remote reference in the linear system
        """

        self.remoteSite = remoteSite
        self.remoteChannels = remoteChannels
        self.remoteSize = len(remoteChannels)
        self.printText(
            "Remote reference set with site {} and channels {}".format(
                self.remoteSite, self.remoteChannels
            )
        )

    def process(self):
        """Process spectra data

        The processing sequence is as below:

        For each decimation level
            Get shared (unmasked) windows for all relevant sites (inSite and outSite)
            For shared unmasked windows
                Calculate out the spectral power data
                Interpolate calculated spectral power data to the evaluation frequencies for the decimation level 
            For each evaluation frequency
                Do the robust processing to calculate the transfer function at that evaluation frequency


        NOTES
        -----
        The spectral power data is smoothed as this tends to improve results. The smoothing can be changed by setting the smoothing parameters. This method is still subject to change in the future as it is an area of active work
        """

        numLevels = self.decParams.numLevels
        dataChans = self.inChannels + self.outChannels
        for iDec in range(0, numLevels):
            # print out some info
            self.printText("Processing decimation level {}".format(iDec))

            fs = self.winSelector.decParams.getSampleFreqLevel(iDec)
            # get the number of all shared windows and the number of unmasked windows
            # unmasked windows are ones that will actually be used in the calculation
            numWindows = self.winSelector.getNumSharedWindows(iDec)
            unmaskedWindows = self.winSelector.getUnmaskedWindowsLevel(iDec)
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
                        iDec
                    )
                )
                continue  # continue to next decimation level
            self.printText("{} windows will be processed".format(numUnmasked))

            # get the evaluation frequencies
            evalFreq = self.decParams.getEvalFrequenciesForLevel(iDec)
            # set some variables
            totalChans = self.inSize + self.outSize
            numEvalFreq = len(evalFreq)
            dataSize = self.winSelector.getDataSize(iDec)
            freq = np.linspace(0, fs / 2, dataSize)
            # get the window smoothing params
            smoothLen = self.getWindowSmooth(datasize=dataSize)

            # create the data array
            # for each evaluation frequency
            # keep the spectral power information for all windows
            evalFreqData = np.empty(
                shape=(numEvalFreq, numWindows, totalChans, self.remoteSize),
                dtype="complex",
            )

            # an array for the in and out channels fourier data
            winDataArray = np.empty(shape=(totalChans, dataSize), dtype="complex")
            # an array for the remote reference fourier data
            winRemoteArray = np.empty(
                shape=(self.remoteSize, dataSize), dtype="complex"
            )
            # an array for the power spectra data
            winSpectraMatrix = np.empty(
                shape=(totalChans, self.remoteSize, dataSize), dtype="complex"
            )

            # loop over shared windows
            localWin = 0
            global2local = {}
            for iWin in unmaskedWindows:
                # do the local to global map
                global2local[iWin] = localWin

                # get the window for the input site
                inSF, inReader = self.winSelector.getSpecReaderForWindow(
                    self.inSite, iDec, iWin
                )
                inData = inReader.readBinaryWindowGlobal(iWin).data
                # get the window and channels for the output site
                if self.outSite != self.inSite:
                    outSF, outReader = self.winSelector.getSpecReaderForWindow(
                        self.outSite, iDec, iWin
                    )
                    outData = outReader.readBinaryWindowGlobal(iWin).data
                else:
                    outData = inData

                # now get the remote reference data - assume this does not equal input or output
                remoteSF, remoteReader = self.winSelector.getSpecReaderForWindow(
                    self.remoteSite, iDec, iWin
                )
                remoteData = remoteReader.readBinaryWindowGlobal(iWin).data

                # get data into the right part of the arrays
                for i in range(0, self.inSize):
                    winDataArray[i] = inData[self.inChannels[i]]
                for i in range(0, self.outSize):
                    winDataArray[self.inSize + i] = outData[self.outChannels[i]]
                for i in range(0, self.remoteSize):
                    winRemoteArray[i] = remoteData[self.remoteChannels[i]]

                # and now can fill the parts of the matrix
                # recall, smooth the power spectra
                for iD, dataChan in enumerate(dataChans):
                    for iR, remoteChan in enumerate(self.remoteChannels):
                        # calculate each one, cannot use complex symmetry here
                        # cannot use conjugate symmetry like with the single site processor
                        winSpectraMatrix[iD, iR] = smooth1d(
                            winDataArray[iD] * np.conjugate(winRemoteArray[iR]),
                            smoothLen,
                            self.win,
                        )
                # after running through all windows, calculate evaluation frequencies
                # calculate frequency array
                evalFreqData[:, localWin] = self.calcEvalFrequencyData(
                    freq, evalFreq, winSpectraMatrix
                )

                # increment local window
                localWin = localWin + 1

            # now all the data has been collected
            # for each evaluation frequency, do the robust processing
            # and get the evaluation frequency data
            for eIdx in range(0, numEvalFreq):
                self.printText(
                    "Processing evaluation frequency = {:.6f} [Hz], period = {:.6f} [s]".format(
                        evalFreq[eIdx], 1.0 / evalFreq[eIdx]
                    )
                )
                # get the constrained windows for the evaluation frequency
                evalFreqWindows = self.winSelector.getWindowsForFreq(iDec, eIdx)
                if len(evalFreqWindows) == 0:  # no windows meet constraints
                    self.printText("No windows found - possibly due to masking")
                    continue
                localWinIndices = []
                for iW in evalFreqWindows:
                    localWinIndices.append(global2local[iW])
                self.printText(
                    "{:d} windows will be solved for".format(len(localWinIndices))
                )
                # restrict processing to data that meets constraints for this evaluation frequency
                # add to class vars
                self.evalFreq.append(evalFreq[eIdx])
                # use process reduced - only the input channels from the remote reference
                numSolveWindows, obs, reg = self.prepareLinearEqn(
                    evalFreqData[eIdx, localWinIndices]
                )
                out, var = self.robustProcess(numSolveWindows, obs, reg)
                self.impedances.append(out)
                self.variances.append(var)

        if len(self.evalFreq) == 0:
            self.printWarning(
                "No data was found at any decimation level for insite {}, outsite {}, remotesite {} and specdir {}".format(
                    self.inSite, self.outSite, self.remoteSite, self.winSelector.specdir
                )
            )
            return

        # write out all the data
        self.writeTF(
            self.winSelector.specdir,
            self.postpend,
            self.evalFreq,
            self.impedances,
            self.variances,
            remotesite=self.remoteSite,
            remotechans=self.remoteChannels,
        )

    def checkRemote(self):
        check = True
        check = check and self.remoteSize == self.inSize
        check = check and self.remoteChannels == self.inChannels
        return check

    # PREPARE ROUTINES
    def prepareLinearEqn(self, data):
        # prepare observations and regressors for linear processing
        numWindows = data.shape[0]
        numWindows, data = self.checkForBadValues(numWindows, data)
        # for each output variable, have ninput regressor variables
        # let's construct our arrays
        obs = np.empty(
            shape=(self.outSize, self.remoteSize * numWindows), dtype="complex"
        )
        reg = np.empty(
            shape=(self.outSize, self.remoteSize * numWindows, self.inSize),
            dtype="complex",
        )
        for iW in range(0, numWindows):
            iOffset = iW * self.remoteSize
            for i in range(0, self.outSize):
                for j in range(0, self.remoteSize):
                    # this is the observation row where,i is the observed output
                    obs[i, iOffset + j] = data[iW, self.inSize + i, j]
                    for k in range(0, self.inSize):
                        reg[i, iOffset + j, k] = data[iW, k, j]
        return numWindows, obs, reg

    # SOLVER ROUTINES
    def robustProcess(self, numWindows, obs, reg):
        # do the chatterjeeMachlerMod robust processing for a single evaluation frequency
        # create array for output
        output = np.empty(shape=(self.outSize, self.inSize), dtype="complex")
        varOutput = np.empty(shape=(self.outSize, self.inSize), dtype="float")
        # solve
        for i in range(0, self.outSize):
            observation = obs[i, :]
            predictors = reg[i, :, :]
            # save the output
            out, resids, weights = chatterjeeMachler(
                predictors, observation, intercept=self.intercept
            )
            # out, resids, scale, weights = mmestimateModel(predictors, observation, intercept=self.intercept)

            # now take the weights, apply to the observations and predictors, stack the appropriate rows and test
            observation2 = np.zeros(shape=(self.remoteSize), dtype="complex")
            predictors2 = np.zeros(
                shape=(self.remoteSize, self.inSize), dtype="complex"
            )
            for iChan in range(0, self.remoteSize):
                # now need to have my indexing array
                indexArray = np.arange(
                    iChan, numWindows * self.remoteSize, self.remoteSize
                )
                weightsLim = weights[indexArray]
                # weightsLim = weightsLim/np.sum(weightsLim) # normalise weights to 1
                observation2[iChan] = (
                    np.sum(obs[i, indexArray] * weightsLim) / numWindows
                )
                # now for the regressors
                for j in range(0, self.inSize):
                    predictors2[iChan, j] = (
                        np.sum(reg[i, indexArray, j] * weightsLim) / numWindows
                    )
            out, resids, weights = chatterjeeMachler(
                predictors2, observation2, intercept=self.intercept
            )
            # out, resids, scale, weights = mmestimateModel(predictors2, observation2, intercept=self.intercept)

            # now calculate out the varainces - have the solution out, have the weights
            # recalculate out the residuals with the final solution
            # calculate standard deviation of residuals
            # and then use chatterjee machler formula to estimate variances
            # this needs work - better to use an empirical bootstrap method, but this will do for now
            resids = np.absolute(observation - np.dot(predictors, out))
            scale = sampleMAD0(
                resids
            )  # some measure of standard deviation, rather than using the standard deviation
            residsVar = scale * scale
            varPred = np.dot(hermitianTranspose(predictors), weights * predictors)
            varPred = np.linalg.inv(varPred)  # this is a pxp matrix
            varOut = 1.91472 * residsVar * varPred
            varOut = np.diag(varOut).real  # this should be a real number

            if self.intercept:
                output[i] = out[1:]
                varOutput[i] = varOut[1:]
            else:
                output[i] = out
                varOutput[i] = varOut

        return output, varOutput

    def olsProcess(self, numWindows, obs, reg):
        # ordinary least squares solution
        # create array for output
        output = np.empty(shape=(self.outSize, self.inSize), dtype="complex")
        # solve
        for i in range(0, self.outSize):
            observation = obs[i, :]
            predictors = reg[i, :, :]
            # save the output
            out, resids, squareResid, rank, s = olsModel(
                predictors, observation, intercept=self.intercept
            )
            if self.intercept:
                output[i] = out[1:]
            else:
                output[i] = out
        return output

    def stackedProcess(self, data):
        # then do various sums
        numWindows = data.shape[0]
        numWindows, data = self.checkForBadValues(numWindows, data)
        # unweighted sum (i.e. normal solution)
        unWeightedSum = np.sum(data, axis=0)
        unWeightedSum = unWeightedSum / numWindows
        # for each output variable, have ninput regressor variables
        # let's construct our arrays
        obs = np.empty(shape=(self.outSize, self.remoteSize), dtype="complex")
        reg = np.empty(
            shape=(self.outSize, self.remoteSize, self.inSize), dtype="complex"
        )
        for i in range(0, self.outSize):
            for j in range(0, self.remoteSize):
                obs[i, j] = unWeightedSum[self.inSize + i, j]
                for k in range(0, self.inSize):
                    reg[i, j, k] = unWeightedSum[k, j]
        # create array for output
        output = np.empty(shape=(self.outSize, self.inSize), dtype="complex")
        for i in range(0, self.outSize):
            observation = obs[i, :]
            predictors = reg[i, :, :]
            # save the output
            out, resids, scale, weights = mmestimateModel(
                predictors, observation, intercept=self.intercept
            )
            if self.intercept:
                output[i] = out[1:]
            else:
                output[i] = out
        return output

    def printList(self) -> List[str]:
        """Class information as a list of strings

        Returns
        -------
        out : list
            List of strings with information
        """

        textLst = []
        textLst.append("In Site = {}".format(self.inSite))
        textLst.append("In Channels = {}".format(self.inChannels))
        textLst.append("Out Site = {}".format(self.outSite))
        textLst.append("Out Channels = {}".format(self.outChannels))
        textLst.append("Remote Site = {}".format(self.remoteSite))
        textLst.append("Remote Channels = {}".format(self.remoteChannels))
        return textLst

