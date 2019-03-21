import os
import random
import numpy as np
import scipy.interpolate as interp
from typing import List, Dict, Tuple

# import from package
from resistics.calculators.calculator import Calculator
from resistics.ioHandlers.transferFunctionWriter import TransferFunctionWriter
from resistics.dataObjects.transferFunctionData import TransferFunctionData
from resistics.utilities.utilsIO import checkAndMakeDir, fileFormatSampleFreq
from resistics.utilities.utilsPrint import generalPrint, warningPrint, blockPrint
from resistics.utilities.utilsSmooth import smooth1d
from resistics.utilities.utilsRobust import (
    sampleMAD0,
    hermitianTranspose,
    olsModel,
    chatterjeeMachler,
    mmestimateModel,
)


class ProcessorSingleSite(Calculator):
    """Performs single site transfer function calculations 

    By default, the ProcessorSingleSite is setup to calculate the impedance tensor using Hx, Hy as input channels and Ex, Ey as output channels. To calculate the Tipper, the appropriate input and output channels have to be set.

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
        The channels to calculate the cross spectra out for
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
        self.inSite: str = "dummy"
        self.inChannels: List[str] = ["Hx", "Hy"]
        self.inSize: int = len(self.inChannels)
        self.outSite: str = "dummy"
        self.outChannels: List[str] = ["Ex", "Ey"]
        self.outSize: int = len(self.outChannels)
        self.allChannels: List[str] = self.inChannels + self.outChannels
        self.crossChannels: List[str] = self.allChannels
        # solution options
        self.intercept: bool = False
        self.method: str = "cm"
        # smoothing options
        self.win: str = "hanning"
        self.winSmooth: int = -1
        # output filename
        self.postpend: str = ""
        # evaluation frequency data
        self.evalFreq = []
        self.impedances = []
        self.variances = []

    def setInput(self, inSite: str, inChannels: List[str]) -> None:
        """Set information about input site and channels
    
        Parameters
        ----------
        inSite : str
            Site to use for input channel data
        inChannels : List[str]
            Channels to use as the input in the linear system
        """

        self.inSite = inSite
        self.inChannels = inChannels
        self.inSize = len(inChannels)
        # set all and cross channels
        self.allChannels = self.inChannels + self.outChannels
        self.crossChannels = self.allChannels

    def setOutput(self, outSite: str, outChannels: List[str]) -> None:
        """Set information about output site and channels
    
        Parameters
        ----------
        inSite : str
            Site to use for output channel data
        inChannels : List[str]
            Channels to use as the output in the linear system
        """

        self.outSite = outSite
        self.outChannels = outChannels
        self.outSize = len(outChannels)
        # set all and cross channels
        self.allChannels = self.inChannels + self.outChannels
        self.crossChannels = self.allChannels

    def process(self) -> None:
        """Process spectra data

        The processing sequence for each decimation level is as below:

        1. Get shared (unmasked) windows for all relevant sites (inSite and outSite)
        2. For shared unmasked windows
            
            - Calculate out the spectral power data
            - Interpolate calculated spectral power data to the evaluation frequencies for the decimation level 
        
        3. For each evaluation frequency
            
            - Do the robust processing to calculate the transfer function at that evaluation frequency


        NOTES
        -----
        The spectral power data is smoothed as this tends to improve results. The smoothing can be changed by setting the smoothing parameters. This method is still subject to change in the future as it is an area of active work
        """

        numLevels: int = self.decParams.numLevels
        for iDec in range(0, numLevels):
            self.printText("Processing decimation level {}".format(iDec))
            fs = self.decParams.getSampleFreqLevel(iDec)
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
            totalSize: int = self.inSize + self.outSize
            numEvalFreq: int = len(evalFreq)
            dataSize: int = self.winSelector.getDataSize(iDec)
            freq: np.ndarray = np.linspace(0, fs / 2, dataSize)
            # get the window smoothing params
            smoothLen: int = self.getWindowSmooth(datasize=dataSize)

            # create the data array for each evaluation frequency
            # keep the spectral power information for all windows
            evalFreqData: np.ndarray = np.empty(
                shape=(numEvalFreq, numWindows, totalSize, totalSize), dtype="complex"
            )

            # an array for the window data
            winSpectraMatrix: np.ndarray = np.empty(
                shape=(totalSize, totalSize, dataSize), dtype="complex"
            )
            winDataArray: np.ndarray = np.empty(
                shape=(totalSize, dataSize), dtype="complex"
            )

            # loop over unmasked windows
            localWin: int = 0
            global2local: Dict = {}
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

                # get data into the right part of the arrays
                for i in range(0, self.inSize):
                    winDataArray[i] = inData[self.inChannels[i]]
                for i in range(0, self.outSize):
                    winDataArray[self.inSize + i] = outData[self.outChannels[i]]

                # and now can fill the parts of the matrix
                # recall, smooth the power spectra
                for i in range(0, totalSize):
                    for j in range(i, totalSize):
                        winSpectraMatrix[i, j] = smooth1d(
                            winDataArray[i] * np.conjugate(winDataArray[j]),
                            smoothLen,
                            self.win,
                        )
                        if i != j:
                            # due to complex symmetry
                            winSpectraMatrix[j, i] = np.conjugate(
                                winSpectraMatrix[i, j]
                            )

                # after running through all windows, calculate various spectral properties at evaluation frequencies
                # using interpolation
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
                        evalFreq[eIdx], 1 / evalFreq[eIdx]
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
                # solution using all components
                numSolveWindows, obs, reg = self.prepareLinearEqn(
                    evalFreqData[eIdx, localWinIndices]
                )
                tmp1, tmp2 = self.robustProcess(numSolveWindows, obs, reg)
                # tmp1, tmp2 = self.olsProcess(numSolveWindows, obs, reg)
                self.impedances.append(tmp1)
                self.variances.append(tmp2)

        if len(self.evalFreq) == 0:
            self.printWarning(
                "No data was found at any decimation level for insite {}, outsite {}, sampling frequency {} and specdir {}".format(
                    self.inSite,
                    self.outSite,
                    self.decParams.getSampleFreqLevel(0),
                    self.winSelector.specdir,
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
        )

    def getWindowSmooth(self, **kwargs):
        """Window smoothing length

        Power spectra data is smoothed. This returns the size of the window smoother. The window itself is given by win.
    
        Parameters
        ----------
        kwargs['datasize] : int
            The size of the data

        Returns
        -------
        smoothLen : int
            Smoothing size
        """

        # check if window size specified by user
        if self.winSmooth != -1 and self.winSmooth > 1:
            return self.winSmooth
        # if not, calculate based on datasize
        if "datasize" in kwargs:
            winSmooth = kwargs["datasize"] * 1.0 / 16.0
            if winSmooth < 3:
                return 3  # minimum smoothing
            # otherwise round to nearest odd number
            winSmooth = np.ceil(winSmooth) // 2  # this is floor division
            return int(winSmooth * 2 + 1)
        # otherwise, return a default value
        return 15

    def calcEvalFrequencyData(self, freq, evalFreq, winDataMatrix):
        """Calculate spectral power data at evaluation frequencies

        Parameters
        ----------
        freq : np.ndarray
            Frequency array of spectra data
        evalFreq : np.ndarray
            Evaluation frequencies for the decimation level
        winDataMatrix : np.ndarray
            Array holding spectral power data at frequencies freq        

        Returns
        -------
        out : np.ndarray
            Spectral power data interpolated to evaluation frequencies 
        """

        inShape: Tuple = winDataMatrix.shape
        data: np.ndarray = np.empty(
            shape=(evalFreq.size, inShape[0], inShape[1]), dtype="complex"
        )
        # get data from winDataMatrix
        for i in range(0, inShape[0]):
            for j in range(0, inShape[1]):
                interpFunc = interp.interp1d(freq, winDataMatrix[i, j])
                interpVals = interpFunc(evalFreq)
                for eIdx, eFreq in enumerate(evalFreq):
                    data[eIdx, i, j] = interpVals[eIdx]
        return data

    def smoothSpectralEstimates(self, data):
        # takes the evaluation frequency data, which is indexed
        # windows, matrix of spectral components
        winSmooth = 9
        totalChans = self.inSize + self.outSize
        for i in range(0, totalChans):
            for j in range(0, totalChans):
                data[:, i, j] = smooth1d(data[:, i, j], winSmooth, self.win)
        return data

    def checkForBadValues(self, numWindows, data):
        finiteArray = np.ones(shape=(numWindows))
        for iW in range(0, numWindows):
            if not np.isfinite(data[iW]).all():
                finiteArray[iW] = 0
        numGoodWindows = sum(finiteArray)
        if numGoodWindows == numWindows:
            return numWindows, data
        self.printWarning(
            "Bad data found...number of windows reduced from {} to {}".format(
                numWindows, numGoodWindows
            )
        )
        goodWindowIndices = np.where(finiteArray == 1)
        return numGoodWindows, data[goodWindowIndices]

    def prepareLinearEqn(self, data):
        # prepare observations and regressors for linear processing
        numWindows = data.shape[0]
        numWindows, data = self.checkForBadValues(numWindows, data)
        crossSize = len(self.crossChannels)
        # for each output variable, have ninput regressor variables
        # construct our arrays
        obs = np.empty(shape=(self.outSize, crossSize * numWindows), dtype="complex")
        reg = np.empty(
            shape=(self.outSize, crossSize * numWindows, self.inSize), dtype="complex"
        )
        for iW in range(0, numWindows):
            iOffset = iW * crossSize
            for i in range(0, self.outSize):
                for j, crossChan in enumerate(self.crossChannels):
                    # this is the observation row where i is the observed output
                    crossIndex = self.allChannels.index(crossChan)
                    obs[i, iOffset + j] = data[iW, self.inSize + i, crossIndex]
                    for k in range(0, self.inSize):
                        reg[i, iOffset + j, k] = data[iW, k, crossIndex]
        return numWindows, obs, reg

    # SOLVER ROUTINES
    def robustProcess(self, numWindows, obs, reg):
        # do the robust processing for a single evaluation frequency
        crossSize = len(self.crossChannels)
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

            # now take the weights, apply to the observations and predictors, stack the appropriate rows
            observation2 = np.zeros(shape=(crossSize), dtype="complex")
            predictors2 = np.zeros(shape=(crossSize, self.inSize), dtype="complex")
            for iChan in range(0, crossSize):
                # now need to have my indexing array
                indexArray = np.arange(iChan, numWindows * crossSize, crossSize)
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
            # out, resids, scale, weights = mmestimateModel(
            #     predictors2, observation2, intercept=self.intercept)

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
            # varPred = np.dot(hermitianTranspose(predictors), weights*predictors) # need to fix this
            varPred = np.dot(hermitianTranspose(predictors), predictors)
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
        varOutput = np.empty(shape=(self.outSize, self.inSize), dtype="float")
        # solve
        for i in range(0, self.outSize):
            observation = obs[i, :]
            predictors = reg[i, :, :]
            # save the output
            out, resids, squareResid, rank, s = olsModel(
                predictors, observation, intercept=self.intercept
            )
            # if self.intercept:
            # 	output[i] = out[1:]
            # else:
            # 	output[i] = out

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
            # varPred = np.dot(hermitianTranspose(predictors), weights*predictors) # need to fix this
            varPred = np.dot(hermitianTranspose(predictors), predictors)
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

    def stackedProcess(self, data):
        # then do various sums
        numWindows = data.shape[0]
        crossSize = len(self.crossChannels)
        # unweighted sum (i.e. normal solution)
        unWeightedSum = np.sum(data, axis=0)
        unWeightedSum = unWeightedSum / numWindows

        # for each output variable, have ninput regressor variables
        # let's construct our arrays
        obs = np.empty(shape=(self.outSize, crossSize), dtype="complex")
        reg = np.empty(shape=(self.outSize, crossSize, self.inSize), dtype="complex")
        for i in range(0, self.outSize):
            for j, crossChan in enumerate(self.crossChannels):
                crossIndex = self.allChannels.index(crossChan)
                obs[i, j] = unWeightedSum[self.inSize + i, crossIndex]
                for k in range(0, self.inSize):
                    reg[i, j, k] = unWeightedSum[k, crossIndex]

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

    def writeTF(self, specdir: str, postpend: str, freq, data, variances, **kwargs):
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
        out : list
            List of strings with information
        """

        textLst = []
        textLst.append("In Site = {}".format(self.inSite))
        textLst.append("In Channels = {}".format(self.inChannels))
        textLst.append("Out Site = {}".format(self.outSite))
        textLst.append("Out Channels = {}".format(self.outChannels))
        return textLst
