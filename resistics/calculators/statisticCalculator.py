import numpy as np
import scipy.stats as stats
import scipy.interpolate as interp
from copy import deepcopy
from typing import List, Dict, Union

# import from package
from resistics.calculators.calculator import Calculator
from resistics.dataObjects.spectrumData import SpectrumData
from resistics.utilities.utilsPrint import listToString, arrayToString
from resistics.utilities.utilsRobust import olsModel
from resistics.utilities.utilsSmooth import smooth1d


class StatisticCalculator(Calculator):
    """Calculate statistics for data restriction

    Statistics are calculated out for each evaluation frequency in each window. Therefore, there are nwindow*nfreq statistics in total.

    This class was written to speed up statistic calculations. Many statistics need the same data, for example power spectra. This class calculates and reuses some common values amongst the various statistics to improve calculation speed.

    Attributes
    ----------
    evalFreq : List
        List of evaluation frequencies
    winLen : int
        Window length for spectra calculations
    winType : str
        Window function to apply to time data before fourier transform
    inChans : List[str]
        Input channels
    inSize : int
        Number of input channels
    outChans : List[str]
        Output channels 
    outSize : int
        Number of output channels
    specChans : List[str] 
        The channels for which to calculate auto and cross power spectra
    remoteChans : List[str]
        Remote reference channels
    psdChans : List[str] 
        Power spectral density channels
    cohPairs : List[List[str]]
        Pairs of channels for coherence calculations
    polDirs : List[List[str]]
        Pairs of channels of polarisation direction calculation
    spec: Dict = {}

    tfCalculated : bool 
        Boolean flag to show that the transfer function has been calculated for a window
    remoteCalculated : bool 
        Boolean flag to show...
    intercept : bool   
        Boolean flag to include an intercept into the 
    outData: Dict = {}

    Methods
    -------
    __init__()
    getEvalFreq()
    getInChans()
    getOutChans()
    getSpecChans()
    getRemoteChans()
    getPSDChans()
    getCohPairs()
    getPolDirs()
    getAutoPower(chan)
    getAutoPowerEval(chan, eIdx)
    getCrossPower(chan1, chan2)
    getCrossPowerEval(chan1, chan2, eIdx)
    getOutData()
    setInChans(inChans)
    setOutChans(outChans)
    setSpectra(freq, winData, evalFreq)
    setIntercept(intercept)
    calculateSpectralMatrix()
    calculateEvalMatrix()
    addRemoteSpec(remoteData, **kwargs)
    calculateRemoteSpectralMatrix()
    calculateRemoteEvalMatrix()
    calculateReferenceSpectralMatrix()
    calculateReferenceEvalMatrix()
    getRemoteAutoPower(chan)
    getRemoteAutoPowerEval(chan, eIdx)
    getRemoteCrossPower(chan1, chan2)
    getRemoteCrossPowerEval(chan1, chan2, eIdx)
    getReferenceCrossPower(dataChan, remoteChan)
    getReferenceCrossPowerEval(dataChan, remoteChan, eIdx)
    interpolateToEvalFreq(data)
    prepareOutDict()
    getDataForStatName(statName)
    winPSD()
    winCoherence()
    winPolarisations()
    winPartials()
    winTransferFunction()
    winRemoteCoherence()
    winRemoteEqnCoherence()
    winRemoteAbsVal()
    winRemoteTransferFunction()
    printList()
        Class information returned as list of strings  
    """

    def __init__(self) -> None:
        """Initialise the statistic calculator"""

        # default evaluation frequencies
        self.evalFreq: List = []
        # power smoothing vals
        self.winLen: int = 13
        self.winType: str = "hanning"
        # set some defaults
        self.inChans: List[str] = ["Hx", "Hy"]
        self.inSize: int = len(self.inChans)
        self.outChans: List[str] = ["Ex", "Ey"]
        self.outSize: int = len(self.outChans)
        self.specChans: List[str] = self.inChans + self.outChans
        self.remoteChans: List[str] = self.inChans
        self.psdChans: List[str] = ["Ex", "Ey", "Hx", "Hy"]
        self.cohPairs: List[List[str]] = [
            ["Ex", "Hx"],
            ["Ex", "Hy"],
            ["Ey", "Hx"],
            ["Ey", "Hy"],
        ]
        self.polDirs: List[List[str]] = [["Ex", "Ey"], ["Hx", "Hy"]]
        # set data presets
        self.freq: Union[np.ndarray, None] = None
        self.spec: Dict[np.ndarray] = {}
        # output data and marker for transfer function calculated
        self.tfCalculated: bool = False
        self.remoteCalculated: bool = False
        self.intercept: bool = False
        self.outData: Dict = {}

    def getEvalFreq(self):
        """Get a copy of the evaluation frequency
        
        Returns
        -------
        List[float]
            List of evaluation frequencies
        """

        return deepcopy(self.evalFreq)

    def getInChans(self) -> List[str]:
        """Get a copy of the input channels
        
        Returns
        -------
        List[str]
            List of input channels
        """

        return deepcopy(self.inChans)

    def getOutChans(self) -> List[str]:
        """Get a copy of the output channels
        
        Returns
        -------
        List[str]
            List of output channels
        """

        return deepcopy(self.outChans)

    def getSpecChans(self):
        """Get a copy of the channels for which to calculate spectra
        
        Returns
        -------
        List[str]
            List of spectra channels
        """

        return deepcopy(self.specChans)

    def getRemoteChans(self) -> List[str]:
        """Get a copy of the remote reference channels
        
        Returns
        -------
        List[str]
            List of remote reference channels
        """

        return deepcopy(self.remoteChans)

    def getPSDChans(self) -> List[str]:
        """Get a copy of the channels to include power spectral density
        
        Returns
        -------
        List[str]
            List of power spectral density channels
        """

        return deepcopy(self.psdChans)

    def getCohPairs(self) -> List[List[str]]:
        """Get a copy of coherence channel pairs to calculate out
        
        Returns
        -------
        List[List[str]]
            List of coherence pairs
        """

        return deepcopy(self.cohPairs)

    def getPolDirs(self) -> List[List[str]]:
        """Get a list of polarisation direction pairs
        
        Returns
        -------
        List[List[str]]
            List of polarisation direction pairs
        """

        return deepcopy(self.polDirs)

    def getAutoPower(self, chan: str) -> np.ndarray:
        """Get the auto power for a channel

        Parameters
        ----------
        chan : str
            The channel for which to get the autopower
        
        Returns
        -------
        np.ndarray
            The auto power for the channel
        """

        idx = self.specChans.index(chan)
        # then return the autopower
        return self.spectralMatrix[idx, idx].real

    def getAutoPowerEval(self, chan: str, eIdx: int) -> float:
        """Get the auto power value for an particular evaluation frequency

        Parameters
        ----------
        chan : str
            The channel for which to get the autopower
        eIdx : int
            The index for the evaluation frequency
        
        Returns
        -------
        np.ndarray
            The auto power for the channel
        """

        idx = self.specChans.index(chan)
        # then return the autopower
        return self.evalMatrix[idx, idx, eIdx].real

    def getCrossPower(self, chan1: str, chan2: str) -> np.ndarray:
        """Get the cross power between two channels

        Parameters
        ----------
        chan1 : str
            The first channel for the cross channels
        chan2 : str
            The second channl for the cross channels
        
        Returns
        -------
        np.ndarray
            The cross power spectral density
        """

        idx1 = self.specChans.index(chan1)
        idx2 = self.specChans.index(chan2)
        # then return the autopower
        return self.spectralMatrix[idx1, idx2]

    def getCrossPowerEval(self, chan1: str, chan2: str, eIdx: int) -> float:
        """Get the cross power between two channels

        Parameters
        ----------
        chan1 : str
            The first channel for the cross channels
        chan2 : str
            The second channl for the cross channels
        eIdex : int
            The index of the evaluation frequency
        
        Returns
        -------
        np.ndarray
            The cross power spectral density
        """

        idx1 = self.specChans.index(chan1)
        idx2 = self.specChans.index(chan2)
        # then return the autopower
        return self.evalMatrix[idx1, idx2, eIdx]

    def getOutData(self) -> Dict:
        """Get the output data

        Returns
        -------
        Dict
            The statistic output data
        """

        return deepcopy(self.outData)

    def setInChans(self, inChans: List[str]) -> None:
        """Set the input channels

        Parameters
        ----------
        inChans : List[str]
            Input channels for the magnetotelluric linear system
        """

        self.inChans = inChans
        self.inSize = len(self.inChans)

    def setOutChans(self, outChans: List[str]):
        """Set the output channels

        Parameters
        ----------
        inChans : List[str]
            Output channels for the magnetotelluric linear system
        """

        self.outChans = outChans
        self.outSize = len(self.outChans)

    def setRemoteChans(self, remoteChans):
        """Set the input channels

        Parameters
        ----------
        inChans : List[str]
            Input channels for the magnetotelluric linear system
        """

        self.remoteChans = remoteChans

    def setPSDChans(self, psdChans: List[str]) -> None:
        """Set the power spectral density channels

        Parameters
        ----------
        psdChans : List[str]
            Power spectral density channels. An example input would be: ["Ex", "Ey", "Hx", "Hy"]
        """

        self.psdChans = psdChans

    def setCohPairs(self, cohPairs: List[List[str]]) -> None:
        """Set the power spectral density channels

        If cohPairs of [["Ex", "Hx"], ["Ex", "Hy"], ["Ey", "Hx"], ["Ey", "Hy"]] are set, the following coherences will be calculated:
        ExHx
        ExHy
        EyHx
        EyHy

        Parameters
        ----------
        cohPairs : List[List[str]]
            Set the coherence pairs using a list of channel pairs, for example: [["Ex", "Hx"], ["Ex", "Hy"], ["Ey", "Hx"], ["Ey", "Hy"]]
        """

        self.cohPairs = cohPairs

    def setPolDirs(self, polDirs: List[List[str]]) -> None:
        """Set the polarisation direction pairs to calculate

        If polDirs of [["Ex", "Ey"], ["Hx", "Hy"]] are set, the following polarisation directions will be calculated:
        Ex Ey
        Hx Hy

        Parameters
        ----------
        polDirs : List[List[str]]
            Set polarisation direction channel pairs, for example: [["Ex", "Ey"], ["Hx", "Hy"]]
        """

        self.polDirs = polDirs

    def setSpectra(
        self, freq: np.ndarray, specData: SpectrumData, evalFreq: np.ndarray
    ) -> None:
        """Set the spectra data

        Parameters
        ----------
        freq : np.ndarray
            The frequency points in the spectra data
        specData : SpectrumData
            Spectrum data (i.e. spectrum data for a window)
        evalFreq : np.ndarray
            Evaluation frequency array
        """

        self.freq = freq
        self.spec: Dict[np.ndarray] = specData.data
        self.evalFreq = evalFreq
        # self.specChans = sorted(self.spec.keys())
        self.numChans = len(self.specChans)
        self.dataSize = specData.dataSize
        # calculate the power matrix
        self.calculateSpectralMatrix()
        self.calculateEvalMatrix()
        # clear the out dictionary and set that transfer function not calculated
        self.prepareOutDict()

    def setIntercept(self, intercept: bool):
        """Set the intercept boolean

        Parameters
        ----------
        intercept : bool
            Boolean flag for having an intercept in the transfer function calculation
        """

        self.intercept = intercept

    def calculateSpectralMatrix(self) -> None:
        """Calculate out the cross power spectral matrix

        The method calculates out the cross powers which will then be used in the other statistic calculations.

        The cross powers spectral matrix is a 3-D matrix of size:
        numChans * numChans * numFrequencies
        The elements of this are calculated by multiplying the spectra of one channel by the complex conjugate of the spectra of another channel.
        """

        # create the 3d array
        self.spectralMatrix = np.empty(
            shape=(self.numChans, self.numChans, self.dataSize), dtype="complex"
        )
        # now need to go through the chans
        for ii in range(0, self.numChans):
            for jj in range(ii, self.numChans):
                chan1 = self.specChans[ii]
                chan2 = self.specChans[jj]
                self.spectralMatrix[ii, jj] = smooth1d(
                    self.spec[chan1] * np.conjugate(self.spec[chan2]),
                    self.winLen,
                    self.winType,
                )
                if ii == jj:
                    # conjugate symmtry
                    self.spectralMatrix[jj, ii] = np.conjugate(
                        self.spectralMatrix[ii, jj]
                    )

    def calculateEvalMatrix(self):
        """Calculate out the cross power spectral matrix at the evaluation frequencies

        The method calculates out the cross powers which will then be used in the other statistic calculations at the evaluation frequencies

        The cross powers spectral matrix for evaluation frequencies is a 3-D matrix of size:
        numChans * numChans * numEvaluationFrequencies
        The elements of this are calculated by taking the cross powers spectral matrix and using the result there to interpolate the values at the evaluation frequencies.
        """

        # create the array
        self.evalMatrix = np.empty(
            shape=(self.numChans, self.numChans, len(self.evalFreq)), dtype="complex"
        )
        for ii in range(0, self.numChans):
            for jj in range(ii, self.numChans):
                self.evalMatrix[ii, jj] = self.interpolateToEvalFreq(
                    self.spectralMatrix[ii, jj]
                )
                if ii != jj:
                    # conjugate symmtry
                    self.evalMatrix[jj, ii] = np.conjugate(self.evalMatrix[ii, jj])

    def addRemoteSpec(self, remoteData: SpectrumData, remoteChans: List[str] = []) -> None:
        """Add coincident remote reference spectrum data

        Parameters
        ----------
        remoteData : SpectrumData
            Spectrum data (i.e. spectrum data for a window)
        remoteChans : List[str]
            The channels to use from remote reference data
        """

        self.remoteSpec = remoteData.data
        if len(remoteChans) > 0:
            self.remoteChans = remoteChans
        # now calculate some remote reference related values
        self.calculateRemoteSpectralMatrix()
        self.calculateRemoteEvalMatrix()
        self.calculateReferenceSpectralMatrix()
        self.calculateReferenceEvalMatrix()

    def calculateRemoteSpectralMatrix(self):
        """Calculate out the cross power spectral matrix for the remote reference data

        The method calculates out the cross powers for the remote reference channels which will then be used in the other statistic calculations.

        The remote reference cross powers spectral matrix is a 3-D matrix of size:
        numRemoteChans * numRemoteChans * numFrequencies
        The elements of this are calculated by multiplying the spectra of one channel by the complex conjugate of the spectra of another channel.
        """

        # create the 3d array
        numRemoteChans = len(self.remoteChans)
        self.remoteSpectralMatrix = np.empty(
            shape=(numRemoteChans, numRemoteChans, self.dataSize), dtype="complex"
        )
        # now need to go through the chans
        for ii in range(0, numRemoteChans):
            for jj in range(ii, numRemoteChans):
                chan1 = self.remoteChans[ii]
                chan2 = self.remoteChans[jj]
                self.remoteSpectralMatrix[ii, jj] = smooth1d(
                    self.remoteSpec[chan1] * np.conjugate(self.remoteSpec[chan2]),
                    self.winLen,
                    self.winType,
                )
                if ii == jj:
                    # conjugate symmtry
                    self.remoteSpectralMatrix[jj, ii] = np.conjugate(
                        self.remoteSpectralMatrix[ii, jj]
                    )

    def calculateRemoteEvalMatrix(self):
        """Calculate out the cross power spectral matrix for the remote reference data at the evaluation frequencies

        Takes the cross power spectral data calculate for the remote reference channels and interpoaltes it to the evaluation frequencies
        """

        # create the array
        numRemoteChans = len(self.remoteChans)
        self.remoteEvalMatrix = np.empty(
            shape=(numRemoteChans, numRemoteChans, len(self.evalFreq)), dtype="complex"
        )
        for ii in range(0, numRemoteChans):
            for jj in range(ii, numRemoteChans):
                self.remoteEvalMatrix[ii, jj] = self.interpolateToEvalFreq(
                    self.remoteSpectralMatrix[ii, jj]
                )
                if ii != jj:
                    # conjugate symmtry
                    self.remoteEvalMatrix[jj, ii] = np.conjugate(
                        self.remoteEvalMatrix[ii, jj]
                    )

    def calculateReferenceSpectralMatrix(self):
        """Calculate out the cross power spectral matrix between the site spectral data and the remote reference spectral data

        The reference cross powers spectral matrix is a 3-D matrix of size:
        numChans * numRemoteChans * numFrequencies
        The elements of this are calculated by multiplying a channel of the site spectral data by the complex conjugate of a channel from the remote reference. 
        """

        # cannot use conjugate symmetry in this case
        self.referenceSpectralMatrix = np.empty(
            shape=(self.numChans, len(self.remoteChans), self.dataSize), dtype="complex"
        )
        for ii, chan1 in enumerate(self.specChans):
            for jj, chan2 in enumerate(self.remoteChans):
                self.referenceSpectralMatrix[ii, jj] = smooth1d(
                    self.spec[chan1] * np.conjugate(self.remoteSpec[chan2]),
                    self.winLen,
                    self.winType,
                )

    def calculateReferenceEvalMatrix(self):
        """Interpolate the remote and site cross powers spectral matrix to the evaluation frequencies.
        """

        self.referenceEvalMatrix = np.empty(
            shape=(self.numChans, len(self.remoteChans), len(self.evalFreq)),
            dtype="complex",
        )
        for ii, chan1 in enumerate(self.specChans):
            for jj, chan2 in enumerate(self.remoteChans):
                self.referenceEvalMatrix[ii, jj] = self.interpolateToEvalFreq(
                    self.referenceSpectralMatrix[ii, jj]
                )

    def getRemoteAutoPower(self, chan: str) -> np.ndarray:
        """Get the auto power of a remote reference channel

        Parameters
        ----------
        chan : str
            The channel for which to get the autopower
        
        Returns
        -------
        np.ndarray
            The autopower array (real for autopowers)
        """

        idx = self.remoteChans.index(chan)
        return self.remoteSpectralMatrix[idx, idx].real

    def getRemoteAutoPowerEval(self, chan: str, eIdx: int) -> float:
        """Get the auto power of a remote reference channel at an evaluation frequency

        Parameters
        ----------
        chan : str
            The channel for which to get the autopower
        eIdx : int
            The evaluation frequency index
        
        Returns
        -------
        float
            The autopower of the channel at the evaluation frequency
        """

        idx = self.remoteChans.index(chan)
        return self.remoteEvalMatrix[idx, idx, eIdx].real

    def getRemoteCrossPower(self, chan1: str, chan2: str) -> np.ndarray:
        """Get the cross power of two remote reference channels

        Parameters
        ----------
        chan1 : str
            The first channel for the cross power
        chan2 : str
            The second channel for the cross power
        
        Returns
        -------
        np.ndarray
            The cross power array 
        """

        idx1 = self.remoteChans.index(chan1)
        idx2 = self.remoteChans.index(chan2)
        return self.remoteSpectralMatrix[idx1, idx2]

    def getRemoteCrossPowerEval(self, chan1: str, chan2: str, eIdx: int) -> float:
        """Get the cross power of two remote reference channels at a single evaluation frequency

        Parameters
        ----------
        chan1 : str
            The first channel for the cross power
        chan2 : str
            The second channel for the cross power
        eIdx : int
            The evaluation frequency index            
        
        Returns
        -------
        float
            The value of the cross power at the evaluation frequency 
        """

        idx1 = self.remoteChans.index(chan1)
        idx2 = self.remoteChans.index(chan2)
        return self.remoteSpectralMatrix[idx1, idx2, eIdx]

    def getReferenceCrossPower(self, dataChan: str, remoteChan: str) -> np.ndarray:
        """Get the cross power of a data channel and a remote reference channel

        Parameters
        ----------
        dataChan : str
            The data channel
        remoteChan : str
            The remote reference channel
        
        Returns
        -------
        np.ndarray
            The cross power array 
        """

        idx1 = self.specChans.index(dataChan)
        idx2 = self.remoteChans.index(remoteChan)
        return self.referenceSpectralMatrix[idx1, idx2]

    def getReferenceCrossPowerEval(
        self, dataChan: str, remoteChan: str, eIdx: int
    ) -> float:
        """Get the cross power of a data channel and a remote reference channel at a single evaluation frequency

        Parameters
        ----------
        dataChan : str
            The data channel
        remoteChan : str
            The remote reference channel
        eIdx : int
            The evaluation frequency index              
        
        Returns
        -------
        float
            The value of the cross power at the evaluation frequency 
        """

        idx1 = self.specChans.index(dataChan)
        idx2 = self.remoteChans.index(remoteChan)
        return self.referenceEvalMatrix[idx1, idx2, eIdx]

    def interpolateToEvalFreq(self, data: np.ndarray) -> np.ndarray:
        """Interpolate data on to the evaluation frequency

        Parameters
        ----------
        data : np.ndarray
            Power spectral data defined at frequency points given in the freqs array

        Returns
        -------
        np.ndarray
            Data interpolated to evaluation frequencies
        """

        interpFunc = interp.interp1d(self.freq, data)
        interpData = interpFunc(self.evalFreq)
        return interpData

    def prepareOutDict(self) -> None:
        """Prepare output statistic output dictionary
        
        The outData dictionary is indexed in the following way:
        outData[evaluation frequency][statistic component] = value
        """

        self.outData = {}
        for e in self.evalFreq:
            self.outData[e] = {}
        # set various calculated flags to false
        self.tfCalculated = False
        self.remoteCalculated = False

    def getDataForStatName(self, statName: str):
        """Return the data for a statistic

        Given a statitic name, this method returns data from the correct internal method.

        Parameters
        ----------
        statName : str
            The name of the statistic to calculate out
        
        Returns
        -------
        Dict
            The output dictionary
        """

        if statName == "absvalEqn":
            return self.winAbsVal()
        elif statName == "coherence":
            return self.winCoherence()
        elif statName == "powerSpectralDensity":
            return self.winPSD()
        elif statName == "polarisationDirection":
            return self.winPolarisations()
        elif statName == "partialCoherence":
            return self.winPartials()
        elif statName == "transferFunction" or statName == "resPhase":
            if self.tfCalculated:
                return self.getOutData()
            return self.winTransferFunction()
        elif statName == "RR_coherence":
            return self.winRemoteCoherence()
        elif statName == "RR_coherenceEqn":
            return self.winRemoteEqnCoherence()
        elif statName == "RR_absvalEqn":
            return self.winRemoteAbsVal()
        elif statName == "RR_transferFunction" or statName == "RR_resPhase":
            if self.remoteCalculated:
                return self.getOutData()
            return self.winRemoteTransferFunction()
        else:
            self.printError(
                "Statistic in getDataForStatName not recognised", quitRun=True
            )
            return self.winCoherence()

    def winPSD(self):
        """Calculate power spectral densities

        Returns
        -------
        Dict : 
            Dictionary with statistic values, indexed by [evaluation frequency][statistic component]
        """

        # need to divide by length of time too
        freqLen = self.freq.size
        timeLen = (freqLen - 1) * 2  # minus 1 because time sections are usually even
        fs = self.freq[-1] * 2  # sampling frequency
        # and then calculate amount of time
        duration = timeLen / fs
        # interpolate onto evaluation frequency and output to outData
        for eIdx, eF in enumerate(self.evalFreq):
            for chan in self.getPSDChans():
                key = "psd{}".format(chan)
                self.outData[eF][key] = self.getAutoPowerEval(chan, eIdx) / duration
        return self.getOutData()

    def winCoherence(self):
        """Calculate spectral coherence pairs

        Returns
        -------
        Dict : 
            Dictionary with statistic values, indexed by [evaluation frequency][statistic component]
        """

        for idx, p in enumerate(self.getCohPairs()):
            c1 = p[0]  # chan1
            c2 = p[1]  # chan2
            for eIdx, eF in enumerate(self.evalFreq):
                # calculate the nominator and denominator
                cohNom = np.power(
                    np.absolute(self.getCrossPowerEval(c1, c2, eIdx)), 2
                ).real
                cohDenom = self.getAutoPowerEval(c1, eIdx) * self.getAutoPowerEval(
                    c2, eIdx
                )
                # save in outData
                key = "coh{}".format(c1 + c2)
                self.outData[eF][key] = cohNom / cohDenom
        return self.getOutData()

    def winPolarisations(self):
        """Calculate polarisation directions

        Returns
        -------
        Dict : 
            Dictionary with statistic values, indexed by [evaluation frequency][statistic component]
        """

        for idx, p in enumerate(self.getPolDirs()):
            c1 = p[0]  # chan1
            c2 = p[1]  # chan2
            for eIdx, eF in enumerate(self.evalFreq):
                # now calculate the nominator and denominator
                cohNom = (
                    2 * self.getCrossPowerEval(c1, c2, eIdx).real
                )  # take the real part of this
                cohDenom = self.getAutoPowerEval(c1, eIdx) - self.getAutoPowerEval(
                    c2, eIdx
                )
                # save to out dictionary
                key = "pol{}".format(c1 + c2)
                self.outData[eF][key] = np.arctan(cohNom / cohDenom) * (180.0 / np.pi)
        return self.getOutData()

    def winPartials(self):
        """Calculate partial coherencies

        Based on paper Weckmann, Magunia Ritter 2005.
        e.g. coherence Ex, Hx w.r.t Hy
        This currently only works for impedance tensor calculations and higher power partial coherencies are not supported.

        Returns
        -------
        Dict : 
            Dictionary with statistic values, indexed by [evaluation frequency][statistic component]

        Notes
        -----
        Based on paper by Weckmann, Magunia Ritter 2005.
        """

        # get the coherences - these will be required later
        winCoherence = self.winCoherence()

        for i, outChan in enumerate(self.outChans):
            for eIdx, eFreq in enumerate(self.evalFreq):
                inChan1 = self.inChans[0]
                inChan2 = self.inChans[1]
                xOutIn1 = self.getCrossPowerEval(outChan, inChan1, eIdx)
                xOutIn2 = self.getCrossPowerEval(outChan, inChan2, eIdx)
                xIn1In2 = self.getCrossPowerEval(inChan1, inChan2, eIdx)
                xIn2In1 = self.getCrossPowerEval(inChan2, inChan1, eIdx)
                # calculate out transFunc components
                denom = (
                    self.getAutoPowerEval(inChan1, eIdx)
                    * self.getAutoPowerEval(inChan2, eIdx)
                    - xIn1In2 * xIn2In1
                )
                # Z1
                Z1nom = (
                    xOutIn1 * self.getAutoPowerEval(inChan2, eIdx) - xIn2In1 * xOutIn2
                )
                Z1 = Z1nom / denom
                # Z2
                Z2nom = (
                    self.getAutoPowerEval(inChan1, eIdx) * xOutIn2 - xIn1In2 * xOutIn1
                )
                Z2 = Z2nom / denom
                # calculate bivariate coherency
                rb = Z1 * self.getCrossPowerEval(
                    inChan1, outChan, eIdx
                ) + Z2 * self.getCrossPowerEval(inChan2, outChan, eIdx)
                rb = rb / self.getAutoPowerEval(outChan, eIdx)
                # now calculate out partials
                # calculate partial inChan, outChan1 with respect to outChan2
                cohkey = "coh{}".format(outChan + inChan2)
                rp1 = (rb - winCoherence[eFreq][cohkey]) / (
                    1.0 - winCoherence[eFreq][cohkey]
                )
                # calculate partial inChan, outChan2 with respect to outChan1
                cohkey = "coh{}".format(outChan + inChan1)
                rp2 = (rb - winCoherence[eFreq][cohkey]) / (
                    1.0 - winCoherence[eFreq][cohkey]
                )
                # now save in outDict
                self.outData[eFreq]["bivar{}".format(outChan)] = rb
                self.outData[eFreq]["par{}".format(outChan + inChan1)] = rp1
                self.outData[eFreq]["par{}".format(outChan + inChan2)] = rp2
        return self.getOutData()

    def winAbsVal(self):
        """Absolute values of the cross power spectral matrix

        This data is often useful for cross plotting

        Returns
        -------
        Dict : 
            Dictionary with statistic values, indexed by [evaluation frequency][statistic component]
        """

        for eIdx, eFreq in enumerate(self.evalFreq):
            for iChan, chan in enumerate(self.specChans):
                # first do the outchans multiplied by every other channel
                for iOut, outChan in enumerate(self.outChans):
                    absval = np.absolute(self.getCrossPowerEval(outChan, chan, eIdx))
                    key = "abs{}{}".format(outChan, chan)
                    self.outData[eFreq][key] = absval

                # then  do the inchans multiplied by every other channel
                for iIn, inChan in enumerate(self.inChans):
                    absval = np.absolute(self.getCrossPowerEval(inChan, chan, eIdx))
                    key = "abs{}{}".format(inChan, chan)
                    self.outData[eFreq][key] = absval
        # return the dictionary
        return self.getOutData()

    def winTransferFunction(self):
        """Calculate transfer function for the spectral data

        Returns
        -------
        Dict : 
            Dictionary with statistic values, indexed by [evaluation frequency][statistic component]
        """

        totalSize = self.inSize + self.outSize
        # now want to calculate the transfer function for each evaluation frequency
        output = np.empty(
            shape=(self.evalFreq.size, self.outSize, self.inSize), dtype="complex"
        )
        for eIdx, eFreq in enumerate(self.evalFreq):
            # solve transfer function
            obs = np.empty(shape=(self.outSize, totalSize), dtype="complex")
            reg = np.empty(
                shape=(self.outSize, totalSize, self.inSize), dtype="complex"
            )
            for i in range(0, self.outSize):
                for j in range(0, totalSize):
                    # this is the observation row where,i is the observed output
                    # idx in the evaluation frequency
                    obs[i, j] = self.getCrossPowerEval(
                        self.outChans[i], self.specChans[j], eIdx
                    )
                    for k in range(0, self.inSize):
                        reg[i, j, k] = self.getCrossPowerEval(
                            self.inChans[k], self.specChans[j], eIdx
                        )

            for i in range(0, self.outSize):
                observation = obs[i, :]
                predictors = reg[i, :, :]
                # now do the solution
                out, resids, squareResid, rank, s = olsModel(
                    predictors, observation, intercept=self.intercept
                )
                # out, resids, scale, weights	= mmestimateModel(predictors, observation, intercept=False)
                # not interested in the intercept (const) term
                if self.intercept:
                    output[eIdx, i] = out[1:]
                else:
                    output[eIdx, i] = out

            # calculate components of transfer function and res and phase
            for i in range(0, self.outSize):
                for j in range(0, self.inSize):
                    period = 1.0 / eFreq
                    res = 0.2 * period * np.power(np.absolute(output[eIdx, i, j]), 2)
                    phase = np.angle(output[eIdx, i, j], deg=True)
                    keyRes = self.outChans[i] + self.inChans[j] + "Res"
                    keyPhase = self.outChans[i] + self.inChans[j] + "Phase"
                    self.outData[eFreq][keyRes] = res
                    self.outData[eFreq][keyPhase] = phase
                    # add the components
                    keyReal = self.outChans[i] + self.inChans[j] + "Real"
                    keyImag = self.outChans[i] + self.inChans[j] + "Imag"
                    self.outData[eFreq][keyReal] = output[eIdx, i, j].real
                    self.outData[eFreq][keyImag] = output[eIdx, i, j].imag
        # set transfer function calculated as true
        # saves having to do it again
        self.tfCalculated = True
        return self.getOutData()

    def winRemoteCoherence(self):
        """Calulate coherence between data channels and remote channels

        For example, this is the coherence of Ex-HxR, Ex-HyR, Ey-HxR, Ey-HyR, Hx-HxR, Hx-HyR, Hy-HxR, Hy-HyR

        Returns
        -------
        Dict : 
            Dictionary with statistic values, indexed by [evaluation frequency][statistic component]
        """

        # now let's calculate coherency
        # abs(crosspower(A,B))^2/autopower(A)*autpower(B)
        for dataChan in self.specChans:
            for remoteChan in self.remoteChans:
                key = "{}{}RR".format(dataChan, remoteChan)
                for eIdx, eFreq in enumerate(self.evalFreq):
                    cohNom = np.power(
                        np.absolute(
                            self.getReferenceCrossPowerEval(dataChan, remoteChan, eIdx)
                        ),
                        2,
                    )
                    cohDenom = self.getAutoPowerEval(
                        dataChan, eIdx
                    ) * self.getRemoteAutoPowerEval(remoteChan, eIdx)
                    coh = cohNom / cohDenom
                    self.outData[eFreq][key] = coh
        return self.getOutData()

    def winRemoteEqnCoherence(self):
        """Calulates coherences for the remote reference solver equations

        For example, this is the coherence of Ex-HxR, Ex-HyR, Ey-HxR, Ey-HyR, Hx-HxR, Hx-HyR, Hy-HxR, Hy-HyR

        todo:
        Write more information in these comments

        Returns
        -------
        Dict : 
            Dictionary with statistic values, indexed by [evaluation frequency][statistic component]
        """

        # now calculate out the relevant coherencies
        # here we calculate the coherency between <Ex,HyR> and <Hy,HyR> for example
        for iOut, outChan in enumerate(self.outChans):
            for iIn, inChan in enumerate(self.inChans):
                for iRemote, remoteChan in enumerate(self.remoteChans):
                    # calculate powers
                    c1c1 = smooth1d(
                        self.getReferenceCrossPower(outChan, remoteChan)
                        * np.conjugate(
                            self.getReferenceCrossPower(outChan, remoteChan)
                        ),
                        self.winLen,
                        self.winType,
                    )
                    c2c2 = smooth1d(
                        self.getReferenceCrossPower(inChan, remoteChan)
                        * np.conjugate(self.getReferenceCrossPower(inChan, remoteChan)),
                        self.winLen,
                        self.winType,
                    )
                    c1c2 = smooth1d(
                        self.getReferenceCrossPower(outChan, remoteChan)
                        * np.conjugate(self.getReferenceCrossPower(inChan, remoteChan)),
                        self.winLen,
                        self.winType,
                    )
                    # now interpolate
                    c1c1 = self.interpolateToEvalFreq(c1c1)
                    c2c2 = self.interpolateToEvalFreq(c2c2)
                    c1c2 = self.interpolateToEvalFreq(c1c2)
                    # now calculate the nominator and denominator
                    cohNom = np.power(np.absolute(c1c2), 2)
                    cohDenom = c1c1 * c2c2
                    coh = (
                        cohNom / cohDenom
                    )  # cast as float - discard complex part (complex part should be zero anyway)
                    # now need the coherencies for the evaluation frequencies
                    # this involves interpolation
                    key = "{}{}R-{}{}R".format(outChan, remoteChan, inChan, remoteChan)
                    for iFreq, eFreq in enumerate(self.evalFreq):
                        self.outData[eFreq][key] = coh[iFreq]
        return self.getOutData()

    def winRemoteAbsVal(self):
        """Absolute values of cross power spectral densities between remote refence channels and data channels

        This data can be useful for cross plotting

        Returns
        -------
        Dict : 
            Dictionary with statistic values, indexed by [evaluation frequency][statistic component]
        """

        for eIdx, eFreq in enumerate(self.evalFreq):
            for iOut, outChan in enumerate(self.outChans):
                for iRemote, remoteChan in enumerate(self.remoteChans):
                    absOut = np.absolute(
                        self.getReferenceCrossPowerEval(outChan, remoteChan, eIdx)
                    )
                    keyOut = "abs{}{}R".format(outChan, remoteChan)
                    self.outData[eFreq][keyOut] = absOut
                    for iIn, inChan in enumerate(self.inChans):
                        absIn = np.absolute(
                            self.getReferenceCrossPowerEval(inChan, remoteChan, eIdx)
                        )
                        keyIn = "abs{}{}R".format(inChan, remoteChan)
                        self.outData[eFreq][keyIn] = absIn
        return self.getOutData()

    def winRemoteTransferFunction(self):
        """Calculate transfer function for the spectral data when remote reference is included too

        Returns
        -------
        Dict : 
            Dictionary with statistic values, indexed by [evaluation frequency][statistic component]
        """

        output = np.empty(
            shape=(self.evalFreq.size, self.outSize, self.inSize), dtype="complex"
        )
        for eIdx, eFreq in enumerate(self.evalFreq):
            # solve transfer function
            obs = np.empty(shape=(self.outSize, self.inSize), dtype="complex")
            reg = np.empty(
                shape=(self.outSize, self.inSize, self.inSize), dtype="complex"
            )
            for i, outChan in enumerate(self.outChans):
                for j, remoteChan in enumerate(self.remoteChans):
                    # this is the observation row where,i is the observed output
                    # eIdx in the evaluation frequency
                    obs[i, j] = self.getReferenceCrossPowerEval(
                        outChan, remoteChan, eIdx
                    )
                    for k, inChan in enumerate(self.inChans):
                        reg[i, j, k] = self.getReferenceCrossPowerEval(
                            inChan, remoteChan, eIdx
                        )

            for i in range(0, self.outSize):
                observation = obs[i, :]
                predictors = reg[i, :, :]
                # now do the solution
                out, resids, squareResid, rank, s = olsModel(
                    predictors, observation, intercept=self.intercept
                )
                # out, resids, scale, weights	= mmestimateModel(predictors, observation, intercept=False)
                # not interested in the intercept (const) term
                if self.intercept:
                    output[eIdx, i] = out[1:]
                else:
                    output[eIdx, i] = out

            # calculate components of transfer function and res and phase
            for i in range(0, self.outSize):
                for j in range(0, self.inSize):
                    period = 1.0 / eFreq
                    res = 0.2 * period * np.power(np.absolute(output[eIdx, i, j]), 2)
                    phase = np.angle(output[eIdx, i, j], deg=True)
                    keyRes = self.outChans[i] + self.inChans[j] + "ResRR"
                    keyPhase = self.outChans[i] + self.inChans[j] + "PhaseRR"
                    self.outData[eFreq][keyRes] = res
                    self.outData[eFreq][keyPhase] = phase
                    # add the components
                    keyReal = self.outChans[i] + self.inChans[j] + "RealRR"
                    keyImag = self.outChans[i] + self.inChans[j] + "ImagRR"
                    self.outData[eFreq][keyReal] = output[eIdx, i, j].real
                    self.outData[eFreq][keyImag] = output[eIdx, i, j].imag
        # set transfer function calculated as true
        # saves having to do it again
        self.remoteCalculated = True
        return self.getOutData()

    def printList(self) -> List[str]:
        """Class information as a list of strings

        Returns
        -------
        out : list
            List of strings with information
        """

        textLst = []
        textLst.append("Default options")
        textLst.append("\tInput Chans = {}".format(listToString(self.getInChans())))
        textLst.append("\tOutput Chans = {}".format(listToString(self.getOutChans())))
        textLst.append(
            "\tRemote Chans = {}".format(listToString(self.getRemoteChans()))
        )
        textLst.append("\tPowers = {}".format(listToString(self.getPSDChans())))
        textLst.append(
            "\tCoherence pairs = {}".format(listToString(self.getCohPairs()))
        )
        textLst.append(
            "\tPartial coherence = {}".format(listToString(self.getPolDirs()))
        )
        if len(self.getEvalFreq()) == 0:
            textLst.append("Evaluation frequencies = {}")
        else:
            textLst.append(
                "Evaluation frequencies = {}".format(arrayToString(self.getEvalFreq()))
            )
        return textLst
