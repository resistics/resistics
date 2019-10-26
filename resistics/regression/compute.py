import numpy as np
from typing import List, Tuple
import scipy.interpolate as interp

from resistics.common.smooth import smooth1d
from resistics.spectra.data import SpectrumData


def evalFrequencyData(freq, evalFreq, winDataMatrix):
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
            for eIdx in range(len(evalFreq)):
                data[eIdx, i, j] = interpVals[eIdx]
    return data


def spectralMatricesWindow(
    inData: SpectrumData,
    outData: SpectrumData,
    inChannels: List[str],
    outChannels: List[str],
    smoothLen: int,
    smoothWin: str,
    evalFreq: List[float],
):
    """Compute the spectral matrices 

    Parameters
    ----------
    inData : SpectrumData
        The input spectrum data
    outData : SpectrumData
        The output spectrum data
    inChannels : List[str]
        The input channels to use
    outChannels : List[str]
        The output channels to use
    smoothLen : int
        The smoothing length
    smoothWin : str
        The smoothing window
    evalFreq : np.ndarray
        The evaluation frequencies to interpolate to

    Returns
    -------
    out : np.ndarray
        Cross spectral matrices interpolated to evaluation frequencies
    """
    inSize = len(inChannels)
    outSize = len(outChannels)
    totalSize = inSize + outSize
    dataSize = inData.dataSize
    # arrays for the window data
    winDataArray: np.ndarray = np.empty(shape=(totalSize, dataSize), dtype="complex")
    winSpectraMatrix: np.ndarray = np.empty(
        shape=(totalSize, totalSize, dataSize), dtype="complex"
    )
    # get data into the right part of the arrays
    for iChan in range(0, inSize):
        winDataArray[iChan] = inData.data[inChannels[iChan]]
    for iChan in range(0, outSize):
        winDataArray[inSize + iChan] = outData.data[outChannels[iChan]]
    # power spectra
    for i in range(0, totalSize):
        for j in range(i, totalSize):
            winSpectraMatrix[i, j] = smooth1d(
                winDataArray[i] * np.conjugate(winDataArray[j]), smoothLen, smoothWin
            )
            if i != j:
                # complex symmetry
                winSpectraMatrix[j, i] = np.conjugate(winSpectraMatrix[i, j])
    # return data interpolated to evaluation frequencies
    return evalFrequencyData(inData.freqArray, evalFreq, winSpectraMatrix)


def spectralMatrices(
    ncores: int,
    inData: List[SpectrumData],
    outData: List[SpectrumData],
    inChannels: List[str],
    outChannels: List[str],
    smoothLen: int,
    smoothWin: str,
    evalFreq: np.ndarray,
):
    """Parallel calculation of spectral matrices
    
    Parameters
    ----------
    ncores : int
        The number of cores to run on
    inData : List[SpectrumData]
        The input spectrum data
    outData : List[SpectrumData]
        The output spectrum data
    inChannels : List[str]
        The input channels to use
    outChannels : List[str]
        The output channels to use
    smoothLen : int
        The smoothing length
    smoothWin : str
        The smoothing window
    evalFreq : np.ndarray
        The evaluation frequencies to interpolate to

    Returns
    -------
    out : List[np.ndarray]
        List of spectral matrices
    """
    if ncores > 0:
        import multiprocessing as mp

        multiTuples = [
            (iD, oD, inChannels, outChannels, smoothLen, smoothWin, evalFreq)
            for iD, oD in zip(inData, outData)
        ]
        with mp.Pool(ncores) as pool:
            out = pool.starmap(spectralMatricesWindow, multiTuples)
    else:
        out = []
        for iD, oD in zip(inData, outData):
            out.append(
                spectralMatricesWindow(
                    iD, oD, inChannels, outChannels, smoothLen, smoothWin, evalFreq
                )
            )
    return out


def remoteMatricesWindow(
    inData: SpectrumData,
    outData: SpectrumData,
    remoteData: SpectrumData,
    inChannels: List[str],
    outChannels: List[str],
    remoteChannels: List[str],
    smoothLen: int,
    smoothWin: str,
    evalFreq: List[float],
):
    """Compute the spectral matrices 

    Parameters
    ----------
    inData : SpectrumData
        The input spectrum data
    outData : SpectrumData
        The output spectrum data
    remoteData : SpectrumData
        The remote spectrum data
    inChannels : List[str]
        The input channels to use
    outChannels : List[str]
        The output channels to use
    remoteChannels : List[str]
        The remote channels to use            
    smoothLen : int
        The smoothing length
    smoothWin : str
        The smoothing window
    evalFreq : np.ndarray
        The evaluation frequencies to interpolate to

    Returns
    -------
    out : np.ndarray
        Cross spectral matrices interpolated to evaluation frequencies
    """
    inSize = len(inChannels)
    outSize = len(outChannels)
    remoteSize = len(remoteChannels)
    totalSize = inSize + outSize
    dataSize = inData.dataSize
    # an array for the in and out channels fourier data
    winDataArray = np.empty(shape=(totalSize, dataSize), dtype="complex")
    # an array for the remote reference fourier data
    winRemoteArray = np.empty(shape=(remoteSize, dataSize), dtype="complex")
    # an array for the power spectra data
    winSpectraMatrix = np.empty(
        shape=(totalSize, remoteSize, dataSize), dtype="complex"
    )
    # get data into the right part of the arrays
    for i in range(0, inSize):
        winDataArray[i] = inData.data[inChannels[i]]
    for i in range(0, outSize):
        winDataArray[inSize + i] = outData.data[outChannels[i]]
    for i in range(0, remoteSize):
        winRemoteArray[i] = remoteData.data[remoteChannels[i]]

    # power spectra
    for iD in range(totalSize):
        for iR in range(remoteSize):
            # cannot use conjugate symmetry unlike single site processor
            winSpectraMatrix[iD, iR] = smooth1d(
                winDataArray[iD] * np.conjugate(winRemoteArray[iR]),
                smoothLen,
                smoothWin,
            )
    # return data interpolated to evaluation frequencies
    return evalFrequencyData(inData.freqArray, evalFreq, winSpectraMatrix)


def remoteMatrices(
    ncores: int,
    inData: List[SpectrumData],
    outData: List[SpectrumData],
    remoteData: List[SpectrumData],
    inChannels: List[str],
    outChannels: List[str],
    remoteChannels: List[str],
    smoothLen: int,
    smoothWin: str,
    evalFreq: np.ndarray,
):
    """Parallel calculation of spectral matrices for remote reference data
    
    Parameters
    ----------
    ncores : int
        The number of cores to run on
    inData : List[SpectrumData]
        The input spectrum data
    outData : List[SpectrumData]
        The output spectrum data
    remoteData : List[SpectrumData]
        The remote reference spectrum data
    inChannels : List[str]
        The input channels to use
    outChannels : List[str]
        The output channels to use
    remoteChannels : List[str]
        The remote channels to use        
    smoothLen : int
        The smoothing length
    smoothWin : str
        The smoothing window
    evalFreq : np.ndarray
        The evaluation frequencies to interpolate to

    Returns
    -------
    out : List[np.ndarray]
        List of spectral matrices
    """
    if ncores > 0:
        import multiprocessing as mp

        multiTuples = [
            (
                iD,
                oD,
                rD,
                inChannels,
                outChannels,
                remoteChannels,
                smoothLen,
                smoothWin,
                evalFreq,
            )
            for iD, oD, rD in zip(inData, outData, remoteData)
        ]
        with mp.Pool(ncores) as pool:
            out = pool.starmap(remoteMatricesWindow, multiTuples)
    else:
        out = []
        for iD, oD, rD in zip(inData, outData, remoteData):
            out.append(
                remoteMatricesWindow(
                    iD,
                    oD,
                    rD,
                    inChannels,
                    outChannels,
                    remoteChannels,
                    smoothLen,
                    smoothWin,
                    evalFreq,
                )
            )
    return out
