"""
Calculate the crosspowers required for the regression solution.
This uses the Python multiprocessing library where requested to speed up calculation.
"""
import numpy as np
from typing import List, Tuple
import scipy.interpolate as interp

from resistics.common.smooth import smooth1d
from resistics.spectra.data import SpectrumData, PowerData


def localCrosspowersWindow(
    inData: SpectrumData,
    outData: SpectrumData,
    primary: List[str],
    secondary: List[str],
    smoothLen: int,
    smoothFunc: str,
    evalFreq: List[float],
) -> PowerData:
    """Compute the crosspowers for a single window 

    Intended for single site or intersite transfer functions with no remote reference

    Parameters
    ----------
    inData : SpectrumData
        The input site spectrum data
    outData : SpectrumData
        The output site spectrum data
    primary : List[str]
        Primary channels in the crosspowers
    secondary : List[str]
        Secondary channels in the crosspowers
    smoothLen : int
        The smoothing length
    smoothFunc : str
        The smoothing window
    evalFreq : np.ndarray
        The evaluation frequencies to interpolate to

    Returns
    -------
    crosspowers : PowerData
        Cross spectral matrices interpolated to evaluation frequencies
    """
    from resistics.spectra.data import mergeSpectra
    from resistics.spectra.calculator import crosspowers

    mergeData = mergeSpectra((inData, outData), channels=None, postpend=("In", "Out"))
    crosspowers = crosspowers(mergeData, primary=primary, secondary=secondary)
    crosspowers.smooth(smoothLen, smoothFunc, inplace=True)
    return crosspowers.interpolate(evalFreq)


def localCrosspowers(
    ncores: int,
    inData: List[SpectrumData],
    inChannels: List[str],
    inCross: List[str],
    outData: List[SpectrumData],
    outChannels: List[str],
    outCross: List[str],
    smoothLen: int,
    smoothFunc: str,
    evalFreq: np.ndarray,
) -> List[PowerData]:
    """Parallel calculation of crosspowers for all windows

    This is for single site or intersite transfer functions with no remote reference
    
    Parameters
    ----------
    ncores : int
        The number of cores to run on
    inData : List[SpectrumData]
        The input spectrum data
    inChannels : List[str]
        The input channels to use
    inCross : List[str]
        The input channels for which to calculate crosspowers
    outData : List[SpectrumData]
        The output spectrum data
    outChannels : List[str]
        The output channels to use
    outCross : List[str]
        The output channels for which to calculate crosspowers
    smoothLen : int
        The smoothing length
    smoothFunc : str
        The smoothing window
    evalFreq : np.ndarray
        The evaluation frequencies to interpolate to

    Returns
    -------
    crosspowerData : List[np.ndarray]
        List of PowerData
    """
    # primary in crosspowers
    inChannels = [iC + "In" for iC in inChannels]
    outChannels = [oC + "Out" for oC in outChannels]
    primary = inChannels + outChannels
    # secondary in crosspowers
    inCross = [iC + "In" for iC in inCross]
    outCross = [oC + "Out" for oC in outCross]
    secondary = inCross + outCross
    # calculate crosspowers
    if ncores > 0:
        import multiprocessing as mp

        multiTuples = [
            (iD, oD, primary, secondary, smoothLen, smoothFunc, evalFreq,)
            for iD, oD in zip(inData, outData)
        ]
        with mp.Pool(ncores) as pool:
            crosspowerData = pool.starmap(localCrosspowersWindow, multiTuples)
    else:
        crosspowerData = []
        for iD, oD in zip(inData, outData):
            crosspowerData.append(
                localCrosspowersWindow(
                    iD, oD, primary, secondary, smoothLen, smoothFunc, evalFreq,
                )
            )
    return crosspowerData


def remoteCrosspowersWindow(
    inData: SpectrumData,
    outData: SpectrumData,
    remoteData: SpectrumData,
    primary: List[str],
    secondary: List[str],
    smoothLen: int,
    smoothFunc: str,
    evalFreq: List[float],
) -> PowerData:
    """Compute the crosspowers for a single window 

    Intended for single site or intersite transfer functions with no remote reference

    Parameters
    ----------
    inData : SpectrumData
        The input site spectrum data
    outData : SpectrumData
        The output site spectrum data
    remoteData : SpectrumData
        The remote site spectrum data
    primary : List[str]
        Primary channels in the crosspowers
    secondary : List[str]
        Secondary channels in the crosspowers
    smoothLen : int
        The smoothing length
    smoothFunc : str
        The smoothing window
    evalFreq : np.ndarray
        The evaluation frequencies to interpolate to

    Returns
    -------
    crosspowers : PowerData
        Cross spectral matrices interpolated to evaluation frequencies
    """
    from resistics.spectra.data import mergeSpectra
    from resistics.spectra.calculator import crosspowers

    mergeData = mergeSpectra(
        (inData, outData, remoteData), channels=None, postpend=("In", "Out", "RR")
    )
    crosspowers = crosspowers(mergeData, primary=primary, secondary=secondary)
    crosspowers.smooth(smoothLen, smoothFunc, inplace=True)
    return crosspowers.interpolate(evalFreq)


def remoteCrosspowers(
    ncores: int,
    inData: List[SpectrumData],
    inChannels: List[str],
    inCross: List[str],
    outData: List[SpectrumData],
    outChannels: List[str],
    outCross: List[str],
    remoteData: List[SpectrumData],
    remoteCross: List[str],
    smoothLen: int,
    smoothFunc: str,
    evalFreq: np.ndarray,
) -> List[PowerData]:
    """Parallel calculation of crosspowers for all windows

    This is for remote reference processing and can also deal with remote reference with intersite.
    
    Parameters
    ----------
    ncores : int
        The number of cores to run on
    inData : List[SpectrumData]
        The input site spectra data
    inChannels : List[str]
        The input channels to use
    inCross : List[str]
        The input channels for which to calculate crosspowers
    outData : List[SpectrumData]
        The output site spectra data
    outChannels : List[str]
        The output channels to use
    outCross : List[str]
        The output channels for which to calculate crosspowers
    remoteData : List[SpectrumData],
        The remote reference site spectra data
    remoteCross : List[str]
        The remote reference site channels for which to calculate crosspowers
    smoothLen : int
        The smoothing length
    smoothFunc : str
        The smoothing window
    evalFreq : np.ndarray
        The evaluation frequencies to interpolate to

    Returns
    -------
    crosspowerData : List[np.ndarray]
        List of PowerData
    """
    # primary in crosspowers
    inChannels = [iC + "In" for iC in inChannels]
    outChannels = [oC + "Out" for oC in outChannels]
    primary = inChannels + outChannels
    # secondary in crosspowers
    inCross = [iC + "In" for iC in inCross]
    outCross = [oC + "Out" for oC in outCross]
    remoteCross = [rC + "RR" for rC in remoteCross]
    secondary = inCross + outCross + remoteCross
    # calculate crosspowers
    if ncores > 0:
        import multiprocessing as mp

        multiTuples = [
            (iD, oD, rD, primary, secondary, smoothLen, smoothFunc, evalFreq,)
            for iD, oD, rD in zip(inData, outData, remoteData)
        ]
        with mp.Pool(ncores) as pool:
            crosspowers = pool.starmap(remoteCrosspowersWindow, multiTuples)
    else:
        crosspowers = []
        for iD, oD, rD in zip(inData, outData, remoteData):
            crosspowers.append(
                remoteCrosspowersWindow(
                    iD, oD, rD, primary, secondary, smoothLen, smoothFunc, evalFreq,
                )
            )
    return crosspowers
