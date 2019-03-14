import numpy as np
import scipy.stats as stats
from typing import List, Dict, Tuple

# import from package
from resistics.utilities.utilsPrint import errorPrint


def getStatNames() -> Tuple[List[str], List[str]]:
    """Get a list of statistic and remotestatistic names

    Returns
    -------
    stats : List[str]
        List of signal site statistics
    remoteStats : List[str]
        List of remote reference statistics
    """

    stats = [
        "absvalEqn",
        "coherence",
        "powerSpectralDensity",
        "polarisationDirection",
        "partialCoherence",
        "transferFunction",
        "resPhase",
    ]
    remoteStats = [
        "RR_coherence",
        "RR_coherenceEqn",
        "RR_absvalEqn",
        "RR_transferFunction",
        "RR_resPhase",
    ]
    return stats, remoteStats


def getStatElements(stat: str) -> List[str]:
    """Get statistic elements for each statistic
    
    Parameters
    ----------
    stat : str
        The statistic for which to get the statistic elements

    Returns
    -------
    Dict[str, List[str]]
        Mapping from a statistic name to the elements of that statistic
    """

    statElements = {
        "absvalEqn": [
            "absExEx",
            "absHyEx",
            "absExEy",
            "absHyEy",
            "absExHx",
            "absHyHx",
            "absExHy",
            "absHyHy",
            "absEyEx",
            "absHxEx",
            "absEyEy",
            "absHxEy",
            "absEyHx",
            "absHxHx",
            "absEyHy",
            "absHxHy",
        ],
        "coherence": ["cohExHx", "cohExHy", "cohEyHx", "cohEyHy"],
        "powerSpectralDensity": ["psdEx", "psdEy", "psdHx", "psdHy"],
        "polarisationDirection": ["polExEy", "polHxHy"],
        "transferFunction": [
            "ExHxReal",
            "ExHxImag",
            "ExHyReal",
            "ExHyImag",
            "EyHxReal",
            "EyHxImag",
            "EyHyReal",
            "EyHyImag",
        ],
        "resPhase": [
            "ExHxRes",
            "ExHxPhase",
            "ExHyRes",
            "ExHyPhase",
            "EyHxRes",
            "EyHxPhase",
            "EyHyRes",
            "EyHyPhase",
        ],
        "partialCoherence": [
            "bivarEx",
            "bivarEy",
            "parExHx",
            "parExHy",
            "parEyHx",
            "parEyHy",
        ],
    }
    # remote reference stat elements
    statElementsRR = {
        "RR_coherence": [
            "ExHxRR",
            "ExHyRR",
            "EyHxRR",
            "EyHyRR",
            "HxHxRR",
            "HxHyRR",
            "HyHxRR",
            "HyHyRR",
        ],
        "RR_coherenceEqn": ["ExHxR-HyHxR", "ExHyR-HyHyR", "EyHxR-HxHxR", "EyHyR-HxHyR"],
        "RR_absvalEqn": [
            "absHyHxR",
            "absExHxR",
            "absHyHyR",
            "absExHyR",
            "absHxHxR",
            "absEyHxR",
            "absHxHyR",
            "absEyHyR",
        ],
        "RR_transferFunction": [
            "ExHxRealRR",
            "ExHxImagRR",
            "ExHyRealRR",
            "ExHyImagRR",
            "EyHxRealRR",
            "EyHxImagRR",
            "EyHyRealRR",
            "EyHyImagRR",
        ],
        "RR_resPhase": [
            "ExHxResRR",
            "ExHxPhaseRR",
            "ExHyResRR",
            "ExHyPhaseRR",
            "EyHxResRR",
            "EyHxPhaseRR",
            "EyHyResRR",
            "EyHyPhaseRR",
        ],
    }

    if stat in statElements:
        return statElements[stat]
    if stat in statElementsRR:
        return statElementsRR[stat]
    errorPrint(
        "utilsStats::getStatElements",
        "Statistic {} not found".format(stat),
        quitRun=True,
    )
    return False


# Window statistics - These are not currently in use
# signal to noise ratio is calculated as mean over std.
# this might be more useful in spectral domain
# can do this in the frequecy domain, on the amplitude
def calcSNR(specData):
    output = {}
    for c in specData.data:
        tmp = np.absolute(specData.data[c])
        output[c] = np.average(tmp) / np.std(tmp)
    return output


# The Pearson correlation coefficient measures the linear relationship
# between two datasets. Strictly speaking, Pearson's correlation requires
# that each dataset be normally distributed.
def pearsonCoefficient(data):
    # construct input matrices
    # this needs to be columns for observations, rows for variables
    # and this needs to be output as a dictionary
    chans = sorted(list(data.keys()))
    numChans = len(chans)  # the channels
    output = {}
    for i in range(0, numChans):
        for j in range(i, numChans):
            key = "{}{}".format(chans[i], chans[j])
            # now calculate the pearson correlation coefficient
            pcc, pval = stats.pearsonr(data[chans[i]], data[chans[j]])
            output[key] = pcc
    return output


def pearsonCoefficientSpec(data):
    # calcaulates the PCC for magnitude and phase separately
    mag = {}
    phase = {}
    for c in data:
        mag[c] = np.absolute(data[c])
        phase[c] = np.unwrap(np.angle(data[c]))
    magPCC = pearsonCoefficient(mag)
    phasePCC = pearsonCoefficient(phase)
    return magPCC, phasePCC
