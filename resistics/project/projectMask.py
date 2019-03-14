import sys
import os
import numpy as np
from datetime import datetime
from typing import Union

# import from package
from resistics.dataObjects.projectData import ProjectData
from resistics.dataObjects.statisticData import StatisticData
from resistics.dataObjects.maskData import MaskData
from resistics.calculators.windowSelector import WindowSelector
from resistics.calculators.maskCalculator import MaskCalculator
from resistics.ioHandlers.statisticIO import StatisticIO
from resistics.ioHandlers.maskIO import MaskIO
from resistics.utilities.utilsChecks import parseKeywords
from resistics.utilities.utilsProject import (
    projectText,
    getDecimationParameters,
    getWindowParameters,
)



def newMaskData(projData: ProjectData, sampleFreq: float) -> MaskData:
    """Get a mask data object that can then be passed onto calculateMask

    Parameters
    ----------
    projData : ProjectData
        A ProjectData instance
    sampleFreq : float
        The sampling frequency to mask
    
    Returns
    -------
    MaskData
        A mask data object with parameters set
    """

    decParams = getDecimationParameters(sampleFreq, projData.config)
    decParams.printInfo()
    return MaskData(
        decParams.sampleFreq, decParams.numLevels, decParams.evalFreqPerLevel
    )


def getMaskData(
    projData: ProjectData,
    site: str,
    maskName: str,
    sampleFreq: Union[float, int],
    **kwargs
) -> MaskData:
    """Get a mask data object

    Parameters
    ----------
    projData : projectData
        A project instance
    site : str
        The site for which to get the mask
    maskName : str
        The name of the mask
    sampleFreq : int, float
        The sampling frequency for which the mask was created
    specdir : str
        The spectra directory for which the mask was created  

    Returns
    -------
    MaskData
        A mask data object with the mask information
    """

    options = {}
    options["specdir"] = projData.config.configParams["Spectra"]["specdir"]
    options = parseKeywords(options, kwargs)

    siteData = projData.getSiteData(site)
    maskIO = MaskIO(siteData.getSpecdirMaskPath(options["specdir"]))
    maskData = maskIO.read(maskName, sampleFreq)
    return maskData


def calculateMask(projData: ProjectData, maskData: MaskData, **kwargs):
    """Calculate masks sites

    Parameters
    ----------
    projData : projectData
        A project instance
    maskData : MaskData
        A mask data instance
    sites : List[str], optional
        A list of sites to calculate statistics for
    specdir : str, optional
        The spectra directory for which to calculate statistics
    """

    options = {}
    options["sites"] = projData.getSites()
    options["specdir"] = projData.config.configParams["Spectra"]["specdir"]
    options = parseKeywords(options, kwargs)

    # create a maskCalculator object
    maskCalc = MaskCalculator(projData, maskData, specdir=options["specdir"])
    maskIO = MaskIO()
    sampleFreq = maskData.sampleFreq

    # loop over sites
    for site in options["sites"]:
        # see if there is a sample freq
        siteData = projData.getSiteData(site)
        siteSampleFreqs = siteData.getSampleFreqs()
        if sampleFreq not in siteSampleFreqs:
            continue

        # decimation and window parameters
        decParams = getDecimationParameters(sampleFreq, projData.config)
        decParams.printInfo()
        winParams = getWindowParameters(decParams, projData.config)

        # clear previous windows from maskCalc
        maskCalc.clearMaskWindows()
        # calculate masked windows
        maskCalc.applyConstraints(site)
        maskCalc.maskData.printInfo()
        maskCalc.maskData.view(0)

        # write maskIO file
        maskIO.datapath = os.path.join(siteData.getSpecdirMaskPath(options["specdir"]))
        maskIO.write(maskCalc.maskData)

        # test with the window selector
        winSelector = WindowSelector(
            projData, sampleFreq, decParams, winParams, specdir=options["specdir"]
        )
        winSelector.setSites([site])
        winSelector.addWindowMask(site, maskData.maskName)
        winSelector.calcSharedWindows()
        winSelector.printInfo()
        winSelector.printDatetimeConstraints()
        winSelector.printWindowMasks()
        winSelector.printSharedWindows()
        winSelector.printWindowsForFrequency()

