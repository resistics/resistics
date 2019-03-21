import sys
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Dict, Union


# import from package
from resistics.dataObjects.projectData import ProjectData
from resistics.dataObjects.statisticData import StatisticData
from resistics.calculators.windowSelector import WindowSelector
from resistics.calculators.statisticCalculator import StatisticCalculator
from resistics.ioHandlers.spectrumReader import SpectrumReader
from resistics.ioHandlers.statisticIO import StatisticIO
from resistics.project.projectMask import getMaskData
from resistics.utilities.utilsIO import fileFormatSampleFreq
from resistics.utilities.utilsPrint import listToString
from resistics.utilities.utilsPlotter import plotOptionsSpec, getPlotRowsAndCols
from resistics.utilities.utilsChecks import parseKeywords
from resistics.utilities.utilsStats import getStatElements
from resistics.utilities.utilsProject import (
    projectText,
    projectWarning,
    projectError,
    getDecimationParameters,
    getWindowParameters,
)


def getStatisticData(
    projData: ProjectData, site: str, meas: str, stat: str, declevel: int = 0, **kwargs
):
    """Get the statistic data for a statistic for a site measurement

    Parameters
    ----------
    projData : ProjectData
        Project instance
    site : str
        The site for which to get the statistic data
    meas : str
        The measurement for which to get the statistic
    stat : str
        The statistic for which to get the measurement
    declevel : int, optional
        The decimation level to read in. Default is 0.
    specdir : str, optional
        The spectra directory

    Returns
    -------
    StatisticData
        A statistic data object
    """

    options = {}
    options["specdir"] = projData.config.configParams["Spectra"]["specdir"]
    options = parseKeywords(options, kwargs)

    siteData = projData.getSiteData(site)
    if not siteData:
        projectError("Unable to find site {} in project".format(site), quitRun=True)

    # load the statistic data
    statIO = StatisticIO()
    statIO.setDatapath(
        os.path.join(siteData.getMeasurementStatPath(meas), options["specdir"])
    )
    statData = statIO.read(stat, declevel)
    return statData


def getStatisticDataForSampleFreq(
    projData: ProjectData,
    site: str,
    sampleFreq: float,
    stat: str,
    declevel: int = 0,
    **kwargs
):
    """Get the statistic data (for a particular decimation level) for all measurements in a site with sampling frequency sampleFreq

    Parameters
    ----------
    projData : ProjectData
        Project instance
    site : str
        The site for which to get the statistic data
    sampleFreq : float
        The sampling frequency
    stat : str
        The statistic for which to get the measurement
    declevel : int, optional
        The decimation level to read in. Default is 0.
    specdir : str, optional
        The spectra directory

    Returns
    -------
    StatisticData : Dict
        A statistic data object
    """

    options = {}
    options["specdir"] = projData.config.configParams["Spectra"]["specdir"]
    options = parseKeywords(options, kwargs)

    siteData = projData.getSiteData(site)
    if not siteData:
        projectError("Unable to find site {} in project".format(site), quitRun=True)

    # load the statistic data
    statData: Dict[str, StatisticData] = {}
    statIO = StatisticIO()
    measurements = siteData.getMeasurements(sampleFreq)
    for meas in measurements:
        statIO.setDatapath(
            os.path.join(siteData.getMeasurementStatPath(meas), options["specdir"])
        )
        # make sure some data was found
        chk = statIO.read(stat, declevel)
        if chk:
            statData[meas] = statIO.read(stat, declevel)
        else:
            projectWarning(
                "No {} data found for site {} and measurement {}".format(
                    stat, site, meas
                )
            )
    return statData


def calculateStatistics(projData: ProjectData, **kwargs):
    """Calculate statistics for sites
    
    Parameters
    ----------
    projData : ProjectData
        A project data instance
    sites : List[str], optional
        A list of sites to calculate statistics for
    sampleFreqs : List[float], optional
        List of sampling frequencies for which to calculate statistics
    chans : List[str], optional
        List of data channels to use
    specdir : str, optional
        The spectra directory for which to calculate statistics
    stats : List[str], optional
        The statistics to calculate out. Acceptable values are: "absvalEqn" "coherence", "psd", "poldir", "transFunc", "resPhase", "partialcoh". Configuration file values are used by default.
    """

    options = {}
    options["sites"] = projData.getSites()
    options["sampleFreqs"] = projData.getSampleFreqs()
    options["chans"] = []
    options["specdir"] = projData.config.configParams["Spectra"]["specdir"]
    options["stats"] = projData.config.configParams["Statistics"]["stats"]
    options = parseKeywords(options, kwargs)

    projectText(
        "Calculating stats: {} for sites: {}".format(
            listToString(options["stats"]), listToString(options["sites"])
        )
    )

    # create the statistic calculator and IO object
    statCalculator = StatisticCalculator()
    statIO = StatisticIO()

    # loop through sites and calculate statistics
    for site in options["sites"]:
        siteData = projData.getSiteData(site)
        measurements = siteData.getMeasurements()

        for meas in measurements:
            sampleFreq = siteData.getMeasurementSampleFreq(meas)
            if sampleFreq not in options["sampleFreqs"]:
                # don't need to calculate statistics for this sampling frequency
                continue

            projectText(
                "Calculating stats for site {}, measurement {}".format(site, meas)
            )

            # decimation parameters
            decParams = getDecimationParameters(sampleFreq, projData.config)
            decParams.printInfo()
            numLevels = decParams.numLevels

            # create the spectrum reader
            specReader = SpectrumReader(
                os.path.join(siteData.getMeasurementSpecPath(meas), options["specdir"])
            )

            # loop through decimation levels
            for iDec in range(0, numLevels):
                # open the spectra file for the current decimation level
                check = specReader.openBinaryForReading("spectra", iDec)
                if not check:
                    # probably because this decimation level not calculated
                    continue
                specReader.printInfo()

                # get windows
                refTime = specReader.getReferenceTime()
                winSize = specReader.getWindowSize()
                winOlap = specReader.getWindowOverlap()
                numWindows = specReader.getNumWindows()
                evalFreq = decParams.getEvalFrequenciesForLevel(iDec)
                sampleFreqDec = specReader.getSampleFreq()
                globalOffset = specReader.getGlobalOffset()
                fArray = specReader.getFrequencyArray()

                statHandlers = {}
                # create the statistic handlers
                for stat in options["stats"]:
                    statElements = getStatElements(stat)
                    statHandlers[stat] = StatisticData(
                        stat, refTime, sampleFreqDec, winSize, winOlap
                    )
                    statHandlers[stat].setStatParams(numWindows, statElements, evalFreq)
                    statHandlers[stat].comments = specReader.getComments()
                    statHandlers[stat].addComment(projData.config.getConfigComment())
                    statHandlers[stat].addComment(
                        "Calculating statistic: {}".format(stat)
                    )
                    statHandlers[stat].addComment(
                        "Statistic components: {}".format(listToString(statElements))
                    )

                # loop over windows and calculate the relevant statistics
                for iW in range(0, numWindows):
                    # get data
                    winData = specReader.readBinaryWindowLocal(iW)
                    globalIndex = iW + globalOffset
                    # give the statistic calculator the spectra
                    statCalculator.setSpectra(fArray, winData, evalFreq)
                    # get the desired statistics
                    for sH in statHandlers:
                        data = statCalculator.getDataForStatName(sH)
                        statHandlers[sH].addStat(iW, globalIndex, data)

                # save statistic
                for sH in statHandlers:
                    statIO.setDatapath(
                        os.path.join(
                            siteData.getMeasurementStatPath(meas), options["specdir"]
                        )
                    )
                    statIO.write(statHandlers[sH], iDec)


def calculateRemoteStatistics(projData: ProjectData, remoteSite: str, **kwargs):
    """Calculate statistics involving a remote reference site

    
    Parameters
    ----------
    projData : ProjectData
        A project data instance
    remoteSite : str
        The name of the site to use as the remote site
    sites : List[str], optional
        A list of sites to calculate statistics for
    sampleFreqs : List[float], optional
        List of sampling frequencies for which to calculate statistics
    chans : List[str], optional
        List of data channels to use
    specdir : str, optional
        The spectra directory for which to calculate statistics
    remotestats : List[str], optional
        The statistics to calculate out. Acceptable statistics are: "coherenceRR", "coherenceRREqn", "absvalRREqn", "transFuncRR", "resPhaseRR". Configuration file values are used by default.
    """

    options = {}
    options["sites"] = projData.getSites()
    options["sampleFreqs"] = projData.getSampleFreqs()
    options["chans"] = []
    options["specdir"] = projData.config.configParams["Spectra"]["specdir"]
    options["remotestats"] = projData.config.configParams["Statistics"]["remotestats"]
    options = parseKeywords(options, kwargs)

    projectText(
        "Calculating stats: {} for sites: {} with remote site {}".format(
            listToString(options["remotestats"]),
            listToString(options["sites"]),
            remoteSite,
        )
    )

    # create the statistic calculator and IO object
    statCalculator = StatisticCalculator()
    statIO = StatisticIO()

    # loop over sites
    for site in options["sites"]:
        siteData = projData.getSiteData(site)
        measurements = siteData.getMeasurements()

        for meas in measurements:
            sampleFreq = siteData.getMeasurementSampleFreq(meas)
            if sampleFreq not in options["sampleFreqs"]:
                # don't need to calculate statistics for this sampling frequency
                continue

            projectText(
                "Calculating stats for site {}, measurement {} with reference {}".format(
                    site, meas, remoteSite
                )
            )

            # decimation and window parameters
            decParams = getDecimationParameters(sampleFreq, projData.config)
            decParams.printInfo()
            numLevels = decParams.numLevels
            winParams = getWindowParameters(decParams, projData.config)

            # create the window selector and find the shared windows
            winSelector = WindowSelector(projData, sampleFreq, decParams, winParams)
            winSelector.setSites([site, remoteSite])
            # calc shared windows between site and remote
            winSelector.calcSharedWindows()
            # create the spectrum reader
            specReader = SpectrumReader(
                os.path.join(siteData.getMeasurementSpecPath(meas), options["specdir"])
            )

            # loop through decimation levels
            for iDec in range(0, numLevels):
                # open the spectra file for the current decimation level
                check = specReader.openBinaryForReading("spectra", iDec)
                if not check:
                    # probably because this decimation level not calculated
                    continue
                specReader.printInfo()

                # get a set of the shared windows at this decimation level
                # these are the global indices
                sharedWindows = winSelector.getSharedWindowsLevel(iDec)

                # get other information regarding only this spectra file
                refTime = specReader.getReferenceTime()
                winSize = specReader.getWindowSize()
                winOlap = specReader.getWindowOverlap()
                numWindows = specReader.getNumWindows()
                evalFreq = decParams.getEvalFrequenciesForLevel(iDec)
                sampleFreqDec = specReader.getSampleFreq()
                globalOffset = specReader.getGlobalOffset()
                fArray = specReader.getFrequencyArray()

                # now want to find the size of the intersection between the windows in this file and the shared windows
                sharedWindowsMeas = sharedWindows.intersection(
                    set(np.arange(globalOffset, globalOffset + numWindows))
                )
                sharedWindowsMeas = sorted(list(sharedWindowsMeas))
                numSharedWindows = len(sharedWindowsMeas)

                statHandlers = {}
                # create the statistic handlers
                for stat in options["remotestats"]:
                    statElements = getStatElements(stat)
                    statHandlers[stat] = StatisticData(
                        stat, refTime, sampleFreqDec, winSize, winOlap
                    )
                    # remember, this is with the remote reference, so the number of windows is number of shared windows
                    statHandlers[stat].setStatParams(
                        numSharedWindows, statElements, evalFreq
                    )
                    statHandlers[stat].comments = specReader.getComments()
                    statHandlers[stat].addComment(projData.config.getConfigComment())
                    statHandlers[stat].addComment(
                        "Calculating remote statistic: {}".format(stat)
                    )
                    statHandlers[stat].addComment(
                        "Statistic components: {}".format(listToString(statElements))
                    )

                # loop over the shared windows between the remote station and local station
                for iW, globalWindow in enumerate(sharedWindowsMeas):
                    # get data and set in the statCalculator
                    winData = specReader.readBinaryWindowGlobal(globalWindow)
                    statCalculator.setSpectra(fArray, winData, evalFreq)
                    # for the remote site, use the reader in win selector
                    remoteSF, remoteReader = winSelector.getSpecReaderForWindow(
                        remoteSite, iDec, globalWindow
                    )
                    winDataRR = remoteReader.readBinaryWindowGlobal(globalWindow)
                    statCalculator.addRemoteSpec(winDataRR)

                    for sH in statHandlers:
                        data = statCalculator.getDataForStatName(sH)
                        statHandlers[sH].addStat(iW, globalWindow, data)

                # save statistic
                for sH in statHandlers:
                    statIO.setDatapath(
                        os.path.join(
                            siteData.getMeasurementStatPath(meas), options["specdir"]
                        )
                    )
                    statIO.write(statHandlers[sH], iDec)


def viewStatistic(
    projData: ProjectData, site: str, sampleFreq: Union[int, float], stat: str, **kwargs
) -> plt.figure:
    """View statistic data for a single sampling frequency of a site
    
    Parameters
    ----------
    projData : ProjectData
        A project instance
    site : str
        The site for which to plot statistics
    stat : str
        The statistic to plot
    sampleFreq : float
        The sampling frequency for which to plot statistics
    declevel : int
        The decimation level to plot
    eFreqI : int
        The evaluation frequency index
    specdir : str
        The spectra directory
    maskname : str
        Mask name         
    clim : List, optional
        Limits for colourbar axis
    xlim : List, optional
        Limits for the x axis
    ylim : List, optional
        Limits for the y axis
    colortitle : str, optional
        Title for the colourbar
    show : bool, optional
        Show the spectra plot
    save : bool, optional
        Save the plot to the images directory
    plotoptions : Dict, optional
        Dictionary of plot options        
    """

    options = {}
    options["declevel"] = 0
    options["eFreqI"] = 0
    options["specdir"] = projData.config.configParams["Spectra"]["specdir"]
    options["maskname"] = ""
    options["clim"] = []
    options["xlim"] = []
    options["ylim"] = []
    options["colortitle"] = ""
    options["show"] = True
    options["save"] = False
    options["plotoptions"] = plotOptionsSpec()
    options = parseKeywords(options, kwargs)

    projectText(
        "Plotting statistic {} for site {} and sampling frequency {}".format(
            stat, site, sampleFreq
        )
    )
    statData = getStatisticDataForSampleFreq(
        projData,
        site,
        sampleFreq,
        stat,
        declevel=options["declevel"],
        specdir=options["specdir"],
    )
    # get the statistics
    statMeas = list(statData.keys())
    # get the evaluation frequency
    eFreq = statData[statMeas[0]].evalFreq[options["eFreqI"]]

    # get the mask data
    maskWindows = []
    if options["maskname"] != "":
        maskData = getMaskData(projData, site, options["maskname"], sampleFreq)
        maskWindows = maskData.getMaskWindowsFreq(
            options["declevel"], options["eFreqI"]
        )

    # setup the figure
    plotfonts = options["plotoptions"]["plotfonts"]
    fig = plt.figure(figsize=options["plotoptions"]["figsize"])

    # get the date limits
    siteData = projData.getSiteData(site)
    if len(options["xlim"]) == 0:
        start = siteData.getMeasurementStart(statMeas[0])
        end = siteData.getMeasurementEnd(statMeas[0])
        for meas in statMeas:
            start = min(start, siteData.getMeasurementStart(meas))
            end = max(end, siteData.getMeasurementEnd(meas))
        options["xlim"] = [start, end]

    # do the plots
    for meas in statMeas:
        statData[meas].view(
            options["eFreqI"],
            fig=fig,
            xlim=options["xlim"],
            ylim=options["ylim"],
            clim=options["clim"],
            label=meas,
            plotfonts=options["plotoptions"]["plotfonts"],
            maskwindows=maskWindows
        )
    # add a legened
    plt.legend()

    # do the title after all the plots
    st = fig.suptitle(
        "{} values for {}, sampling frequency = {:.2f} Hz, decimation level = {} and evaluation frequency {} Hz".format(
            stat, site, sampleFreq, options["declevel"], eFreq
        ),
        fontsize=plotfonts["suptitle"],
    )
    st.set_y(0.98)

    # plot format
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)

    # plot show and save
    if options["save"]:
        impath = projData.imagePath
        sampleFreqStr = fileFormatSampleFreq(sampleFreq)
        filename = "stat_{:s}_{:s}_{:s}_dec{:d}_efreq{:d}_{:s}".format(
            stat,
            site,
            sampleFreqStr,
            options["declevel"],
            options["eFreqI"],
            options["specdir"],
        )
        if options["maskname"] != "":
            filename = "{}_{}".format(filename, options["maskname"])
        filepath = os.path.join(impath, filename)
        fig.savefig(filepath)
        projectText("Image saved to file {}".format(filename))
    if options["show"]:
        plt.show(block=options["plotoptions"]["block"])
    return fig


def viewStatisticHistogram(
    projData: ProjectData, site: str, sampleFreq: float, stat: str, **kwargs
) -> None:
    """View statistic histograms for a single sampling frequency of a site
    
    Parameters
    ----------
    projData : ProjectData
        A project instance
    site : str
        The site for which to plot statistics
    stat : str
        The statistic to plot
    sampleFreq : float
        The sampling frequency for which to plot statistics
    declevel : int
        The decimation level to plot
    eFreqI : int
        The evaluation frequency index       
    specdir : str
        The spectra directory        
    maskname : str
        Mask name 
    numbins : int
        The number of bins for the histogram data binning
    xlim : List, optional
        Limits for the x axis
    maxcols : int
        The maximum number of columns in the plots
    show : bool, optional
        Show the spectra plot
    save : bool, optional
        Save the plot to the images directory
    plotoptions : Dict, optional
        Dictionary of plot options        
    """

    options = {}
    options["declevel"] = 0
    options["eFreqI"] = 0
    options["specdir"] = projData.config.configParams["Spectra"]["specdir"]
    options["maskname"] = ""
    options["numbins"] = 40
    options["xlim"] = []
    options["maxcols"] = 4
    options["show"] = True
    options["save"] = False
    options["plotoptions"] = plotOptionsSpec()
    options = parseKeywords(options, kwargs)

    projectText(
        "Plotting histogram for statistic {}, site {} and sampling frequency {}".format(
            stat, site, sampleFreq
        )
    )

    statData = getStatisticDataForSampleFreq(
        projData,
        site,
        sampleFreq,
        stat,
        declevel=options["declevel"],
        specdir=options["specdir"],
    )
    statMeas = list(statData.keys())
    # get the statistic components
    statComponents = statData[statMeas[0]].winStats
    # get the evaluation frequency
    eFreq = statData[statMeas[0]].evalFreq[options["eFreqI"]]

    # get the mask data
    maskWindows = []
    if options["maskname"] != "":
        maskData = getMaskData(projData, site, options["maskname"], sampleFreq)
        maskWindows = maskData.getMaskWindowsFreq(
            options["declevel"], options["eFreqI"]
        )

    # plot information
    nrows, ncols = getPlotRowsAndCols(options["maxcols"], len(statComponents))
    numbins = options["numbins"]

    plotfonts = options["plotoptions"]["plotfonts"]
    fig = plt.figure(figsize=options["plotoptions"]["figsize"])
    # suptitle
    st = fig.suptitle(
        "{} histogram for {}, sampling frequency {} Hz, decimation level {} and evaluation frequency {} Hz".format(
            stat, site, sampleFreq, options["declevel"], eFreq
        ),
        fontsize=plotfonts["suptitle"],
    )
    st.set_y(0.98)

    # now plot the data
    for idx, val in enumerate(statComponents):
        ax = plt.subplot(nrows, ncols, idx + 1)
        plt.title("Histogram {}".format(val), fontsize=plotfonts["title"])

        plotData = np.empty(shape=(0))
        for meas in statMeas:
            stats = statData[meas].getStats(maskwindows=maskWindows)
            plotData = np.concatenate(
                (plotData, np.squeeze(stats[:, options["eFreqI"], idx]))
            )
        # remove infinities and nans
        plotData = plotData[np.isfinite(plotData)]

        # x axis options
        xlim = (
            kwargs["xlim"] if "xlim" in kwargs else [np.min(plotData), np.max(plotData)]
        )
        plt.xlim(xlim)
        plt.xlabel("Value", fontsize=plotfonts["axisLabel"])
        # now plot with xlim in mind
        plt.hist(plotData, numbins, range=xlim, facecolor="red", alpha=0.75)
        plt.grid()
        # y axis options
        plt.ylabel("Count", fontsize=plotfonts["axisLabel"])
        # set tick sizes
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(plotfonts["axisTicks"])

    # plot format
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)

    # plot show and save
    if options["save"]:
        impath = projData.imagePath
        sampleFreqStr = fileFormatSampleFreq(sampleFreq)
        filename = "statHist_{:s}_{:s}_{:s}_dec{:d}_efreq{:d}_{:s}".format(
            stat,
            site,
            sampleFreqStr,
            options["declevel"],
            options["eFreqI"],
            options["specdir"],
        )
        if options["maskname"] != "":
            filename = "{}_{}".format(filename, options["maskname"])        
        filepath = os.path.join(impath, filename)
        fig.savefig(filepath)
        projectText("Image saved to file {}".format(filename))
    if options["show"]:
        plt.show(block=options["plotoptions"]["block"])
    return fig
