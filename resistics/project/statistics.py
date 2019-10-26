import sys
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.figure import Figure
from typing import Dict, Union, List, Any

from resistics.project.data import ProjectData
from resistics.statistics.data import StatisticData
from resistics.statistics.calculator import StatisticCalculator
from resistics.statistics.utils import getStatElements
from resistics.window.selector import WindowSelector
from resistics.spectra.io import SpectrumReader
from resistics.spectra.data import SpectrumData
from resistics.project.mask import getMaskData
from resistics.common.io import fileFormatSampleFreq
from resistics.common.print import listToString
from resistics.common.checks import parseKeywords
from resistics.project.utils import projectText, projectWarning, projectError


def getStatisticData(
    projData: ProjectData, site: str, meas: str, stat: str, declevel: int = 0, **kwargs
) -> StatisticData:
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
    from resistics.statistics.io import StatisticIO

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
) -> List[StatisticData]:
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
    Dict[str, StatisticData]
        A statistic data object
    """
    from resistics.statistics.io import StatisticIO

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
    specdir : str, optional
        The spectra directory for which to calculate statistics
    stats : List[str], optional
        The statistics to calculate out. Acceptable values are: "absvalEqn" "coherence", "psd", "poldir", "transFunc", "resPhase", "partialcoh". Configuration file values are used by default.
    ncores : int, optional
        The number of cores to run the transfer function calculations on        
    """
    from resistics.statistics.io import StatisticIO
    from resistics.project.shortcuts import getDecimationParameters

    options = {}
    options["sites"] = projData.getSites()
    options["sampleFreqs"] = projData.getSampleFreqs()
    options["chans"] = []
    options["specdir"] = projData.config.configParams["Spectra"]["specdir"]
    options["stats"] = projData.config.configParams["Statistics"]["stats"]
    options["ncores"] = projData.config.getStatisticCores()
    options = parseKeywords(options, kwargs)

    projectText(
        "Calculating stats: {} for sites: {}".format(
            listToString(options["stats"]), listToString(options["sites"])
        )
    )
    # loop through sites and calculate statistics
    statIO = StatisticIO()
    for site in options["sites"]:
        siteData = projData.getSiteData(site)
        measurements = siteData.getMeasurements()

        for meas in measurements:
            sampleFreq = siteData.getMeasurementSampleFreq(meas)
            if sampleFreq not in options["sampleFreqs"]:
                continue
            projectText(
                "Calculating stats for site {}, measurement {}".format(site, meas)
            )
            decParams = getDecimationParameters(sampleFreq, projData.config)
            numLevels = decParams.numLevels
            specReader = SpectrumReader(
                os.path.join(siteData.getMeasurementSpecPath(meas), options["specdir"])
            )

            # calculate statistics for decimation level if spectra file exists
            for declevel in range(0, numLevels):
                check = specReader.openBinaryForReading("spectra", declevel)
                if not check:
                    continue
                refTime = specReader.getReferenceTime()
                winSize = specReader.getWindowSize()
                winOlap = specReader.getWindowOverlap()
                numWindows = specReader.getNumWindows()
                sampleFreqDec = specReader.getSampleFreq()
                evalFreq = decParams.getEvalFrequenciesForLevel(declevel)

                # dictionary for saving statistic data
                statData = {}
                for stat in options["stats"]:
                    statElements = getStatElements(stat)
                    statData[stat] = StatisticData(
                        stat, refTime, sampleFreqDec, winSize, winOlap
                    )
                    statData[stat].setStatParams(numWindows, statElements, evalFreq)
                    statData[stat].comments = specReader.getComments()
                    statData[stat].addComment(projData.config.getConfigComment())
                    statData[stat].addComment("Calculating statistic: {}".format(stat))
                    statData[stat].addComment(
                        "Statistic components: {}".format(listToString(statElements))
                    )
                # get all the spectra data in batch and process all the windows
                spectraData, globalIndices = specReader.readBinaryBatchGlobal()
                if options["ncores"] > 0:
                    out = multiStatistics(
                        options["ncores"], spectraData, evalFreq, options["stats"]
                    )
                    for iW in range(numWindows):
                        for stat in options["stats"]:
                            statData[stat].addStat(iW, globalIndices[iW], out[iW][stat])
                else:
                    statCalculator = StatisticCalculator()
                    for iW in range(numWindows):
                        winSpecData = spectraData[iW]
                        winStatData = calculateWindowStatistics(
                            winSpecData,
                            evalFreq,
                            options["stats"],
                            statCalculator=statCalculator,
                        )
                        for stat in options["stats"]:
                            statData[stat].addStat(
                                iW, globalIndices[iW], winStatData[stat]
                            )
                specReader.closeFile()

                # save statistic
                for stat in options["stats"]:
                    statIO.setDatapath(
                        os.path.join(
                            siteData.getMeasurementStatPath(meas), options["specdir"]
                        )
                    )
                    statIO.write(statData[stat], declevel)


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
    specdir : str, optional
        The spectra directory for which to calculate statistics
    remotestats : List[str], optional
        The statistics to calculate out. Acceptable statistics are: "RR_coherence", "RR_coherenceEqn", "RR_absvalEqn", "RR_transferFunction", "RR_resPhase". Configuration file values are used by default.
    """
    from resistics.statistics.io import StatisticIO
    from resistics.statistics.calculator import StatisticCalculator
    from resistics.project.shortcuts import (
        getDecimationParameters,
        getWindowParameters,
        getWindowSelector,
    )

    options = {}
    options["sites"] = projData.getSites()
    options["sampleFreqs"] = projData.getSampleFreqs()
    options["chans"] = []
    options["specdir"] = projData.config.configParams["Spectra"]["specdir"]
    options["remotestats"] = projData.config.configParams["Statistics"]["remotestats"]
    options["ncores"] = projData.config.getStatisticCores()
    options = parseKeywords(options, kwargs)

    projectText(
        "Calculating stats: {} for sites: {} with remote site {}".format(
            listToString(options["remotestats"]),
            listToString(options["sites"]),
            remoteSite,
        )
    )

    statIO = StatisticIO()
    for site in options["sites"]:
        siteData = projData.getSiteData(site)
        measurements = siteData.getMeasurements()

        for meas in measurements:
            sampleFreq = siteData.getMeasurementSampleFreq(meas)
            if sampleFreq not in options["sampleFreqs"]:
                continue
            projectText(
                "Calculating stats for site {}, measurement {} with reference {}".format(
                    site, meas, remoteSite
                )
            )
            # decimation and window parameters
            decParams = getDecimationParameters(sampleFreq, projData.config)
            numLevels = decParams.numLevels
            winParams = getWindowParameters(decParams, projData.config)
            # create the window selector and find the shared windows
            winSelector = getWindowSelector(projData, decParams, winParams)
            winSelector.setSites([site, remoteSite])
            winSelector.calcSharedWindows()
            # create the spectrum reader
            specReader = SpectrumReader(
                os.path.join(siteData.getMeasurementSpecPath(meas), options["specdir"])
            )

            # calculate statistics for decimation level if spectra file exists
            for declevel in range(0, numLevels):
                check = specReader.openBinaryForReading("spectra", declevel)
                if not check:
                    continue
                # information regarding only this spectra file
                refTime = specReader.getReferenceTime()
                winSize = specReader.getWindowSize()
                winOlap = specReader.getWindowOverlap()
                numWindows = specReader.getNumWindows()
                evalFreq = decParams.getEvalFrequenciesForLevel(declevel)
                sampleFreqDec = specReader.getSampleFreq()
                globalOffset = specReader.getGlobalOffset()

                # find size of the intersection between the windows in this spectra file and the shared windows
                sharedWindows = winSelector.getSharedWindowsLevel(declevel)
                sharedWindowsMeas = sharedWindows.intersection(
                    set(np.arange(globalOffset, globalOffset + numWindows))
                )
                sharedWindowsMeas = sorted(list(sharedWindowsMeas))
                numSharedWindows = len(sharedWindowsMeas)

                statData = {}
                # create the statistic handlers
                for stat in options["remotestats"]:
                    statElements = getStatElements(stat)
                    statData[stat] = StatisticData(
                        stat, refTime, sampleFreqDec, winSize, winOlap
                    )
                    # with remote reference the number of windows is number of shared windows
                    statData[stat].setStatParams(
                        numSharedWindows, statElements, evalFreq
                    )
                    statData[stat].comments = specReader.getComments()
                    statData[stat].addComment(projData.config.getConfigComment())
                    statData[stat].addComment(
                        "Calculating remote statistic: {}".format(stat)
                    )
                    statData[stat].addComment(
                        "Statistic components: {}".format(listToString(statElements))
                    )

                # collect the spectra data
                spectraData, _globalIndices = specReader.readBinaryBatchGlobal(
                    sharedWindowsMeas
                )
                remoteData = []
                for globalWindow in sharedWindowsMeas:
                    _, remoteReader = winSelector.getSpecReaderForWindow(
                        remoteSite, declevel, globalWindow
                    )
                    remoteData.append(remoteReader.readBinaryWindowGlobal(globalWindow))

                # calculate
                if options["ncores"] > 0:
                    out = multiStatistics(
                        options["ncores"],
                        spectraData,
                        evalFreq,
                        options["remotestats"],
                        remoteData=remoteData,
                    )
                    for iW, globalWindow in enumerate(sharedWindowsMeas):
                        for stat in options["remotestats"]:
                            statData[stat].addStat(iW, globalWindow, out[iW][stat])
                else:
                    statCalculator = StatisticCalculator()
                    for iW, globalWindow in enumerate(sharedWindowsMeas):
                        winStatData = calculateWindowStatistics(
                            spectraData[iW],
                            evalFreq,
                            options["remotestats"],
                            remoteSpecData=remoteData[iW],
                            statCalculator=statCalculator,
                        )
                        for stat in options["remotestats"]:
                            statData[stat].addStat(iW, globalWindow, winStatData[stat])

                # save statistic
                for stat in options["remotestats"]:
                    statIO.setDatapath(
                        os.path.join(
                            siteData.getMeasurementStatPath(meas), options["specdir"]
                        )
                    )
                    statIO.write(statData[stat], declevel)


def calculateWindowStatistics(
    winSpecData: SpectrumData,
    evalFreq: np.ndarray,
    stats: List[str],
    remoteSpecData: Union[SpectrumData, None] = None,
    statCalculator: Union[StatisticCalculator, None] = None,
) -> Dict[str, Any]:
    """Calculate the statistics for a single window
    
    This method is meant to be used with multiprocessing

    Parameters
    ----------
    winSpecData : SpectrumData
        Spectrum data for a single window
    evalFreq : np.ndarray
        Array for evaluation frequencies
    stats : List[str]
        List of statistics to calculate
    remoteSpecData : None, SpectrumData
        Optionally add remote spectrum data to calculate remote statistics
    statCalculator : None, StatisticCalculator
        Provide a statistic calculator to avoid multiple creation of objects
    """
    if statCalculator is None:
        statCalculator = StatisticCalculator()
    statCalculator.setSpectra(winSpecData.freqArray, winSpecData, evalFreq)
    if remoteSpecData is not None:
        statCalculator.addRemoteSpec(remoteSpecData)
    data = {}
    for stat in stats:
        data[stat] = statCalculator.getDataForStatName(stat)
    return data


def multiStatistics(
    ncores: int,
    spectraData: List[SpectrumData],
    evalFreq: np.ndarray,
    stats: List[str],
    remoteData: Union[List[SpectrumData], None] = None,
):
    """Multiprocessing of statistics
    
    Parameters
    ----------
    ncores : int
        The number of cores to use    
    spectraData : List[SpectrumData]
        List of spectrum data to process
    evalFreq : np.ndarray
        The evaluation frequencies
    stats : List[str]
        The statistics to calculate
    remoteData : None, List[SpectrumData]
        Remote data in case of remote reference statistics. None is default

    Returns
    -------
    List[Dict[str, Any]]
        The statistic data returned as a list with an entry for every window. The dictionary maps the statistic name to the data for that window.
    """
    import multiprocessing as mp

    # prepare the lists
    if remoteData is None:
        multiTuples = [(winSpecData, evalFreq, stats) for winSpecData in spectraData]
    else:
        multiTuples = [
            (winSpecData, evalFreq, stats, remoteSpecData)
            for winSpecData, remoteSpecData in zip(spectraData, remoteData)
        ]
    # multiprocess
    projectText("Running statistic calculations on {} cores".format(ncores))
    with mp.Pool(ncores) as pool:
        out = pool.starmap(calculateWindowStatistics, multiTuples)
    return out


def viewStatistic(
    projData: ProjectData, site: str, sampleFreq: Union[int, float], stat: str, **kwargs
) -> Union[Figure, None]:
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

    Returns
    -------
    matplotlib.pyplot.figure or None
        A matplotlib figure unless the plot is not shown and is saved, in which case None and the figure is closed.
    """
    from resistics.common.plot import savePlot, plotOptionsSpec, getPlotRowsAndCols

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
            maskwindows=maskWindows,
        )
    # add a legened
    plt.legend(markerscale=4, fontsize=plotfonts["legend"])

    # do the title after all the plots
    fig.suptitle(
        "{} values for {}, sampling frequency = {:.2f} Hz, decimation level = {} and evaluation frequency {} Hz".format(
            stat, site, sampleFreq, options["declevel"], eFreq
        ),
        fontsize=plotfonts["suptitle"],
    )

    # plot format, show and save
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])
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
        savename = savePlot(impath, filename, fig)
        projectText("Image saved to file {}".format(savename))
    if options["show"]:
        plt.show(block=options["plotoptions"]["block"])
    if not options["show"] and options["save"]:
        plt.close(fig)
        return None
    return fig


def viewStatisticHistogram(
    projData: ProjectData, site: str, sampleFreq: float, stat: str, **kwargs
) -> Union[Figure, None]:
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

    Returns
    -------
    matplotlib.pyplot.figure or None
        A matplotlib figure unless the plot is not shown and is saved, in which case None.
    """
    from resistics.common.plot import savePlot, plotOptionsSpec, getPlotRowsAndCols

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
            options["xlim"]
            if len(options["xlim"]) > 0
            else [np.min(plotData), np.max(plotData)]
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

    # plot format, show and save
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])
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
        savename = savePlot(impath, filename, fig)
        projectText("Image saved to file {}".format(savename))
    if options["show"]:
        plt.show(block=options["plotoptions"]["block"])
    if not options["show"] and options["save"]:
        plt.close(fig)
        return None
    return fig


def viewStatisticCrossplot(
    projData: ProjectData,
    site: str,
    sampleFreq: Union[int, float],
    stat: str,
    crossplots: List[List[str]],
    **kwargs
) -> Union[Figure, None]:
    """View statistic crossplot data for a single sampling frequency of a site
    
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
    crossplots : List[List[str]]
        The statistic element pairs to crossplot
    declevel : int
        The decimation level to plot
    eFreqI : int
        The evaluation frequency index
    specdir : str
        The spectra directory
    maskname : str
        Mask name         
    xlim : List, optional
        Limits for the x axis
    ylim : List, optional
        Limits for the y axis
    maxcols : int
        The maximum number of columns in the plots        
    show : bool, optional
        Show the spectra plot
    save : bool, optional
        Save the plot to the images directory
    plotoptions : Dict, optional
        Dictionary of plot options    

    Returns
    -------
    matplotlib.pyplot.figure or None
        A matplotlib figure unless the plot is not shown and is saved, in which case None and the figure is closed.
    """
    from resistics.common.plot import savePlot, plotOptionsSpec, getPlotRowsAndCols

    options = {}
    options["declevel"] = 0
    options["eFreqI"] = 0
    options["specdir"] = projData.config.configParams["Spectra"]["specdir"]
    options["maskname"] = ""
    options["xlim"] = []
    options["ylim"] = []
    options["maxcols"] = 2
    options["show"] = True
    options["save"] = False
    options["plotoptions"] = plotOptionsSpec()
    options = parseKeywords(options, kwargs)

    projectText(
        "Plotting crossplot for statistic {}, site {} and sampling frequency {}".format(
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
    nrows, ncols = getPlotRowsAndCols(options["maxcols"], len(crossplots))

    plotfonts = options["plotoptions"]["plotfonts"]
    fig = plt.figure(figsize=options["plotoptions"]["figsize"])
    # suptitle
    st = fig.suptitle(
        "{} crossplots for {}, sampling frequency {} Hz, decimation level {} and evaluation frequency {} Hz".format(
            stat, site, sampleFreq, options["declevel"], eFreq
        ),
        fontsize=plotfonts["suptitle"],
    )
    st.set_y(0.98)

    # now plot the data
    for idx, cplot in enumerate(crossplots):
        ax = plt.subplot(nrows, ncols, idx + 1)
        plt.title("Crossplot {}".format(cplot), fontsize=plotfonts["title"])

        for meas in statMeas:
            stats = statData[meas].getStats(maskwindows=maskWindows)
            plotI1 = statData[meas].winStats.index(cplot[0])
            plotData1 = np.squeeze(stats[:, options["eFreqI"], plotI1])
            plotI2 = statData[meas].winStats.index(cplot[1])
            plotData2 = np.squeeze(stats[:, options["eFreqI"], plotI2])
            scat = plt.scatter(
                plotData1, plotData2, edgecolors="none", marker="o", s=12, label=meas
            )

        # x axis options
        if len(options["xlim"]) > 0:
            plt.xlim(options["xlim"])
        if len(options["ylim"]) > 0:
            plt.ylim(options["ylim"])
        plt.xlabel(cplot[0], fontsize=plotfonts["axisLabel"])
        plt.ylabel(cplot[1], fontsize=plotfonts["axisLabel"])
        plt.grid(True)
        # set tick sizes
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(plotfonts["axisTicks"])
        plt.legend(loc=2, markerscale=4, fontsize=plotfonts["legend"])

    # plot format, show and save
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])
    if options["save"]:
        impath = projData.imagePath
        sampleFreqStr = fileFormatSampleFreq(sampleFreq)
        filename = "statCrossplot_{:s}_{:s}_{:s}_dec{:d}_efreq{:d}_{:s}".format(
            stat,
            site,
            sampleFreqStr,
            options["declevel"],
            options["eFreqI"],
            options["specdir"],
        )
        if options["maskname"] != "":
            filename = "{}_{}".format(filename, options["maskname"])
        savename = savePlot(impath, filename, fig)
        projectText("Image saved to file {}".format(savename))
    if options["show"]:
        plt.show(block=options["plotoptions"]["block"])
    if not options["show"] and options["save"]:
        plt.close(fig)
        return None
    return fig


def viewStatisticDensityplot(
    projData: ProjectData,
    site: str,
    sampleFreq: Union[int, float],
    stat: str,
    crossplots: List[List[str]],
    **kwargs
) -> Union[Figure, None]:
    """View statistic data as a density plot for a single sampling frequency of a site
    
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
    crossplots : List[List[str]]
        The statistic element pairs to crossplot
    declevel : int
        The decimation level to plot
    eFreqI : int
        The evaluation frequency index
    specdir : str
        The spectra directory
    maskname : str
        Mask name         
    xlim : List, optional
        Limits for the x axis
    ylim : List, optional
        Limits for the y axis
    maxcols : int
        The maximum number of columns in the plots        
    show : bool, optional
        Show the spectra plot
    save : bool, optional
        Save the plot to the images directory
    plotoptions : Dict, optional
        Dictionary of plot options    

    Returns
    -------
    matplotlib.pyplot.figure or None
        A matplotlib figure unless the plot is not shown and is saved, in which case None and the figure is closed.
    """
    from resistics.common.plot import (
        savePlot,
        plotOptionsSpec,
        getPlotRowsAndCols,
        colorbar2dSpectra,
    )
    from scipy.stats import kde
    from scipy.interpolate import interpn
    import matplotlib.colors as mcolors

    options = {}
    options["declevel"] = 0
    options["eFreqI"] = 0
    options["specdir"] = projData.config.configParams["Spectra"]["specdir"]
    options["maskname"] = ""
    options["xlim"] = []
    options["ylim"] = []
    options["maxcols"] = 2
    options["show"] = True
    options["save"] = False
    options["plotoptions"] = plotOptionsSpec()
    options = parseKeywords(options, kwargs)

    projectText(
        "Plotting density plot for statistic {}, site {} and sampling frequency {}".format(
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
    nrows, ncols = getPlotRowsAndCols(options["maxcols"], len(crossplots))

    plotfonts = options["plotoptions"]["plotfonts"]
    fig = plt.figure(figsize=options["plotoptions"]["figsize"])
    # suptitle
    st = fig.suptitle(
        "{} density plots for {}, sampling frequency {} Hz, decimation level {} and evaluation frequency {} Hz".format(
            stat, site, sampleFreq, options["declevel"], eFreq
        ),
        fontsize=plotfonts["suptitle"],
    )
    st.set_y(0.98)

    # now plot the data
    for idx, cplot in enumerate(crossplots):
        ax = plt.subplot(nrows, ncols, idx + 1)
        plt.title("Crossplot {}".format(cplot), fontsize=plotfonts["title"])

        plotAll1 = []
        plotAll2 = []
        for meas in statMeas:
            stats = statData[meas].getStats(maskwindows=maskWindows)
            plotI1 = statData[meas].winStats.index(cplot[0])
            plotData1 = np.squeeze(stats[:, options["eFreqI"], plotI1])
            plotI2 = statData[meas].winStats.index(cplot[1])
            plotData2 = np.squeeze(stats[:, options["eFreqI"], plotI2])
            # add to all data
            if plotData1.size == 0:
                continue
            if plotData1.size == 1:
                plotAll1 = plotAll1 + [float(plotData1)]
                plotAll2 = plotAll2 + [float(plotData2)]
            else:
                plotAll1 = plotAll1 + plotData1.tolist()
                plotAll2 = plotAll2 + plotData2.tolist()

        plotAll1 = np.array(plotAll1)
        plotAll2 = np.array(plotAll2)

        nbins = 200
        if len(options["xlim"]) > 0:
            plt.xlim(options["xlim"])
            rangex = options["xlim"]
        else:
            minx = np.percentile(plotAll1, 2)
            maxx = np.percentile(plotAll1, 98)
            ax.set_xlim(minx, maxx)
            rangex = [minx, maxx]

        if len(options["ylim"]) > 0:
            plt.ylim(options["ylim"])
            rangey = options["ylim"]
        else:
            miny = np.percentile(plotAll2, 2)
            maxy = np.percentile(plotAll2, 98)
            ax.set_ylim(miny, maxy)
            rangey = [miny, maxy]

        plt.hist2d(
            plotAll1,
            plotAll2,
            bins=(nbins, nbins),
            range=[rangex, rangey],
            cmap=plt.cm.inferno,
        )

        # axis format
        plt.xlabel(cplot[0], fontsize=plotfonts["axisLabel"])
        plt.ylabel(cplot[1], fontsize=plotfonts["axisLabel"])
        plt.grid(True)
        # set tick sizes
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(plotfonts["axisTicks"])

    # plot format, show and save
    # fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])
    if options["save"]:
        impath = projData.imagePath
        sampleFreqStr = fileFormatSampleFreq(sampleFreq)
        filename = "statDensityplot_{:s}_{:s}_{:s}_dec{:d}_efreq{:d}_{:s}".format(
            stat,
            site,
            sampleFreqStr,
            options["declevel"],
            options["eFreqI"],
            options["specdir"],
        )
        if options["maskname"] != "":
            filename = "{}_{}".format(filename, options["maskname"])
        savename = savePlot(impath, filename, fig)
        projectText("Image saved to file {}".format(savename))
    if options["show"]:
        plt.show(block=options["plotoptions"]["block"])
    if not options["show"] and options["save"]:
        plt.close(fig)
        return None
    return fig
