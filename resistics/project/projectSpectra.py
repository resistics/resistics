import sys
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
from typing import List, Union, Dict

# import from package
from resistics.dataObjects.projectData import ProjectData
from resistics.dataObjects.siteData import SiteData
from resistics.dataObjects.spectrumData import SpectrumData
from resistics.calculators.calibrator import Calibrator
from resistics.calculators.decimator import Decimator
from resistics.calculators.windower import Windower
from resistics.calculators.spectrumCalculator import SpectrumCalculator
from resistics.ioHandlers.spectrumReader import SpectrumReader
from resistics.ioHandlers.spectrumWriter import SpectrumWriter
from resistics.utilities.utilsChecks import parseKeywords, isMagnetic
from resistics.utilities.utilsPrint import listToString, breakComment
from resistics.utilities.utilsPlotter import colorbar2dSpectra, plotOptionsSpec
from resistics.utilities.utilsProject import (
    projectText,
    projectError,
    getCalibrator,
    getDecimationParameters,
    getWindowParameters,
    applyCalibrationOptions,
    applyFilterOptions,
    applyNotchOptions,
)


def getSpecReader(
    projData: ProjectData, site: str, meas: str, **kwargs
) -> SpectrumReader:
    """Get the spectrum reader for a measurement

    Parameters
    ----------
    site : str
        Site for which to get the spectra reader
    meas : str
        The measurement
    options : Dict
        Options in a dictionary
    declevel : int, optional
        Decimation level for which to get data
    specdir : str, optional
        String that specifies spectra directory for the measurement

    Returns
    -------
    SpectrumReader
        The spectrum reader object
    """

    options = {}
    options["declevel"]: int = 0
    options["specdir"]: str = projData.config.configParams["Spectra"]["specdir"]
    options = parseKeywords(options, kwargs)

    siteData = projData.getSiteData(site)
    measurements = siteData.getMeasurements()
    if meas not in measurements:
        projectError("Measurement directory {} not found".format(meas), quitRun=True)

    # create the spectrum reader
    specReader = SpectrumReader(
        os.path.join(siteData.getMeasurementSpecPath(meas), options["specdir"])
    )
    specReader.printInfo()

    # open the spectra file for the current decimation level
    check = specReader.openBinaryForReading("spectra", options["declevel"])
    if not check:
        # probably because this decimation level not calculated
        projectError(
            "Spectra file does not exist at level {}".format(options["declevel"]),
            quitRun=True,
        )
    return specReader


def calculateSpectra(projData: ProjectData, **kwargs) -> None:
    """Calculate spectra for the project time data

    The philosophy is that spectra are calculated out for all data and later limited using statistics and time constraints

    Parameters
    ----------
    projData : ProjectData
        A project data object
    sites : str, List[str], optional
        Either a single site or a list of sites
    sampleFreqs : int, float, List[float], optional
        The frequencies in Hz for which to calculate the spectra. Either a single frequency or a list of them.
    chans : List[str], optional
        The channels for which to calculate out the spectra
    calibrate : bool, optional
        Flag whether to calibrate the data or not
    notch : List[float], optional
        List of frequencies to notch
    filter : Dict, optional
        Filter parameters
    specdir : str, optional
        The spectra directory to save the spectra data in
    """

    # default options
    options = {}
    options["sites"] = projData.getSites()
    options["sampleFreqs"] = projData.getSampleFreqs()
    options["chans"] = []
    options["calibrate"] = True
    options["notch"] = []
    options["filter"] = {}
    options["specdir"] = projData.config.configParams["Spectra"]["specdir"]
    options = parseKeywords(options, kwargs)

    # prepare calibrator
    cal = getCalibrator(projData.calPath, projData.config)
    if options["calibrate"]:
        cal.printInfo()

    datetimeRef = projData.refTime
    for site in options["sites"]:
        siteData = projData.getSiteData(site)
        siteData.printInfo()

        # calculate spectra for each frequency
        for sampleFreq in options["sampleFreqs"]:
            measurements = siteData.getMeasurements(sampleFreq)
            projectText(
                "Site {} has {:d} measurement(s) at sampling frequency {:.2f}".format(
                    site, len(measurements), sampleFreq
                )
            )
            if len(measurements) == 0:
                continue  # no data files at this sample rate

            for meas in measurements:
                projectText(
                    "Calculating spectra for site {} and measurement {}".format(
                        site, meas
                    )
                )
                # get measurement start and end times - this is the time of the first and last sample
                reader = siteData.getMeasurement(meas)
                startTime = siteData.getMeasurementStart(meas)
                stopTime = siteData.getMeasurementEnd(meas)
                dataChans = (
                    options["chans"]
                    if len(options["chans"]) > 0
                    else reader.getChannels()
                )
                timeData = reader.getPhysicalData(startTime, stopTime, chans=dataChans)
                timeData.addComment(breakComment())
                timeData.addComment("Calculating project spectra")
                timeData.addComment(projData.config.getConfigComment())

                # apply various options
                applyCalibrationOptions(options, cal, timeData, reader)
                applyFilterOptions(options, timeData)
                applyNotchOptions(options, timeData)

                # define decimation and window parameters
                decParams = getDecimationParameters(sampleFreq, projData.config)
                decParams.printInfo()
                numLevels = decParams.numLevels
                winParams = getWindowParameters(decParams, projData.config)
                dec = Decimator(timeData, decParams)
                timeData.addComment(
                    "Decimating with {} levels and {} frequencies per level".format(
                        numLevels, decParams.freqPerLevel
                    )
                )

                # loop through decimation levels
                for iDec in range(0, numLevels):
                    # get the data for the current level
                    check = dec.incrementLevel()
                    if not check:
                        break  # not enough data
                    timeData = dec.timeData

                    # create the windower and give it window parameters for current level
                    sampleFreqDec = dec.sampleFreq
                    win = Windower(
                        datetimeRef,
                        timeData,
                        winParams.getWindowSize(iDec),
                        winParams.getOverlap(iDec),
                    )
                    if win.numWindows < 2:
                        break  # do no more decimation

                    # add some comments
                    timeData.addComment(
                        "Evaluation frequencies for this level {}".format(
                            listToString(decParams.getEvalFrequenciesForLevel(iDec))
                        )
                    )
                    timeData.addComment(
                        "Windowing with window size {} samples and overlap {} samples".format(
                            winParams.getWindowSize(iDec), winParams.getOverlap(iDec)
                        )
                    )

                    # create the spectrum calculator and statistics calculators
                    specCalc = SpectrumCalculator(
                        sampleFreqDec, winParams.getWindowSize(iDec)
                    )
                    # get ready a file to save the spectra
                    specPath = os.path.join(
                        siteData.getMeasurementSpecPath(meas), options["specdir"]
                    )
                    specWrite = SpectrumWriter(specPath, datetimeRef)
                    specWrite.openBinaryForWriting(
                        "spectra",
                        iDec,
                        sampleFreqDec,
                        winParams.getWindowSize(iDec),
                        winParams.getOverlap(iDec),
                        win.winOffset,
                        win.numWindows,
                        dataChans,
                    )

                    # loop though windows, calculate spectra and save
                    for iW in range(0, win.numWindows):
                        # get the window data
                        winData = win.getData(iW)
                        # calculate spectra
                        specData = specCalc.calcFourierCoeff(winData)
                        # write out spectra
                        specWrite.writeBinary(specData)

                    # close spectra file
                    specWrite.writeCommentsFile(timeData.getComments())
                    specWrite.closeFile()


def viewSpectra(projData: ProjectData, site: str, meas: str, **kwargs) -> plt.figure:
    """View spectra for a measurement

    Parameters
    ----------
    projData : projecData
        The project data
    site : str
        The site to view
    meas: str
        The measurement of the site to view    
    chans : List[str], optional
        Channels to plot
    declevel : int, optional
        Decimation level to plot
    plotwindow : int, str, Dict, optional
        Windows to plot (local). If int, the window with local window plotwindow will be plotted. If string and "all", all the windows will be plotted if there are less than 20 windows, otherwise 20 windows throughout the whole spectra dataset will be splotted. If a dictionary, needs to have start and stop to define a range.
    specdir : str, optional
        String that specifies spectra directory for the measurement
    show : bool, optional
        Show the spectra plot
    save : bool, optional
        Save the plot to the images directory
    plotoptions : Dict, optional
        Dictionary of plot options
    
    Returns
    -------
    matplotlib.pyplot.figure
        Figure handle
    """

    options = {}
    options["chans"]: List[str] = []
    options["declevel"]: int = 0
    options["plotwindow"]: Union[int, Dict, str] = [0]
    options["specdir"]: str = projData.config.configParams["Spectra"]["specdir"]
    options["show"]: bool = True
    options["save"]: bool = False
    options["plotoptions"]: Dict = plotOptionsSpec()
    options = parseKeywords(options, kwargs)

    projectText("Plotting spectra for measurement {} and site {}".format(meas, site))
    specReader = getSpecReader(projData, site, meas, **options)

    # channels
    dataChans = specReader.getChannels()
    if len(options["chans"]) > 0:
        dataChans = options["chans"]
    numChans = len(dataChans)

    # get windows
    numWindows = specReader.getNumWindows()
    sampleFreqDec = specReader.getSampleFreq()

    # get the window data
    windows = options["plotwindow"]
    if isinstance(windows, str) and windows == "all":
        if numWindows > 20:
            windows = list(
                np.linspace(0, numWindows, 20, endpoint=False, dtype=np.int32)
            )
        else:
            windows = list(np.arange(0, numWindows))
    elif isinstance(windows, int):
        windows = [windows]  # if an integer, make it into a list
    elif isinstance(windows, dict):
        windows = list(np.arange(windows["start"], windows["stop"] + 1))

    # create a figure
    plotfonts = options["plotoptions"]["plotfonts"]
    fig = plt.figure(figsize=options["plotoptions"]["figsize"])
    for iW in windows:
        if iW >= numWindows:
            break
        winData = specReader.readBinaryWindowLocal(iW)
        winData.view(
            fig=fig,
            chans=dataChans,
            label="{} to {}".format(
                winData.startTime.strftime("%m-%d %H:%M:%S"),
                winData.stopTime.strftime("%m-%d %H:%M:%S"),
            ),
            plotfonts=plotfonts,
        )

    st = fig.suptitle(
        "Spectra plot, site = {}, meas = {}, fs = {:.2f} [Hz], decimation level = {:2d}".format(
            site, meas, sampleFreqDec, options["declevel"]
        ),
        fontsize=plotfonts["suptitle"],
    )
    st.set_y(0.98)

    # put on axis labels etc
    for idx, chan in enumerate(dataChans):
        ax = plt.subplot(numChans, 1, idx + 1)
        plt.title("Amplitude {}".format(chan), fontsize=plotfonts["title"])
        if len(options["plotoptions"]["amplim"]) == 2:
            ax.set_ylim(options["plotoptions"]["amplim"])
        ax.set_xlim(0, specReader.getSampleFreq() / 2.0)
        plt.grid(True)

    # fig legend and formatting
    ax = plt.gca()
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, ncol=numChans, loc="lower center", fontsize=plotfonts["legend"])
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.subplots_adjust(bottom=0.08 + 0.03 * len(windows) / numChans)

    # plot show and save
    if options["save"]:
        impath = projData.imagePath
        filename = "spectraData_{}_{}_dec{}_{}".format(
            site, meas, options["declevel"], options["specdir"]
        )
        filepath = os.path.join(impath, filename)
        fig.savefig(filepath)
        projectText("Image saved to file {}".format(filename))
    if options["show"]:
        plt.show(block=options["plotoptions"]["block"])
    return fig


def viewSpectraSection(projData: ProjectData, site: str, meas: str, **kwargs):
    """View spectra section for a measurement

    Parameters
    ----------
    projData : projecData
        The project data
    site : str
        The site to view
    meas: str
        The measurement of the site to view    
    chans : List[str], optional
        Channels to plot
    declevel : int, optional
        Decimation level to plot
    specdir : str, optional
        String that specifies spectra directory for the measurement
    show : bool, optional
        Show the spectra plot
    save : bool, optional
        Save the plot to the images directory
    plotoptions : Dict, optional
        Dictionary of plot options
    
    Returns
    -------
    matplotlib.pyplot.figure
        Figure handle
    """

    options = {}
    options["chans"] = []
    options["declevel"] = 0
    options["specdir"] = projData.config.configParams["Spectra"]["specdir"]
    options["show"] = True
    options["save"] = False
    options["plotoptions"] = plotOptionsSpec()
    options = parseKeywords(options, kwargs)

    projectText(
        "Plotting spectra section for measurement {} and site {}".format(meas, site)
    )
    specReader = getSpecReader(projData, site, meas, **options)

    # channels
    dataChans = specReader.getChannels()
    if len(options["chans"]) > 0:
        dataChans = options["chans"]

    # get windows
    numWindows = specReader.getNumWindows()
    sampleFreqDec = specReader.getSampleFreq()

    # freq array
    f = specReader.getFrequencyArray()

    # now if plotting a section, ignore plotwindow for now
    if numWindows > 250:
        windows = list(np.linspace(0, numWindows, 250, endpoint=False, dtype=np.int32))
    else:
        windows = np.arange(0, 250)

    # create figure
    plotfonts = options["plotoptions"]["plotfonts"]
    fig = plt.figure(figsize=options["plotoptions"]["figsize"])
    st = fig.suptitle(
        "Spectra section, site = {}, meas = {}, fs = {:.2f} [Hz], decimation level = {:2d}, windows = {:d}, {} to {}".format(
            site,
            meas,
            sampleFreqDec,
            options["declevel"],
            len(windows),
            windows[0],
            windows[-1],
        ),
        fontsize=plotfonts["suptitle"],
    )
    st.set_y(0.98)

    # collect the data
    specData = np.empty(
        shape=(len(windows), len(dataChans), specReader.getDataSize()), dtype="complex"
    )
    dates = []
    for idx, iW in enumerate(windows):
        winData = specReader.readBinaryWindowLocal(iW)
        for cIdx, chan in enumerate(dataChans):
            specData[idx, cIdx, :] = winData.data[chan]
        dates.append(winData.startTime)

    ampLim = options["plotoptions"]["amplim"]
    for idx, chan in enumerate(dataChans):
        ax = plt.subplot(1, len(dataChans), idx + 1)
        plotData = np.transpose(np.absolute(np.squeeze(specData[:, idx, :])))
        if len(ampLim) == 2:
            plt.pcolor(
                dates,
                f,
                plotData,
                norm=LogNorm(vmin=ampLim[0], vmax=ampLim[1]),
                cmap=colorbar2dSpectra(),
            )
        else:
            plt.pcolor(
                dates,
                f,
                plotData,
                norm=LogNorm(vmin=plotData.min(), vmax=plotData.max()),
                cmap=colorbar2dSpectra(),
            )
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=plotfonts["axisTicks"])
        # set axis limits
        ax.set_ylim(0, specReader.getSampleFreq() / 2.0)
        ax.set_xlim([dates[0], dates[-1]])
        if isMagnetic(chan):
            plt.title("Amplitude {} [nT]".format(chan), fontsize=plotfonts["title"])
        else:
            plt.title("Amplitude {} [mV/km]".format(chan), fontsize=plotfonts["title"])
        ax.set_ylabel("Frequency [Hz]", fontsize=plotfonts["axisLabel"])
        ax.set_xlabel("Time", fontsize=plotfonts["axisLabel"])
        # set tick sizes
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(plotfonts["axisTicks"])
        plt.grid(True)

    # plot format
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)

    # plot show and save
    if options["save"]:
        impath = projData.imagePath
        filename = "spectraSection_{}_{}_dec{}_{}".format(
            site, meas, options["declevel"], options["specdir"]
        )
        filepath = os.path.join(impath, filename)
        fig.savefig(filepath)
        projectText("Image saved to file {}".format(filename))
    if options["show"]:
        plt.show(block=options["plotoptions"]["block"])
    return fig


def viewSpectraStack(
    projData: ProjectData, site: str, meas: str, **kwargs
) -> plt.figure:
    """View spectra stacks for a measurement

    Parameters
    ----------
    projData : projecData
        The project data
    site : str
        The site to view
    meas: str
        The measurement of the site to view
    chans : List[str], optional
        Channels to plot
    declevel : int, optional
        Decimation level to plot
    numstacks : int, optional
        The number of windows to stack
    coherences : List[List[str]], optional
        A list of coherences to add, specified as [["Ex", "Hy"], ["Ey", "Hx"]] 
    specdir : str, optional
        String that specifies spectra directory for the measurement
    show : bool, optional
        Show the spectra plot
    save : bool, optional
        Save the plot to the images directory
    plotoptions : Dict, optional
        Dictionary of plot options
    
    Returns
    -------
    matplotlib.pyplot.figure
        Figure handle
    """

    options = {}
    options["chans"] = []
    options["declevel"] = 0
    options["numstacks"] = 10
    options["coherences"] = []
    options["specdir"] = projData.config.configParams["Spectra"]["specdir"]
    options["show"] = True
    options["save"] = False
    options["plotoptions"] = plotOptionsSpec()
    options = parseKeywords(options, kwargs)

    projectText(
        "Plotting spectra stack for measurement {} and site {}".format(meas, site)
    )
    specReader = getSpecReader(projData, site, meas, **options)

    # channels
    dataChans = specReader.getChannels()
    if len(options["chans"]) > 0:
        dataChans = options["chans"]
    numChans = len(dataChans)

    # get windows
    numWindows = specReader.getNumWindows()
    sampleFreqDec = specReader.getSampleFreq()
    f = specReader.getFrequencyArray()

    # calculate num of windows to stack in each set
    stackSize = int(np.floor(1.0 * numWindows / options["numstacks"]))

    # calculate number of rows - in case interested in coherences too
    nrows = (
        2
        if len(options["coherences"]) == 0
        else 2 + np.ceil(1.0 * len(options["coherences"]) / numChans)
    )

    # setup the figure
    plotfonts = options["plotoptions"]["plotfonts"]
    fig = plt.figure(figsize=options["plotoptions"]["figsize"])
    st = fig.suptitle(
        "Spectra stack, fs = {:.6f} [Hz], decimation level = {:2d}, windows in each set = {:d}".format(
            sampleFreqDec, options["declevel"], stackSize
        ),
        fontsize=plotfonts["suptitle"],
    )
    st.set_y(0.98)

    # do the stacking
    for iP in range(0, options["numstacks"]):
        stackStart = iP * stackSize
        stackStop = min(stackStart + stackSize, numWindows)
        # dictionaries to hold data for this section
        stackedData = {}
        ampData = {}
        phaseData = {}
        powerData = {}

        # assign initial zeros
        for c in dataChans:
            stackedData[c] = np.zeros(shape=(specReader.getDataSize()), dtype="complex")
            ampData[c] = np.zeros(shape=(specReader.getDataSize()), dtype="complex")
            phaseData[c] = np.zeros(shape=(specReader.getDataSize()), dtype="complex")
            for c2 in dataChans:
                powerData[c + c2] = np.zeros(
                    shape=(specReader.getDataSize()), dtype="complex"
                )

        # now stack the data and create nice plots
        for iW in range(stackStart, stackStop):
            winData = specReader.readBinaryWindowLocal(iW)
            for c in dataChans:
                stackedData[c] += winData.data[c]
                ampData[c] += np.absolute(winData.data[c])
                phaseData[c] += np.angle(winData.data[c]) * (180.0 / np.pi)
                # get coherency data
                for c2 in dataChans:
                    powerData[c + c2] += winData.data[c] * np.conjugate(
                        winData.data[c2]
                    )
            if iW == stackStart:
                startTime = winData.startTime
            if iW == stackStop - 1:
                stopTime = winData.stopTime

        # scale powers and stacks
        ampLim = options["plotoptions"]["amplim"]
        for idx, c in enumerate(dataChans):
            stackedData[c] = stackedData[c] / (stackStop - stackStart)
            ampData[c] = ampData[c] / (stackStop - stackStart)
            phaseData[c] = phaseData[c] / (stackStop - stackStart)
            for c2 in dataChans:
                # normalisation
                powerData[c + c2] = 2 * powerData[c + c2] / (stackStop - stackStart)
                # normalisation
                powerData[c + c2][[0, -1]] = powerData[c + c2][[0, -1]] / 2

            # plot
            ax1 = plt.subplot(nrows, numChans, idx + 1)
            plt.title("Amplitude {}".format(c), fontsize=plotfonts["title"])
            h = ax1.semilogy(
                f,
                ampData[c],
                label="{} to {}".format(
                    startTime.strftime("%m-%d %H:%M:%S"),
                    stopTime.strftime("%m-%d %H:%M:%S"),
                ),
            )
            if len(ampLim) > 2:
                ax1.set_ylim(ampLim)
            else:
                ax1.set_ylim(0.01, 1000)
            ax1.set_xlim(0, sampleFreqDec / 2.0)
            if isMagnetic(c):
                ax1.set_ylabel("Amplitude [nT]", fontsize=plotfonts["axisLabel"])
            else:
                ax1.set_ylabel("Amplitude [mV/km]", fontsize=plotfonts["axisLabel"])
            ax1.set_xlabel("Frequency [Hz]", fontsize=plotfonts["axisLabel"])
            plt.grid(True)

            # set tick sizes
            for label in ax1.get_xticklabels() + ax1.get_yticklabels():
                label.set_fontsize(plotfonts["axisTicks"])
            # plot phase
            ax2 = plt.subplot(nrows, numChans, numChans + idx + 1)
            plt.title("Phase {}".format(c), fontsize=plotfonts["title"])
            ax2.plot(
                f,
                phaseData[c],
                label="{} to {}".format(
                    startTime.strftime("%m-%d %H:%M:%S"),
                    stopTime.strftime("%m-%d %H:%M:%S"),
                ),
            )
            ax2.set_ylim(-180, 180)
            ax2.set_xlim(0, sampleFreqDec / 2.0)
            ax2.set_ylabel("Phase [degrees]", fontsize=plotfonts["axisLabel"])
            ax2.set_xlabel("Frequency [Hz]", fontsize=plotfonts["axisLabel"])
            plt.grid(True)
            # set tick sizes
            for label in ax2.get_xticklabels() + ax2.get_yticklabels():
                label.set_fontsize(plotfonts["axisTicks"])

        # plot coherences
        for idx, coh in enumerate(options["coherences"]):
            c = coh[0]
            c2 = coh[1]
            cohNom = np.power(np.absolute(powerData[c + c2]), 2)
            cohDenom = powerData[c + c] * powerData[c2 + c2]
            coherence = cohNom / cohDenom
            ax = plt.subplot(nrows, numChans, 2 * numChans + idx + 1)
            plt.title("Coherence {} - {}".format(c, c2), fontsize=plotfonts["title"])
            ax.plot(
                f,
                coherence,
                label="{} to {}".format(
                    startTime.strftime("%m-%d %H:%M:%S"),
                    stopTime.strftime("%m-%d %H:%M:%S"),
                ),
            )
            ax.set_ylim(0, 1.1)
            ax.set_xlim(0, sampleFreqDec / 2)
            ax.set_ylabel("Coherence", fontsize=plotfonts["axisLabel"])
            ax.set_xlabel("Frequency [Hz]", fontsize=plotfonts["axisLabel"])
            plt.grid(True)
            # set tick sizes
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(plotfonts["axisTicks"])

    # fig legend and layout
    ax = plt.gca()
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, ncol=numChans, loc="lower center", fontsize=plotfonts["legend"])
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.subplots_adjust(bottom=0.05 + 0.03 * options["numstacks"] / numChans)

    # plot show and save
    if options["save"]:
        impath = projData.imagePath
        filename = "spectraStack_{}_{}_dec{}_{}".format(
            site, meas, options["declevel"], options["specdir"]
        )
        filepath = os.path.join(impath, filename)
        fig.savefig(filepath)
        projectText("Image saved to file {}".format(filename))
    if options["show"]:
        plt.show(block=options["plotoptions"]["block"])
    return fig
