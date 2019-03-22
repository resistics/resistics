import sys
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Union

# import from package
from resistics.dataObjects.projectData import ProjectData
from resistics.dataObjects.siteData import SiteData
from resistics.dataObjects.transferFunctionData import TransferFunctionData
from resistics.calculators.decimator import Decimator
from resistics.calculators.windowSelector import WindowSelector
from resistics.calculators.processorSingleSite import ProcessorSingleSite
from resistics.calculators.processorRemoteReference import ProcessorRemoteReference
from resistics.ioHandlers.spectrumReader import SpectrumReader
from resistics.ioHandlers.transferFunctionReader import TransferFunctionReader
from resistics.ioHandlers.transferFunctionWriter import TransferFunctionWriter
from resistics.utilities.utilsChecks import parseKeywords, isElectric, isMagnetic
from resistics.utilities.utilsIO import checkFilepath, fileFormatSampleFreq
from resistics.utilities.utilsPrint import arrayToString
from resistics.utilities.utilsPlotter import (
    plotOptionsTransferFunction,
    getTransferFunctionFigSize,
    plotOptionsTipper,
    transferFunctionColours,
)
from resistics.utilities.utilsProject import (
    projectText,
    projectWarning,
    projectError,
    getDecimationParameters,
    getWindowParameters,
    getSingleSiteProcessor,
    getRemoteReferenceProcessor,
)


def getTransferFunctionData(
    projData: ProjectData, site: str, sampleFreq: float, **kwargs
) -> TransferFunctionData:
    """Get transfer function data

    Parameters
    ----------
    projData : projecData
        The project data
    site : str
        Site to get the transfer functiond data for
    sampleFreq : int, float
        The sampling frequency for which to get the transfer function data
    specdir : str, optional
        The spectra directories used
    postpend : str, optional
        The postpend on the transfer function files
    """

    options = {}
    options["specdir"] = projData.config.configParams["Spectra"]["specdir"]
    options["postpend"] = ""
    options = parseKeywords(options, kwargs)

    # deal with the postpend
    if options["postpend"] != "":
        postpend = "_{}".format(options["postpend"])
    else:
        postpend = options["postpend"]

    siteData = projData.getSiteData(site)
    sampleFreqStr = fileFormatSampleFreq(sampleFreq)
    path = os.path.join(
        siteData.transFuncPath,
        "{:s}".format(sampleFreqStr),
        "{}_fs{:s}_{}{}".format(site, sampleFreqStr, options["specdir"], postpend),
    )
    # check path
    if not checkFilepath(path):
        projectWarning("No transfer function file with name {}".format(path))
        return False

    projectText(
        "Reading transfer function for site {}, sample frequency {}, file {}".format(
            site, sampleFreq, path
        )
    )

    tfReader = TransferFunctionReader(path)
    tfReader.printInfo()
    return tfReader.tfData


def processProject(projData: ProjectData, **kwargs) -> None:
    """Process a project

    Parameters
    ----------
    projData : ProjectData
        The project data instance for the project    
    sites : List[str], optional
        List of sites 
    sampleFreqs : List[float], optional
        List of sample frequencies to plot
    specdir : str, optional
        The spectra directories to use
    inchans : List[str], optional
        Channels to use as the input of the linear system
    inputsite : str, optional
        Site from which to take the input channels. The default is to use input and output channels from the same site        
    outchans : List[str], optional
        Channels to use as the output of the linear system
    remotesite : str, optional
        The site to use as the remote site
    remotechans : List[str], optional
        Channels to use from the remote reference site
    crosschannels : List[str], optional
        List of channels to use for cross powers
    masks : Dict, optional
        Masks dictionary for passing mask data        
    postpend : str, optional
        String to postpend to the transfer function output
    """

    options = {}
    options["sites"] = projData.getSites()
    options["sampleFreqs"] = projData.getSampleFreqs()
    options["specdir"] = projData.config.configParams["Spectra"]["specdir"]
    options["inchans"] = ["Hx", "Hy"]
    options["inputsite"] = ""
    options["outchans"] = ["Ex", "Ey"]
    options["remotesite"] = ""
    options["remotechans"] = options["inchans"]
    options["crosschannels"] = []
    options["masks"] = {}
    options["postpend"] = ""
    options = parseKeywords(options, kwargs)

    for site in options["sites"]:
        siteData = projData.getSiteData(site)
        siteFreqs = siteData.getSampleFreqs()
        for sampleFreq in siteFreqs:
            # check if not included
            if sampleFreq not in options["sampleFreqs"]:
                continue
            processSite(projData, site, sampleFreq, **options)


def processSite(
    projData: ProjectData, site: str, sampleFreq: Union[int, float], **kwargs
):
    """Process a single sampling frequency for a site

    The site passed is assumed to be the output site (the output channels will come from this site). If channels from a different site are desired to be used as the input channels, this can be done by specifying the optional inputsite argument.

    .. todo:: Give a few different examples here

    Parameters
    ----------
    projData : ProjectData
        The project data instance for the project
    site : str
        Site to process 
    sampleFreq : float, int
        Sample frequency to process
    specdir : str, optional
        The spectra directories to use
    inchans : List[str], optional
        Channels to use as the input of the linear system
    inputsite : str, optional
        Site from which to take the input channels. The default is to use input and output channels from the same site
    outchans : List[str], optional
        Channels to use as the output of the linear system
    remotesite : str, optional
        The site to use as the remote site
    remotechans : List[str], optional
        Channels to use from the remote reference site
    crosschannels : List[str], optional
        List of channels to use for cross powers
    masks : Dict, optional
        Masks dictionary for passing mask data
    postpend : str, optional
        String to postpend to the transfer function output
    """

    options = {}
    options["specdir"] = projData.config.configParams["Spectra"]["specdir"]
    options["inchans"] = ["Hx", "Hy"]
    options["inputsite"] = ""
    options["outchans"] = ["Ex", "Ey"]
    options["remotesite"] = ""
    options["remotechans"] = options["inchans"]
    options["crosschannels"] = []
    options["masks"] = {}
    options["postpend"] = ""
    options = parseKeywords(options, kwargs)
    if options["inputsite"] == "":
        options["inputsite"] = site

    projectText("Processing site {}, sampling frequency {}".format(site, sampleFreq))
    siteData = projData.getSiteData(site)

    # define decimation parameters
    decParams = getDecimationParameters(sampleFreq, projData.config)
    decParams.printInfo()
    winParams = getWindowParameters(decParams, projData.config)

    # window selector
    winSelector = WindowSelector(
        projData, sampleFreq, decParams, winParams, specdir=options["specdir"]
    )

    # if two sites are duplicated (e.g. input site and output site), winSelector only uses distinct sites. Hence using site and inputSite is no problem even if they are the same
    processSites = []
    if options["remotesite"]:
        processSites = [site, options["inputsite"], options["remotesite"]]
        winSelector.setSites(processSites)
    else:
        # if no remote site, then single site processing
        processSites = [site, options["inputsite"]]
        winSelector.setSites(processSites)

    # add window masks
    if len(list(options["masks"].keys())) > 0:
        for maskSite in options["masks"]:
            if maskSite not in processSites:
                # there is a site in the masks dictionary which is of no interest
                continue
            if isinstance(options["masks"][maskSite], str):
                # a single mask
                winSelector.addWindowMask(maskSite, options["masks"][maskSite])
                continue
            if all(isinstance(item, str) for item in options["masks"][maskSite]):
                # list of masks for the site
                for mask in options["masks"][maskSite]:
                    winSelector.addWindowMask(maskSite, mask)

    # calculate the shared windows and print info
    winSelector.calcSharedWindows()
    winSelector.printInfo()
    winSelector.printDatetimeConstraints()
    winSelector.printWindowMasks()
    winSelector.printSharedWindows()
    winSelector.printWindowsForFrequency()

    # now have the windows, pass the winSelector to processors
    outPath = siteData.transFuncPath
    if options["remotesite"]:
        projectText(
            "Remote reference processing with sites: in = {}, out = {}, reference = {}".format(
                options["inputsite"], site, options["remotesite"]
            )
        )
        processor = getRemoteReferenceProcessor(winSelector, outPath, projData.config)
        processor.setRemote(options["remotesite"], options["remotechans"])
    else:
        projectText(
            "Single site processing with sites: in = {}, out = {}".format(
                options["inputsite"], site
            )
        )
        processor = getSingleSiteProcessor(winSelector, outPath, projData.config)

    # add the input and output site
    processor.setInput(options["inputsite"], options["inchans"])
    processor.setOutput(site, options["outchans"])
    if len(options["crosschannels"]) > 0:
        processor.crossChannels = options["crosschannels"]
    processor.postpend = options["postpend"]
    processor.printInfo()
    processor.process()


def viewTransferFunction(projData: ProjectData, **kwargs) -> None:
    """View transfer function data

    Parameters
    ----------
    projData : projecData
        The project data
    sites : List[str], optional
        List of sites to plot transfer functions for
    sampleFreqs : List[float], optional 
        List of samples frequencies for which to plot transfer functions
    polarisations : List[str], optional 
        A list of polarisations to plot. For example, ["ExHx", "ExHy", "EyHx", "EyHy"]
    specdir : str, optional
        The spectra directories used
    postpend : str, optional
        The postpend on the transfer function files
    oneplot : bool, optional
        Plot the polarisation on a single plot
    show : bool, optional
        Show the spectra plot
    save : bool, optional
        Save the plot to the images directory
    plotoptions : Dict
        A dictionary of plot options. For example, set the resistivity y limits using res_ylim, set the phase y limits using phase_ylim and set the xlimits using xlim
    """

    options = {}
    options["sites"] = projData.getSites()
    options["sampleFreqs"] = projData.getSampleFreqs()
    options["polarisations"] = ["ExHx", "ExHy", "EyHx", "EyHy"]
    options["specdir"] = projData.config.configParams["Spectra"]["specdir"]
    options["postpend"] = ""
    options["oneplot"] = True
    options["save"] = False
    options["show"] = True
    options["plotoptions"] = plotOptionsTransferFunction()
    options = parseKeywords(options, kwargs)

    # loop over sites
    for site in options["sites"]:
        siteData = projData.getSiteData(site)
        sampleFreqs = set(siteData.getSampleFreqs())
        # find the intersection with the options["freqs"]
        sampleFreqs = sampleFreqs.intersection(options["sampleFreqs"])
        sampleFreqs = sorted(list(sampleFreqs))
        print(sampleFreqs)

        # if prepend is a string, then make it a list
        if isinstance(options["postpend"], str):
            options["postpend"] = [options["postpend"]]

        plotfonts = options["plotoptions"]["plotfonts"]
        # now loop over the postpend options
        for pp in options["postpend"]:
            # add an underscore if not empty
            postpend = "_{}".format(pp) if pp != "" else pp

            if options["plotoptions"]["figsize"] is None:
                figsize = getTransferFunctionFigSize(
                    options["oneplot"], len(options["polarisations"])
                )
            else:
                figsize = options["plotoptions"]["figsize"]
            fig = plt.figure(figsize=figsize)
            mks = ["o", "*", "d", "^", "h"]
            lstyles = ["solid", "dashed", "dashdot", "dotted"]
            colours = transferFunctionColours()

            # loop over sampling frequencies
            includedFreqs = []
            for idx, sampleFreq in enumerate(sampleFreqs):

                tfData = getTransferFunctionData(
                    projData, site, sampleFreq, specdir=options["specdir"], postpend=pp
                )
                if not tfData:
                    continue

                includedFreqs.append(sampleFreq)
                projectText(
                    "Plotting transfer function for site {}, sample frequency {}".format(
                        site, sampleFreq
                    )
                )

                # plot
                mk = mks[idx % len(mks)]
                ls = lstyles[idx % len(lstyles)]
                tfData.view(
                    fig=fig,
                    polarisations=options["polarisations"],
                    mk=mk,
                    ls=ls,
                    colours=colours,
                    oneplot=options["oneplot"],
                    res_ylim=options["plotoptions"]["res_ylim"],
                    phase_ylim=options["plotoptions"]["phase_ylim"],
                    xlim=options["plotoptions"]["xlim"],
                    label="{}".format(sampleFreq),
                )

            # check if any files found
            if len(includedFreqs) == 0:
                continue

            # sup title
            sub = "Site {}: {}".format(site, options["specdir"] + postpend)
            sub = "{}\nfs = {}".format(sub, arrayToString(includedFreqs, decimals=3))
            st = fig.suptitle(sub, fontsize=plotfonts["suptitle"])
            st.set_y(0.99)
            fig.tight_layout()
            fig.subplots_adjust(top=0.92)

            if options["save"]:
                imPath = projData.imagePath
                filename = "transFunction_{}_{}{}.png".format(
                    site, options["specdir"], postpend
                )
                fig.savefig(os.path.join(imPath, filename))
                projectText("Image saved to file {}".format(filename))

        if not options["show"]:
            plt.close("all")
        else:
            plt.show(block=options["plotoptions"]["block"])


def viewTipper(projData: ProjectData, **kwargs) -> None:
    """View transfer function data

    Parameters
    ----------
    projData : projecData
        The project data
    sites : List[str], optional
        List of sites to plot transfer functions for
    sampleFreqs : List[float], optional 
        List of samples frequencies for which to plot transfer functions
    specdir : str, optional
        The spectra directories used
    postpend : str, optional
        The postpend on the transfer function files
    cols : bool, optional
        Boolean flag, True to arrange tipper plot as 1 row with 3 columns
    show : bool, optional
        Show the spectra plot
    save : bool, optional
        Save the plot to the images directory
    plotoptions : Dict
        A dictionary of plot options. For example, set the resistivity y limits using res_ylim, set the phase y limits using phase_ylim and set the xlimits using xlim
    """

    options = {}
    options["sites"] = projData.getSites()
    options["sampleFreqs"] = projData.getSampleFreqs()
    options["specdir"] = projData.config.configParams["Spectra"]["specdir"]
    options["postpend"] = ""
    options["cols"] = True
    options["save"] = False
    options["show"] = True
    options["plotoptions"] = plotOptionsTipper()
    options = parseKeywords(options, kwargs)

    # loop over sites
    for site in options["sites"]:
        siteData = projData.getSiteData(site)
        sampleFreqs = set(siteData.getSampleFreqs())
        # find the intersection with the options["freqs"]
        sampleFreqs = sampleFreqs.intersection(options["sampleFreqs"])
        sampleFreqs = sorted(list(sampleFreqs))

        # if prepend is a string, then make it a list
        if isinstance(options["postpend"], str):
            options["postpend"] = [options["postpend"]]

        plotfonts = options["plotoptions"]["plotfonts"]
        # now loop over the postpend options
        for pp in options["postpend"]:
            # add an underscore if not empty
            postpend = "_{}".format(pp) if pp != "" else pp

            fig = plt.figure(figsize=options["plotoptions"]["figsize"])
            mks = ["o", "*", "d", "^", "h"]
            lstyles = ["solid", "dashed", "dashdot", "dotted"]

            # loop over sampling frequencies
            includedFreqs = []
            for idx, sampleFreq in enumerate(sampleFreqs):

                tfData = getTransferFunctionData(
                    projData, site, sampleFreq, specdir=options["specdir"], postpend=pp
                )
                if not tfData:
                    continue

                includedFreqs.append(sampleFreq)
                projectText(
                    "Plotting tipper for site {}, sample frequency {}".format(
                        site, sampleFreq
                    )
                )

                mk = mks[idx % len(mks)]
                ls = lstyles[idx % len(lstyles)]
                tfData.viewTipper(
                    fig=fig,
                    rows=options["cols"],
                    mk=mk,
                    ls=ls,
                    label="{}".format(sampleFreq),
                    xlim=options["plotoptions"]["xlim"],
                    length_ylim=options["plotoptions"]["length_ylim"],
                    angle_ylim=options["plotoptions"]["angle_ylim"],
                )

            # check if any files found
            if len(includedFreqs) == 0:
                continue

            # sup title
            sub = "Site {} tipper: {}".format(site, options["specdir"] + postpend)
            sub = "{}\nfs = {}".format(sub, arrayToString(includedFreqs, decimals=3))
            st = fig.suptitle(sub, fontsize=plotfonts["suptitle"])
            st.set_y(0.99)
            fig.tight_layout()
            fig.subplots_adjust(top=0.85)

            if options["save"]:
                imPath = projData.imagePath
                filename = "tipper_{}_{}{}.png".format(
                    site, options["specdir"], postpend
                )
                fig.savefig(os.path.join(imPath, filename))
                projectText("Image saved to file {}".format(filename))

        if not options["show"]:
            plt.close("all")
        else:
            plt.show(block=options["plotoptions"]["block"])

