import os
from resistics.project.projectIO import loadProject
from resistics.project.projectTransferFunction import (
    processProject,
    viewTransferFunction,
)

# need the project path for loading
projectPath = os.path.join("tutorialProject")
projData = loadProject(projectPath)

# calculate another set of spectra for the 128 Hz data with notching at 50Hz and 16.667Hz
from resistics.project.projectSpectra import calculateSpectra

calculateSpectra(projData, sampleFreqs=[128], notch=[16.667, 50], specdir="notch")
projData.refresh()

# view the spectra
from resistics.utilities.utilsPlotter import plotOptionsSpec, getPaperFonts
from resistics.project.projectSpectra import viewSpectraSection

plotOptions = plotOptionsSpec(plotfonts=getPaperFonts())
viewSpectraSection(
    projData,
    "site1",
    "meas_2012-02-10_11-30-00",
    specdir="notch",
    plotoptions=plotOptions,
    show=True,
    save=True,
)

# process the new set of spectra
from resistics.project.projectTransferFunction import processProject

processProject(projData, sites=["site1"], specdir="notch")
projData.refresh()
# plot the transfer functions, again with specifying the relevant specdir
from resistics.project.projectTransferFunction import viewTransferFunction

viewTransferFunction(
    projData,
    sites=["site1"],
    sampleFreqs=[128],
    oneplot=False,
    specdir="notch",
    save=True,
)
# and compare to the original
viewTransferFunction(
    projData,
    sites=["site1"],
    sampleFreqs=[128],
    oneplot=False,
    specdir="spectra",
    save=True,
)

