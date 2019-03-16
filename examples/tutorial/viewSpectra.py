import os
from resistics.project.projectIO import loadProject

# load the project
projectPath = os.path.join("tutorialProject")
projData = loadProject(projectPath)

# setting the plot options
from resistics.utilities.utilsPlotter import plotOptionsSpec, getPaperFonts
plotOptions = plotOptionsSpec(plotfonts=getPaperFonts())

# view the spectra at 128 Hz
from resistics.project.projectSpectra import viewSpectra
viewSpectra(
    projData,
    "site1",
    "meas_2012-02-10_11-30-00",
    plotoptions=plotOptions,
    show=True,
    save=True,
)

from resistics.project.projectSpectra import viewSpectraSection
plotOptions = plotOptionsSpec(plotfonts=getPaperFonts())
viewSpectraSection(
    projData,
    "site1",
    "meas_2012-02-10_11-30-00",
    plotoptions=plotOptions,
    show=True,
    save=True,
)

from resistics.project.projectSpectra import viewSpectraStack
viewSpectraStack(
    projData,
    "site1",
    "meas_2012-02-10_11-30-00",
    coherences=[["Ex", "Hy"], ["Ey", "Hx"]],
    plotoptions=plotOptions,
    show=True,
    save=True,
)

# view the spectra at 4096 Hz
viewSpectra(
    projData,
    "site1",
    "meas_2012-02-10_11-05-00",
    plotoptions=plotOptions,
    show=True,
    save=True,
)

viewSpectraSection(
    projData,
    "site1",
    "meas_2012-02-10_11-05-00",
    plotoptions=plotOptions,
    show=True,
    save=True,
)

viewSpectraStack(
    projData,
    "site1",
    "meas_2012-02-10_11-05-00",
    coherences=[["Ex", "Hy"], ["Ey", "Hx"]],
    plotoptions=plotOptions,
    show=True,
    save=True,
)