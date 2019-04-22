import os
from resistics.project.projectIO import loadProject

# load the project
projectPath = "asciiProject"
projData = loadProject(projectPath)
fig = projData.view()
fig.savefig(os.path.join(projectPath, "images", "projectTimeline"))

# view site data
siteData = projData.getSiteData("site1")
fig = siteData.view()
fig.savefig(os.path.join(projectPath, "images", "siteTimeline"))

from resistics.project.projectTime import viewTime
from resistics.utilities.utilsPlotter import plotOptionsTime, getPaperFonts

plotOptions = plotOptionsTime(plotfonts=getPaperFonts())
viewTime(
    projData,
    "2018-01-03 00:00:00",
    "2018-01-05 00:00:00",
    plotoptions=plotOptions,
    save=True,
)

# calculate spectrum using standard options
from resistics.project.projectSpectra import calculateSpectra

calculateSpectra(projData, calibrate=False)
projData.refresh()

from resistics.project.projectSpectra import viewSpectraStack
from resistics.utilities.utilsPlotter import plotOptionsSpec

plotOptions = plotOptionsSpec(plotfonts=getPaperFonts())
viewSpectraStack(
    projData,
    "site1",
    "meas",
    coherences=[["Ex", "Hy"], ["Ey", "Hx"]],
    plotoptions=plotOptions,
    save=True,
    show=False,
)

# process the spectra to estimate the transfer function
from resistics.project.projectTransferFunction import processProject

processProject(projData, outchans=["Ex", "Ey"])

# plot impedance tensor and save the plot
from resistics.project.projectTransferFunction import viewImpedance
from resistics.utilities.utilsPlotter import plotOptionsTransferFunction

plotoptions = plotOptionsTransferFunction()
plotoptions["xlim"] = [0.01, 1000000]
plotoptions["phase_ylim"] = [-360, 360]
viewImpedance(
    projData,
    sites=["site1"],
    oneplot=True,
    polarisations=["ExHy", "EyHx"],
    plotoptions=plotoptions,
    save=True,
)

# calculate the tipper
processProject(projData, outchans=["Ex", "Ey", "Hz"], postpend="withHz")

# plot the tipper
from resistics.project.projectTransferFunction import viewTipper
from resistics.utilities.utilsPlotter import plotOptionsTipper

plotoptions = plotOptionsTipper()
plotoptions["xlim"] = [0.01, 1000000]
viewTipper(
    projData, sites=["site1"], postpend="withHz", plotoptions=plotoptions, save=True
)