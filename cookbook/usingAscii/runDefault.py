import os
from resistics.project.projectIO import loadProject

# load the project
projectPath = os.path.join("asciiProject")
projData = loadProject(projectPath)
projData.view()

# view site data
siteData = projData.getSiteData("site1")
siteData.view()

from resistics.project.projectTime import viewTime

viewTime(projData, "2018-01-03 00:00:00", "2018-01-04 00:00:00")

# calculate spectrum using standard options
from resistics.project.projectSpectra import calculateSpectra

calculateSpectra(projData, calibrate=False)

# process the spectra
from resistics.project.projectTransferFunction import processProject

processProject(projData, outchans=["Ex", "Ey"])

# plot transfer function and save the plot
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

# plot a single file
from resistics.project.projectTransferFunction import viewTipper
from resistics.utilities.utilsPlotter import plotOptionsTipper

plotoptions = plotOptionsTipper()
plotoptions["xlim"] = [0.01, 1000000]
viewTipper(
    projData, sites=["site1"], postpend="withHz", plotoptions=plotoptions, save=True
)
