import os
from resistics.project.projectIO import loadProject

# load the project
projectPath = os.path.join("asciiProject")
projData = loadProject(projectPath)

# calculate spectrum using standard options
from resistics.project.projectSpectra import calculateSpectra

calculateSpectra(projData, calibrate=False)
projData.refresh()

# process the spectra
from resistics.project.projectTransferFunction import processProject

processProject(projData)
projData.refresh()

# plot transfer function and save the plot
from resistics.project.projectTransferFunction import viewTransferFunction

viewTransferFunction(projData, sites=["site1"], save=True)

# or keep the two most important polarisations on the same plot
from resistics.utilities.utilsPlotter import plotOptionsTransferFunction

plotoptions = plotOptionsTransferFunction()
plotoptions["xlim"] = [0.01, 1000000]
viewTransferFunction(
    projData,
    sites=["site1"],
    polarisations=["ExHy", "EyHx"],
    plotoptions=plotoptions,
    save=True,
)

# this plot is quite busy, let's plot all the components on separate plots
viewTransferFunction(
    projData, sites=["site1"], oneplot=False, plotoptions=plotoptions, save=True
)

# plot a single file
from resistics.project.projectTransferFunction import getTransferFunctionData

tfData = getTransferFunctionData(projData, "site1", 0.5)
tfData.view(oneplot=True, polarisations=["ExHy", "EyHx"], save=True)
