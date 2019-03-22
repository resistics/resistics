import os
from resistics.project.projectIO import loadProject

#  load the project and also provide a config file
projectPath = os.path.join("asciiProject")
projData = loadProject(projectPath, configFile="asciiconfig.ini")
projData.printInfo()

# calculate spectrum using the new configuration
from resistics.project.projectSpectra import calculateSpectra

calculateSpectra(projData, calibrate=False)
projData.refresh()
# process the spectra
from resistics.project.projectTransferFunction import (
    processProject,
    viewTransferFunction,
)

processProject(projData)
projData.refresh()
# plot transfer function and save the plot
from resistics.utilities.utilsPlotter import plotOptionsTransferFunction

plotoptions = plotOptionsTransferFunction()
plotoptions["xlim"] = [0.01, 1000000]
viewTransferFunction(projData, sites=["site1"], oneplot=False, plotoptions=plotoptions, save=True)
