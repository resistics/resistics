import os
from resistics.project.projectIO import loadProject

#  load the project and also provide a config file
projectPath = os.path.join("asciiProject")
projData = loadProject(projectPath, configFile="asciiconfig.ini")
projData.printInfo()

# calculate spectrum using the new configuration
from resistics.project.projectSpectra import calculateSpectra

calculateSpectra(projData, calibrate=False)

# calculate transfer function
from resistics.project.projectTransferFunction import processProject

processProject(projData, outchans=["Ex", "Ey"])

# plot transfer function and save the plot
from resistics.project.projectTransferFunction import viewImpedance
from resistics.utilities.utilsPlotter import plotOptionsTransferFunction

plotoptions = plotOptionsTransferFunction()
plotoptions["xlim"] = [0.01, 1000000]
plotoptions["phase_ylim"] = [-360, 360]
viewImpedance(projData, sites=["site1"], oneplot=True, polarisations=["ExHy", "EyHx"], plotoptions=plotoptions, save=True)

# from resistics.project.projectTransferFunction import viewTipper
# from resistics.utilities.utilsPlotter import plotOptionsTipper

# plotoptions = plotOptionsTipper()
# plotoptions["xlim"] = [0.01, 1000000]
# viewTipper(projData, sites=["site1"], plotoptions=plotoptions, save=True)
