from resistics.utilities.utilsConfig import copyDefaultConfig
copyDefaultConfig("ex1_config.ini")

import os
from resistics.project.projectIO import loadProject

#  load the project and also provide a config file
projectPath = os.path.join("tutorialProject")
projData = loadProject(projectPath, configFile="ex1_04_config.ini")
projData.printInfo()

# calculate spectrum using the new configuration
from resistics.project.projectSpectra import calculateSpectra
calculateSpectra(projData)
projData.refresh()
# process the spectra
from resistics.project.projectTransferFunction import processProject, viewTransferFunction
processProject(projData)
projData.refresh()
# plot transfer function and save the plot
viewTransferFunction(projData, sites=["site1"], oneplot=False, save=True)
