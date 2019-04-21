import os
from resistics.project.projectIO import loadProject

#  load the project and also provide a config file
projectPath = os.path.join("tutorialProject")
projData = loadProject(projectPath, configFile="tutorialConfig.ini")
projData.printInfo()

# calculate spectrum using the new configuration
from resistics.project.projectSpectra import calculateSpectra

calculateSpectra(projData)

# process the spectra
from resistics.project.projectTransferFunction import processProject, viewImpedance

processProject(projData)
viewImpedance(projData, sites=["site1"], oneplot=False, save=True)