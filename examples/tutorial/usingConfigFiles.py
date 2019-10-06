from datapaths import projectPath, imagePath
from resistics.project.io import loadProject

#  load the project and also provide a config file
projData = loadProject(projectPath, configFile="tutorialConfig.ini")
projData.printInfo()

# calculate spectrum using the new configuration
from resistics.project.spectra import calculateSpectra

calculateSpectra(projData)
projData.refresh()

# process the spectra
from resistics.project.transfunc import processProject, viewImpedance

processProject(projData)
figs = viewImpedance(projData, sites=["site1"], oneplot=False, save=False, show=False)
figs[0].savefig(imagePath / "usingConfigFiles_viewimp")
