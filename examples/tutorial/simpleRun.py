import os
from resistics.project.projectIO import loadProject

# load the project
projectPath = os.path.join("tutorialProject")
projData = loadProject(projectPath)

# calculate spectrum using standard options
from resistics.project.projectSpectra import calculateSpectra

calculateSpectra(projData)
projData.refresh()

# process the spectra
from resistics.project.projectTransferFunction import processProject

processProject(projData)

# plot transfer function and save the plot
from resistics.project.projectTransferFunction import viewImpedance

viewImpedance(projData, sites=["site1"], save=True)

# or keep the two most important polarisations on the same plot
viewImpedance(
    projData, sites=["site1"], polarisations=["ExHy", "EyHx"], save=True
)

# this plot is quite busy, let's plot all the components on separate plots
viewImpedance(projData, sites=["site1"], oneplot=False, save=True)

# get a transfer function data object
from resistics.project.projectTransferFunction import getTransferFunctionData

tfData = getTransferFunctionData(projData, "site1", 128)
fig = tfData.viewImpedance(oneplot=True, polarisations=["ExHy", "EyHx"], save=True)
fig.savefig(os.path.join("tutorialProject", "images", "transferFunctionViewExample"))
