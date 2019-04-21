import os
from resistics.project.projectIO import loadProject

# load project and configuration file
projectPath = os.path.join("tutorialProject")
projData = loadProject(projectPath, configFile="tutorialConfig.ini")

# Process the data with this mask
from resistics.project.projectTransferFunction import processProject

processProject(
    projData,
    sampleFreqs=[4096, 128],
    sites=["site1"],
    outchans=["Ex", "Ey"],
    masks={"site1": "coh70_100"},
    postpend="coh70_100",
)

from resistics.project.projectTransferFunction import viewImpedance

viewImpedance(
    projData, sites=["site1"], postpend="coh70_100", oneplot=False, save=True, show=True
)

processProject(
    projData,
    sampleFreqs=[4096, 128],
    sites=["site1"],
    outchans=["Ex", "Ey"],
    masks={"site1": "coh70_100_tfConstrained"},
    postpend="coh70_100_tfConstrained",
)

viewImpedance(
    projData,
    sites=["site1"],
    postpend="coh70_100_tfConstrained",
    oneplot=False,
    save=True,
    show=True,
)