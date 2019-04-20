import os
from resistics.project.projectIO import loadProject

# load the project
projectPath = os.path.join("tutorialProject")
projData = loadProject(projectPath)

# process the spectra with tippers
from resistics.project.projectTransferFunction import processProject

processProject(
    projData, sites=["site1"], outchans=["Ex", "Ey", "Hz"], postpend="with_Hz"
)
projData.refresh()

# plot the tippers
from resistics.project.projectTransferFunction import viewTipper

viewTipper(projData, sites=["site1"], postpend="with_Hz", save=True)

# plot the transfer function
from resistics.project.projectTransferFunction import viewImpedance

viewImpedance(
    projData,
    sites=["site1"],
    polarisations=["ExHy", "EyHx"],
    postpend="with_Hz",
    save=True,
)

# process only the tippers
processProject(projData, sites=["site1"], outchans=["Hz"], postpend="only_Hz")
projData.refresh()
viewTipper(projData, sites=["site1"], postpend="only_Hz", save=True)
