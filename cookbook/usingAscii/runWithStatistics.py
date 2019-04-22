import os
from resistics.project.projectIO import loadProject

#  load the project and also provide a config file
projectPath = os.path.join("asciiProject")
projData = loadProject(projectPath, configFile="asciiconfig.ini")
projData.printInfo()

# calculate statistics
from resistics.project.projectStatistics import calculateStatistics

calculateStatistics(projData)

# create a mask based on coherence
from resistics.project.projectMask import newMaskData, calculateMask

maskData = newMaskData(projData, 0.5)
maskData.setStats(["coherence"])
maskData.addConstraint("coherence", {"cohExHy": [0.3, 1.0], "cohEyHx": [0.3, 1.0]})
maskData.maskName = "coh30_100"
calculateMask(projData, maskData, sites=["site1"])
fig = maskData.view(0)
fig.savefig(os.path.join(projectPath, "images", "maskcoh"))

# calculate impedance tensor
from resistics.project.projectTransferFunction import processProject

processProject(
    projData, outchans=["Ex", "Ey"], masks={"site1": maskData.maskName}, postpend=maskData.maskName
)
projData.refresh()

# plot transfer function and save the plot
from resistics.project.projectTransferFunction import viewImpedance
from resistics.utilities.utilsPlotter import plotOptionsTransferFunction

plotoptions = plotOptionsTransferFunction()
plotoptions["xlim"] = [0.01, 1000000]
plotoptions["phase_ylim"] = [-10, 100]
viewImpedance(
    projData,
    sites=["site1"],
    postpend=maskData.maskName,
    oneplot=True,
    polarisations=["ExHy", "EyHx"],
    plotoptions=plotoptions,
    save=True,
)