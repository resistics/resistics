from datapaths import projectPath, imagePath
from resistics.project.io import loadProject

#  load the project and also provide a config file
projData = loadProject(projectPath, configFile="asciiconfig.ini")
projData.printInfo()

# calculate statistics
from resistics.project.statistics import calculateStatistics

calculateStatistics(projData)

# create a mask based on coherence
from resistics.project.mask import newMaskData, calculateMask

maskData = newMaskData(projData, 0.5)
maskData.setStats(["coherence"])
maskData.addConstraint("coherence", {"cohExHy": [0.3, 1.0], "cohEyHx": [0.3, 1.0]})
maskData.maskName = "coh30_100"
calculateMask(projData, maskData, sites=["site1"])
fig = maskData.view(0)
fig.savefig(imagePath / "maskcoh")

# calculate impedance tensor
from resistics.project.transfunc import processProject

processProject(
    projData, outchans=["Ex", "Ey"], masks={"site1": maskData.maskName}, postpend=maskData.maskName
)

# plot transfer function and save the plot
from resistics.project.transfunc import viewImpedance
from resistics.common.plot import plotOptionsTransferFunction, getPaperFonts

plotoptions = plotOptionsTransferFunction(plotfonts=getPaperFonts())
plotoptions["xlim"] = [0.01, 1000000]
plotoptions["phase_ylim"] = [-10, 100]
figs = viewImpedance(
    projData,
    sites=["site1"],
    postpend=maskData.maskName,
    oneplot=True,
    polarisations=["ExHy", "EyHx"],
    plotoptions=plotoptions,
    save=False,
)
figs[0].savefig(imagePath / "impedance_config_masked")