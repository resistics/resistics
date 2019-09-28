from configuration import intersitePath, intersiteImages
from resistics.project.projectIO import loadProject

# headers for MT stations
from resistics.ioHandlers.dataReaderLemiB423 import folderB423Headers

folderB423Headers(
    intersitePath / "timeData" / "site1_mt",
    500,
    hxSensor=712,
    hySensor=710,
    hzSensor=714,
    hGain=16,
    dx=60,
    dy=60.7,
)

# headers for telluric only stations
from resistics.ioHandlers.dataReaderLemiB423E import folderB423EHeaders

folderB423EHeaders(
    intersitePath / "timeData" / "site2_te", 500, ex="E1", ey="E2", dx=60, dy=60.7
)

# load the project
proj = loadProject(intersitePath, "customconfig.ini")
proj.printInfo()
fig = proj.view()
fig.savefig(intersiteImages / "timeline.png")

# view time
from resistics.project.projectTime import viewTime

fig = viewTime(
    proj,
    startDate="2019-05-27 14:15:00",
    endDate="2019-05-27 15:00:00",
    filter={"lpfilt": 4},
    save=False,
    show=False,
)
fig.savefig(intersiteImages / "viewTime.png")

# calculate spectra
from resistics.project.projectSpectra import calculateSpectra

calculateSpectra(proj, sites=["site1_mt"])
calculateSpectra(
    proj, sites=["site2_te"], chans=["Ex", "Ey"], polreverse={"Ey": True}
)
proj.refresh()

# calculate statistics for MT site
from resistics.project.projectStatistics import calculateStatistics

calculateStatistics(proj, sites=["site1_mt"])

# intersite
from resistics.project.projectTransferFunction import (
    processProject,
    processSite,
    viewImpedance,
)
from resistics.utilities.utilsPlotter import plotOptionsTransferFunction, getPaperFonts

plotOptions = plotOptionsTransferFunction(figsize=(24, 12), plotfonts=getPaperFonts())
plotOptions["res_ylim"] = [1, 1000000]
processSite(
    proj,
    "site2_te",
    500,
    inputsite="site1_mt",
    postpend="intersite",
)
figs = viewImpedance(
    proj,
    sites=["site2_te"],
    postpend="intersite",
    plotoptions=plotOptions,
    oneplot=False,
    save=False,
    show=False,
)
figs[0].savefig(intersiteImages / "intersiteTransferFunction.png")

# now try again with some statistics for the dead band
from resistics.project.projectMask import newMaskData, calculateMask

maskData = newMaskData(proj, 500)
maskData.setStats(["coherence"])
maskData.addConstraintLevel(
    "coherence", {"cohExHy": [0.8, 1.0], "cohEyHx": [0.8, 1.0]}, 0
)
maskData.addConstraintLevel(
    "coherence", {"cohExHy": [0.8, 1.0], "cohEyHx": [0.8, 1.0]}, 1
)
maskData.addConstraintLevel(
    "coherence", {"cohExHy": [0.7, 1.0], "cohEyHx": [0.7, 1.0]}, 2
)
maskData.addConstraintLevel(
    "coherence", {"cohExHy": [0.6, 1.0], "cohEyHx": [0.6, 1.0]}, 3
)
maskData.addConstraintLevel(
    "coherence", {"cohExHy": [0.5, 1.0], "cohEyHx": [0.5, 1.0]}, 4
)
maskData.addConstraintLevel(
    "coherence", {"cohExHy": [0.5, 1.0], "cohEyHx": [0.5, 1.0]}, 5
)
maskData.addConstraintLevel(
    "coherence", {"cohExHy": [0.4, 1.0], "cohEyHx": [0.4, 1.0]}, 6
)
maskData.maskName = "coh80_100"
calculateMask(proj, maskData, sites=["site1_mt"])

# process with mask
processSite(
    proj,
    "site2_te",
    500,
    inputsite="site1_mt",
    masks={"site1_mt": "coh80_100"},
    postpend="intersite_coh80_100",
)
figs = viewImpedance(
    proj,
    sites=["site2_te"],
    postpend="intersite_coh80_100",
    plotoptions=plotOptions,
    oneplot=False,
    save=False,
    show=False,
)
figs[0].savefig(intersiteImages / "intersiteTransferFunctionMask.png")