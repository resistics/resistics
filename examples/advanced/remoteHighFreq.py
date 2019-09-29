from configuration import remotePath
from resistics.project.projectIO import loadProject
from resistics.project.projectSpectra import calculateSpectra
from resistics.project.projectTransferFunction import processProject, viewImpedance
from resistics.project.projectStatistics import (
    calculateStatistics,
    calculateRemoteStatistics,
    viewStatistic,
)
from resistics.project.projectMask import newMaskData, calculateMask

projData = loadProject(remotePath, "remoteConfig.ini")

calculateStatistics(projData, sites=["M6"], sampleFreqs=[512, 4096, 16384, 65536])
# generate mask
for fs in [512, 4096, 16384, 65536]:
    maskData = newMaskData(projData, fs)
    maskData.setStats(["coherence"])
    maskData.addConstraint("coherence", {"cohExHy": [0.8, 1.0], "cohEyHx": [0.8, 1.0]})
    # finally, lets give maskData a name, which will relate to the output file
    maskData.maskName = "coh_80_100"
    calculateMask(projData, maskData, sites=["M6"])
    maskData.printInfo()

processProject(
    projData,
    sites=["M6"],
    sampleFreqs=[512, 4096, 16384, 65536],
    masks={"M6": "coh_80_100"},
    postpend="coh_80_100",
)

viewImpedance(
    projData,
    sites=["M6"],
    sampleFreqs=[512, 4096, 16384, 65536],
    postpend="coh_80_100",
    oneplot=False,
    save=True,
    show=False,
)

# # try this
# from resistics.utilities.utilsPlotter import plotOptionsTransferFunction, getPaperFonts

# plotOptions = plotOptionsTransferFunction(figsize=(24, 12), plotfonts=getPaperFonts())
# print(plotOptions)

# projectPath = Path("remoteProject")
# projData = loadProject(projectPath, "manualWindowsConfig.ini")
# viewImpedance(
#     projData,
#     sites=["M6"],
#     sampleFreqs=[65536, 4096, 128],
#     postpend="rr_cohEqn_80_100_night",
#     oneplot=False,
#     plotoptions=plotOptions,
#     save=True,
#     show=False,
# )
