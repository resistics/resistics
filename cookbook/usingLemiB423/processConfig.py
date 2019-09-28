from configuration import projectPath, imagePath
from resistics.project.projectIO import loadProject

proj = loadProject(projectPath, "customconfig.ini")
proj.printInfo()

from resistics.project.projectSpectra import calculateSpectra
from resistics.project.projectStatistics import calculateStatistics
from resistics.project.projectTransferFunction import processProject, viewImpedance
from resistics.utilities.utilsPlotter import plotOptionsTransferFunction, getPaperFonts

calculateSpectra(proj)
proj.refresh()
calculateStatistics(proj)
processProject(proj)
plotOptions = plotOptionsTransferFunction(figsize=(24, 12), plotfonts=getPaperFonts())
plotOptions["res_ylim"] = [1, 1000000]
figs = viewImpedance(
    proj, oneplot=False, plotoptions=plotOptions, show=False, save=False
)
figs[0].savefig(imagePath / "impedance_config.png")

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

# process the site with the mask
processProject(proj, masks={"site1_mt": "coh80_100"}, postpend="coh80_100")
figs = viewImpedance(
    proj,
    postpend="coh80_100",
    oneplot=False,
    plotoptions=plotOptions,
    show=False,
    save=False,
)
figs[0].savefig(imagePath / "impedance_config_masks.png")