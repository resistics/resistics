from datapaths import projectPath, imagePath
from resistics.project.io import loadProject

# load project and configuration file
projData = loadProject(projectPath, configFile="tutorialConfig.ini")

# Process the data with this mask
from resistics.project.transfunc import processProject

processProject(
    projData,
    sampleFreqs=[4096, 128],
    sites=["site1"],
    outchans=["Ex", "Ey"],
    masks={"site1": "coh70_100"},
    postpend="coh70_100",
)

from resistics.project.transfunc import viewImpedance

figs = viewImpedance(
    projData,
    sites=["site1"],
    postpend="coh70_100",
    oneplot=False,
    save=False,
    show=False,
)
figs[0].savefig(imagePath / "runWithMask_coh70_100")

processProject(
    projData,
    sampleFreqs=[4096, 128],
    sites=["site1"],
    outchans=["Ex", "Ey"],
    masks={"site1": "coh70_100_tfConstrained"},
    postpend="coh70_100_tfConstrained",
)

figs = viewImpedance(
    projData,
    sites=["site1"],
    postpend="coh70_100_tfConstrained",
    oneplot=False,
    save=False,
    show=False,
)
figs[0].savefig(imagePath / "runWithMask_coh70_100_tfConstrained")