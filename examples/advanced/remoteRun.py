from pathlib import Path
from resistics.project.projectIO import loadProject
from resistics.project.projectSpectra import calculateSpectra
from resistics.project.projectTransferFunction import processProject, viewImpedance
from resistics.project.projectStatistics import calculateStatistics, viewStatistic
from resistics.project.projectMask import newMaskData, calculateMask
from resistics.utilities.utilsPlotter import plotOptionsTransferFunction, getPaperFonts

plotOptions = plotOptionsTransferFunction(plotfonts=getPaperFonts())
projectPath = Path("remoteProject")
proj = loadProject(projectPath)

# calculate spectrum using standard options
calculateSpectra(proj, sites=["M6", "Remote"])
proj.refresh()

# single site processing
processProject(proj, sites=["M6", "Remote"])

# 128 Hz impedance tensor estimates
figs = viewImpedance(
    proj,
    sites=["M6", "Remote"],
    sampleFreqs=[128],
    oneplot=False,
    plotoptions=plotOptions,
    show=False,
    save=False,
)
figs[0].savefig(Path(proj.imagePath, "singleSiteM6_128.png"))
figs[1].savefig(Path(proj.imagePath, "singleSiteRemote_128.png"))

# all sampling frequencies for M6
figs = viewImpedance(
    proj, sites=["M6"], oneplot=False, plotoptions=plotOptions, show=False, save=False
)
figs[0].savefig(Path(proj.imagePath, "singleSiteM6_all.png"))

# perform standard remote reference runs - remember to call the output something else
processProject(
    proj, sites=["M6"], sampleFreqs=[128], remotesite="Remote", postpend="rr"
)
figs = viewImpedance(
    proj,
    sites=["M6"],
    postpend="rr",
    oneplot=False,
    plotoptions=plotOptions,
    show=False,
    save=True,
)
figs[0].savefig(Path(proj.imagePath, "remoteReferenceM6.png"))

# let's calculate some single site statistics
calculateStatistics(proj, sites=["M6", "Remote"], stats=["coherence", "transferFunction"])

# calculate masks
maskData = newMaskData(proj, 128)
maskData.setStats(["coherence"])
maskData.addConstraint("coherence", {"cohExHy": [0.8, 1.0], "cohEyHx": [0.8, 1.0]})
maskData.maskName = "coh80_100"
maskData.printInfo()
maskData.printConstraints()
# calculate
calculateMask(proj, maskData, sites=["M6", "Remote"])

# single site processing again
processProject(
    proj,
    sites=["M6", "Remote"],
    sampleFreqs=[128],
    masks={"M6": "coh80_100", "Remote": "coh80_100"},
    postpend="coh80_100",
)
figs = viewImpedance(
    proj,
    sites=["M6", "Remote"],
    sampleFreqs=[128],
    postpend="coh80_100",
    oneplot=False,
    plotoptions=plotOptions,
    show=False,
    save=False,
)
figs[0].savefig(Path(proj.imagePath, "singleSiteM6_128_coh80.png"))
figs[1].savefig(Path(proj.imagePath, "singleSiteRemote_128_coh80.png"))

# remote reference processing with masks
processProject(
    proj,
    sites=["M6"],
    sampleFreqs=[128],
    remotesite="Remote",
    masks={"Remote": "coh80_100"},
    postpend="rr_coh80_100",
)

figs = viewImpedance(
    proj,
    sites=["M6"],
    sampleFreqs=[128],
    postpend="rr_coh80_100",
    oneplot=False,
    plotoptions=plotOptions,
    show=False,
    save=False,
)
figs[0].savefig(Path(proj.imagePath, "remoteReferenceM6_128_coh80.png"))

# remote reference processing with datetime constraints
processProject(
    proj,
    sites=["M6"],
    sampleFreqs=[128],
    remotesite="Remote",
    datetimes=[{"type": "time", "start": "20:00:00", "stop": "07:00:00"}],
    postpend="rr_night",
)

figs = viewImpedance(
    proj,
    sites=["M6"],
    sampleFreqs=[128],
    postpend="rr_night",
    oneplot=False,
    plotoptions=plotOptions,
    show=False,
    save=False,
)
figs[0].savefig(Path(proj.imagePath, "remoteReferenceM6_128_coh80_datetimes.png"))