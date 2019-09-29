from configuration import remotePath, remoteImages
from resistics.project.projectIO import loadProject
from resistics.project.projectSpectra import calculateSpectra
from resistics.project.projectTransferFunction import processProject, viewImpedance
from resistics.project.projectStatistics import calculateRemoteStatistics, viewStatistic
from resistics.project.projectMask import newMaskData, calculateMask
from resistics.utilities.utilsPlotter import plotOptionsTransferFunction, getPaperFonts

plotOptions = plotOptionsTransferFunction(plotfonts=getPaperFonts())
proj = loadProject(remotePath, "remoteConfig.ini")

calculateSpectra(proj, sites=["M6", "Remote"])
proj.refresh()

# single site processing
processProject(proj, sites=["M6", "Remote"])
figs = viewImpedance(
    proj,
    sites=["M6", "Remote"],
    sampleFreqs=[128],    
    oneplot=False,
    plotoptions=plotOptions,
    save=False,
    show=False,
)
figs[0].savefig(remoteImages / "singleSiteM6_128_dec8_5.png")
figs[1].savefig(remoteImages / "singleSiteRemote_128_dec8_5.png")

# calculate the statistic we are interested in
calculateRemoteStatistics(proj, "Remote", sites=["M6"], sampleFreqs=[128])

# generate mask
maskData = newMaskData(proj, 128)
maskData.setStats(["RR_coherenceEqn"])
maskData.addConstraint(
    "RR_coherenceEqn", {"ExHyR-HyHyR": [0.8, 1.0], "EyHxR-HxHxR": [0.8, 1.0]}
)
# finally, lets give maskData a name, which will relate to the output file
maskData.maskName = "rr_cohEqn_80_100"
calculateMask(proj, maskData, sites=["M6"])
maskData.printInfo()

# process with masks
processProject(
    proj,
    sites=["M6"],
    sampleFreqs=[128],
    remotesite="Remote",
    masks={"M6": "rr_cohEqn_80_100"},
    datetimes=[
        {"type": "time", "start": "20:00:00", "stop": "07:00:00", "levels": [0, 1]}
    ],
    postpend="rr_cohEqn_80_100_night",
)

figs = viewImpedance(
    proj,
    sites=["M6"],
    sampleFreqs=[128],
    postpend="rr_cohEqn_80_100_night",
    oneplot=False,
    plotoptions=plotOptions,
    save=False,
    show=False,
)
figs[0].savefig(remoteImages / "remoteReferenceM6_128_RR_dec8_5_coh_datetime_01.png")

