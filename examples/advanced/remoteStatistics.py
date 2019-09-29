from configuration import remotePath, remoteImages
from resistics.project.projectIO import loadProject
from resistics.project.projectStatistics import (
    calculateRemoteStatistics,
    viewStatistic,
    viewStatisticHistogram,
    viewStatisticDensityplot,
)
from resistics.project.projectMask import newMaskData, calculateMask
from resistics.project.projectTransferFunction import processProject, viewImpedance
from resistics.utilities.utilsPlotter import plotOptionsStandard, getPaperFonts

plotOptions = plotOptionsStandard(plotfonts=getPaperFonts())
proj = loadProject(remotePath)

calculateRemoteStatistics(
    proj,
    "Remote",
    sites=["M6"],
    sampleFreqs=[128],
    remotestats=[
        "RR_coherence",
        "RR_coherenceEqn",
        "RR_absvalEqn",
        "RR_transferFunction",
        "RR_resPhase",
    ],
)

lims = {0: 200, 1: 120, 2: 50, 3: 30}
for declevel in range(0, 4):
    fig = viewStatisticDensityplot(
        proj,
        "M6",
        128,
        "RR_transferFunction",
        declevel=declevel,
        crossplots=[["ExHyRealRR", "ExHyImagRR"], ["EyHxRealRR", "EyHxImagRR"]],
        xlim=[-lims[declevel], lims[declevel]],
        ylim=[-lims[declevel], lims[declevel]],
        plotoptions=plotOptions,
        show=False,
    )
    fig.savefig(remoteImages / "densityPlot_remoteRef_{}.png".format(declevel))
    fig = viewStatisticDensityplot(
        proj,
        "M6",
        128,
        "transferFunction",
        declevel=declevel,
        crossplots=[["ExHyReal", "ExHyImag"], ["EyHxReal", "EyHxImag"]],
        xlim=[-lims[declevel], lims[declevel]],
        ylim=[-lims[declevel], lims[declevel]],
        plotoptions=plotOptions,
        show=False,
    )
    fig.savefig(remoteImages / "densityPlot_singleSite_{}.png".format(declevel))

# create a mask that uses some of the remote reference statistics that were calculated
# get a mask data object and specify the sampling frequency to mask (128Hz)
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
    postpend="rr_cohEqn_80_100",
)

from resistics.utilities.utilsPlotter import plotOptionsTransferFunction

plotOptionsTF = plotOptionsTransferFunction(plotfonts=getPaperFonts())
figs = viewImpedance(
    proj,
    sites=["M6"],
    sampleFreqs=[128],
    postpend="rr_cohEqn_80_100",
    oneplot=False,
    plotoptions=plotOptionsTF,
    save=False,
    show=False,
)
figs[0].savefig(remoteImages / "remoteReferenceM6_128_RR_coh.png")

# see how the masks changed the density plot
lims = {0: 200, 1: 120, 2: 50, 3: 30}
for declevel in range(0, 4):
    fig = viewStatisticDensityplot(
        proj,
        "M6",
        128,
        "RR_transferFunction",
        declevel=declevel,
        crossplots=[["ExHyRealRR", "ExHyImagRR"], ["EyHxRealRR", "EyHxImagRR"]],
        maskname="rr_cohEqn_80_100",
        xlim=[-lims[declevel], lims[declevel]],
        ylim=[-lims[declevel], lims[declevel]],
        plotoptions=plotOptions,
        show=False,
    )
    fig.savefig(remoteImages / "densityPlot_remoteRef_{}_mask.png".format(declevel))

# process with masks and datetime constraints
processProject(
    proj,
    sites=["M6"],
    sampleFreqs=[128],
    remotesite="Remote",
    masks={"M6": "rr_cohEqn_80_100"},
    datetimes=[{"type": "time", "start": "20:00:00", "stop": "07:00:00"}],
    postpend="rr_cohEqn_80_100_night",
)

figs = viewImpedance(
    proj,
    sites=["M6"],
    sampleFreqs=[128],
    postpend="rr_cohEqn_80_100_night",
    oneplot=False,
    plotoptions=plotOptionsTF,    
    save=False,
    show=False,
)
figs[0].savefig(remoteImages / "remoteReferenceM6_128_RR_coh_datetime.png")

# process with masks and datetime constraints, but only for decimation levels 0 and 1
processProject(
    proj,
    sites=["M6"],
    sampleFreqs=[128],
    remotesite="Remote",
    masks={"M6": "rr_cohEqn_80_100"},
    datetimes=[
        {"type": "time", "start": "20:00:00", "stop": "07:00:00", "levels": [0, 1]}
    ],
    postpend="rr_cohEqn_80_100_night2",
)

figs = viewImpedance(
    proj,
    sites=["M6"],
    sampleFreqs=[128],
    postpend="rr_cohEqn_80_100_night2",
    oneplot=False,
    plotoptions=plotOptionsTF,    
    save=False,
    show=False,
)
figs[0].savefig(remoteImages / "remoteReferenceM6_128_RR_coh_datetime_01.png")