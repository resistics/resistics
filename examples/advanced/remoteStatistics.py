from pathlib import Path
from resistics.project.projectIO import loadProject
from resistics.project.projectStatistics import (
    calculateRemoteStatistics,
    viewStatistic,
    viewStatisticHistogram,
)
from resistics.project.projectMask import newMaskData, calculateMask
from resistics.project.projectTransferFunction import processProject, viewImpedance


projectPath = Path("remoteProject")
projData = loadProject(projectPath)

calculateRemoteStatistics(
    projData,
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

# for stat in ["RR_coherence", "RR_coherenceEqn", "RR_absvalEqn", "RR_transferFunction", "RR_resPhase"]:
# for stat in ["RR_coherenceEqn"]:
#     for declevel in range(0, 4):
#         viewStatistic(projData, "M6", 128, stat, declevel=declevel)
#         viewStatisticHistogram(projData, "M6", 128, stat, declevel=declevel)

# create a mask that uses some of the remote reference statistics that were calculated
# get a mask data object and specify the sampling frequency to mask (128Hz)
maskData = newMaskData(projData, 128)
maskData.setStats(["RR_coherenceEqn"])
maskData.addConstraint(
    "RR_coherenceEqn", {"ExHyR-HyHyR": [0.8, 1.0], "EyHxR-HxHxR": [0.8, 1.0]}
)
# finally, lets give maskData a name, which will relate to the output file
maskData.maskName = "rr_cohEqn_80_100"
calculateMask(projData, maskData, sites=["M6"])
maskData.printInfo()

# process with masks
processProject(
    projData,
    sites=["M6"],
    sampleFreqs=[128],
    remotesite="Remote",
    masks={"M6": "rr_cohEqn_80_100"},
    postpend="rr_cohEqn_80_100",
)

viewImpedance(
    projData,
    sites=["M6"],
    postpend="rr_cohEqn_80_100",
    oneplot=False,
    save=True,
    show=False,
)

# process with masks and datetime constraints
processProject(
    projData,
    sites=["M6"],
    sampleFreqs=[128],
    remotesite="Remote",
    masks={"M6": "rr_cohEqn_80_100"},
    datetimes=[{"type": "time", "start": "20:00:00", "stop": "07:00:00"}],
    postpend="rr_cohEqn_80_100_night",
)

viewImpedance(
    projData,
    sites=["M6"],
    sampleFreqs=[128],
    postpend="rr_cohEqn_80_100_night",
    oneplot=False,
    save=True,
    show=False,
)

# process with masks and datetime constraints, but only for decimation levels 0 and 1
processProject(
    projData,
    sites=["M6"],
    sampleFreqs=[128],
    remotesite="Remote",
    masks={"M6": "rr_cohEqn_80_100"},
    datetimes=[
        {"type": "time", "start": "20:00:00", "stop": "07:00:00", "levels": [0, 1]}
    ],
    postpend="rr_cohEqn_80_100_night2",
)

viewImpedance(
    projData,
    sites=["M6"],
    sampleFreqs=[128],
    postpend="rr_cohEqn_80_100_night2",
    oneplot=False,
    save=True,
    show=False,
)
