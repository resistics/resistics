from pathlib import Path
from resistics.project.projectIO import loadProject
from resistics.project.projectSpectra import calculateSpectra
from resistics.project.projectTransferFunction import processProject, viewImpedance
from resistics.project.projectStatistics import calculateRemoteStatistics, viewStatistic
from resistics.project.projectMask import newMaskData, calculateMask

projectPath = Path("remoteProject")
projData = loadProject(projectPath, "remoteConfig.ini")
calculateSpectra(projData, sites=["M6", "Remote"])
projData.refresh()
processProject(projData, sites=["M6", "Remote"])
viewImpedance(projData, sites=["M6", "Remote"], oneplot=False, save=True, show=False)

# calculate the statistic we are interested in
calculateRemoteStatistics(projData, "Remote", sites=["M6"], sampleFreqs=[128])

# generate mask
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
    datetimes=[
        {"type": "time", "start": "20:00:00", "stop": "07:00:00", "levels": [0, 1]}
    ],
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
