from pathlib import Path
from resistics.project.projectIO import loadProject
from resistics.project.projectSpectra import calculateSpectra
from resistics.project.projectTransferFunction import processProject, viewImpedance
from resistics.project.projectStatistics import calculateStatistics, viewStatistic
from resistics.project.projectMask import newMaskData, calculateMask

projectPath = Path("remoteProject")
projData = loadProject(projectPath)

# calculate spectrum using standard options
calculateSpectra(projData, sites=["M6", "Remote"])
projData.refresh()

# begin with single site processing
processProject(projData, sites=["M6", "Remote"])
viewImpedance(projData, sites=["M6", "Remote"], oneplot=False, show=False, save=True)

# perform standard remote reference runs - remember to call the output something else
processProject(
    projData, sites=["M6"], sampleFreqs=[128], remotesite="Remote", postpend="rr"
)
viewImpedance(
    projData, sites=["M6"], postpend="rr", oneplot=False, show=False, save=True
)

# let's calculate some single site statistics
calculateStatistics(projData, sites=["M6", "Remote"], stats=["coherence"])

# calculate masks
maskData = newMaskData(projData, 128)
maskData.setStats(["coherence"])
maskData.addConstraint("coherence", {"cohExHy": [0.8, 1.0], "cohEyHx": [0.8, 1.0]})
maskData.maskName = "coh80_100"
maskData.printInfo()
maskData.printConstraints()
# calculate
calculateMask(projData, maskData, sites=["M6", "Remote"])

# single site processing again
processProject(
    projData,
    sites=["M6", "Remote"],
    sampleFreqs=[128],
    masks={"M6": "coh80_100", "Remote": "coh80_100"},
    postpend="coh80_100",
)
viewImpedance(
    projData,
    sites=["M6", "Remote"],
    sampleFreqs=[128],
    postpend="coh80_100",
    oneplot=False,
    show=False,
    save=True,
)

# remote reference processing with masks
processProject(
    projData,
    sites=["M6"],
    sampleFreqs=[128],
    remotesite="Remote",
    masks={"M6": "coh80_100", "Remote": "coh80_100"},
    postpend="rr_coh80_100",
)

viewImpedance(
    projData,
    sites=["M6"],
    sampleFreqs=[128],
    postpend="rr_coh80_100",
    oneplot=False,
    show=False,
    save=True,
)

# remote reference processing with datetime constraints
processProject(
    projData,
    sites=["M6"],
    sampleFreqs=[128],
    remotesite="Remote",
    datetimes=[{"type": "time", "start": "20:00:00", "stop": "07:00:00"}],
    postpend="rr_night",
)

viewImpedance(
    projData,
    sites=["M6"],
    sampleFreqs=[128],
    postpend="rr_night",
    oneplot=False,
    show=False,
    save=True,
)
