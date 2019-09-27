from pathlib import Path
from resistics.project.projectIO import loadProject
from resistics.project.projectSpectra import calculateSpectra
from resistics.project.projectTransferFunction import processProject, viewImpedance
from resistics.project.projectStatistics import (
    calculateStatistics,
    calculateRemoteStatistics,
    viewStatistic,
)
from resistics.project.projectMask import newMaskData, calculateMask
from resistics.utilities.utilsPlotter import plotOptionsTransferFunction, getPaperFonts

plotOptions = plotOptionsTransferFunction(plotfonts=getPaperFonts())
projectPath = Path("remoteProject")
proj = loadProject(projectPath, "manualWindowsConfig.ini")

# calculate spectra
calculateSpectra(proj, sites=["M6", "Remote"])
proj.refresh()
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
figs[0].savefig(Path(proj.imagePath, "singleSiteM6_128_man8_5.png"))
figs[1].savefig(Path(proj.imagePath, "singleSiteRemote_128_man8_5.png"))

# calculate the statistic we are interested in
calculateStatistics(proj, sites=["M6", "Remote"], sampleFreqs=[128])
calculateRemoteStatistics(proj, "Remote", sites=["M6"], sampleFreqs=[128])

# generate single site masks
maskData = newMaskData(proj, 128)
maskData.setStats(["coherence"])
maskData.addConstraint("coherence", {"cohExHy": [0.8, 1.0], "cohEyHx": [0.8, 1.0]})
# finally, lets give maskData a name, which will relate to the output file
maskData.maskName = "coh_80_100"
calculateMask(proj, maskData, sites=["M6", "Remote"])
maskData.printInfo()

# process remote on its own to compare to the first processing we did
processProject(
    proj,
    sites=["Remote"],
    sampleFreqs=[128],
    masks={"Remote": "coh_80_100"},
    postpend="coh_80_100",
)
figs = viewImpedance(
    proj,
    sites=["Remote"],
    sampleFreqs=[128],
    postpend="coh_80_100",
    oneplot=False,
    plotoptions=plotOptions,    
    save=False,
    show=False,
)
figs[0].savefig(Path(proj.imagePath, "singleSiteRemote_128_man8_5_coh80.png"))

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
figs[0].savefig(Path(proj.imagePath, "remoteReferenceM6_128_RR_man8_5_coh_datetime_01.png"))

# one more example with multiple masks
# let's use standard coherence as well for Remote only
processProject(
    proj,
    sites=["M6"],
    sampleFreqs=[128],
    remotesite="Remote",
    masks={"M6": ["rr_cohEqn_80_100"], "Remote": "coh_80_100"},
    datetimes=[
        {"type": "time", "start": "20:00:00", "stop": "07:00:00", "levels": [0, 1]}
    ],
    postpend="rr_cohEqn_80_100_and_coh_80_100_night",
)
figs = viewImpedance(
    proj,
    sites=["M6"],
    sampleFreqs=[128],
    postpend="rr_cohEqn_80_100_and_coh_80_100_night",
    oneplot=False,
    plotoptions=plotOptions,    
    save=False,
    show=False,
)
figs[0].savefig(Path(proj.imagePath, "remoteReferenceM6_128_RR_man8_5_2coh_datetime_01.png"))

# try one more where the Remote coherence mask is variable to avoid losing two many windows and long periods
maskData = newMaskData(proj, 128)
maskData.setStats(["coherence"])
maskData.addConstraintLevel("coherence", {"cohExHy": [0.9, 1.0], "cohEyHx": [0.9, 1.0]}, 0)
maskData.addConstraintLevel("coherence", {"cohExHy": [0.8, 1.0], "cohEyHx": [0.8, 1.0]}, 1)
maskData.addConstraintLevel("coherence", {"cohExHy": [0.5, 1.0], "cohEyHx": [0.5, 1.0]}, 2)
maskData.addConstraintLevel("coherence", {"cohExHy": [0.3, 1.0], "cohEyHx": [0.3, 1.0]}, 3)
maskData.addConstraintLevel("coherence", {"cohExHy": [0.2, 1.0], "cohEyHx": [0.2, 1.0]}, 4)
# finally, lets give maskData a name, which will relate to the output file
maskData.maskName = "coh_variable"
calculateMask(proj, maskData, sites=["M6", "Remote"])
maskData.printInfo()

processProject(
    proj,
    sites=["M6"],
    sampleFreqs=[128],
    remotesite="Remote",
    masks={"M6": ["rr_cohEqn_80_100"], "Remote": "coh_variable"},
    datetimes=[
        {"type": "time", "start": "20:00:00", "stop": "07:00:00", "levels": [0, 1]}
    ],
    postpend="rr_cohEqn_80_100_and_coh_variable_night",
)
figs = viewImpedance(
    proj,
    sites=["M6"],
    sampleFreqs=[128],
    postpend="rr_cohEqn_80_100_and_coh_variable_night",
    oneplot=False,
    plotoptions=plotOptions,    
    save=False,
    show=False,
)
figs[0].savefig(Path(proj.imagePath, "remoteReferenceM6_128_RR_man8_5_cohvar_datetime_01.png"))