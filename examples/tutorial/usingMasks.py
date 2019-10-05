from datapaths import projectPath, imagePath
from resistics.project.io import loadProject

# load project and configuration file
projData = loadProject(projectPath, configFile="tutorialConfig.ini")

# get a mask data object and specify the sampling frequency to mask (128Hz)
from resistics.project.mask import newMaskData

maskData = newMaskData(projData, 128)
# set the statistics to use in our masking - these must already be calculated out
maskData.setStats(["coherence"])
# window must have the coherence parameters "cohExHy" and "cohEyHx" both between 0.7 and 1.0
maskData.addConstraint("coherence", {"cohExHy": [0.7, 1.0], "cohEyHx": [0.7, 1.0]})
# give maskData a name, which will relate to the output file
maskData.maskName = "coh70_100"
# print info to see what has been added
maskData.printInfo()
maskData.printConstraints()

# calculate a file of masked windows for the sampling frequency associated with the maskData
from resistics.project.mask import calculateMask

calculateMask(projData, maskData, sites=["site1"])
fig = maskData.view(0)
fig.savefig(imagePath / "usingMasks_maskData_128_coh_dec0")

# do the same for 4096 Hz
maskData = newMaskData(projData, 4096)
maskData.setStats(["coherence"])
maskData.addConstraint("coherence", {"cohExHy": [0.7, 1.0], "cohEyHx": [0.7, 1.0]})
maskData.maskName = "coh70_100"
maskData.printInfo()
maskData.printConstraints()
calculateMask(projData, maskData, sites=["site1"])
fig = maskData.view(0)
fig.savefig(imagePath / "usingMasks_maskData_4096_coh_dec0")

# calculate out statistics again, but this time use both transfer function and coherence
maskData = newMaskData(projData, 128)
maskData.setStats(["coherence", "transferFunction"])
maskData.addConstraint("coherence", {"cohExHy": [0.7, 1.0], "cohEyHx": [0.7, 1.0]})
maskData.addConstraint(
    "transferFunction",
    {
        "ExHyReal": [-500, 500],
        "ExHyImag": [-500, 500],
        "EyHxReal": [-500, 500],
        "EyHxImag": [-500, 500],
    },
)
maskData.maskName = "coh70_100_tfConstrained"
calculateMask(projData, maskData, sites=["site1"])
fig = maskData.view(0)
fig.savefig(imagePath / "usingMasks_maskData_128_coh_tf_dec0")

maskData = newMaskData(projData, 4096)
maskData.setStats(["coherence", "transferFunction"])
maskData.addConstraint("coherence", {"cohExHy": [0.7, 1.0], "cohEyHx": [0.7, 1.0]})
maskData.addConstraint(
    "transferFunction",
    {
        "ExHyReal": [-500, 500],
        "ExHyImag": [-500, 500],
        "EyHxReal": [-500, 500],
        "EyHxImag": [-500, 500],
    },
)
maskData.maskName = "coh70_100_tfConstrained"
calculateMask(projData, maskData, sites=["site1"])
fig = maskData.view(0)
fig.savefig(imagePath / "usingMasks_maskData_4096_coh_tf_dec0")

