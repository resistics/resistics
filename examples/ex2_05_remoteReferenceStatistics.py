import os
from resistics.project.projectIO import loadProject
from resistics.project.projectMask import newMaskData, calculateMask
from resistics.project.projectTransferFunction import processProject, viewTransferFunction


"""Using Masks

Once statistics have been calculated out, they can be used to mask windows by providing statistic constraints. Masking windows removes them from processing when the window does not meet the constraints.
Mask data is stored in the maskData folder and is associated with spectra directories. When masks are calculated, a new file is produced for each unique sampling frequency in a particular set of spectra files. In the example below, the set of spectra files calculated from the config file is selected (specdir = "config8_5").

Mask data can be found as follows
project -> maskData -> site -> specdir -> maskData (a different file for each unique sampling frequency)
"""

# need the project path for loading
# let's use our configuration file
projectPath = os.path.join("exampleProject2")
projData = loadProject(projectPath)

# # get a mask data object and specify the sampling frequency to mask (128Hz)
# maskData = getMaskData(projData, 128)
# # set the statistics to use in our masking
# maskData.setStats(["coherence", "transferFunction"])
# # initially, let's set it that the window must have the statistic "cohExHy" and "cohEyHx" both between 0.7 and 1.0
# maskData.addConstraint("coherence", {"cohExHy": [0.7, 1.0], "cohEyHx": [0.7, 1.0]})
# # finally, lets give maskData a name, which will relate to the output file
# maskData.maskName = "coh70_100"
# # print info to see what has been added
# maskData.printInfo()
# maskData.printConstraints()
# Now that this has been done, we can use it to make a file of masked windows
# this will do it only for the sampling frequency associated with the maskData
# calculateMask(projData, maskData, sites=["M1", "M13"])
# processProject(
#     projData,
#     sites=["M1", "M13"],
#     sampleFreqs=[128],
#     remotesite="RemoteResampled",
#     masks={"M1": "coh70_100", "M13": "coh70_100"},
#     postpend="rr_coh70_100",
# )
# projData.refresh()
# viewTransferFunction(
#     projData,
#     sites=["M1", "M13"],
#     postpend="rr_coh70_100",
#     oneplot=False,
#     save=True,
#     show=True,
# )

# let's create another mask that uses some of the remote reference statistics that were calculated
# get a mask data object and specify the sampling frequency to mask (128Hz)
maskData = newMaskData(projData, 128)
# set the statistics to use in our masking
maskData.setStats(["RR_coherence", "RR_transferFunction"])
# initially, let's set it that the window must have the statistic "cohExHy" and "cohEyHx" both between 0.7 and 1.0
maskData.addConstraint(
    "RR_transferFunction", {"cohExHy": [0.7, 1.0], "cohEyHx": [0.7, 1.0]}
)
# finally, lets give maskData a name, which will relate to the output file
maskData.maskName = "rr_tfcoh70_100"
# print info to see what has been added
# maskData.printInfo()
# maskData.printConstraints()
# Now that this has been done, we can use it to make a file of masked windows
# this will do it only for the sampling frequency associated with the maskData
calculateMask(projData, maskData, sites=["M1", "M13"])
processProject(
    projData,
    sites=["M1", "M13"],
    sampleFreqs=[128],
    remotesite="RemoteResampled",
    masks={"M1": "coh70_100", "M13": "coh70_100"},
    postpend="rr_coh70_100",
)
projData.refresh()
viewTransferFunction(
    projData,
    sites=["M1", "M13"],
    postpend="rr_coh70_100",
    oneplot=False,
    save=True,
    show=True,
)
