import os
from resistics.project.projectIO import loadProject
from resistics.project.projectMask import getMaskData, calculateMask
from resistics.project.projectTransferFunctionCalc import processProject
from resistics.project.projectView import viewTransferFunction

# need the project path for loading
projectPath = os.path.join("exampleProject")
projData = loadProject(projectPath)

# # get a mask data object and specify the sampling frequency to mask
# maskData = getMaskData(projData, 128) 
# # set the statistics to use in our masking
# maskData.setStats(["cohStat", "tfStat"])
# # initially, let's set it that the window must have the statistic "cohExHy" and "cohEyHx" both between 0.7 and 1.0
# maskData.addConstraint("cohStat", {"cohExHy": [0.7,1.0], "cohEyHx": [0.7, 1.0]})
# # finally, lets give maskData a name, which will relate to the output file
# maskData.maskName = "coh70_100"
# # # print info to see what has been added
# maskData.printInfo()
# maskData.printConstraints()

# # # Now that this has been done, we can use it to make a file of masked windows
# # # this will do it only for the sampling frequency associated with the maskData
# calculateMask(projData, maskData)

# now the mask has been created and can be found in the stats project directory
# let's process the data with this mask
processProject(projData, sampleFreqs=[128], sites=["site1"], outchans=["Ex", "Ey", "Hz"], masks={"site1": "coh70_100"}, postpend="coh70_100")
# viewTransferFunction(projData, sites=["site1"], postpend="coh70_100", oneplot=False, save=True, show=True)

# we can also make another mask. When looking at the histograms in the statistics, there are large outliers
# beyond the standard bell curve. 
# these could manifest as leverage points or high residual points, which subsequently cause issues with the statistics
# let's try and remove these in addition to the coherence thresholding
# start again by getting a maskData object
# maskData = pMask.getMaskData(128)
# now we start adding information to our maskData object
# first let's add coherence thresholding
# but this time, we can changes the threshold by decimation level - have it higer for lower levels
# as lower levels have more windows
# maskData.setStats(["cohStat", "tfStat"])
# maskData.addConstraintLevel("cohStat", {"cohExHy": [0.9,1.0], "cohEyHx": [0.9, 1.0]}, 0)
# maskData.addConstraintLevel("cohStat", {"cohExHy": [0.8,1.0], "cohEyHx": [0.8, 1.0]}, 1)
# maskData.addConstraintLevel("cohStat", {"cohExHy": [0.7,1.0], "cohEyHx": [0.7, 1.0]}, 2)
# maskData.addConstraintLevel("cohStat", {"cohExHy": [0.6,1.0], "cohEyHx": [0.6, 1.0]}, 3)
# maskData.addConstraintLevel("cohStat", {"cohExHy": [0.5,1.0], "cohEyHx": [0.5, 1.0]}, 4)
# maskData.addConstraint("tfStat", {"ExHyReal": [-200,200], "ExHyImag": [-200,200], "EyHxReal": [-200,200], "EyHxImag": [-200,200]})
# set the maskName
# maskData.maskName ="cohVar_tf0_200"
# maskData.printConstraints()
# pMask.calculate(maskData)
