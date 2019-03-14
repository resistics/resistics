"""Using Masks

Once statistics have been calculated out, they can be used to mask windows by providing statistic constraints. Masking windows removes them from processing when the window does not meet the constraints.
Mask data is stored in the maskData folder and is associated with spectra directories. When masks are calculated, a new file is produced for each unique sampling frequency in a particular set of spectra files. In the example below, the set of spectra files calculated from the config file is selected (specdir = "config8_5").

Mask data can be found as follows
project -> maskData -> site -> specdir -> maskData (a different file for each unique sampling frequency)
"""

import os
from resistics.project.projectIO import loadProject

# load project and configuration file
projectPath = os.path.join("exampleProject")
projData = loadProject(projectPath, configFile="ex1_04_config.ini")

# get a mask data object and specify the sampling frequency to mask (128Hz)
from resistics.project.projectMask import newMaskData

maskData = newMaskData(projData, 128)
# set the statistics to use in our masking - these must already be calculated out
maskData.setStats(["coherence", "transferFunction"])
# window must have the coherence parameters "cohExHy" and "cohEyHx" both between 0.7 and 1.0
maskData.addConstraint("coherence", {"cohExHy": [0.7, 1.0], "cohEyHx": [0.7, 1.0]})
# give maskData a name, which will relate to the output file
maskData.maskName = "coh70_100"
# print info to see what has been added
maskData.printInfo()
maskData.printConstraints()
# calculate a file of masked windows for the sampling frequency associated with the maskData
from resistics.project.projectMask import calculateMask

calculateMask(projData, maskData, sites=["site1"])

# do the same for 4096 Hz
maskData = newMaskData(projData, 4096)
maskData.setStats(["coherence", "transferFunction"])
maskData.addConstraint("coherence", {"cohExHy": [0.7, 1.0], "cohEyHx": [0.7, 1.0]})
maskData.maskName = "coh70_100"
maskData.printInfo()
maskData.printConstraints()
calculateMask(projData, maskData, sites=["site1"])

# get statistic data
from resistics.project.projectStatistics import getStatisticData

statData = getStatisticData(projData, "site1", "meas_2012-02-10_11-30-00", "coherence")

# get mask data and the masked windows for decimation level 0 and evaluation frequency index 0
from resistics.project.projectMask import getMaskData

maskData = getMaskData(projData, "site1", "coh70_100", 128)
maskWindows = maskData.getMaskWindowsFreq(0, 0)

# view statistic data again but this time exclude the masked windows
statData.view(0, maskwindows=maskWindows, ylim=[0, 1])
statData.histogram(0, maskwindows=maskWindows, xlim=[0, 1])
statData.crossplot(
    0,
    maskwindows=maskWindows,
    crossplots=[["cohExHy", "cohEyHx"], ["cohExHx", "cohEyHy"]],
    xlim=[0, 1],
    ylim=[0, 1],
)

# if there are more than one data folder of the same site for a sampling frequency, the better way to plot statistics with masks is using the methods in projectStatistics
from resistics.project.projectStatistics import viewStatistic, viewStatisticHistogram

viewStatistic(
    projData,
    "site1",
    128,
    "coherence",
    maskname="coh70_100",
    ylim=[0, 1],
    save=True,
    show=False,
)
viewStatisticHistogram(
    projData,
    "site1",
    128,
    "coherence",
    maskname="coh70_100",
    xlim=[0, 1],
    save=True,
    show=False,
)

# Process the data with this mask
from resistics.project.projectTransferFunction import processProject, viewTransferFunction

processProject(
    projData,
    sampleFreqs=[4096, 128],
    sites=["site1"],
    outchans=["Ex", "Ey", "Hz"],
    masks={"site1": "coh70_100"},
    postpend="coh70_100",
)
projData.refresh()
viewTransferFunction(
    projData, sites=["site1"], postpend="coh70_100", oneplot=False, save=True, show=True
)

