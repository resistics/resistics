import os
from resistics.project.projectIO import loadProject

# load project and configuration file
projectPath = os.path.join("tutorialProject")
projData = loadProject(projectPath, configFile="tutorialConfig.ini")

# get mask data
from resistics.project.projectMask import getMaskData

maskData = getMaskData(projData, "site1", "coh70_100", 128)
# get the masked windows for decimation level 0 and evaluation frequency index 0
maskWindows = maskData.getMaskWindowsFreq(0, 0)

# get statistic data
from resistics.project.projectStatistics import getStatisticData

statData = getStatisticData(
    projData, "site1", "meas_2012-02-10_11-30-00", "transferFunction"
)

# view masked statistic data again but this with constraints on both coherence and transfer function
statData.view(0, ylim=[-2000, 2000])
statData.view(0, maskwindows=maskWindows, ylim=[-2000, 2000])
# histogram
statData.histogram(0, xlim=[-500, 500])
statData.histogram(0, maskwindows=maskWindows, xlim=[-500, 500])
# crossplot
statData.crossplot(
    0,
    crossplots=[
        ["ExHxReal", "ExHxImag"],
        ["ExHyReal", "ExHyImag"],
        ["EyHxReal", "EyHxImag"],
        ["EyHyReal", "EyHyImag"],
    ],
    xlim=[-2500, 2500],
    ylim=[-2500, 2500],
)
statData.crossplot(
    0,
    maskwindows=maskWindows,
    crossplots=[
        ["ExHxReal", "ExHxImag"],
        ["ExHyReal", "ExHyImag"],
        ["EyHxReal", "EyHxImag"],
        ["EyHyReal", "EyHyImag"],
    ],
    xlim=[-2500, 2500],
    ylim=[-2500, 2500],
)

# view statistic data again but this time exclude the masked windows
maskData = getMaskData(projData, "site1", "coh70_100_tfConstrained", 128)
maskWindows = maskData.getMaskWindowsFreq(0, 0)
statData.view(0, maskwindows=maskWindows, ylim=[-2000, 2000])
statData.histogram(0, maskwindows=maskWindows, xlim=[-500, 500])
statData.crossplot(
    0,
    maskwindows=maskWindows,
    crossplots=[
        ["ExHxReal", "ExHxImag"],
        ["ExHyReal", "ExHyImag"],
        ["EyHxReal", "EyHxImag"],
        ["EyHyReal", "EyHyImag"],
    ],
    xlim=[-2500, 2500],
    ylim=[-2500, 2500],
)

# if there are more than one data folder of the same site for a sampling frequency
# the better way to plot statistics with masks is using the methods in projectStatistics
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