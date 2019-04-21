import os
from resistics.project.projectIO import loadProject

# load project and configuration file
projectPath = os.path.join("tutorialProject")
projData = loadProject(projectPath, configFile="tutorialConfig.ini")

# get mask data
from resistics.project.projectMask import getMaskData

maskData = getMaskData(projData, "site1", "coh70_100", 4096)
# get the masked windows for decimation level 0 and evaluation frequency index 0
maskWindows = maskData.getMaskWindowsFreq(0, 0)

# get statistic data
from resistics.project.projectStatistics import getStatisticData

statData = getStatisticData(
    projData, "site1", "meas_2012-02-10_11-05-00", "transferFunction"
)

# view masked statistic data again but this with constraints on both coherence and transfer function
fig = statData.view(0, ylim=[-2000, 2000])
fig.savefig(os.path.join("tutorialProject", "images", "statistic_4096_nomask_view"))
fig = statData.view(0, maskwindows=maskWindows, ylim=[-2000, 2000])
fig.savefig(os.path.join("tutorialProject", "images", "statistic_4096_maskcoh_view"))
# histogram
fig = statData.histogram(0, xlim=[-1000, 1000])
fig.savefig(os.path.join("tutorialProject", "images", "statistic_4096_nomask_hist"))
fig = statData.histogram(0, maskwindows=maskWindows, xlim=[-1000, 1000])
fig.savefig(os.path.join("tutorialProject", "images", "statistic_4096_maskcoh_hist"))
# crossplot
fig = statData.crossplot(
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
fig.savefig(os.path.join("tutorialProject", "images", "statistic_4096_nomask_crossplot"))
fig = statData.crossplot(
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
fig.savefig(os.path.join("tutorialProject", "images", "statistic_4096_maskcoh_crossplot"))

# view statistic data again but this time exclude the masked windows
maskData = getMaskData(projData, "site1", "coh70_100_tfConstrained", 4096)
maskWindows = maskData.getMaskWindowsFreq(0, 0)
fig = statData.view(0, maskwindows=maskWindows, ylim=[-2000, 2000])
fig.savefig(os.path.join("tutorialProject", "images", "statistic_4096_maskcoh_tf_view"))
fig = statData.histogram(0, maskwindows=maskWindows, xlim=[-1000, 1000])
fig.savefig(os.path.join("tutorialProject", "images", "statistic_4096_maskcoh_tf_hist"))
fig = statData.crossplot(
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
fig.savefig(os.path.join("tutorialProject", "images", "statistic_4096_maskcoh_tf_crossplot"))

# if there are more than one data folder for the same site at the same sampling frequency
# the better way to plot statistics with masks is using the methods in projectStatistics
from resistics.project.projectStatistics import viewStatistic, viewStatisticHistogram
from resistics.utilities.utilsPlotter import plotOptionsStandard, getPaperFonts

plotOptions = plotOptionsStandard(plotfonts=getPaperFonts())
viewStatistic(
    projData,
    "site1",
    128,
    "coherence",
    maskname="coh70_100",
    ylim=[0, 1],
    save=True,
    show=False,
    plotoptions=plotOptions,
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
    plotoptions=plotOptions,    
)