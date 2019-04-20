import os
from resistics.project.projectIO import loadProject

# need the project path for loading
projectPath = os.path.join("tutorialProject")
# projData = loadProject(projectPath)

# # get default statistic names
# from resistics.utilities.utilsStats import getStatNames

# stats, remotestats = getStatNames()

# # calculate statistics
# from resistics.project.projectStatistics import calculateStatistics

# calculateStatistics(projData, stats=stats)

# calculate statistics for a different spectra directory
# load the project with the tutorial configuration file
projData = loadProject(projectPath, configFile="tutorialConfig.ini")
projData.printInfo()
# calculateStatistics(projData, stats=stats)

# to get statistic data, we need the site, the measurement and the statistic we want
from resistics.project.projectStatistics import getStatisticData

# coherence statistic data
statData = getStatisticData(
    projData, "site1", "meas_2012-02-10_11-30-00", "coherence", declevel=0
)
# view statistic value over time
statData.view(0)
# view statistic histogram
statData.histogram(0)
# view statistic crossplot
statData.crossplot(0, crossplots=[["cohExHy", "cohEyHx"], ["cohExHx", "cohEyHy"]])

# transfer function statistic data
statData = getStatisticData(
    projData, "site1", "meas_2012-02-10_11-30-00", "transferFunction", declevel=0
)
# view statistic value over time
statData.view(0, ylim=[-2000, 2000])
# view statistic histogram
statData.histogram(0, xlim=[-500, 500])
# view statistic crossplot
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
# look at the next evaluation frequency
statData.view(1, ylim=[-2000, 2000])

# plot statistics for all data of a of a sampling frequency for a site
from resistics.project.projectStatistics import viewStatistic, viewStatisticHistogram

# statistic in time
viewStatistic(
    projData,
    "site1",
    128,
    "transferFunction",
    ylim=[-2000, 2000],
    save=True,
    show=False,
)
# statistic histogram
viewStatisticHistogram(
    projData, "site1", 128, "transferFunction", xlim=[-500, 500], save=True, show=False
)
# change the evaluation frequency
viewStatistic(
    projData,
    "site1",
    128,
    "transferFunction",
    ylim=[-2000, 2000],
    eFreqI=1,
    save=True,
    show=False,
)
# change the decimation level
viewStatistic(
    projData,
    "site1",
    128,
    "transferFunction",
    ylim=[-2000, 2000],
    declevel=1,
    eFreqI=0,
    save=True,
    show=False,
)
