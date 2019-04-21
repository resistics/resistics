import os
from resistics.project.projectIO import loadProject

# need the project path for loading
projectPath = os.path.join("tutorialProject")
projData = loadProject(projectPath)

# get default statistic names
from resistics.utilities.utilsStats import getStatNames

stats, remotestats = getStatNames()

# calculate statistics
from resistics.project.projectStatistics import calculateStatistics

calculateStatistics(projData, stats=stats)

# calculate statistics for a different spectra directory
# load the project with the tutorial configuration file
projData = loadProject(projectPath, configFile="tutorialConfig.ini")
projData.printInfo()
calculateStatistics(projData, stats=stats)

# to get statistic data, we need the site, the measurement and the statistic we want
from resistics.project.projectStatistics import getStatisticData

# coherence statistic data
statData = getStatisticData(
    projData, "site1", "meas_2012-02-10_11-30-00", "coherence", declevel=0
)
# view statistic value over time
fig = statData.view(0)
fig.savefig(os.path.join("tutorialProject", "images", "statistic_coherence_view"))
# view statistic histogram
fig = statData.histogram(0)
fig.savefig(os.path.join("tutorialProject", "images", "statistic_coherence_histogram"))
# view statistic crossplot
fig = statData.crossplot(0, crossplots=[["cohExHy", "cohEyHx"], ["cohExHx", "cohEyHy"]])
fig.savefig(os.path.join("tutorialProject", "images", "statistic_coherence_crossplot"))


# transfer function statistic data
statData = getStatisticData(
    projData, "site1", "meas_2012-02-10_11-30-00", "transferFunction", declevel=0
)
# view statistic value over time
fig = statData.view(0, ylim=[-2000, 2000])
fig.savefig(os.path.join("tutorialProject", "images", "statistic_transferfunction_view"))
# view statistic histogram
fig = statData.histogram(0, xlim=[-500, 500])
fig.savefig(os.path.join("tutorialProject", "images", "statistic_transferfunction_histogram"))
# view statistic crossplot
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
fig.savefig(os.path.join("tutorialProject", "images", "statistic_transferfunction_crossplot"))
# look at the next evaluation frequency
fig = statData.view(1, ylim=[-2000, 2000])
fig.savefig(os.path.join("tutorialProject", "images", "statistic_transferfunction_view_eval1"))

# plot statistic values in time for all data of a specified sampling frequency in a site
from resistics.project.projectStatistics import viewStatistic

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

# plot statistic histogram for all data of a specified sampling frequency in a site
from resistics.project.projectStatistics import viewStatisticHistogram

# statistic histogram
viewStatisticHistogram(
    projData, "site1", 128, "transferFunction", xlim=[-500, 500], save=True, show=False
)

# more examples
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
