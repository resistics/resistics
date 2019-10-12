from datapaths import projectPath, imagePath
from resistics.project.io import loadProject

# need the project path for loading
projData = loadProject(projectPath)

# get default statistic names
from resistics.statistics.utils import getStatNames

stats, remotestats = getStatNames()

# calculate statistics
from resistics.project.statistics import calculateStatistics

calculateStatistics(projData, stats=stats)

# calculate statistics for a different spectra directory
# load the project with the tutorial configuration file
projData = loadProject(projectPath, configFile="tutorialConfig.ini")
projData.printInfo()
calculateStatistics(projData, stats=stats)

# to get statistic data, we need the site, the measurement and the statistic we want
from resistics.project.statistics import getStatisticData

# coherence statistic data
statData = getStatisticData(
    projData, "site1", "meas_2012-02-10_11-30-00", "coherence", declevel=0
)
# view statistic value over time
fig = statData.view(0)
fig.savefig(imagePath / "usingStats_statistic_coherence_view")
# view statistic histogram
fig = statData.histogram(0)
fig.savefig(imagePath / "usingStats_statistic_coherence_histogram")
# view statistic crossplot
fig = statData.crossplot(0, crossplots=[["cohExHy", "cohEyHx"], ["cohExHx", "cohEyHy"]])
fig.savefig(imagePath / "usingStats_statistic_coherence_crossplot")
# view statistic density plot
fig = statData.densityplot(
    0,
    crossplots=[["cohExHy", "cohEyHx"], ["cohExHx", "cohEyHy"]],
    xlim=[0, 1],
    ylim=[0, 1],
)
fig.savefig(imagePath / "usingStats_statistic_coherence_densityplot")

# transfer function statistic data
statData = getStatisticData(
    projData, "site1", "meas_2012-02-10_11-30-00", "transferFunction", declevel=0
)
# view statistic value over time
fig = statData.view(0, ylim=[-2000, 2000])
fig.savefig(imagePath / "usingStats_statistic_transferfunction_view")
# view statistic histogram
fig = statData.histogram(0, xlim=[-500, 500])
fig.savefig(imagePath / "usingStats_statistic_transferfunction_histogram")
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
fig.savefig(imagePath / "usingStats_statistic_transferfunction_crossplot")
# view statistic densityplot
fig = statData.densityplot(
    0,
    crossplots=[
        ["ExHxReal", "ExHxImag"],
        ["ExHyReal", "ExHyImag"],
        ["EyHxReal", "EyHxImag"],
        ["EyHyReal", "EyHyImag"],
    ],
    xlim=[-60, 60],
    ylim=[-60, 60],
)
fig.savefig(imagePath / "usingStats_statistic_transferfunction_densityplot")

# look at the next evaluation frequency
fig = statData.view(1, ylim=[-2000, 2000])
fig.savefig(imagePath / "usingStats_statistic_transferfunction_view_eval1")

# plot statistic values in time for all data of a specified sampling frequency in a site
from resistics.project.statistics import viewStatistic

# statistic in time
fig = viewStatistic(
    projData,
    "site1",
    128,
    "transferFunction",
    ylim=[-2000, 2000],
    save=False,
    show=False,
)
fig.savefig(imagePath / "usingStats_projstat_transfunction_view")

# plot statistic histogram for all data of a specified sampling frequency in a site
from resistics.project.statistics import viewStatisticHistogram

# statistic histogram
fig = viewStatisticHistogram(
    projData, "site1", 128, "transferFunction", xlim=[-500, 500], save=False, show=False
)
fig.savefig(imagePath / "usingStats_projstat_transfunction_hist")

# more examples
# change the evaluation frequency
fig = viewStatistic(
    projData,
    "site1",
    128,
    "transferFunction",
    ylim=[-2000, 2000],
    eFreqI=1,
    save=False,
    show=False,
)
fig.savefig(imagePath / "usingStats_projstat_transfunction_view_efreq")

# change the decimation level
fig = viewStatistic(
    projData,
    "site1",
    128,
    "transferFunction",
    ylim=[-2000, 2000],
    declevel=1,
    eFreqI=0,
    save=False,
    show=False,
)
fig.savefig(imagePath / "usingStats_projstat_transfunction_view_declevel")
