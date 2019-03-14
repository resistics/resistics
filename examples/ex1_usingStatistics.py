"""Using statistics

One of the main features of the package is the ability to calculate statistics on an evaluation frequency basis. Currently only built-in statistics are supported, but the plan is to extend this to allow for custom statistics, though how this will be technically done is yet to be decided.

It is possible to calculate statistics for standard magnetotelluric processing and remote reference processing. In this example, only standard magnetotelluric processing will be considered. 

Statistic data is calculated for each evaluation frequency for each spectra file. The data is stored in the following path:
project -> statData -> site -> measurement -> specdir -> statName -> statData
"""

import os
from resistics.project.projectIO import loadProject
from resistics.project.projectStatistics import calculateStatistics
from resistics.utilities.utilsStats import getStatNames

# need the project path for loading
projectPath = os.path.join("exampleProject")
projData = loadProject(projectPath)
stats, remotestats = getStatNames()
calculateStatistics(projData, stats=stats)

# calculate statistics for a different spectra directory
projData = loadProject(projectPath, configFile="ex1_04_config.ini")
projData.printInfo()
calculateStatistics(projData, stats=stats)

# to get statistic data, we need the site, the measurement and the statistic we want
from resistics.project.projectStatistics import getStatisticData
statData = getStatisticData(projData, "site1", "meas_2012-02-10_11-30-00", "coherence", declevel=0)
# view statistic value over time
statData.view(0)
# view statistic histogram
statData.histogram(0)
# view statistic crossplot
statData.crossplot(0, crossplots=[["cohExHy", "cohEyHx"], ["cohExHx", "cohEyHy"]])

# plot statistics for all data of a of a sampling frequency for a site
from resistics.project.projectStatistics import viewStatistic, viewStatisticHistogram
from resistics.utilities.utilsPlotter import plotOptionsSpec
options = plotOptionsSpec()
options["figsize"] = (20,7)
# statistic in time
viewStatistic(projData, "site1", 128, "transferFunction", save=True, show=False)
# statistic histogram
viewStatisticHistogram(
    projData, "site1", 128, "transferFunction", save=True, show=False
)
# change the evaluation frequency
viewStatistic(
    projData, "site1", 128, "transferFunction", eFreqI=1, save=True, show=False
)
# change the decimation level 
viewStatistic(
    projData, "site1", 128, "transferFunction", declevel=1, eFreqI=0, save=True, show=False
)
