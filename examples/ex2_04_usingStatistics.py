import os
from resistics.project.projectIO import loadProject
from resistics.project.projectStatCalc import calculateStatistics, calculateRemoteStatistics
from resistics.project.projectView import viewStatistic, viewStatisticHistogram
from resistics.utilities.utilsStats import getStatNames


"""Using statistics

One of the main features of the package is the ability to calculate statistics on an evaluation frequency basis. Currently only built-in statistics are supported, but the plan is to extend this to allow for custom statistics, though how this will be technically done is yet to be decided.

It is possible to calculate statistics for standard magnetotelluric processing and remote reference processing. In this example, only standard magnetotelluric processing will be considered. 

Statistic data is calculated for each evaluation frequency for each spectra file. The data is stored in the following path:
project -> statData -> site -> measurement -> specdir -> statName -> statData
"""

# need the project path for loading
projectPath = os.path.join("exampleProject2")
projData = loadProject(projectPath)

# get a list of all statistic names
stats, remotestats = getStatNames()

# now calculate out statistics
# these statistics can be found under: project -> statData -> site -> measurement -> spectra -> statName -> statData
# calculateStatistics(projData, sites=["M1", "M13", "RemoteResampled"])

# remote statistics
calculateRemoteStatistics(projData, "RemoteResampled", sites=["M1", "M13"], remotestats=remotestats, sampleFreqs=[128])

projData.refresh()
# let's view a statistic
# viewStatistic(projData, "M1", 128, "transferFunction", declevel=0, eFreqI=3, show=False, save=True)
# viewStatisticHistogram(projData, "M1", 128, "transferFunction", declevel=0, eFreqI=3, xlim=[-1000, 1000], show=False, save=True)
# viewStatistic(projData, "M13", 128, "transferFunction", declevel=0, eFreqI=3, show=False, save=True)
# viewStatisticHistogram(projData, "M13", 128, "transferFunction", declevel=0, eFreqI=3, xlim=[-1000, 1000], show=False, save=True)

# and can also plot remote reference statitsics
for statName in remotestats:
    viewStatistic(projData, "M1", 128, statName, declevel=0, eFreqI=3, show=False, save=True)
    viewStatisticHistogram(projData, "M1", 128, statName, declevel=0, eFreqI=3, xlim=[-1500, 1500], show=False, save=True)
    viewStatistic(projData, "M13", 128, statName, declevel=0, eFreqI=3, show=False, save=True)
    viewStatisticHistogram(projData, "M13", 128, statName, declevel=0, eFreqI=3, xlim=[-1500, 1500], show=False, save=True)