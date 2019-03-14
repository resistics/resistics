import os
from resistics.project.projectIO import loadProject
from resistics.project.projectStatistics import (
    calculateStatistics,
    getStatisticData,
    viewStatistic,
    viewStatisticHistogram,
)
from resistics.utilities.utilsStats import getStatNames
from resistics.utilities.utilsPlotter import plotOptionsSpec

"""Using statistics

One of the main features of the package is the ability to calculate statistics on an evaluation frequency basis. Currently only built-in statistics are supported, but the plan is to extend this to allow for custom statistics, though how this will be technically done is yet to be decided.

It is possible to calculate statistics for standard magnetotelluric processing and remote reference processing. In this example, only standard magnetotelluric processing will be considered. 

Statistic data is calculated for each evaluation frequency for each spectra file. The data is stored in the following path:
project -> statData -> site -> measurement -> specdir -> statName -> statData

"absvalEqn",
"coherence",
"powerSpectralDensity",
"polarisationDirection",
"partialCoherence",
"transferFunction",
"resPhase",
"""


# need the project path for loading
projectPath = os.path.join("exampleProject")
projData = loadProject(projectPath, configFile="ex1_04_config.ini")
projData.printInfo()
stats, remotestats = getStatNames()

# plot the examples we need
options = plotOptionsSpec()
options["figsize"] = (20, 7)
options["plotfonts"]["suptitle"] = 18
options["plotfonts"]["title"] = 16

# coherence
viewStatistic(
    projData, "site1", 128, "coherence", save=True, show=False, plotoptions=options
)
viewStatisticHistogram(
    projData, "site1", 128, "coherence", save=True, show=False, plotoptions=options
)

# polarisation directions
options["figsize"] = (17, 12)
viewStatistic(
    projData,
    "site1",
    128,
    "polarisationDirection",
    save=True,
    show=False,
    ylim=[-100, 100],
    plotoptions=options
)
viewStatisticHistogram(
    projData,
    "site1",
    128,
    "polarisationDirection",
    save=True,
    show=False,
    xlim=[-100, 100],    
    plotoptions=options
)

# power spectral density
options["figsize"] = (17, 12)
viewStatistic(
    projData,
    "site1",
    128,
    "powerSpectralDensity",
    save=True,
    show=False,
    plotoptions=options
)
viewStatisticHistogram(
    projData,
    "site1",
    128,
    "powerSpectralDensity",
    save=True,
    show=False, 
    plotoptions=options
)

# absvalEqn
options["figsize"] = (17, 12)
viewStatistic(
    projData,
    "site1",
    128,
    "absvalEqn",
    save=True,
    show=False,
    plotoptions=options
)
viewStatisticHistogram(
    projData,
    "site1",
    128,
    "absvalEqn",
    save=True,
    show=False, 
    plotoptions=options
)

# partial coherences
options["figsize"] = (20, 13)
viewStatistic(
    projData,
    "site1",
    128,
    "partialcoherence",
    save=True,
    show=False,
    plotoptions=options,
)
viewStatisticHistogram(
    projData,
    "site1",
    128,
    "partialcoherence",
    save=True,
    show=False,
    plotoptions=options,
)

# transfer function 
viewStatistic(
    projData,
    "site1",
    128,
    "transferFunction",
    save=True,
    show=False,
    ylim=[-1000, 1000],
    plotoptions=options
)
viewStatisticHistogram(
    projData,
    "site1",
    128,
    "transferFunction",
    save=True,
    show=False,
    xlim=[-200, 200],
    plotoptions=options
)

# resistivity and phase
viewStatistic(
    projData,
    "site1",
    128,
    "resPhase",
    save=True,
    show=False,
    ylim=[-1000, 1000],
    plotoptions=options
)
viewStatisticHistogram(
    projData,
    "site1",
    128,
    "resPhase",
    save=True,
    show=False,
    xlim=[-300, 300],
    plotoptions=options
)

