from datapaths import projectPath, imagePath
from resistics.project.io import loadProject

# load project and configuration file
projData = loadProject(projectPath, configFile="tutorialConfig.ini")

# get decimation parameters
from resistics.project.shortcuts import getDecimationParameters

decimationParameters = getDecimationParameters(4096, projData.config)
decimationParameters.printInfo()

# get the window parameters
from resistics.project.shortcuts import getWindowParameters

windowParameters = getWindowParameters(decimationParameters, projData.config)
windowParameters.printInfo()

from resistics.project.shortcuts import getWindowSelector

decimationParameters = getDecimationParameters(128, projData.config)
windowParameters = getWindowParameters(decimationParameters, projData.config)
selector = getWindowSelector(projData, decimationParameters, windowParameters)
selector.printInfo()

# add a site and print the information to the terminal
selector.setSites(["site1"])
selector.printInfo()

# calculate shared windows
selector.calcSharedWindows()
selector.printSharedWindows()

# currently using every window, but how might this change with a date or time constraint
selector.addDateConstraint("2012-02-11")
selector.printDatetimeConstraints()
selector.calcSharedWindows()
selector.printSharedWindows()

# reset datetime constraints
selector.resetDatetimeConstraints()
selector.addDatetimeConstraint("2012-02-10 19:00:00", "2012-02-11 08:00:00")
selector.printDatetimeConstraints()
selector.calcSharedWindows()
selector.printSharedWindows()

# can add a mask
selector.addWindowMask("site1", "coh70_100")
selector.printWindowMasks()
selector.calcSharedWindows()
selector.printSharedWindows()

# the number of windows for each evaluation frequency will be different when masks are applied
# loop over the evaluation frequencies and print the number of windows that will be processed
# save the information to a file
with open("usingWindowSelector.txt", "w") as f:
    numDecimationLevels = decimationParameters.numLevels
    numFrequenciesPerLevel = decimationParameters.freqPerLevel
    for declevel in range(numDecimationLevels):
        evaluationFrequencies = decimationParameters.getEvalFrequenciesForLevel(
            declevel
        )
        for eIdx in range(numFrequenciesPerLevel):
            f.write(
                "Decimation level = {}, evaluation frequency = {:.3f}, number of windows = {:d}\n".format(
                    declevel,
                    evaluationFrequencies[eIdx],
                    len(selector.getWindowsForFreq(declevel, eIdx)),
                )
            )