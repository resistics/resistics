from pathlib import Path
from resistics.project.projectIO import loadProject
from resistics.project.projectSpectra import calculateSpectra
from resistics.project.projectTransferFunction import processProject, viewImpedance
from resistics.project.projectStatistics import calculateStatistics

projectPath = Path("remoteProject")
projData = loadProject(projectPath)

# calculate spectrum using standard options
# calculateSpectra(projData, sites=["M6", "Remote"])
# projData.refresh()

# begin with single site processing
# processProject(projData, sites=["M6", "Remote"])

# plot the transfer functions
# viewImpedance(projData, sites=["M6", "Remote"], oneplot=False, save=True)

# perform standard remote reference runs - remember to call the output something else
# processProject(projData, sites=["M6"], sampleFreqs=[128], remotesite="Remote", postpend="rr")
# viewImpedance(projData, sites=["M6"], postpend="rr", oneplot=False, save=True)

# let's calculate some single site statistics
calculateStatistics(projData, sites=["M6", "Remote"], stats=["coherence"])