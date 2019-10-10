from datapaths import projectPath
from resistics.project.io import loadProject

proj = loadProject(projectPath)
proj.printInfo()

# calculate statistics
from resistics.project.spectra import calculateSpectra
from resistics.project.statistics import calculateStatistics, calculateRemoteStatistics
from resistics.statistics.utils import getStatNames

calculateSpectra(proj)
proj.refresh()
stats, remotestats = getStatNames()
calculateStatistics(proj, stats=stats)
calculateRemoteStatistics(proj, "Remote_M1", sites=["M1"], remotestats=remotestats)
