from datapaths import projectPath, remoteImagePath
from resistics.project.io import loadProject
from resistics.project.statistics import getStatisticData
from resistics.statistics.utils import getStatNames

proj = loadProject(projectPath)
stats, remoteStats = getStatNames()
statCrossplots = dict()
statCrossplots["RR_transferFunction"] = [
    ["ExHxRealRR", "ExHxImagRR"],
    ["ExHyRealRR", "ExHyImagRR"],
    ["EyHxRealRR", "EyHxImagRR"],
    ["EyHyRealRR", "EyHyImagRR"],
]

for stat in remoteStats:
    statData = getStatisticData(
        proj, "M1", "meas_2016-03-26_02-35-00", stat, declevel=1
    )
    fig = statData.view(0)
    fig.savefig(remoteImagePath / "M1_{}_view_128".format(stat))
    fig = statData.histogram(0)
    fig.savefig(remoteImagePath / "M1_{}_histogram_128".format(stat))
    if stat in statCrossplots:
        fig = statData.crossplot(0, crossplots=statCrossplots[stat])
        fig.savefig(remoteImagePath / "M1_{}_crossplot_128".format(stat))
        fig = statData.densityplot(0, crossplots=statCrossplots[stat])
        fig.savefig(remoteImagePath / "M1_{}_densityplot_128".format(stat))
