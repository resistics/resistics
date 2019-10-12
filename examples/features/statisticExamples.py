from datapaths import projectPath, statImagePath
from resistics.project.io import loadProject
from resistics.project.statistics import getStatisticData
from resistics.statistics.utils import getStatNames

proj = loadProject(projectPath)
stats, remoteStats = getStatNames()
statCrossplots = dict()
statCrossplots["coherence"] = [["cohExHy", "cohEyHx"], ["cohExHx", "cohEyHy"]]
statCrossplots["transferFunction"] = [
    ["ExHxReal", "ExHxImag"],
    ["ExHyReal", "ExHyImag"],
    ["EyHxReal", "EyHxImag"],
    ["EyHyReal", "EyHyImag"],
]

# for stat in stats:
#     statData = getStatisticData(proj, "M1", "meas_2016-03-23_02-00-00", stat)
#     fig = statData.view(0)
#     fig.savefig(statImagePath / "M1_{}_view_4096".format(stat))
#     fig = statData.histogram(0)
#     fig.savefig(statImagePath / "M1_{}_histogram_4096".format(stat))
#     if stat in statCrossplots:
#         fig = statData.crossplot(0, crossplots=statCrossplots[stat])
#         fig.savefig(statImagePath / "M1_{}_crossplot_4096".format(stat))
#         fig = statData.densityplot(0, crossplots=statCrossplots[stat])
#         fig.savefig(statImagePath / "M1_{}_densityplot_4096".format(stat))


for stat in stats:
    statData = getStatisticData(proj, "Remote_M1", "run4_2016-03-25_02-35-00", stat)
    fig = statData.view(0)
    fig.savefig(statImagePath / "Remote_{}_view_128".format(stat))
    fig = statData.histogram(0)
    fig.savefig(statImagePath / "Remote_{}_histogram_128".format(stat))
    if stat in statCrossplots:
        fig = statData.crossplot(0, crossplots=statCrossplots[stat])
        fig.savefig(statImagePath / "Remote_{}_crossplot_128".format(stat))
        fig = statData.densityplot(0, crossplots=statCrossplots[stat])
        fig.savefig(statImagePath / "Remote_{}_densityplot_128".format(stat))


