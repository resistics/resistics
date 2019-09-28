from configuration import preprocessPath, preprocessImages
from resistics.project.projectIO import loadProject

proj = loadProject(preprocessPath)
proj.printInfo()

from resistics.utilities.utilsPlotter import plotOptionsTime, getPresentationFonts
plotOptions = plotOptionsTime(plotfonts=getPresentationFonts())

# polarity reverse the Ey channel
from resistics.project.projectTime import preProcess, viewTime

preProcess(
    proj,
    sites="site1",
    polreverse={"Ey": True},
    outputsite="site1_polreverse",
    prepend="",
)
proj.refresh()
proj.printInfo()

fig = viewTime(
    proj,
    "2012-02-10 11:05:00",
    "2012-02-10 11:05:01",
    sites=["site1", "site1_polreverse"],
    chans=["Ex", "Ey"],
    show=False,
    plotoptions=plotOptions,
)
fig.savefig(preprocessImages / "viewTimePolarityReversal.png")

preProcess(
    proj,
    sites="site1",
    scale={"Ex": -2, "Ey": 0.5},
    outputsite="site1_scale",
    prepend="",
)
proj.refresh()
proj.printInfo()

fig = viewTime(
    proj,
    "2012-02-10 11:05:00",
    "2012-02-10 11:05:01",
    sites=["site1", "site1_scale"],
    chans=["Ex", "Ey"],
    show=False,
    plotoptions=plotOptions,
)
fig.savefig(preprocessImages / "viewTimeScale.png")

# normalisation
preProcess(
    proj,
    sites="site1",
    normalise=True,
    outputsite="site1_norm",
    prepend="",
)
proj.refresh()
proj.printInfo()

fig = viewTime(
    proj,
    "2012-02-10 11:05:00",
    "2012-02-10 11:05:01",
    sites=["site1", "site1_norm"],
    chans=["Ex", "Ey"],
    show=False,
    plotoptions=plotOptions,
)
fig.savefig(preprocessImages / "viewTimeNorm.png")