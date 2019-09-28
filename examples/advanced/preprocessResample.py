from configuration import preprocessPath, preprocessImages
from resistics.project.projectIO import loadProject

proj = loadProject(preprocessPath)
proj.printInfo()

from resistics.utilities.utilsPlotter import plotOptionsTime, getPresentationFonts
plotOptions = plotOptionsTime(plotfonts=getPresentationFonts())

# resample to 1024 Hz and save in new site
from resistics.project.projectTime import preProcess

preProcess(
    proj, sites="site1", resamp={4096: 1024}, outputsite="site1_resample", prepend=""
)
proj.refresh()
proj.printInfo()

# let's view the time series
from resistics.project.projectTime import viewTime

fig = viewTime(
    proj,
    "2012-02-10 11:05:00",
    "2012-02-10 11:05:03",
    sites=["site1", "site1_resample"],
    chans=["Hx", "Hy", "Hz"],
    show=False,
    plotoptions=plotOptions,
)
fig.savefig(preprocessImages / "viewTimeResample.png")