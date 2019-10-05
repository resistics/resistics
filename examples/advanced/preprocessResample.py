from datapaths import preprocessPath, preprocessImages
from resistics.project.io import loadProject

proj = loadProject(preprocessPath)
proj.printInfo()

from resistics.common.plot import plotOptionsTime, getPresentationFonts
plotOptions = plotOptionsTime(plotfonts=getPresentationFonts())

# resample to 1024 Hz and save in new site
from resistics.project.time import preProcess

preProcess(
    proj, sites="site1", resamp={4096: 1024}, outputsite="site1_resample", prepend=""
)
proj.refresh()
proj.printInfo()

# let's view the time series
from resistics.project.time import viewTime

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