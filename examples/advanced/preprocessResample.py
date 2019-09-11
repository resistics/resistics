from pathlib import Path
from resistics.project.projectIO import loadProject

projectPath = Path("preprocessProject")
proj = loadProject(projectPath)
proj.printInfo()

# resample to 1024 Hz and save in new site
from resistics.project.projectTime import preProcess

preProcess(proj, sites="site1", resamp={4096:1024}, outputsite="site1_resample", prepend="")
proj.refresh()
proj.printInfo()

# let's view the time series
from resistics.project.projectTime import viewTime

viewTime(
    proj,
    "2012-02-10 11:05:00",
    "2012-02-10 11:05:03",
    sites=["site1", "site1_lowpass"],
    chans=["Hx", "Hy", "Hz"],
    show=True,
    save=True,
)