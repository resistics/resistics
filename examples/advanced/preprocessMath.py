from pathlib import Path
from resistics.project.projectIO import loadProject

projectPath = Path("preprocessProject")
proj = loadProject(projectPath)
proj.printInfo()

# resample to 1024 Hz and save in new site
from resistics.project.projectTime import preProcess, viewTime

preProcess(proj, sites="site1", polreverse={"Ey": True}, outputsite="site1_polreverse", prepend="")
proj.refresh()
proj.printInfo()

viewTime(
    proj,
    "2012-02-10 11:05:00",
    "2012-02-10 11:05:01",
    sites=["site1", "site1_polreverse"],
    chans=["Ex", "Ey"],
    show=True,
    save=True,
)

preProcess(proj, sites="site1", scale={"Ex": -2, "Ey": 0.5}, outputsite="site1_scale", prepend="")
proj.refresh()
proj.printInfo()

viewTime(
    proj,
    "2012-02-10 11:05:00",
    "2012-02-10 11:05:01",
    sites=["site1", "site1_scale"],
    chans=["Ex", "Ey"],
    show=True,
    save=True,
)