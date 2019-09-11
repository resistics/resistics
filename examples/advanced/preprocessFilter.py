from pathlib import Path
from resistics.project.projectIO import loadProject

projectPath = Path("preprocessProject")
proj = loadProject(projectPath)
proj.printInfo()

# resample to 1024 Hz and save in new site
from resistics.project.projectTime import preProcess, viewTime

preProcess(proj, sites="site1", filter={"lpfilt": 512}, outputsite="site1_lowpass", prepend="")
proj.refresh()
proj.printInfo()

viewTime(
    proj,
    "2012-02-10 11:05:00",
    "2012-02-10 11:05:03",
    sites=["site1", "site1_lowpass"],
    chans=["Hx", "Hy", "Hz"],
    show=True,
    save=True,
)

preProcess(proj, sites="site1", filter={"hpfilt": 512}, outputsite="site1_highpass", prepend="")
proj.refresh()
proj.printInfo()

viewTime(
    proj,
    "2012-02-10 11:05:00",
    "2012-02-10 11:05:03",
    sites=["site1", "site1_highpass"],
    chans=["Hx", "Hy", "Hz"],
    show=True,
    save=True,
)

preProcess(proj, sites="site1", filter={"bpfilt": [100, 2000]}, outputsite="site1_bandpass", prepend="")
proj.refresh()
proj.printInfo()

viewTime(
    proj,
    "2012-02-10 11:05:00",
    "2012-02-10 11:05:03",
    sites=["site1", "site1_bandpass"],
    chans=["Hx", "Hy", "Hz"],
    show=True,
    save=True,
)

preProcess(proj, sites="site1", notch=[50], outputsite="site1_notch", prepend="")
proj.refresh()
proj.printInfo()

viewTime(
    proj,
    "2012-02-10 11:05:00",
    "2012-02-10 11:05:03",
    sites=["site1", "site1_notch"],
    chans=["Hx", "Hy", "Hz"],
    show=True,
    save=True,
)

preProcess(proj, sites="site1", calibrate=True, outputsite="site1_calibrate", prepend="")
proj.refresh()
proj.printInfo()

viewTime(
    proj,
    "2012-02-10 11:05:00",
    "2012-02-10 11:05:03",
    sites=["site1", "site1_calibrate"],
    chans=["Ex", "Hx",],
    show=True,
    save=True,
)