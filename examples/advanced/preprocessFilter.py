from datapaths import preprocessPath, preprocessImages
from resistics.project.io import loadProject

proj = loadProject(preprocessPath)
proj.printInfo()

from resistics.common.plot import plotOptionsTime, getPresentationFonts
plotOptions = plotOptionsTime(plotfonts=getPresentationFonts())

# resample to 1024 Hz and save in new site
from resistics.project.time import preProcess, viewTime

preProcess(
    proj, sites="site1", filter={"lpfilt": 32}, outputsite="site1_lowpass", prepend=""
)
proj.refresh()
proj.printInfo()

fig = viewTime(
    proj,
    "2012-02-10 11:05:00",
    "2012-02-10 11:05:03",
    sites=["site1", "site1_lowpass"],
    chans=["Hx", "Hy", "Hz"],
    show=False,
    plotoptions=plotOptions,
)
fig.savefig(preprocessImages / "viewTimeLowpass.png")

preProcess(
    proj, sites="site1", filter={"hpfilt": 512}, outputsite="site1_highpass", prepend=""
)
proj.refresh()
proj.printInfo()

fig = viewTime(
    proj,
    "2012-02-10 11:05:00",
    "2012-02-10 11:05:03",
    sites=["site1", "site1_highpass"],
    chans=["Hx", "Hy", "Hz"],
    show=False,
    plotoptions=plotOptions,
)
fig.savefig(preprocessImages / "viewTimeHighpass.png")

preProcess(
    proj,
    sites="site1",
    filter={"bpfilt": [100, 2000]},
    outputsite="site1_bandpass",
    prepend="",
)
proj.refresh()
proj.printInfo()

fig = viewTime(
    proj,
    "2012-02-10 11:05:00",
    "2012-02-10 11:05:03",
    sites=["site1", "site1_bandpass"],
    chans=["Hx", "Hy", "Hz"],
    show=False,
    plotoptions=plotOptions,
)
fig.savefig(preprocessImages / "viewTimeBandpass.png")

preProcess(proj, sites="site1", notch=[50], outputsite="site1_notch", prepend="")
proj.refresh()
proj.printInfo()

fig = viewTime(
    proj,
    "2012-02-10 11:05:00",
    "2012-02-10 11:05:03",
    sites=["site1", "site1_notch"],
    chans=["Hx", "Hy", "Hz"],
    show=False,
    plotoptions=plotOptions,
)
fig.savefig(preprocessImages / "viewTimeNotch.png")

preProcess(
    proj, sites="site1", calibrate=True, outputsite="site1_calibrate", prepend=""
)
proj.refresh()
proj.printInfo()

fig = viewTime(
    proj,
    "2012-02-10 11:05:00",
    "2012-02-10 11:05:03",
    sites=["site1", "site1_calibrate"],
    chans=["Ex", "Hx"],
    show=False,
    plotoptions=plotOptions,
)
fig.savefig(preprocessImages / "viewTimeCalibrate.png")