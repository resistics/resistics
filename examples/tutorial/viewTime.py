from datapaths import projectPath, imagePath
from resistics.project.io import loadProject

# load the project
projData = loadProject(projectPath)

# load the viewing method
from resistics.project.time import viewTime

# view data between certain date range
fig = viewTime(
    projData, "2012-02-11 01:00:00", "2012-02-11 01:10:00", show=False, save=False
)
fig.savefig(imagePath / "viewTime_projtime_view")

# explicitly define sites and channels to plot
fig = viewTime(
    projData,
    "2012-02-11 01:00:00",
    "2012-02-11 01:02:00",
    sites=["site1"],
    chans=["Ex", "Hy"],
    show=False,
    save=False,
)
fig.savefig(imagePath / "viewTime_projtime_view_chans")

# calibrate magnetic channels
fig = viewTime(
    projData,
    "2012-02-11 01:00:00",
    "2012-02-11 01:02:00",
    sites=["site1"],
    chans=["Ex", "Hy"],
    calibrate=True,
    show=False,
    save=False,
)
fig.savefig(imagePath / "viewTime_projtime_view_calibrate")

# low pass filter
fig = viewTime(
    projData,
    "2012-02-11 01:00:00",
    "2012-02-11 01:02:00",
    sites=["site1"],
    chans=["Ex", "Hy"],
    calibrate=True,
    filter={"lpfilt": 0.5},
    show=False,
    save=False,
)
fig.savefig(imagePath / "viewTime_projtime_view_calibrate_lpfilt")

# high pass filter
fig = viewTime(
    projData,
    "2012-02-11 01:00:00",
    "2012-02-11 01:02:00",
    sites=["site1"],
    chans=["Ex", "Hy"],
    calibrate=True,
    filter={"hpfilt": 10},
    show=False,
    save=False,
)
fig.savefig(imagePath / "viewTime_projtime_view_calibrate_hpfilt")

# band pass filter
fig = viewTime(
    projData,
    "2012-02-11 01:00:00",
    "2012-02-11 01:02:00",
    sites=["site1"],
    chans=["Ex", "Hy"],
    calibrate=True,
    filter={"bpfilt": [1, 10]},
    show=False,
    save=False,
)
fig.savefig(imagePath / "viewTime_projtime_view_calibrate_bpfilt")

# notch
fig = viewTime(
    projData,
    "2012-02-11 01:00:00",
    "2012-02-11 01:02:00",
    sites=["site1"],
    chans=["Ex", "Hy"],
    calibrate=True,
    notch=[16.6667, 50],
    show=False,
    save=False,
)
fig.savefig(imagePath / "viewTime_projtime_view_calibrate_notch")

# normalise
fig = viewTime(
    projData,
    "2012-02-11 01:00:00",
    "2012-02-11 01:02:00",
    sites=["site1"],
    chans=["Ex", "Hy"],
    calibrate=True,
    normalise=True,
    show=False,
    save=False,
)
fig.savefig(imagePath / "viewTime_projtime_view_calibrate_normalise")

# save with band pass filter
fig = viewTime(
    projData,
    "2012-02-11 01:00:00",
    "2012-02-11 01:02:00",
    sites=["site1"],
    chans=["Ex", "Hy"],
    calibrate=True,
    filter={"bpfilt": [1, 8]},
    save=True,
    show=False,
)
fig.savefig(imagePath / "viewTime_projtime_view_calibrate_bpfilt_save")
