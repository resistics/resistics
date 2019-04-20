import os
from resistics.project.projectIO import loadProject

# load the project
projectPath = os.path.join("tutorialProject")
projData = loadProject(projectPath)

# load the viewing method
from resistics.project.projectTime import viewTime

# view data between certain date range
viewTime(projData, "2012-02-11 01:00:00", "2012-02-11 01:10:00", save=True)

# explicitly define sites and channels to plot
viewTime(
    projData,
    "2012-02-11 01:00:00",
    "2012-02-11 01:02:00",
    sites=["site1"],
    chans=["Ex", "Hy"],
    save=True,
)

# calibrate magnetic channels
viewTime(
    projData,
    "2012-02-11 01:00:00",
    "2012-02-11 01:02:00",
    sites=["site1"],
    chans=["Ex", "Hy"],
    calibrate=True,
    save=True,
)

# low pass filter
viewTime(
    projData,
    "2012-02-11 01:00:00",
    "2012-02-11 01:02:00",
    sites=["site1"],
    chans=["Ex", "Hy"],
    calibrate=True,
    filter={"lpfilt": 0.5},
    save=True,
)

# high pass filter
viewTime(
    projData,
    "2012-02-11 01:00:00",
    "2012-02-11 01:02:00",
    sites=["site1"],
    chans=["Ex", "Hy"],
    calibrate=True,
    filter={"hpfilt": 10},
    save=True,
)

# band pass filter
viewTime(
    projData,
    "2012-02-11 01:00:00",
    "2012-02-11 01:02:00",
    sites=["site1"],
    chans=["Ex", "Hy"],
    calibrate=True,
    filter={"bpfilt": [1, 10]},
    save=True,
)

# notch
viewTime(
    projData,
    "2012-02-11 01:00:00",
    "2012-02-11 01:02:00",
    sites=["site1"],
    chans=["Ex", "Hy"],
    calibrate=True,
    notch=[16.6667, 50],
    save=True,
)

# normalise
viewTime(
    projData,
    "2012-02-11 01:00:00",
    "2012-02-11 01:02:00",
    sites=["site1"],
    chans=["Ex", "Hy"],
    calibrate=True,
    normalise=True,
    save=True,
)

# save with band pass filter
viewTime(
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
