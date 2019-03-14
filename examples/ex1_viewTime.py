import os
from resistics.project.projectIO import loadProject

# load the project
projectPath = os.path.join("exampleProject")
projData = loadProject(projectPath)

# load the viewing method
from resistics.project.projectTime import viewTime
# view data between certain date range
viewTime(projData, "2012-02-11 01:00:00", "2012-02-11 01:10:00")

# explicitly define sites and channels to plot
viewTime(
    projData,
    "2012-02-11 01:00:00",
    "2012-02-11 01:02:00",
    sites=["site1"],
    chans=["Ex", "Hy"],
)

# calibrate magnetic channels
viewTime(
    projData,
    "2012-02-11 01:00:00",
    "2012-02-11 01:02:00",
    sites=["site1"],
    chans=["Ex", "Hy"], 
    calibrate=True,
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
