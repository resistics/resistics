from pathlib import Path
from resistics.project.projectIO import loadProject
from resistics.project.projectTime import preProcess, viewTime

projectPath = Path("remoteProject")
proj = loadProject(projectPath)
proj.printInfo()
# get information about the local site
siteLocal = proj.getSiteData("M6")
siteLocal.printInfo()
# get information about the SPAM data remote reference
siteRemote = proj.getSiteData("RemoteSPAM")
siteRemote.printInfo()

# want to resample the RemoteSPAM data to 128 Hz
# SPAM data want to resample onto the second
# to make sure the times are matching, we can loop through the 128Hz measurements of M6 and use those times
meas128 = siteLocal.getMeasurements(128)
for meas in meas128:
    start = siteLocal.getMeasurementStart(meas).strftime("%Y-%m-%d %H:%M:%S")
    stop = siteLocal.getMeasurementEnd(meas).strftime("%Y-%m-%d %H:%M:%S")
    postpend = siteLocal.getMeasurementStart(meas).strftime("%Y-%m-%d_%H-%M-%S")
    print("Processing data from {} to {}".format(start, stop))
    preProcess(proj, sites="RemoteSPAM", start=start, stop=stop, interp=True, resamp={250: 128}, outputsite="Remote", prepend="", postpend=postpend)

viewTime(proj, sites=["M6", "Remote"], startDate="2016-02-17 03:20:00", endDate="2016-02-17 03:35:00", chans=["Hx", "Hy"], save=True)

viewTime(proj, sites=["M6", "Remote"], startDate="2016-02-17 03:20:00", endDate="2016-02-17 03:35:00", filter={"lpfilt": 4}, chans=["Hx", "Hy"], save=True)