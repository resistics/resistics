import os
from resistics.project.projectIO import loadProject
from resistics.project.projectTime import preProcess, viewTime

"""Resample and interpolate

Look at the project data.
"""

# need the project path for loading
projectPath = os.path.join("exampleProject2")
# load the project
projData = loadProject(projectPath)

# print information about each site
for site in projData.getSites():
    siteData = projData.getSiteData(site)
    siteData.printInfo()
    siteData.view()

projData.view()

# looking at the site information, it is clear that the remote reference is sampled at 250Hz, whereas the sites M1 and M13 have sample rates at 65536Hz, 16384Hz, 4096Hz, 512Hz and 128Hz.
# The RemoteRef data can act as a remote refence for M1 and M13 but needs to be resampled to 128Hz first.
# The other thing to notice here as that the RemoteRef SPAM data does not start on a full second unlike the ATS M1 and M13 site data.
# To avoid issues going forward, it would be best to interpolate the data onto the second.
# Let's save the resampled and interpolated data to a new site
# preProcess(projData, sites=["RemoteRef"], interp=True, resamp={250: 128}, outputsite="RemoteResampled")
# # refresh the project
# projData.refresh()

# print site information for RemoteResampled
# projData.getSiteData("RemoteResampled").printInfo()

# # finally, let's look at the time series data
viewTime(projData, "2016-03-19 00:20:00", "2016-03-19 00:20:10", sites=["RemoteRef", "RemoteResampled"], filter={"lpfilt": 4}, chans=["Ex", "Ey", "Hx", "Hy"], save=True, show=True)
viewTime(projData, "2016-03-22 23:20:00", "2016-03-22 23:20:10", sites=["RemoteRef", "RemoteResampled"], filter={"lpfilt": 4}, chans=["Ex", "Ey", "Hx", "Hy"], save=True, show=True)
