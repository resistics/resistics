import os
from resistics.project.projectIO import loadProject
from resistics.project.projectTime import preProcess, viewTime

"""Interpolation and Resampling

In some cases and particularly when dealing with multiple different formats, it is necessary to preprocess data. Preprocessing can mean filtering, resampling or interpolating so that sampling is coincident with seconds. For example, ATS data sampling starts on a full second and samples onwards i.e.
sample period 0.2s : 1.0 1.2 1.4 1.6 1.8 2.0 
However, for SPAM data, this is not always the case, so a similar SPAM dataset might actually sample at:
sample period 0.2s : 0.9 1.1 1.3 1.5 1.7 1.9

To avoid causing problems with window time definitions, it is possible to interpolate data onto the second so that all sites will have consistent sample times.

Note that by default pre-processed timeseries data is saved to the same site, but can be given a prepend to the measurement folder name and a postpend. Suppose the measurement folder "measA" of site "site1" is being resampled. 
The input data is in:
project -> timeData -> site -> measA
The resampled output data will be in
project -> timeData -> site -> prepend_measA_postpend

Date is written out in the internal data format.

In many cases, the new time data should be moved to a new site. In order to do this, provide the outputsite option to the preProcess function. If the outputsite does not exist, it will be created and the pre processed time series data will be saved to the output site, i.e.

project -> timeData -> outputsite -> prepend_measA_postpend
"""

# need the project path for loading
# let's use our configuration file
projectPath = os.path.join("exampleProject")
projData = loadProject(projectPath)

# let's notch the data and save it to a new time file
preProcess(projData, sampleFreqs=[128], notch=[50], postpend="notchCal")
# refresh the project to pick up the new data file
projData.refresh()
# and let's look at the data
viewTime(projData, "2012-02-11 01:00:00", "2012-02-11 01:02:00", sites=["site1"], sampleFreqs=[128])

# filter the 4096 data
preProcess(projData, filter={"lpfilt": 400}, sampleFreqs=[4096], postpend="lp1000")
# refresh the project to pick up the new data file
projData.refresh()
# and let's look at the data
viewTime(projData, "2012-02-10 11:10:00", "2012-02-10 11:10:20", sites=["site1"], sampleFreqs=[4096])

# try resampling the data - but this time save the output to a different site
# if the site does not already exist, it will be created
preProcess(projData, sampleFreqs=[128], resample={128: 16}, outputsite="resampleSite", postpend="resamp8")
# refresh the project to pick up the new data file
projData.refresh()
# and let's look at the data
viewTime(projData, "2012-02-11 01:00:00", "2012-02-11 01:02:00", sites=["site1"], sampleFreqs=[128])