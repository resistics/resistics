import os
from resistics.ioHandlers.dataReaderPhoenix import DataReaderPhoenix

# read in spam data
phoenixPath = os.path.join("timeData", "phoenix")
phoenixReader = DataReaderPhoenix(phoenixPath)
phoenixReader.printDataFileInfo()
phoenixReader.printInfo()

# get some data
startTime = "2011-11-14 02:00:00"
stopTime = "2011-11-14 03:00:00"
unscaledData = phoenixReader.getUnscaledData(startTime, stopTime)
print(unscaledData)

# plot data
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(16, 3 * unscaledData.numChans))
unscaledData.view(fig=fig, sampleEnd=20000)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()
fig.savefig(os.path.join("images", "phoenixUnscaled.png"))

# let's try physical data and view it
physicalData = phoenixReader.getPhysicalData(startTime, stopTime)
physicalData.printInfo()
fig = plt.figure(figsize=(16, 3 * physicalData.numChans))
physicalData.view(fig=fig, sampleEnd=20000)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()
fig.savefig(os.path.join("images", "phoenixPhysical.png"))

# can filter the data
from resistics.utilities.utilsFilter import highPass

filteredData = highPass(physicalData, 4, inplace=False)
fig = plt.figure(figsize=(16, 3 * physicalData.numChans))
filteredData.view(fig=fig, sampleEnd=20000)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()
fig.savefig(os.path.join("images", "phoenixFiltered.png"))

# reformat the continuous sampling frequency
phoenix_2internal = os.path.join("timeData", "phoenixInternal")
phoenixReader.reformatContinuous(phoenix_2internal)

# reading output
from resistics.ioHandlers.dataReaderInternal import DataReaderInternal

internalReader = DataReaderInternal(
    os.path.join(phoenix_2internal, "meas_ts5_2011-11-13-17-04-02_2011-11-14-14-29-46")
)
internalReader.printInfo()
internalReader.printComments()

# read in physical data
internalData = internalReader.getPhysicalData(startTime, stopTime)

# plot the two together
fig = plt.figure(figsize=(16, 3 * physicalData.numChans))
physicalData.view(fig=fig, label="Phoenix format data")
internalData.view(fig=fig, label="Internal format data", legend=True)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()
fig.savefig(os.path.join("images", "phoenix_vs_internal_continuous.png"))
