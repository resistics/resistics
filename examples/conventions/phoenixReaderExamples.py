import os
from resistics.ioHandlers.dataReaderPhoenix import DataReaderPhoenix
# read in spam data
phoenixPath = os.path.join("timeData", "phoenix")
phoenixReader = DataReaderPhoenix(phoenixPath)
phoenixReader.printInfo()

# get some data
startTime = "2011-11-14 02:00:00"
stopTime = "2011-11-14 03:00:00"
unscaledData = phoenixReader.getUnscaledData(startTime, stopTime)
unscaledData.printInfo()
unscaledData.view(sampleEnd=20000)

# let's try physical data
physicalData = phoenixReader.getPhysicalData(startTime, stopTime)
physicalData.printInfo()
physicalData.view(sampleEnd=20000)

# all we see is 50Hz and 16Hz noise - apply low pass filter
from resistics.utilities.utilsFilter import highPass
filteredData = highPass(physicalData, 4, inplace=False)
filteredData.view(sampleEnd=20000)

# only reformat the continuous for now
phoenix_2internal = os.path.join("timeData", "phoenixInternal")
phoenixReader.reformatContinuous(phoenix_2internal)

# reading output
from resistics.ioHandlers.dataReaderInternal import DataReaderInternal
internalReader = DataReaderInternal(os.path.join(phoenix_2internal, "meas_ts5_2011-11-13-17-04-02_2011-11-14-14-29-46"))
internalReader.printInfo()
internalReader.printComments()
internalData = internalReader.getPhysicalData(startTime, stopTime)

# plot this against the original
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(20, 2*physicalData.numChans))
physicalData.view(fig = fig, label="Phoenix format data")
internalData.view(fig = fig, label="Internal format data", legend=True)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()
fig.savefig(os.path.join("images", "phoenix_vs_internal_continuous.png"))