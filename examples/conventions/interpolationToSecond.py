import os
from resistics.ioHandlers.dataReaderSpam import DataReaderSPAM

# read in spam data
spamPath = os.path.join("timeData", "spam")
spamReader = DataReaderSPAM(spamPath)
spamReader.printInfo()
spamData = spamReader.getPhysicalSamples()
spamData.printInfo()

# interpolate to second
from resistics.utilities.utilsInterp import interpolateToSecond

interpData = interpolateToSecond(spamData, inplace=False)
interpData.printInfo()

# can now write out the interpolated dataset
from resistics.ioHandlers.dataWriterInternal import DataWriterInternal

interpPath = os.path.join("timeData", "spamInterp")
headers = spamReader.getHeaders()
chanHeaders, chanMap = spamReader.getChanHeaders()
writer = DataWriterInternal()
writer.setOutPath(interpPath)
writer.writeData(
    headers,
    chanHeaders,
    interpData,
    physical=True,
)
writer.printInfo()

# read in the internal data
from resistics.ioHandlers.dataReaderInternal import DataReaderInternal

interpReader = DataReaderInternal(interpPath)
interpReader.printInfo()
interpReader.printComments()

# get data between a time range
startTime = "2016-02-07 02:10:00"
stopTime = "2016-02-07 02:30:00"
spamData = spamReader.getPhysicalData(startTime, stopTime)
interpData = interpReader.getPhysicalData(startTime, stopTime)

# plot the datasets
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 8))
plt.plot(spamData.getDateArray()[0:100], spamData.data["Ex"][0:100], "o--")
plt.plot(interpData.getDateArray()[0:100], interpData.data["Ex"][0:100], "x:")
plt.legend(["Original", "Interpolated"], loc=2)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()
fig.savefig(os.path.join("images", "interpolation.png"))