from datapaths import timePath, timeImages
from resistics.time.reader_spam import TimeReaderSPAM

# read in spam data
spamPath = timePath / "spam"
spamReader = TimeReaderSPAM(spamPath)
spamReader.printInfo()
spamData = spamReader.getPhysicalSamples()
spamData.printInfo()

# interpolate to second
from resistics.time.interp import interpolateToSecond

interpData = interpolateToSecond(spamData, inplace=False)
interpData.printInfo()

# can now write out the interpolated dataset
from resistics.time.writer_internal import TimeWriterInternal

interpPath = timePath / "spamInterp"
headers = spamReader.getHeaders()
chanHeaders, chanMap = spamReader.getChanHeaders()
writer = TimeWriterInternal()
writer.setOutPath(interpPath)
writer.writeData(
    headers,
    chanHeaders,
    interpData,
    physical=True,
)
writer.printInfo()

# read in the internal data
from resistics.time.reader_internal import TimeReaderInternal

interpReader = TimeReaderInternal(interpPath)
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
fig.savefig(timeImages / "interpolation.png")