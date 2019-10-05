from datapaths import timePath, timeImages
from resistics.time.reader_ats import TimeReaderATS

# read ats data
atsPath = timePath / "ats"
atsReader = TimeReaderATS(atsPath)
atsReader.printInfo()

# get unscaled data
startTime = "2016-02-21 03:00:00"
stopTime = "2016-02-21 04:00:00"
unscaledData = atsReader.getUnscaledData(startTime, stopTime)
unscaledData.printInfo()

# view unscaled data
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(16, 3 * unscaledData.numChans))
unscaledData.view(fig=fig, sampleStop=20000)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()
fig.savefig(timeImages / "ats_unscaledData.png")

# get physical data, which is converted to field units
physicalATSData = atsReader.getPhysicalData(startTime, stopTime)
physicalATSData.printInfo()
fig = plt.figure(figsize=(16, 3 * physicalATSData.numChans))
fig = physicalATSData.view(fig=fig, sampleStop=20000)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()
fig.savefig(timeImages / "ats_physicalData.png")

# all we see is 50Hz and 16Hz noise - apply low pass filter
from resistics.time.filter import lowPass

filteredATSData = lowPass(physicalATSData, 4, inplace=False)
fig = plt.figure(figsize=(16, 3 * filteredATSData.numChans))
fig = filteredATSData.view(fig=fig, sampleStop=20000)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()
fig.savefig(timeImages / "ats_filteredData.png")

# now write out as internal format
from resistics.time.writer_internal import TimeWriterInternal

ats_2intenrnal = timePath / "atsInternal"
writer = TimeWriterInternal()
writer.setOutPath(ats_2intenrnal)
writer.writeDataset(atsReader, physical=True)

# read in internal format
from resistics.time.reader_internal import TimeReaderInternal

internalReader = TimeReaderInternal(ats_2intenrnal)
internalReader.printInfo()
internalReader.printComments()
physicalInternalData = internalReader.getPhysicalData(startTime, stopTime)
physicalInternalData.printInfo()

# now plot the two datasets together
fig = plt.figure(figsize=(16, 3 * physicalATSData.numChans))
physicalATSData.view(fig=fig, sampleStop=200, label="ATS format", legend=True)
physicalInternalData.view(fig=fig, sampleStop=200, label="Internal format", legend=True)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()
fig.savefig(timeImages / "ats_vs_internal.png")

# now write out as ascii format
from resistics.time.writer_ascii import TimeWriterAscii

ats_2ascii = timePath / "atsAscii"
writer = TimeWriterAscii()
writer.setOutPath(ats_2ascii)
writer.writeDataset(atsReader, physical=True)

# read in ascii format
from resistics.time.reader_ascii import TimeReaderAscii

asciiReader = TimeReaderAscii(ats_2ascii)
asciiReader.printInfo()
asciiReader.printComments()
physicalAsciiData = asciiReader.getPhysicalData(startTime, stopTime)
physicalAsciiData.printInfo()

# now plot the two datasets together
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(16, 3 * physicalATSData.numChans))
physicalATSData.view(fig=fig, sampleStop=200, label="ATS format", legend=True)
physicalAsciiData.view(fig=fig, sampleStop=200, label="Ascii format", legend=True)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()
fig.savefig(timeImages / "ats_vs_ascii.png")

