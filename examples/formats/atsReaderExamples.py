import os
from resistics.ioHandlers.dataReaderATS import DataReaderATS

# read ats data
atsPath = os.path.join("timeData", "ats")
atsReader = DataReaderATS(atsPath)
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
fig.savefig(os.path.join("images", "ats_unscaledData.png"))

# get physical data, which is converted to field units
physicalATSData = atsReader.getPhysicalData(startTime, stopTime)
physicalATSData.printInfo()
fig = plt.figure(figsize=(16, 3 * physicalATSData.numChans))
fig = physicalATSData.view(fig=fig, sampleStop=20000)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()
fig.savefig(os.path.join("images", "ats_physicalData.png"))

# all we see is 50Hz and 16Hz noise - apply low pass filter
from resistics.utilities.utilsFilter import lowPass

filteredATSData = lowPass(physicalATSData, 4, inplace=False)
fig = plt.figure(figsize=(16, 3 * filteredATSData.numChans))
fig = filteredATSData.view(fig=fig, sampleStop=20000)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()
fig.savefig(os.path.join("images", "ats_filteredData.png"))

# now write out as internal format
from resistics.ioHandlers.dataWriterInternal import DataWriterInternal

ats_2intenrnal = os.path.join("timeData", "atsInternal")
writer = DataWriterInternal()
writer.setOutPath(ats_2intenrnal)
writer.writeDataset(atsReader, physical=True)

# read in internal format
from resistics.ioHandlers.dataReaderInternal import DataReaderInternal

internalReader = DataReaderInternal(ats_2intenrnal)
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
fig.savefig(os.path.join("images", "ats_vs_internal.png"))

# now write out as ascii format
from resistics.ioHandlers.dataWriterAscii import DataWriterAscii

ats_2ascii = os.path.join("timeData", "atsAscii")
writer = DataWriterAscii()
writer.setOutPath(ats_2ascii)
writer.writeDataset(atsReader, physical=True)

# read in ascii format
from resistics.ioHandlers.dataReaderAscii import DataReaderAscii

asciiReader = DataReaderAscii(ats_2ascii)
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
fig.savefig(os.path.join("images", "ats_vs_ascii.png"))

