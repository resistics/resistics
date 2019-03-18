import os
from resistics.ioHandlers.dataReaderATS import DataReaderATS
# read ats data
atsPath = os.path.join("testData", "ats")
atsReader = DataReaderATS(atsPath)
atsReader.printInfo()

# get unscaled data
startTime = "2016-02-21 03:00:00"
stopTime = "2016-02-21 04:00:00"
unscaledData = atsReader.getUnscaledData(startTime, stopTime)
unscaledData.printInfo()
# view unscaled data
fig = unscaledData.view(sampleStop=20000)
fig.savefig(os.path.join("images", "ats_unscaledData.png"))

# get physical data, which is converted to field units 
physicalData = atsReader.getPhysicalData(startTime, stopTime)
physicalData.printInfo()
fig = physicalData.view(sampleStop=20000)
fig.savefig(os.path.join("images", "ats_physicalData.png"))

# all we see is 50Hz and 16Hz noise - apply low pass filter
from resistics.utilities.utilsFilter import lowPass
filteredData = lowPass(physicalData, 4, inplace=False)
fig = filteredData.view(sampleStop=20000)
fig.savefig(os.path.join("images", "ats_filteredData.png"))

# now write out as internal format
from resistics.ioHandlers.dataWriterInternal import DataWriterInternal
ats_2intenrnal = os.path.join("testData", "atsInternal")
writer = DataWriterInternal()
writer.setOutPath(ats_2intenrnal)
writer.writeDataset(atsReader)

# now write out as ascii format
from resistics.ioHandlers.dataWriterAscii import DataWriterAscii
ats_2ascii = os.path.join("testData", "atsAscii")
writer = DataWriterAscii()
writer.setOutPath(ats_2ascii)
writer.writeDataset(atsReader)




