import os
from resistics.ioHandlers.dataReaderInternal import DataReaderInternal
from resistics.ioHandlers.dataReaderPhoenix import DataReaderPhoenix
from resistics.ioHandlers.dataWriterInternal import DataWriterInternal
from resistics.utilities.utilsFilter import lowPass

# read in spam data
phoenixPath = os.path.join("testData", "phoenix")
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
filteredData = lowPass(physicalData, 4, inplace=False)
filteredData.view(sampleEnd=20000)

# only reformat the continuous for now
phoenix_2internal = os.path.join("testData", "phoenixInternal")
phoenixReader.reformatContinuous(phoenix_2internal)

# reformat the higher frequency recordings - this will produce many data folders
# phoenixReader.reformatHigh(phoenix_2internal, ts=[4])
