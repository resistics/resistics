import os
from resistics.ioHandlers.dataReaderSpam import DataReaderSPAM
# read in spam data
spamPath = os.path.join("testData", "spam")    
spamReader = DataReaderSPAM(spamPath)
spamReader.printInfo()

# get unscaled data and view
startTime = "2016-02-07 02:10:00"
stopTime = "2016-02-07 03:10:00"
unscaledData = spamReader.getUnscaledData(startTime, stopTime)
unscaledData.printInfo()
fig = unscaledData.view(sampleStop=20000)
fig.savefig(os.path.join("images", "spam_unscaledData.png"))

# get physical data
physicalData = spamReader.getPhysicalData(startTime, stopTime)
physicalData.printInfo()
fig = physicalData.view(sampleStop=20000)
fig.savefig(os.path.join("images", "spam_physicalData.png"))

# all we see is 50Hz and 16Hz noise - apply a band pass filter
from resistics.utilities.utilsFilter import bandPass
filteredData = bandPass(physicalData, 0.5, 4, inplace=False)
fig = filteredData.view(sampleStop=20000)
fig.savefig(os.path.join("images", "spam_filteredData.png"))

# plot filtered data and physical data on the same plot
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(16,8))
physicalData.view(fig = fig, sampleStop = 20000, label="Original")
filteredData.view(fig = fig, sampleStop = 20000, label="Filtered", legend=True)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()
fig.savefig(os.path.join("images", "spam_jointData.png"))

# write out as the internal format
from resistics.ioHandlers.dataWriterInternal import DataWriterInternal
spam_2internal = os.path.join("testData", "spamInternal")
writer = DataWriterInternal()
writer.setOutPath(spam_2internal)
writer.writeDataset(spamReader)

# read in again and see what's in the comments
from resistics.ioHandlers.dataReaderInternal import DataReaderInternal
reader = DataReaderInternal(spam_2internal)
reader.printInfo()
reader.printComments()

# write out a smaller subset of data
spam_2internalSubset = os.path.join("testData", "spamInternalSubset")
writer.setOutPath(spam_2internalSubset)
physicalData.printInfo()
chanHeaders, chanMap = spamReader.getChanHeaders()
writer.writeData(spamReader.getHeaders(), chanHeaders, physicalData)

# let's try reading in again
reader = DataReaderInternal(spam_2internalSubset)
reader.printInfo()
reader.printComments()

# plot this against the original
internalSubset = reader.getUnscaledSamples()
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(16,8))
physicalData.view(fig = fig, sampleStop = 20000)
internalSubset.view(fig = fig, sampleStop = 20000)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()
fig.savefig(os.path.join("images", "spam_vsInternal.png"))

# write out the filtered subset of data
spam_2filteredSubset = os.path.join("testData", "spamFilteredSubset")
writer.setOutPath(spam_2filteredSubset)
filteredData.printInfo()
chanHeaders, chanMap = spamReader.getChanHeaders()
writer.writeData(spamReader.getHeaders(), chanHeaders, filteredData)

# let's try reading in again
reader = DataReaderInternal(spam_2filteredSubset)
reader.printInfo()
reader.printComments()

