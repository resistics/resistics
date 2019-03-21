import os
from resistics.ioHandlers.dataReaderSpam import DataReaderSPAM
# read in spam data
spamPath = os.path.join("timeData", "spam")    
spamReader = DataReaderSPAM(spamPath)
spamReader.printInfo()

# write out as the internal format
from resistics.ioHandlers.dataWriterInternal import DataWriterInternal
spam_2internal = os.path.join("timeData", "spamInternal")
writer = DataWriterInternal()
writer.setOutPath(spam_2internal)
writer.writeDataset(spamReader, physical=True)

# get physical data from SPAM
import matplotlib.pyplot as plt
startTime = "2016-02-07 02:10:00"
stopTime = "2016-02-07 02:30:00"
physicalSPAMData = spamReader.getPhysicalData(startTime, stopTime, remnans=True)
physicalSPAMData.printInfo()
fig = plt.figure(figsize=(20, 2*physicalSPAMData.numChans))
physicalSPAMData.view(fig = fig, sampleStop=2000)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()
fig.savefig(os.path.join("images", "spam.png"))

# read in the internal format dataset and see what's in the comments
from resistics.ioHandlers.dataReaderInternal import DataReaderInternal
internalReader = DataReaderInternal(spam_2internal)
internalReader.printInfo()
internalReader.printComments()
physicalInternalData = internalReader.getPhysicalData(startTime, stopTime)
physicalInternalData.printInfo()

# now plot the two datasets together
fig = plt.figure(figsize=(20, 2*physicalSPAMData.numChans))
physicalSPAMData.view(fig = fig, sampleStop=500, label="SPAM format")
physicalInternalData.view(fig = fig, sampleStop=500, label="Internal format", legend=True)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()
fig.savefig(os.path.join("images", "spam_vs_internal.png"))

# all we see is 50Hz and 16Hz noise - apply a band pass filter
from resistics.utilities.utilsFilter import bandPass, lowPass
filteredSPAMData = bandPass(physicalSPAMData, 0.2, 16, inplace=False)
filteredSPAMData.printInfo()

# write out a filtered data - this is a subset of the data
spam_2filteredSubset = os.path.join("timeData", "spamInternalFiltered")
writer.setOutPath(spam_2filteredSubset)
chanHeaders, chanMap = spamReader.getChanHeaders()
writer.writeData(spamReader.getHeaders(), chanHeaders, filteredSPAMData, physical=True)

# let's try reading in again
internalReaderFiltered = DataReaderInternal(spam_2filteredSubset)
internalReaderFiltered.printInfo()
internalReaderFiltered.printComments()

# get the internal formatted filtered data
filteredInternalData = internalReaderFiltered.getPhysicalSamples()
filteredInternalData.printInfo()

# plot this against the original
fig = plt.figure(figsize=(20, 2*physicalSPAMData.numChans))
filteredSPAMData.view(fig = fig, sampleStop=5000, label="filtered SPAM format")
filteredInternalData.view(fig = fig, sampleStop=5000, label="filtered internal format", legend=True)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()
fig.savefig(os.path.join("images", "spam_vs_internal_filtered.png"))
