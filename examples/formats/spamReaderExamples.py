from datapaths import timePath, timeImages
from resistics.time.reader_spam import TimeReaderSPAM

# read in spam data
spamPath = timePath / "spam"
spamReader = TimeReaderSPAM(spamPath)
spamReader.printInfo()

# get physical data from SPAM
import matplotlib.pyplot as plt

startTime = "2016-02-07 02:10:00"
stopTime = "2016-02-07 02:30:00"
physicalSPAMData = spamReader.getPhysicalData(startTime, stopTime, remnans=True)
physicalSPAMData.printInfo()
fig = plt.figure(figsize=(16, 3 * physicalSPAMData.numChans))
physicalSPAMData.view(fig=fig, sampleStop=2000)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()
fig.savefig(timeImages / "spam.png")

# write out as the internal format
from resistics.time.writer_internal import TimeWriterInternal

spam_2internal = timePath / "spamInternal"
writer = TimeWriterInternal()
writer.setOutPath(spam_2internal)
writer.writeDataset(spamReader, physical=True)

# read in the internal format dataset and see what's in the comments
from resistics.time.reader_internal import TimeReaderInternal

internalReader = TimeReaderInternal(spam_2internal)
internalReader.printInfo()
internalReader.printComments()
physicalInternalData = internalReader.getPhysicalData(startTime, stopTime)
physicalInternalData.printInfo()

# now plot the two datasets together
fig = plt.figure(figsize=(16, 3 * physicalSPAMData.numChans))
physicalSPAMData.view(fig=fig, sampleStop=500, label="SPAM format")
physicalInternalData.view(fig=fig, sampleStop=500, label="Internal format", legend=True)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()
fig.savefig(timeImages / "spam_vs_internal.png")

# all we see is 50Hz and 16Hz noise - apply a band pass filter
from resistics.time.filter import bandPass

filteredSPAMData = bandPass(physicalSPAMData, 0.2, 16, inplace=False)
filteredSPAMData.printInfo()

# write out a filtered data - this is a subset of the data
spam_2filteredSubset = timePath / "spamInternalFiltered"
writer.setOutPath(spam_2filteredSubset)
chanHeaders, chanMap = spamReader.getChanHeaders()
writer.writeData(spamReader.getHeaders(), chanHeaders, filteredSPAMData, physical=True)

# let's try reading in again
internalReaderFiltered = TimeReaderInternal(spam_2filteredSubset)
internalReaderFiltered.printInfo()
internalReaderFiltered.printComments()

# get the internal formatted filtered data
filteredInternalData = internalReaderFiltered.getPhysicalSamples()
filteredInternalData.printInfo()

# plot this against the original
fig = plt.figure(figsize=(16, 3 * physicalSPAMData.numChans))
filteredSPAMData.view(fig=fig, sampleStop=5000, label="filtered SPAM format")
filteredInternalData.view(
    fig=fig, sampleStop=5000, label="filtered internal format", legend=True
)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()
fig.savefig(timeImages / "spam_vs_internal_filtered.png")
