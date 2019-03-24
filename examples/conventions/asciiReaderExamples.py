import os
from resistics.ioHandlers.dataWriter import DataWriter

asciiPath = os.path.join("timeData", "ascii")
writer = DataWriter()
writer.setOutPath(asciiPath)
chan2FileMap = {
    "Ex": "exmuVm.ascii",
    "Ey": "eymuVm.ascii",
    "Hx": "bxnT.ascii",
    "Hy": "bynT.ascii",
    "Hz": "bznT.ascii",
}
startDate = "2018-01-01 12:00:00"
writer.writeTemplateHeaderFiles(
    ["Ex", "Ey", "Hx", "Hy", "Hz"], chan2FileMap, 0.5, 430000, startDate
)

# read in ascii format
from resistics.ioHandlers.dataReaderAscii import DataReaderAscii

asciiReader = DataReaderAscii(asciiPath)
asciiReader.printInfo()

# get data and view
import matplotlib.pyplot as plt

asciiData = asciiReader.getPhysicalSamples()
asciiData.printInfo()
fig = plt.figure(figsize=(20, 2 * asciiData.numChans))
asciiData.view(fig=fig, label="ASCII format", legend=True)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()
fig.savefig(os.path.join("images", "ascii.png"))

# now write out as internal format
from resistics.ioHandlers.dataWriterInternal import DataWriterInternal

ascii_2intenrnal = os.path.join("timeData", "asciiInternal")
writer = DataWriterInternal()
writer.setOutPath(ascii_2intenrnal)
writer.writeDataset(asciiReader, physical=True)

# read in internal format
from resistics.ioHandlers.dataReaderInternal import DataReaderInternal

internalReader = DataReaderInternal(ascii_2intenrnal)
internalReader.printInfo()
internalReader.printComments()
internalData = internalReader.getPhysicalSamples()
internalData.printInfo()

# now plot the two datasets together
fig = plt.figure(figsize=(20, 2 * asciiData.numChans))
asciiData.view(fig=fig, sampleStop=500, label="ASCII format", legend=True)
internalData.view(fig=fig, sampleStop=500, label="Internal format", legend=True)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()
fig.savefig(os.path.join("images", "ascii_vs_internal.png"))
