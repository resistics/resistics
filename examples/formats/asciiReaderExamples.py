from datapaths import timePath, timeImages
from resistics.time.writer import TimeWriter

asciiPath = timePath / "ascii"
writer = TimeWriter()
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
from resistics.time.reader_ascii import TimeReaderAscii

asciiReader = TimeReaderAscii(asciiPath)
asciiReader.printInfo()

# get data and view
import matplotlib.pyplot as plt

asciiData = asciiReader.getPhysicalSamples()
asciiData.printInfo()
fig = plt.figure(figsize=(16, 3 * asciiData.numChans))
asciiData.view(fig=fig, label="ASCII format", legend=True)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()
fig.savefig(timeImages / "ascii.png")

# now write out as internal format
from resistics.time.writer_internal import TimeWriterInternal

ascii_2intenrnal = timePath / "asciiInternal"
writer = TimeWriterInternal()
writer.setOutPath(ascii_2intenrnal)
writer.writeDataset(asciiReader, physical=True)

# read in internal format
from resistics.time.reader_internal import TimeReaderInternal

internalReader = TimeReaderInternal(ascii_2intenrnal)
internalReader.printInfo()
internalReader.printComments()
internalData = internalReader.getPhysicalSamples()
internalData.printInfo()

# now plot the two datasets together
fig = plt.figure(figsize=(16, 3 * asciiData.numChans))
asciiData.view(fig=fig, sampleStop=500, label="ASCII format", legend=True)
internalData.view(fig=fig, sampleStop=500, label="Internal format", legend=True)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()
fig.savefig(timeImages / "ascii_vs_internal.png")
