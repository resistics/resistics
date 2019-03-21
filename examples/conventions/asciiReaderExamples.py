import os
from resistics.ioHandlers.dataWriterAscii import DataWriterAscii

asciiPath = os.path.join("timeData", "ascii")
writer = DataWriterAscii()
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
asciiReader.printComments()
asciiData = asciiReader.getPhysicalSamples()
asciiData.printInfo()
asciiData.view()

