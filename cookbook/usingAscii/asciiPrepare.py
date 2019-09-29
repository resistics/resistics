from configuration import projectPath
from resistics.ioHandlers.dataWriter import DataWriter

asciiPath = projectPath / "timeData" / "site1" / "meas"
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