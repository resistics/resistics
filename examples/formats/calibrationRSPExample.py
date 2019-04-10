import os
from resistics.ioHandlers.calibrationIO import CalibrationIO
from resistics.ioHandlers.calibrationIO import CalibrationData

# read RSP calibration data - this is a broadband calibraition - no chopper on or off
filepath = os.path.join("calData", "Metronix_Coil-----TYPE-006_BB-ID-000307.RSP")
calIO = CalibrationIO(filepath, "rsp", extend=False)
calData = calIO.read()
calData.printInfo()

# plot
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 8))
calData.view(fig=fig, label="RSP calibration", legend=True)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()
fig.savefig(os.path.join("images", "calibrationRSP.png"))

# plot
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 8))
calData.view(fig=fig, label="RSP calibration", degrees=True, legend=True)
calData.view(
    fig=fig, staticgain=False, degrees=True, label="RSP no static gain", legend=True
)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()
fig.savefig(os.path.join("images", "calibrationRSP_staticGainAndDegrees.png"))

# write as the ASCII format
rsp2ascii = os.path.join("calData", "rsp2ascii.TXT")
calIO.writeInternalFormat(calData, rsp2ascii)

# can read this again
calIO.refresh(rsp2ascii, "induction", extend=False)
calDataAscii = calIO.read()
calDataAscii.printInfo()

# plot together
fig = plt.figure(figsize=(8, 8))
calData.view(fig=fig, label="RSP calibration", legend=True)
calDataAscii.view(fig=fig, label="ASCII calibration", legend=True)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()
fig.savefig(os.path.join("images", "calibrationRSPvsASCII.png"))
