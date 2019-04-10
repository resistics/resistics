import os
from resistics.ioHandlers.calibrationIO import CalibrationIO
from resistics.ioHandlers.calibrationIO import CalibrationData

# read RSPX calibration data
filepath = os.path.join("calData", "Metronix_Coil-----TYPE-006_HF-ID-000133.RSPX")
calIO = CalibrationIO(filepath, "rspx", extend=False)
calData = calIO.read()
calData.printInfo()

# plot
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 8))
calData.view(fig=fig, label="RSPX calibration", legend=True)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()
fig.savefig(os.path.join("images", "calibrationRSPX.png"))

# plot
fig = plt.figure(figsize=(8, 8))
calData.view(fig=fig, label="RSPX calibration", degrees=True, legend=True)
calData.view(
    fig=fig, staticgain=False, degrees=True, label="RSPX calibration", legend=True
)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()
fig.savefig(os.path.join("images", "calibrationRSPX_staticGainAndDegrees.png"))

# this time read in with extend
calIO.refresh(filepath, "rspx", extend=True)
calDataExtended = calIO.read()
calDataExtended.printInfo()

# plot these two together
fig = plt.figure(figsize=(8, 8))
calDataExtended.view(fig=fig, degrees=True, label="RSPX calibration", legend=True)
calData.view(fig=fig, label="RSPX calibration", degrees=True, legend=True)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()
fig.savefig(os.path.join("images", "calibrationRSPX_extended.png"))

# write as the ASCII format
rspx2ascii = os.path.join("calData", "rspx2ascii.TXT")
calIO.writeInternalFormat(calData, rspx2ascii)

# can read this again
calIO.refresh(rspx2ascii, "induction", chopper=False, extend=False)
calDataAscii = calIO.read()
calDataAscii.printInfo()

# plot together
fig = plt.figure(figsize=(8, 8))
calData.view(fig=fig, label="RSPX calibration", legend=True)
calDataAscii.view(fig=fig, label="ASCII calibration", legend=True)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()
fig.savefig(os.path.join("images", "calibrationRSPXvsASCII.png"))
