from datapaths import calPath, calImages
from resistics.calibrate.io import CalibrationIO

# read metronix calibration data with chopper off
filepath = calPath / "Hz_MFS06307.TXT"
calIO = CalibrationIO(filepath, "metronix", chopper=False, extend=False)
calDataChopperOff = calIO.read()
calDataChopperOff.printInfo()

# plot
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 8))
calDataChopperOff.view(fig=fig, label="Chopper off", legend=True)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()
fig.savefig(calImages / "calibrationMetronixChopperOff.png")

# extend True
calIO.refresh(filepath, "metronix", chopper=False, extend=True)
calDataExtend = calIO.read()
calDataExtend.printInfo()

# plot the two together
fig = plt.figure(figsize=(8, 8))
calDataExtend.view(fig=fig, label="Chopper off, Extend True", legend=True)
calDataChopperOff.view(fig=fig, label="Chopper off", legend=True)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()
fig.savefig(calImages / "calibrationMetronixExtend.png")

# read in the chopper on
calIO.refresh(filepath, "metronix", chopper=True, extend=False)
calDataChopperOn = calIO.read()
calDataChopperOn.printInfo()

# plot the three together
fig = plt.figure(figsize=(8, 8))
calDataChopperOff.view(fig=fig, label="Chopper off", legend=True)
calDataChopperOn.view(fig=fig, label="Chopper on", legend=True)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()
fig.savefig(calImages / "calibrationMetronixChopperOn.png")

# write as the ASCII format
calIO.writeInternalFormat(calDataChopperOn, calPath / "metronix2ascii.TXT")
