import os
import matplotlib.pyplot as plt
from resistics.ioHandlers.calibrationIO import CalibrationIO
from resistics.ioHandlers.calibrationIO import CalibrationData

fig = plt.figure(figsize=(8, 8))

# read metronix calibration data with chopper off
filepath = os.path.join("calData", "Hz_MFS06307.TXT")
calIO = CalibrationIO(filepath, "metronix", extend=False)
calData = calIO.read()
print(calData)
calData.view(fig=fig, label="metronix chopper off")

# read metronix calibration data with chopper on
filepath = os.path.join("calData", "Hz_MFS06307.TXT")
calIO.refresh(filepath, "metronix", chopper=True, extend=True)
calData = calIO.read()
print(calData)
calData.view(fig=fig, label="metronix chopper on")
calIO.writeInternalFormat(calData, os.path.join("calData", "IC_307.TXT"))

# read internal format calibration data
filepath = os.path.join("calData", "IC_307.TXT")
calIO.refresh(filepath, "induction", extend=True)
calData = calIO.read()
print(calData)
calData.view(fig=fig, label="induction")

fig.tight_layout()
plt.show()


fig = plt.figure(figsize=(8, 8))

# read RSP calibration data with chopper off
# these can often be broadband
filepath = os.path.join("calData", "Metronix_Coil-----TYPE-006_BB-ID-000307.RSP")
calIO.refresh(filepath, "rsp", extend=False)
calData = calIO.read()
print(calData)
calData.view(fig=fig, label="rsp BB")

# read RSPX calibration data with chopper off
# here can often have HF and LF board
filepath = os.path.join("calData", "Metronix_Coil-----TYPE-006_HF-ID-000133.RSPX")
calIO.refresh(filepath, "rspx", extend=False)
calData = calIO.read()
print(calData)
calData.view(fig=fig, label="rspx HF")

fig.tight_layout()
plt.show()
