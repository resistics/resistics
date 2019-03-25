import os
import matplotlib.pyplot as plt
from resistics.ioHandlers.calibrationIO import CalibrationIO
from resistics.ioHandlers.calibrationIO import CalibrationData

fig = plt.figure(figsize=(8, 8))

# read RSPX calibration data with chopper off
# here can often have HF and LF board
filepath = os.path.join("calData", "Metronix_Coil-----TYPE-006_HF-ID-000133.RSPX")
calIO = CalibrationIO(filepath, "rspx", extend=False)
calData = calIO.read()
print(calData)
calData.view(fig=fig, label="rspx HF")

fig.tight_layout()
plt.show()
