import os
import matplotlib.pyplot as plt
from resistics.ioHandlers.calibrationIO import CalibrationIO
from resistics.ioHandlers.calibrationIO import CalibrationData

fig = plt.figure(figsize=(8, 8))

# read internal format calibration data
filepath = os.path.join("calData", "IC_307.TXT")
calIO = CalibrationIO(filepath, "induction", extend=False)
calData = calIO.read()
print(calData)
calData.view(fig=fig, label="induction")

fig.tight_layout()
plt.show()