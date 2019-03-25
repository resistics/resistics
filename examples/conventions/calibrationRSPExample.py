import os
import matplotlib.pyplot as plt
from resistics.ioHandlers.calibrationIO import CalibrationIO
from resistics.ioHandlers.calibrationIO import CalibrationData

fig = plt.figure(figsize=(8, 8))

# read RSP calibration data with chopper off
# these can often be broadband
filepath = os.path.join("calData", "Metronix_Coil-----TYPE-006_BB-ID-000307.RSP")
calIO = CalibrationIO(filepath, "rsp", extend=False)
calData = calIO.read()
print(calData)
calData.view(fig=fig, label="rsp BB")