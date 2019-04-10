import os
from resistics.ioHandlers.calibrationIO import CalibrationIO
from resistics.ioHandlers.calibrationIO import CalibrationData

# we can write a template into which to paste our data
writepath = os.path.join("calData", "ascii.txt")
calIO = CalibrationIO()
calIO.writeInternalTemplate(writepath, 307, "MFS06", 1)

# read back the internal template file
filepath = os.path.join("calData", "asciiWithData.txt")
calIO.refresh(filepath, "induction", extend=False)
calData = calIO.read()
calData.printInfo()

# plot
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 8))
calData.view(fig=fig, label="Internal ASCII format", legend=True)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()
fig.savefig(os.path.join("images", "calibrationASCII.png"))