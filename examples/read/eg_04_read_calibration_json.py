"""
Calibration data JSON
^^^^^^^^^^^^^^^^^^^^^

The preferred format for calibration data is JSON file. Howeer, they are not
always as easy to handwrite, so it is possible to use txt/ASCII calibration
files too.
"""
from pathlib import Path
import json
from resistics.time import ChanMetadata
from resistics.calibrate import SensorCalibrationJSON
import plotly

# %%
# Define the calibration data path. This is dependent on where the data is
# stored.
cal_data_path = Path("..", "..", "data", "calibration", "example.json")

# %%
# Inspect the contents of the calibration file
with cal_data_path.open("r") as f:
    file_contents = json.load(f)
print(json.dumps(file_contents, indent=4, sort_keys=True))

# %%
# Read the data using the appropriate calibration data reader. As calibration
# data can be dependent on certain sensor parameters, channel metadata needs
# to be passed to the method.
chan_metadata = ChanMetadata(name="Hx", chan_type="magnetic", data_files=[])
cal_data = SensorCalibrationJSON().read_calibration_data(cal_data_path, chan_metadata)

# %%
# Plot the calibration data.
fig = cal_data.plot(color="maroon")
fig.update_layout(height=700)
plotly.io.show(fig)
