"""
Calibration data JSON
^^^^^^^^^^^^^^^^^^^^^

An alternative to text calibration files is to use a JSON calibration file.
"""
from dotenv import load_dotenv
import os
from pathlib import Path
import json
import plotly
from resistics.time import ChanMetadata
from resistics.calibrate import SensorCalibrationJSON

# %%
# Define the calibration data path. This is dependent on where the data is
# stored. Here, the data path is being read from an environment variable.
load_dotenv()
data_path = Path(os.getenv("EXAMPLES_DATA_PATH"))
cal_data_path = data_path / "calibration_json" / "example.json"

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
