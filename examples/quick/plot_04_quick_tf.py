"""
Transfer functions
^^^^^^^^^^^^^^^^^^

When doing field work, it can be useful to quickly estimate the transfer
function from a single continuous recording.

This example shows estimation of the ImpedanceTensor for a single 128 Hz
measurement and a single 4096 Hz measurement.
"""
from dotenv import load_dotenv
import os
from pathlib import Path
import plotly
import resistics.letsgo as letsgo
from resistics.config import Configuration
from resistics.calibrate import SensorCalibrator, SensorCalibrationJSON
from resistics.decimate import DecimationSetup
from resistics.window import WindowerTarget
from resistics.transfunc import ImpedanceTensor


# %%
# Define the data path. This is dependent on where the data is stored. Here, the
# data path is being read from an environment variable.
load_dotenv()
data_path = Path(os.getenv("EXAMPLES_DATA_PATH"))
time_data_128_path = data_path / "quick_tf" / "fs128"
time_data_4096_path = data_path / "quick_tf" / "fs4096"
calibration_data_path = data_path / "quick_tf"

# %%
# To help us setup the configuration, let's having a look at the calibration
# file naming. For this example, they follow the naming scheme
# 'serial{serial#}.json'
print(list(x.name for x in calibration_data_path.glob("*.json")))

# %%
# Set some basic configuration parameters before calculating the impedance
# tensor. This example is using JSON format calibration files, numpy time
# data and a non-standard windower. The windower chooses window sizes to create
# a target number of windows. The windower is not advised for project based
# processing, especially when needing to align windows across sites.
#
# Note that configurations can be saved to JSON files and simply loaded in later
# as required.
calibrator = SensorCalibrator(
    readers=[SensorCalibrationJSON(file_str="serial$serial$extension")]
)
config = Configuration(
    name="example",
    dec_setup=DecimationSetup(n_levels=6, per_level=5),
    sensor_calibrator=calibrator,
    windower=WindowerTarget(target=5_000),
    tf=ImpedanceTensor(),
)
config.summary()

# %%
# Confirm the transfer function that is being calculated
config.tf.summary()
print(config.tf)

# %%
# Now calculate the transfer function, in this case the impedance tensor
soln128 = letsgo.quick_tf(time_data_128_path, config, calibration_data_path)
soln4096 = letsgo.quick_tf(time_data_4096_path, config, calibration_data_path)
fig = soln128.tf.plot(
    soln128.freqs,
    soln128.components,
    to_plot=["ExHy", "EyHx"],
    x_lim=[-4, 4],
    res_lim=[-1, 3],
    legend="128",
    symbol="circle",
)
fig = soln4096.tf.plot(
    soln4096.freqs,
    soln4096.components,
    fig=fig,
    to_plot=["ExHy", "EyHx"],
    legend="4096",
    symbol="cross",
)
fig.update_layout(height=800)
plotly.io.show(fig)
