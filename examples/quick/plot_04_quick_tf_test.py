"""
Transfer functions test
^^^^^^^^^^^^^^^^^^^^^^^

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
from resistics.time import InterpolateNans, Multiply
from resistics.decimate import DecimationSetup
from resistics.window import WindowerTarget
from resistics.transfunc import ImpedanceTensor

# from resistics.regression import SolverScikitRANSAC


# %%
# Define the data path. This is dependent on where the data is stored. Here, the
# data path is being read from an environment variable.
load_dotenv()
data_path = Path(os.getenv("EXAMPLES_DATA_PATH"))
time_data_path = data_path / "time_numpy_kap157"

# %%
# Set some basic configuration parameters before calculating the impedance
# tensor. This example is using JSON format calibration files, numpy time
# data and a non-standard windower. The windower chooses window sizes to create
# a target number of windows. The windower is not advised for project based
# processing, especially when needing to align windows across sites.
#
# Note that configurations can be saved to JSON files and simply loaded in later
# as required.
config = Configuration(
    name="example",
    time_processors=[InterpolateNans(), Multiply(multiplier={"Ex": -1, "Ey": -1})],
    dec_setup=DecimationSetup(n_levels=7, per_level=3),
    windower=WindowerTarget(target=1_000),
    tf=ImpedanceTensor(),
    # solver=SolverScikitRANSAC()
)
config.summary()

# %%
# Confirm the transfer function that is being calculated
config.tf.summary()
print(config.tf)

# %%
# Now calculate the transfer function, in this case the impedance tensor
soln = letsgo.quick_tf(time_data_path, config)
fig = soln.tf.plot(
    soln.freqs,
    soln.components,
    to_plot=["ExHy", "EyHx"],
    x_lim=[0, 5],
    res_lim=[1, 4],
    legend="128",
    symbol="circle",
)
fig.update_layout(height=800)
plotly.io.show(fig)
