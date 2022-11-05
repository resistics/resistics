"""
Quick configuration
^^^^^^^^^^^^^^^^^^^

If no configuration is passed, the quick processing functions in resistics will
use the default configuration. However, it is possible to use a different
configuration if preferred.

The dataset in this example has been provided for use by the SAMTEX consortium.
For more information, please refer to [Jones2009]_. Additional details about the
dataset can be found at https://www.mtnet.info/data/kap03/kap03.html.
"""
from pathlib import Path
import seedir as sd
import resistics.letsgo as letsgo
from resistics.config import Configuration
from resistics.time import InterpolateNans, RemoveMean, Multiply
from resistics.decimate import DecimationSetup
from resistics.window import WindowerTarget
import plotly

# %%
# Define the data path. This is dependent on where the data is stored.
time_data_path = Path("..", "..", "data", "time", "quick", "kap123")
sd.seedir(str(time_data_path), style="emoji")

# %%
# Quick calculation of the transfer function using default parameters.
soln_default = letsgo.quick_tf(time_data_path)
fig = soln_default.tf.plot(
    soln_default.freqs,
    soln_default.components,
    to_plot=["ExHy", "EyHx"],
    x_lim=[1, 5],
    res_lim=[0, 4],
    legend="Default config",
    symbol="circle",
)
fig.update_layout(height=800)
fig

# %%
# Looking at the transfer function, it's clear that the phases are in the wrong
# quadrants. A new time process can be added to correct this by multiplying the
# electric channels by -1.
#
# Further, let's use a different windower that will change the window size
# (subject to a minimum) to try and generate a target number of windows. The
# WindowTarget ignores the min_size in the WindowSetup and uses its own. This
# alternative windower will be combined with a modified decimation setup.
config = Configuration(
    name="custom",
    time_processors=[
        InterpolateNans(),
        RemoveMean(),
        Multiply(multiplier={"Ex": -1, "Ey": -1}),
    ],
    dec_setup=DecimationSetup(n_levels=3, per_level=7),
    windower=WindowerTarget(target=2_000, min_size=180),
)
config.summary()

# %%
# Quick calculate the impedance tensor using the new custom configuration and
# plot the result.
soln_custom = letsgo.quick_tf(time_data_path, config)
fig = soln_custom.tf.plot(
    soln_custom.freqs,
    soln_custom.components,
    to_plot=["ExHy", "EyHx"],
    x_lim=[1, 5],
    res_lim=[0, 4],
    phs_lim=[0, 100],
    legend="Custom config",
    symbol="diamond",
)
fig.update_layout(height=800)
plotly.io.show(fig)
