"""
Transfer functions
^^^^^^^^^^^^^^^^^^

When doing field work, it can be useful to quickly estimate the transfer
function from a single continuous recording. This example shows estimation of
the transfer function using all default settings. The default transfer function
is the impedance tensor and this will be calculated. Later, the data will be
re-processed using an alternative configuration.

The dataset in this example has been provided for use by the SAMTEX consortium.
For more information, please refer to [Jones2009]_. Additional details about the
dataset can be found at https://www.mtnet.info/data/kap03/kap03.html.
"""
from pathlib import Path
import seedir as sd
import resistics.letsgo as letsgo
import plotly


# %%
# Define the data path. This is dependent on where the data is stored.
time_data_path = Path("..", "..", "data", "time", "quick", "kap123")
sd.seedir(str(time_data_path), style="emoji")

# %%
# Now calculate the transfer function, in this case the impedance tensor
soln = letsgo.quick_tf(time_data_path)
fig = soln.tf.plot(
    soln.freqs,
    soln.components,
    to_plot=["ExHy", "EyHx"],
    x_lim=[1, 5],
    res_lim=[0, 4],
    legend="128",
    symbol="circle",
)
fig.update_layout(height=900)
plotly.io.show(fig)
