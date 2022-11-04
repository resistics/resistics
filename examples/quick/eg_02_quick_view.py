"""
Viewing time data
^^^^^^^^^^^^^^^^^

With the quick viewing functionality, it is possible to view time series data
without having to setup a project or explicitly read the data first. The
quickview decimation option provides an easy way to see the time series at
multiple sampling frequencies (decimated to lower frequencies).

.. warning::

    The time series data is downsampled for viewing using the LTTB algorithm,
    which tries to capture the features of the time series using a given number
    of data points. Setting max_pts to None will try and plot all points which
    can cause serious performance issues for large datasets.

    Those looking to view non downsampled data are advised to use the quick
    reading functionality and then plot specific subsections of data.

The dataset in this example has been provided for use by the SAMTEX consortium.
For more information, please refer to [Jones2009]_. Additional details about the
dataset can be found at https://www.mtnet.info/data/kap03/kap03.html.
"""
# sphinx_gallery_thumbnail_number = 2
from pathlib import Path
import seedir as sd
import resistics.letsgo as letsgo
import plotly


# %%
# Define the data path. This is dependent on where the data is stored.
time_data_path = Path("..", "..", "data", "time", "quick", "kap123")
sd.seedir(str(time_data_path), style="emoji")

# %%
# Quickly view the time series data
fig = letsgo.quick_view(time_data_path, max_pts=1_000)
fig.update_layout(height=700)
plotly.io.show(fig)

# %%
# In many cases, data plotting at its recording frequency can be quite nosiy.
# The quickview function has the option to plot multiple decimation levels
# so the data can be compared at multiple sampling frequencies.
fig = letsgo.quick_view(time_data_path, max_pts=1_000, decimate=True)
fig.update_layout(height=700)
plotly.io.show(fig)
