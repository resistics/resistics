"""
Reading time data
^^^^^^^^^^^^^^^^^

Resistics can quickly read a single continuous recording using the quick reading
functionality. This can be useful for inspecting the metadata and having a
look at the data when in the field.

Further details about the data can be found in [Jones2009]_.
"""
from pathlib import Path
import seedir as sd
import plotly
import resistics.letsgo as letsgo

# %%
# Define the data path. This is dependent on where the data is stored.
time_data_path = Path("..", "..", "data", "time", "quick", "kap123")
sd.seedir(str(time_data_path), style="emoji")

# %%
# Quickly read the time series data and inspect the metadata
time_data = letsgo.quick_read(time_data_path)
time_data.metadata.summary()

# %%
# Take a subsection of the data and inspect the metadata for the subsection
time_data_sub = time_data.subsection("2003-11-20 12:00:00", "2003-11-21 00:00:00")
time_data_sub.metadata.summary()

# %%
# Plot the full time data with LTTB downsampling and a subsection without any
# downsampling. Comparing the downsampled and original data, there is clearly
# some loss but the LTTB downsampled data does a reasonable job capaturing the
# main features whilst showing a greater amount of data.
fig = time_data.plot(max_pts=1_000)
fig = time_data_sub.plot(
    fig, chans=time_data.metadata.chans, color="red", legend="Subsection", max_pts=None
)
fig.update_layout(height=700)
plotly.io.show(fig)
