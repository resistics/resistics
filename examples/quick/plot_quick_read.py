"""
Reading time data
^^^^^^^^^^^^^^^^^

Resistics can quickly read a single continuous recording using the quick reading
functionality. This can be useful for inspecting the metadata and having a quick
look at the data.
"""
from dotenv import load_dotenv
import os
from pathlib import Path
import plotly
import resistics.letsgo as letsgo

# %%
# Define the data path. This is dependent on where the data is stored. Here, the
# data path is being read from an environment variable.
load_dotenv()
data_path = Path(os.getenv("EXAMPLES_DATA_PATH"))
time_data_path = data_path / "time_numpy"

# %%
# Quickly read the time series data and inspect the metadata
time_data = letsgo.quick_read(time_data_path)
time_data.metadata.summary()

# %%
# Take a subsection of the data and inspect the metadata for the subsection
time_data_sub = time_data.subsection("2012-02-10 11:32:45", "2012-02-10 11:33:15")
time_data_sub.metadata.summary()

# %%
# Plot the full time data with downsampling and a subsection without any
# downsampling. Comparing the downsampled and original data, there is clearly
# some loss but the LTTB downsampled data does a reasonable job capaturing the
# main features whilst showing a greater amount of data.
fig = time_data.plot(max_pts=1_000)
fig = time_data_sub.plot(
    fig, chans=time_data.metadata.chans, color="red", legend="Subsection", max_pts=None
)
fig.update_layout(height=700)
plotly.io.show(fig)
