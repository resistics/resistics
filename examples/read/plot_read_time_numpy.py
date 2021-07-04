"""
Time data numpy
^^^^^^^^^^^^^^^

Numpy formatted time series data is the default time data format written out by
resistics. For this reason, it usually comes with metadata. If it does not,
refer to reading ascii data for an example on how to make metadata.
"""
from dotenv import load_dotenv
import os
from pathlib import Path
import plotly
from resistics.time import TimeReaderNumpy, RemoveMean, LowPass

# %%
# Define the data path. This is dependent on where the data is stored. Here, the
# data path is being read from an environment variable.
load_dotenv()
data_path = Path(os.getenv("EXAMPLES_DATA_PATH"))
time_data_path = data_path / "time_numpy"

# %%
# Read the numpy formatted time data using the appropriate time data reader.
time_data = TimeReaderNumpy().run(time_data_path)
time_data.metadata.summary()

# %%
# Next, plot the data. By default, the data is downsampled using lttb so that it
# is possible to plot the full timeseries. Before plotting, remove the mean and
# apply a low pass filter
time_data = RemoveMean().run(time_data)
time_data = LowPass(cutoff=4).run(time_data)
fig = time_data.plot(max_pts=1_000)
fig.update_layout(height=700)
plotly.io.show(fig)
