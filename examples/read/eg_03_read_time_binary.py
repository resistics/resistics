"""
Time data binary
^^^^^^^^^^^^^^^^

If a data file is available in npy binary format, this can be read in using the
TimeReaderNumpy reader as long as a metadata file can be made.

Information about the recording will be required to make the metadata file. In
the below example, a metadata file is made and then the data is read.

The dataset in this example has been provided for use by the SAMTEX consortium.
For more information, please refer to [Jones2009]_. Additional details about the
dataset can be found at https://www.mtnet.info/data/kap03/kap03.html.

The dataset is KAP130. A couple of notes:

- The data has a sample every 5 seconds, meaning a 0.2 Hz sampling frequency.
- Values of 1E32 have been replaced by NaN
"""
from pathlib import Path
import numpy as np
import pandas as pd
from resistics.time import TimeMetadata, ChanMetadata, TimeReaderNumpy
from resistics.time import InterpolateNans, LowPass

# %%
# Define the data path. This is dependent on where the data is stored. Here, the
# data path is being read from an environment variable.
time_data_path = Path("..", "..", "data", "time", "binary")
binary_data_path = time_data_path / "kap130as.npy"


# %%
# Define key pieces of recording information. This is known.
fs = 0.2
chans = ["Hx", "Hy", "Hz", "Ex", "Ey"]
first_time = pd.Timestamp("2003-10-17 15:30:00")

# %%
# Note that the metadata requires the number of samples. This can be found by
# loading the data in memory mapped mode. In most cases, it is likely that this
# be known.
data = np.load(binary_data_path, mmap_mode="r")
n_samples = data.shape[1]
last_time = first_time + (n_samples - 1) * pd.Timedelta(1 / fs, "s")

# %%
# The next step is to create a TimeMetadata object. The TimeMetdata has
# information about the recording and channels. Let's construct the
# TimeMetadata and save it as a JSON along with the time series data file.
chans_metadata = {}
for chan in chans:
    chan_type = "electric" if chan in ["Ex", "Ey"] else "magnetic"
    chans_metadata[chan] = ChanMetadata(
        name=chan, chan_type=chan_type, data_files=[binary_data_path.name]
    )
time_metadata = TimeMetadata(
    fs=fs,
    chans=chans,
    n_samples=n_samples,
    first_time=first_time,
    last_time=last_time,
    chans_metadata=chans_metadata,
)
time_metadata.summary()
time_metadata.write(time_data_path / "metadata.json")


# %%
# Read the numpy formatted time data using the appropriate time data reader.
time_data = TimeReaderNumpy().run(time_data_path)
time_data.metadata.summary()

# %%
# Next remove any NaN values and plot the data. By default, the data is
# downsampled using lttb so that it is possible to plot the full timeseries. A
# second plot will be added with the same data filtered with a (1/(24*3600)) Hz
# or 1 day period low pass filter.
time_data = InterpolateNans().run(time_data)
fig = time_data.plot(max_pts=1_000, legend="original")
filtered_data = LowPass(cutoff=1 / (24 * 3_600)).run(time_data)
fig = filtered_data.plot(
    max_pts=1_000, fig=fig, chans=chans, legend="filtered", color="red"
)
fig.update_layout(height=700)
fig
