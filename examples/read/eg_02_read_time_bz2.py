"""
Time data bz2
^^^^^^^^^^^^^

This example will show how to read time data from a compressed ASCII file using
the default ASCII data reader. In this case, the data has been compressed using
bz2.

To read such a compressed ASCII data file, a metadata file is required. The
example shows how an appropriate metadata file can be created and the
information required to create such a metadata file.

The dataset in this example has been provided for use by the SAMTEX consortium.
For more information, please refer to [Jones2009]_. Additional details about the
dataset can be found at https://www.mtnet.info/data/kap03/kap03.html.

The dataset is KAP148. A couple of notes:

- The data has a sample every 5 seconds, meaning a 0.2 Hz sampling frequency.
- Values of 1E32 have been replaced by NaN
"""
from pathlib import Path
import bz2
import plotly
import pandas as pd
from resistics.time import ChanMetadata, TimeMetadata, TimeReaderAscii, InterpolateNans

# %%
# Define the data path. This is dependent on where the data is stored. Here, the
# data path is being read from an environment variable.
time_data_path = Path("..", "..", "data", "time", "bz2")
ascii_data_path = time_data_path / "kap148as.ts.bz2"

# %%
# The folder contains a single ascii data file. Let's have a look at the
# contents of the file.
with bz2.open(ascii_data_path, "rt") as f:
    for line_number, line in enumerate(f):
        print(line.strip("\n"))
        if line_number >= 130:
            break

# %%
# Note that the metadata requires the number of samples. Pandas can be useful
# for this purpose.
df = pd.read_csv(ascii_data_path, header=None, skiprows=123, delim_whitespace=True)
n_samples = len(df.index)
print(df)

# %%
# Define other key pieces of recording information
fs = 0.2
chans = ["Hx", "Hy", "Hz", "Ex", "Ey"]
first_time = pd.Timestamp("2003-10-25 11:30:00")
last_time = first_time + (n_samples - 1) * pd.Timedelta(1 / fs, "s")

# %%
# The next step is to create a TimeMetadata object. The TimeMetdata has
# information about the recording and channels. Let's construct the
# TimeMetadata and save it as a JSON along with the time series data file.
chans_metadata = {}
for chan in chans:
    chan_type = "electric" if chan in ["Ex", "Ey"] else "magnetic"
    chans_metadata[chan] = ChanMetadata(
        name=chan, chan_type=chan_type, data_files=[ascii_data_path.name]
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
# Now the data is ready to be read in by resistics. Read it in and print the
# first and last sample values.
reader = TimeReaderAscii(extension=".bz2", n_header=123)
time_data = reader.run(time_data_path)
print(time_data.data[:, 0])
print(time_data.data[:, -1])

# %%
# There are some invalid values in the data that have been replaced by NaN
# values. Interpolate the NaN values.
time_data = InterpolateNans().run(time_data)

# %%
# Finally plot the data. By default, the data is downsampled using the LTTB
# algorithm to avoid slow and large plots.
fig = time_data.plot(max_pts=1_000)
fig.update_layout(height=700)
plotly.io.show(fig)
