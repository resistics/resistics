"""
Time data ASCII
^^^^^^^^^^^^^^^

This example will show how to read time data from an ASCII file
"""
from dotenv import load_dotenv
import os
from pathlib import Path
import plotly
import pandas as pd
from resistics.time import ChanMetadata, TimeMetadata, TimeReaderAscii

# %%
# Define the data path. This is dependent on where the data is stored. Here, the
# data path is being read from an environment variable.
load_dotenv()
data_path = Path(os.getenv("EXAMPLES_DATA_PATH"))
time_data_path = data_path / "time_ascii"

# %%
# In the folder there is a single data.txt file. Let's have a look at the first
# few lines of the data.txt file.
list(x.name for x in time_data_path.glob("*"))
with (time_data_path / "data.txt").open("r") as f:
    for line_number, line in enumerate(f):
        print(line.strip("\n"))
        if line_number >= 6:
            break

# %%
# Apart from the single header line, the rest of the file is data. To read the
# data using resistics a metadata file is required. The metadata file has
# information about the recording and channels. Let's construct the
# metadata and save it along with the data.txt file.
#
# Note that the metadata requires the number of samples. When counting the lines
# in a file, this could be incorrect if there are blank lines at the end. Pandas
# can be useful for this purpose. For this example, the number of samples is
# known.
chans = ["Ex", "Ey", "Hx", "Hy", "Hz"]
n_samples = 20_001
fs = 128
first_time = pd.Timestamp("2020-01-01 00:00:00")
last_time = first_time + (n_samples - 1) * pd.Timedelta(1 / fs, "s")

chans_metadata = {}
for chan in chans:
    chan_type = "electric" if chan in ["Ex", "Ey"] else "magnetic"
    chans_metadata[chan] = ChanMetadata(
        name=chan, chan_type=chan_type, data_files=["data.txt"]
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
# first sample values.
time_data = TimeReaderAscii(delimiter="\t", n_header=1).run(time_data_path)
print(time_data.data[:, 0])

# %%
# Finally plot the data. By default, the data is downsampled using the LTTB
# algorithm to avoid slow and large plots.
fig = time_data.plot(max_pts=1_000)
fig.update_layout(height=700)
plotly.io.show(fig)
