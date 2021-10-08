"""
Time data ASCII2
^^^^^^^^^^^^^^^^

This example will show how to read time data from an ASCII file.

.. note::

    The dataset in this example has provided for use by SAMTEX consortium
    Area selection for diamonds using magnetotellurics: Examples from southern Africa.
    Lithos, 112S, 83-92, doi: 10.1016/j.lithos.2009.06.011
    Jones et al. (2009):

    The dataset is KAP103
    Values of 1E32 have been replaced by NaN
"""
from dotenv import load_dotenv
import os
from pathlib import Path
import plotly
import pandas as pd
from resistics.time import ChanMetadata, TimeMetadata, TimeReaderAscii, InterpolateNans

# %%
# Define the data path. This is dependent on where the data is stored. Here, the
# data path is being read from an environment variable.
load_dotenv()
data_path = Path(os.getenv("EXAMPLES_DATA_PATH"))
time_data_path = data_path / "time_ascii2"
ascii_file_name = "kap163as.ts"

# %%
# The folder contains a single data file. Let's have a look at the contents of
# the file.
with (time_data_path / ascii_file_name).open("r") as f:
    for line_number, line in enumerate(f):
        print(line.strip("\n"))
        if line_number >= 130:
            break

# %%
# To read the data using resistics a metadata file is required. The metadata
# file has information about the recording and channels. Let's construct the
# metadata and save it along with the kap103as.ts file.
#
# Note that the metadata requires the number of samples. When counting the lines
# in a file, this could be incorrect if there are blank lines at the end. Pandas
# can be useful for this purpose. For this example, the number of samples is
# known.
chans = ["Hx", "Hy", "Hz", "Ex", "Ey"]
n_samples = 461_094  # kap103
# n_samples = 449_819 # kap106 good
# n_samples = 429_642 # kap152
# n_samples = 414_498 # kap172 good
# n_samples = 342_425 # kap125 good
# n_samples = 463_807 # kap163 good
fs = 0.2
first_time = pd.Timestamp("2003-11-08 16:30:00")  # kap103
# first_time = pd.Timestamp("2003-11-08 11:00:00") # kap106 good
# first_time = pd.Timestamp("2003-11-03 16:00:00") # kap152
# first_time = pd.Timestamp("2003-10-30 13:00:00") # kap172 good
# first_time = pd.Timestamp("2003-11-11 14:30:00") # kap125 good
# first_time = pd.Timestamp("2003-10-28 15:30:00") # kap163 good
last_time = first_time + (n_samples - 1) * pd.Timedelta(1 / fs, "s")

chans_metadata = {}
for chan in chans:
    chan_type = "electric" if chan in ["Ex", "Ey"] else "magnetic"
    chans_metadata[chan] = ChanMetadata(
        name=chan, chan_type=chan_type, data_files=[ascii_file_name]
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
# reader = TimeReaderAscii(extension=".ts", n_header=121) # 106, 125
reader = TimeReaderAscii(extension=".ts", n_header=122)  # 172 152 163
time_data = reader.run(time_data_path)
print(time_data.data[:, 0])

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
