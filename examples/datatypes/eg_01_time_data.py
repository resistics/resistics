"""
Time data
^^^^^^^^^

Time data objects store time series data, usually from a single recording at a
site. The data array has the shape: number of chans x number of samples.

The order of channels in the data array matches the order of the channels in
the time data chans attribute.
"""
from pathlib import Path
import resistics.letsgo as letsgo

# %%
# Define the data path. This is dependent on where the data is stored.
time_data_path = Path("..", "..", "data", "time", "quick", "kap123")

# %%
# Let's inspect information that will tell us about the shape of the data array
time_data = letsgo.quick_read(time_data_path)
print(f"{time_data.metadata.chans=}")
print(f"{time_data.metadata.n_chans=}")
print(f"{time_data.metadata.n_samples=}")


# %%
# Look at the shape of the data array
print(time_data.data.shape)

# %%
# The order of the data in the array matches that of the channels list
print(time_data["Ex"][0:5])
ex_index = time_data.metadata.chans.index("Ex")
print(time_data.data[ex_index][0:5])
