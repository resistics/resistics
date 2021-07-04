"""
Getting spectra data
^^^^^^^^^^^^^^^^^^^^

It can often be useful to have a look at the spectral content of time data. The
quick functions make it easy to get the spectra data of a single time series
recording.

Note that spectra data are calculated after decimation and spectra data objects
include data for multiple decimation levels.
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
# Get the spectra data.
spec_data = letsgo.quick_spectra(time_data_path)

# %%
# Once the spectra data has been calculated, it can be plotted in a variety of
# ways. The default plotting function plots the spectral data for multiple
# decimation levels.
fig = spec_data.plot()
fig.update_layout(height=900)
plotly.io.show(fig)

# %%
# It is also possible to plot spectra data for a particular decimation level.
# In the below example, an optional grouping is being used to stack spectra data
# for the decimation level into certain time groups
fig = spec_data.plot_level_stack(level=0, grouping="1T")
fig.update_layout(height=900)
plotly.io.show(fig)


# %%
# It is also possible to plot spectra heatmaps for a decimation level.
fig = spec_data.plot_level_section(level=0, grouping="5S")
fig.update_layout(height=900)
plotly.io.show(fig)
