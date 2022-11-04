"""
Getting spectra data
^^^^^^^^^^^^^^^^^^^^

It can often be useful to have a look at the spectral content of time data. The
quick functions make it easy to get the spectra data of a single time series
recording.

Note that spectra data are calculated after decimation and spectra data objects
include data for multiple decimation levels.

The dataset in this example has been provided for use by the SAMTEX consortium.
For more information, please refer to [Jones2009]_. Additional details about the
dataset can be found at https://www.mtnet.info/data/kap03/kap03.html.
"""
# sphinx_gallery_thumbnail_number = 3
from pathlib import Path
import seedir as sd
import resistics.letsgo as letsgo
import plotly

# %%
# Define the data path. This is dependent on where the data is stored.
time_data_path = Path("..", "..", "data", "time", "quick", "kap123")
sd.seedir(str(time_data_path), style="emoji")

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
fig = spec_data.plot_level_stack(level=0, grouping="3D")
fig.update_layout(height=900)
plotly.io.show(fig)


# %%
# It is also possible to plot spectra heatmaps for a decimation level. Here, the
# sphinx_gallery_defer_figures
fig = spec_data.plot_level_section(level=0, grouping="6H")
fig.update_layout(height=900)
plotly.io.show(fig)
