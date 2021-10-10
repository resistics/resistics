"""
Making a project
^^^^^^^^^^^^^^^^

The quick reading functionality of resistics focuses on analysis of single
continuous recordings. When there are multiple recordings at a site or multiple
sites, it can be more convenient to use a resistics project. This is generally
easier to manage and use, especially when doing remote reference or intersite
processing.

The data in this example has been provided for use by the SAMTEX consortium.
For more information, please refer to [Jones2009]_. Additional details about the
data can be found at https://www.mtnet.info/data/kap03/kap03.html.
"""
from pathlib import Path
import seedir as sd
import shutil
import plotly
import resistics.letsgo as letsgo

# %%
# Define the path where the project will be created and any extra project
# metadata. The only required piece of metadata is the reference time but there
# are other optional fields.
project_path = Path("..", "..", "data", "project", "kap03")
project_info = {
    "ref_time": "2003-10-15 00:00:00",
    "year": 2003,
    "country": "South Africa",
}

# %%
# Create the new project and look at the directory structure. There are no data
# files in the project yet, so there is not much to see.
letsgo.new(project_path, project_info)
sd.seedir(str(project_path), style="emoji")

# %%
# Load the project and have a look. When loading a project, a resistics
# environment is returned. This is a combination of a resistics project and a
# configuration.
resenv = letsgo.load(project_path)
resenv.proj.summary()

# %%
# Now let's copy some time series data into the project and look at the
# directory structure. Copy the data does not have to be done using Python and
# users can simply copy and paste the time series data into the time folder
copy_from = Path("..", "..", "data", "time", "kap03")
for site in copy_from.glob("*"):
    shutil.copytree(site, project_path / "time" / site.stem)
sd.seedir(str(project_path), style="emoji")

# %%
# Reload the project and print a new summary.
resenv = letsgo.reload(resenv)
resenv.proj.summary()

# %%
# Finally, plot the project timeline.
fig = resenv.proj.plot()
fig.update_layout(height=700)
plotly.io.show(fig)
