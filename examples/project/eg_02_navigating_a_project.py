"""
Navigating a project
^^^^^^^^^^^^^^^^^^^^

After creating a project and copying in the time data into the time folder, it
is useful to be able to navigate a project. This example shows the various types
of objects available in resistics that can help navigate a project and access
data.

The data in this example has been provided for use by the SAMTEX consortium.
For more information, please refer to [Jones2009]_. Additional details about the
data can be found at https://www.mtnet.info/data/kap03/kap03.html.
"""
# sphinx_gallery_thumbnail_number = 2
from pathlib import Path
import seedir as sd
import resistics.letsgo as letsgo

# %%
# Let's remind ourselves of the project contents, load the project and have a
# look at its contents.
project_path = Path("..", "..", "data", "project", "kap03")
sd.seedir(str(project_path), style="emoji")
resenv = letsgo.load(project_path)

# %%
# Project summaries can be quite verbose. Instead, let's convert it to a pandas
# DataFrame and see the information in tabular form.
print(resenv.proj.to_dataframe())

# %%
# The project has three sites, each with a single recording. Another way to look
# at the sites in the project is to make a list of them.
sites = [site.name for site in resenv.proj]
print(sites)

# %%
# To get more information about a single site, get the corresponding Site
# object.
site = resenv.proj["kap160"]
print(type(site))

# %%
# Like most objects in resistics, the Site object has a summary method, which
# prints out a comprehensive summary of the site.
site.summary()

# %%
# Sometimes, it can be more convenient to access the information from the Site
# object directly.
print(site.name)
print(site.begin_time)
print(site.end_time)
measurements = [meas.name for meas in site]
print(measurements)

# %%
# It's also possible to plot the timeline of a single site.
fig = site.plot()
fig

# %%
# There is only a single measurement in this site named "meas01". Let's get its
# Measurement object.
meas = site["meas01"]
print(type(meas))

# %%
# Unsurprisingly, Measurement objects also have a summary method.
meas.summary()

# %%
# Measurement objects only hold metadata to avoid loading in lots of data when
# projects are loaded. However, it is possible to read the data from the
# measurement.
time_data = meas.reader.run(meas.dir_path, metadata=meas.metadata)
time_data.summary()

# %%
# Let's plot the time data.
fig = time_data.plot()
fig.update_layout(height=700)
fig
