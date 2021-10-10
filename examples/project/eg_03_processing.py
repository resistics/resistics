"""
Processing a project
^^^^^^^^^^^^^^^^^^^^

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
import plotly
import resistics.letsgo as letsgo

# %%
# Let's remind ourselves of the project contents and then load the project.
project_path = Path("..", "..", "data", "project", "kap03")
sd.seedir(str(project_path), style="emoji")
resenv = letsgo.load(project_path)

# %%
# Inspect the current configuration. As no custom configuration has been
# specified, this will be the default configuration.
resenv.config.summary()

# %%
# And it's always useful to know what transfer function will be calculated out.
print(resenv.config.tf)

# %%
# Now let's run single site processing on a site and then look at the directory
# structure again. Begin by transforming to frequency domain and reducing to the
# evaluation frequencies. Note that whilst there is only a single measurement
# for this site, the below is written to work when there are more measurements.
site = resenv.proj["kap160"]
for meas in site:
    letsgo.process_time_to_evals(resenv, "kap160", meas.name)
sd.seedir(str(project_path), style="emoji")

# %%
# Now let's run single site processing on a site and then look at the directory
# structure again. To run the transfer function calculation, the sampling
# frequency to process needs to be specified. In this case, it's 0.2 Hz.
letsgo.process_evals_to_tf(resenv, 0.2, "kap160")
sd.seedir(str(project_path), style="emoji")

# %%
# Get the transfer function
soln = letsgo.get_solution(
    resenv,
    "kap160",
    resenv.config.name,
    0.2,
    resenv.config.tf.name,
    resenv.config.tf.variation,
)
fig = soln.tf.plot(
    soln.freqs,
    soln.components,
    to_plot=["ExHy", "EyHx"],
    x_lim=[1, 5],
    res_lim=[1, 4],
    legend="128",
    symbol="circle",
)
fig.update_layout(height=900)
plotly.io.show(fig)
