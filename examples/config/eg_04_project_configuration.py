"""
Project configuration
^^^^^^^^^^^^^^^^^^^^^

Alternative configurations can also be used with projects. When using cutom
configurations in the project environment, the name of the configuration is key
as this will determine where any data is saved. The below example shows what
happens when using different configurations with a project.

The dataset in this example has been provided for use by the SAMTEX consortium.
For more information, please refer to [Jones2009]_. Additional details about the
dataset can be found at https://www.mtnet.info/data/kap03/kap03.html.
"""
from pathlib import Path
import seedir as sd
import plotly
from resistics.config import Configuration
import resistics.letsgo as letsgo
from resistics.time import TimeReaderNumpy, InterpolateNans, RemoveMean, Multiply
from resistics.decimate import DecimationSetup
from resistics.window import WindowSetup

# The first thing to do is define the configuration to use.
myconfig = letsgo.Configuration(
    name="myconfig",
    time_readers=[TimeReaderNumpy()],
    time_processors=[
        InterpolateNans(),
        RemoveMean(),
        Multiply(multiplier={"Ex": -1, "Ey": -1}),
    ],
    dec_setup=DecimationSetup(n_levels=7, per_level=3),
    win_setup=WindowSetup(min_size=64, min_olap=16),
)
myconfig.summary()

# %%
# Save the configuration to a file. This is to imitate scenarios where users
# have an existing configuration file that they want to load in and use.
myconfig_path = Path("..", "..", "data", "config", "myconfig.json")
with myconfig_path.open("w") as f:
    f.write(myconfig.json())

# %%
# Let's remind ourselves of the project contents. Note that some
# processing with default parameters has already taken place.
project_path = Path("..", "..", "data", "project", "kap03")
sd.seedir(str(project_path), style="emoji")

# %%
# Now load our configuration and the project with myconfig.
config = Configuration.parse_file(myconfig_path)
resenv = letsgo.load(project_path, config=config)
resenv.config.summary()


# %%
# Now calculate the evaluation frequency spectral data and view the directory
# structure. This shows how resistics handles saving data for different
# configurations. The data is placed in a new folder with the same name as the
# the configuration. This is why the configuration name is important.
site = resenv.proj["kap160"]
for meas in site:
    letsgo.process_time_to_evals(resenv, "kap160", meas.name)
sd.seedir(str(project_path), style="emoji")

# %%
# Let's calculate the impedance tensor with this configuration. The sampling
# frequency to process is 0.2 (Hz)
letsgo.process_evals_to_tf(resenv, 0.2, "kap160")
sd.seedir(str(project_path), style="emoji")

# %%
# Finally, let's plot our the impedance tensor for this configuration
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
    phs_lim=[0, 100],
    legend="128",
    symbol="circle",
)
fig.update_layout(height=900)
plotly.io.show(fig)
