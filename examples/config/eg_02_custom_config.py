"""
Custom configuration
^^^^^^^^^^^^^^^^^^^^

It is possible to customise the configuration and save it for use later.
Configurations can either be customised at initialisation or after
initialisation.

Configurations can be saved as JSON files and later reloaded. This allows users
to keep a library of configurations that can be used depending on the use case
or the survey.
"""
from pathlib import Path
from resistics.config import Configuration
from resistics.time import Add
from resistics.transfunc import TransferFunction

# %%
# Creating a new configuration requires only a name. In this instance, default
# parameters will be used for everything else.
custom_config = Configuration(name="example")
custom_config.summary()

# %%
# However, it is possible to customise more at initialisation time.
custom_config = Configuration(
    name="example",
    time_processors=[Add(add=5)],
    tf=TransferFunction(in_chans=["A", "B"], out_chans=["M", "N"]),
)
custom_config.summary()

# %%
# A configuration can be updated after it has been initialised. For example,
# let's update a windowing parameter. First, have a look at the summary of the
# windowing parameters. Then they can be updated and the summary can be
# inspected again.
custom_config.win_setup.summary()
custom_config.win_setup.min_size = 512
custom_config.win_setup.summary()

# %%
# Configuration information can be saved to JSON files.
save_path = Path("..", "..", "data", "config", "custom_config.json")
with save_path.open("w") as f:
    f.write(custom_config.json())

# %%
# Configurations can also be loaded from JSON files.
reloaded_config = Configuration.parse_file(save_path)
reloaded_config.summary()
