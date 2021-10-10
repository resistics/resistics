"""
Customising configuration
^^^^^^^^^^^^^^^^^^^^^^^^^

It is possible to customise the configuration and save it for use later.
"""
from resistics.config import Configuration
from resistics.time import Add
from resistics.transfunc import TransferFunction

# %%
# Creating a new configuration requires only a name. In this instance, default
# parameters will be used for everything else.
config = Configuration(name="example")
config.summary()

# %%
# However, it is possible to customise more at initialisation time.
config = Configuration(
    name="example",
    time_processors=[Add(add=5)],
    tf=TransferFunction(in_chans=["A", "B"], out_chans=["M", "N"]),
)
config.summary()

# %%
# A configuration can be updated after it has been initialised
config.win_setup.min_size = 512
config.summary()


# %%
# Configuration can be saved to JSON files

# %%
# Configurations can also be loaded from JSON files
