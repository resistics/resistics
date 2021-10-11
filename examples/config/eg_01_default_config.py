"""
Default configuration
^^^^^^^^^^^^^^^^^^^^^

This example shows the default resistics configuration. The configuration
defines the processing sequence and parameterisation that will be used to
process the data.
"""
from resistics.config import get_default_configuration

# %%
# Get the default configuration and print the summary.
default_config = get_default_configuration()
default_config.summary()

# %%
# By default, the configuration includes two time data readers. These will be
# used to try and read any data. Each has parameters that can be altered
# depending on the type of data. More time readers for particular data formats
# are available in the resistics-readers package.
for time_reader in default_config.time_readers:
    time_reader.summary()

# %%
# The default transfer function is the magnetotelluric impedance tensor. It can
# be printed out to help show the relationship.
default_config.tf.summary()
print(default_config.tf)

# %%
# Other important parameters include those related to decimation setup and
# windowing setup.
default_config.win_setup.summary()
default_config.dec_setup.summary()
