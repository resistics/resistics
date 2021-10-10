"""
Default configuration
^^^^^^^^^^^^^^^^^^^^^

This example shows the default resistics configuration.
"""
from resistics.config import get_default_configuration

# %%
# Get the default configuration and print the summary
default_config = get_default_configuration()
default_config.summary()
