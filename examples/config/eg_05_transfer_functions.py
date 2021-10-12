"""
Transfer functions
^^^^^^^^^^^^^^^^^^

Transfer functions can be customised too depending on needs. There are built-in
transfer functions, which have the added benefit of having plotting functions
meaning the results can be visualised correctly, for example the impedance
tensor.

However, if a completely custom transfer function is required, this can be done
with the caveat that there will be no plotting function available. A better
solution might be to write a custom transfer function if required. For more
about writing custom transfer functions, see the advanced usage.
"""
from resistics.transfunc import TransferFunction

# %%
# To initialise a new transfer function, the input and channels need to be
# defined.
tf = TransferFunction(in_chans=["Cat", "Dog"], out_chans=["Tiger", "Wolf"])
print(tf)
tf.summary()

# %%
# It is also possible to set the channels that will be used to calculate out the
# cross spectra. Note that these channels should be available in the input site,
# output site and cross site respectively.
tf = TransferFunction(
    name="Jungle",
    in_chans=["Cat", "Dog"],
    out_chans=["Tiger", "Wolf"],
    cross_chans=["Lizard", "Crocodile"],
)
print(tf)
tf.summary()

# %%
# In scenarios where the core transfer function stays the same (input and
# output channels), but the cross channels will be changed, there is an
# additional variation property that helps separate them.
tf = TransferFunction(
    name="Jungle",
    variation="Birds",
    in_chans=["Cat", "Dog"],
    out_chans=["Tiger", "Wolf"],
    cross_chans=["Owl", "Eagle"],
)
print(tf)
tf.summary()
