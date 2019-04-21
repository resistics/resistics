Processing with masks
---------------------

Now that statistics and masks have been calculated, the final step is to include those masks in the transfer function calculation. There is a simple addition to the standard transfer function calculation workflow presented in :doc:`Up and running <up-and-running>` and :doc:`Tippers <tipper>`.

As always, load the project. 

.. literalinclude:: ../../../examples/tutorial/runWithMasks.py
    :linenos:
    :language: python
    :lines: 1-6
    :lineno-start: 1

Now perform the transfer function calculation with the addition of a mask.

.. literalinclude:: ../../../examples/tutorial/runWithMasks.py
    :linenos:
    :language: python
    :lines: 8-18
    :lineno-start: 8

The elements of note in this code block are:

- masks keyword. A mask needs to be associated with a site and provided in a dictionary. More than a single mask file can be supplied for a site by passing a list of masks.
- postpend keyword. This is the postpend on the output transfer function data file to avoid overwriting existing transfer function calculation files.

The resultant impedance tensor result can be visualised in the same way as demonstrated in previous sections using the :meth:`~resistics.project.projectTransferFunction.viewImpedance` method of the :mod:`~resistics.project.projectTransferFunction` module. 

.. literalinclude:: ../../../examples/tutorial/runWithMasks.py
    :linenos:
    :language: python
    :lines: 20-24
    :lineno-start: 1

.. figure:: ../../../examples/tutorial/tutorialProject/images/transFunction_site1_dec8_5_coh70_100.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Impedance tensor estimate with the coherence masking

Repeating the process for the mask with additional transfer function constraints gives:

.. literalinclude:: ../../../examples/tutorial/runWithMasks.py
    :linenos:
    :language: python
    :lines: 26-42
    :lineno-start: 26

.. figure:: ../../../examples/tutorial/tutorialProject/images/transFunction_site1_dec8_5_coh70_100_tfConstrained.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Impedance tensor estimate with both coherence and transfer function masking

.. note::

    This was the last section of the tutorial. To explore more advanced use cases or user provided examples, please visit the :doc:`Advanced <../advanced>` and :doc:`Cookbook <../cookbook>` sections respectively.

Complete example script
~~~~~~~~~~~~~~~~~~~~~~~

For the purposes of clarity, the complete example script is provided below.

.. literalinclude:: ../../../examples/tutorial/runWithMasks.py
    :linenos:
    :language: python