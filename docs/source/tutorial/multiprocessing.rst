.. |Ex| replace:: E\ :sub:`x`
.. |Ey| replace:: E\ :sub:`y`
.. |Hx| replace:: H\ :sub:`x`
.. |Hy| replace:: H\ :sub:`y`
.. |Hz| replace:: H\ :sub:`z`
.. |Zxy| replace:: Z\ :sub:`xy`
.. |Zxx| replace:: Z\ :sub:`xx`
.. |Zyx| replace:: Z\ :sub:`yx`
.. |Zyy| replace:: Z\ :sub:`yy`
.. |fs| replace:: f\ :sub:`s`
.. role:: python(code)
   :language: python

Multiprocessing
---------------

Processing magnetotelluric field surveys can be demanding if there many sites and long recordings. To help speed up processing, resistics supports multiprocessor computation using the python multiprocessing module (`read more here <https://docs.python.org/3/library/multiprocessing.html>`_). Multiprocessing can be used in the following processing steps:

- Spectra calculation
- Statistic calculation
- Transfer function estimation (robust regression)

.. important::

    For Windows users, multiprocessing requires scripts using resistics and multiprocessing to be protected using :python:`if __name__ == '__main__'`. For a more detailed explanation of why, please see the `offical python documentation <https://docs.python.org/3/library/multiprocessing.html#multiprocessing-programming>`_. What this means is illustrated in the below :ref:`example <tutorial/multiprocessing:Example>`.

Multiprocessing is activated by specifying the :python:`ncores` keyword in the following methods:

- :meth:`~resistics.project.spectra.calculateSpectra`
- :meth:`~resistics.project.statistics.calculateStatistics`
- :meth:`~resistics.project.transfer.processProject` or :meth:`~resistics.project.transfer.processSite`

The optional argument :python:`ncores` is expected to be an integer. It is recommend to use less cores than cores available on the system. This will leave spare processing power for system tasks and for interaction with the system.

Alternatively and probably more usefully, the :python:`ncores` options can be set in configuration files, where it can be either a subparameter or a global parameter.

For example, setting :python:`ncores` as a global parameter:

.. literalinclude:: ../_static/examples/tutorial/multiconfig.ini
    :linenos:
    :language: text

By setting a global :python:`ncores` option, any methods that support multiprocessing will use it on the globally specified number of cores. 

Another option is to define :python:`ncores` separately for individual tasks. The :python:`ncores` option can be supplied to the following sections separately.

- Spectra
- Statistics
- Solver

The following configuration specifies different cores for each section.

.. literalinclude:: ../_static/examples/tutorial/multiconfigSeparate.ini
    :linenos:
    :language: text

In the above config file, spectra calculations will use 4 cores, statistic calculations will use 5 cores and the solver (robust regression) will use only a single core.

Example 
~~~~~~~
The below configuration and processing script runs spectra, statistic and transfer function calculations on 6 cores. 

.. literalinclude:: ../_static/examples/tutorial/multiconfig.ini
    :linenos:
    :language: text

.. literalinclude:: ../../../examples/tutorial/multiProc.py
    :linenos:
    :language: python

.. warning::

    This script was run on the Windows platform. Note the use of :python:`if __name__ == '__main__'` to ensure no multiprocessing issues are encountered. Forgetting this part would cause many processes to be spawned which will consume system resources. 

The impedance tensor estimates are shown below with and without masking. 

.. figure:: ../_static/examples/tutorial/multproc_standard_process.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Multiprocessing using all standard options

.. figure:: ../_static/examples/tutorial/multproc_mask_process.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Multiprocessing with masks

Complete example script
~~~~~~~~~~~~~~~~~~~~~~~
For the purposes of clarity, the complete example script is provided below.

.. literalinclude:: ../../../examples/tutorial/multiProc.py
    :linenos:
    :language: python
