API reference
-------------

Resistics is made up of a number of sub packages that deal with different functions or data types, such as time data, frequency data, statistic data and more. Detailed information about each can be found in the relevant sections.

- :ref:`modules:common`
- :ref:`modules:config`
- :ref:`modules:time`
- :ref:`modules:spectra`
- :ref:`modules:statistics`
- :ref:`modules:mask`
- :ref:`modules:regression`
- :ref:`modules:transfunc`
- :ref:`modules:site`
- :ref:`modules:project`
- :ref:`modules:calibrate`
- :ref:`modules:window`
- :ref:`modules:decimate`

For most users, the API functionality of interest will be in :ref:`modules:project`, which holds methods for interacting with resistics projects. 

common
~~~~~~
Methods in the :mod:`~resistics.common` are used across the package. They include utility functions for print, checking io, plotting and more. 

[:ref:`back to top <modules:API reference>`]

.. toctree::
    :maxdepth: 3

    api/resistics.common.rst

config
~~~~~~
:mod:`~resistics.config` has methods and classes for the resistics configuration data and parameters. 

[:ref:`back to top <modules:API reference>`]

.. toctree::
    :maxdepth: 3

    api/resistics.config.rst    

time
~~~~
Methods and classes in :mod:`~resistics.time` relate to reading, writing, holding and processing of time data. 

[:ref:`back to top <modules:API reference>`]

.. toctree::
    :maxdepth: 3

    api/resistics.time.rst

spectra
~~~~~~~
Methods and classes in :mod:`~resistics.spectra` deal with calculating fourier spectra, frequency data reading and writing and spectra data objects. 

[:ref:`back to top <modules:API reference>`]

.. toctree::
    :maxdepth: 3

    api/resistics.spectra.rst

statistics
~~~~~~~~~~
:mod:`~resistics.statistics` provides functionality for calculating local site and remote reference statistics, statistic reading and writing and statistic data objects. 

[:ref:`back to top <modules:API reference>`]

.. toctree::
    :maxdepth: 3

    api/resistics.statistics.rst

mask
~~~~
:mod:`~resistics.mask` data in resistics is used to remove windows from further processing. It provides classes for taking in statistic data together with constraints and finding the windows which fail the constraints. 

[:ref:`back to top <modules:API reference>`]

.. toctree::
    :maxdepth: 3

    api/resistics.mask.rst

regression
~~~~~~~~~~
Methods and classes in :mod:`~resistics.regression` use the frequency data to estimate transfer functions. 

[:ref:`back to top <modules:API reference>`]

.. toctree::
    :maxdepth: 3

    api/resistics.regression.rst

transfunc
~~~~~~~~~
Methods and classes in :mod:`~resistics.transfunc` deal with the input and output of transfer function data and provide data objects for transfer function data. 

[:ref:`back to top <modules:API reference>`]

.. toctree::
    :maxdepth: 3

    api/resistics.transfunc.rst

site
~~~~
:mod:`~resistics.site` provides methods for interacting with site data. 

[:ref:`back to top <modules:API reference>`]

.. toctree::
    :maxdepth: 3

    api/resistics.site.rst

project
~~~~~~~
:mod:`~resistics.project` provides methods for interacting with resistics projects. For most users, these are the most interesting methods and APIs. 

[:ref:`back to top <modules:API reference>`]

.. toctree::
    :maxdepth: 3

    api/resistics.project.rst

calibrate
~~~~~~~~~
Methods and classes for handling calibration data can be found in :mod:`~resistics.calibrate`. 

[:ref:`back to top <modules:API reference>`]

.. toctree::
    :maxdepth: 3

    api/resistics.calibrate.rst

window
~~~~~~
Window parameters, timeseries windowing and window selection functionality is provided in :mod:`~resistics.window`.

[:ref:`back to top <modules:API reference>`]

.. toctree::
    :maxdepth: 3

    api/resistics.window.rst

decimate
~~~~~~~~
Decimation parameters and decimation functionality is provided in :mod:`~resistics.window`.

[:ref:`back to top <modules:API reference>`]

.. toctree::
    :maxdepth: 3

    api/resistics.decimate.rst