Resistics is made up of a number of submodules that deal with different functions, such as handling the reading and writing of data, holding data or manipulating data. These are described in more detail below.

ioHandlers
~~~~~~~~~~

:mod:`~resistics.ioHandlers` are used to read and write data. This includes timeseries data, spectra data, statistic data, mask data and transfer function data.

dataObjects
~~~~~~~~~~~

:mod:`~resistics.dataObjects` are objects that hold data, such as :class:`~resistics.dataObjects.projectData.ProjectData`, :class:`~resistics.dataObjects.siteData.SiteData` or :class:`~resistics.dataObjects.timeData.TimeData`.

calculators
~~~~~~~~~~~

:mod:`~resistics.calculators` perform modifications, transformations and calculations given data objects. For example, calibration, spectra calculation, statistic calculation and transfer function calculation.

project
~~~~~~~

The :mod:`project` submodule provides functions for the batch processing of magnetotelluric data when using the resistics project environment. For more information on the project environment and batch processing, see the :doc:`tutorial <tutorial>`. 


utilities
~~~~~~~~~

:mod:`~resistics.utilities` are a set of submodules that provide helper functions used across the package. This includes various checks and printing functions.