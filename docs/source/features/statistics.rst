.. |Ex| replace:: E\ :sub:`x`
.. |Ey| replace:: E\ :sub:`y`
.. |Hx| replace:: H\ :sub:`x`
.. |Hy| replace:: H\ :sub:`y`
.. |Hz| replace:: H\ :sub:`z`

Statistics
----------

There are two main motivating factors for the calculation of statistics:

- Investigating data quality  
- Selecting windows for further processing

When processing magnetotelluric data, it can often be difficult to understand why transfer function estimates are poor. Statistics help users investigate data quality by showing the variation of a selection of parameters over the course of the recording. Based on the statistic data, users can specify criteria for selecting windows based on the parameter values. The most familiar example is coherence statistics, which are widely used already in magnetotelluric data processing. 

The philosophy of resistics is that all timeseries data is windowed and transformed into the frequency domain. Selection of windows happens after this point using either date and time constraints or constraints based on the calculation of statistics. Statistics are calculated on an evaluation frequency basis, meaning that different windows can be selected for each evaluation frequency.

There are a number of statistics already included in resistics. These are:

.. toctree::
    :maxdepth: 2
    :titlesonly:
    :glob:

    statistics/absval.rst
    statistics/powerSpectralDensity.rst
    statistics/coherence.rst
    statistics/partialCoherence.rst
    statistics/polarisationDirection.rst
    statistics/transferFunction.rst
    statistics/resistivityPhase.rst

| **A good paper detailing and demonstrating the use of various statistics is:**
| `Effective noise separation for magnetotelluric single site data processing using a frequency domain selection scheme <https://academic.oup.com/gji/article/161/3/635/557193>`_
| U. Weckmann  A. Maguina  O. Ritter
| *Geophysical Journal International, Volume 161, Issue 3, June, 2005* 
| *Pages 635-652* 
| https://doi.org/10.1111/j.1365-246X.2005.02621.x


The following sections dive into each one of the statistics in more detail. The intention is to support custom statistics in the future but the architecture of this has not yet been decided.

Standard statitsics support local site processing. For remote reference processing continue :doc:`here <remote-statistics>`.