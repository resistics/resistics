Statistics
==========

There are two motivating factors for the calculation of statistics:

- Selecting windows for further processing
- Investigating data quality  

Statistics are calculated on an evaluation frequency basis, meaning that different windows can be selected for each evaluation frequency.

Selection of windows
~~~~~~~~~~~~~~~~~~~~
The philosophy of Resistics is that all time series data is windowed and transformed into the frequency domain. Selection of windows happens after this point using either date and time constraints or constraints based on the calculation of statistics. 

There are a number of statistics already included in Resistics. These are:

- Coherence between channels
- Transfer function components
- Apparent resistivity and phase 
- Polarisation direction
- Partial coherence

In addition, statistics have been added that include remote reference data:



The following sections dive into each one of the statistics in more detail. The intention is to support custom statistics in the future but the architecture of this has not yet been decided.


Data investigation
~~~~~~~~~~~~~~~~~~
The second reason behind the calculation of statistics in Resistics is to allow transparent investigation of the data. The ability to see the variation in time of certain statistic values offers insight into 

.. toctree::
    :maxdepth: 2
    :titlesonly:
    :glob:
    :hidden:

    3a_powerSpectralDensity.rst
    3a_coherences.rst
    3a_partialCoherence.rst
    3a_polarisationDirection.rst
    3a_transferFunction.rst
    3a_resistivityPhase.rst

.. |Ex| replace:: E\ :sub:`x`
.. |Ey| replace:: E\ :sub:`y`
.. |Hx| replace:: H\ :sub:`x`
.. |Hy| replace:: H\ :sub:`y`
.. |Hz| replace:: H\ :sub:`z`