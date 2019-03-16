.. |Ex| replace:: E\ :sub:`x`
.. |Ey| replace:: E\ :sub:`y`
.. |Hx| replace:: H\ :sub:`x`
.. |Hy| replace:: H\ :sub:`y`
.. |Hz| replace:: H\ :sub:`z`

Conventions
-----------

Resistics adopts a number of conventions which are outlined below:

- Data is assumed to be sampled on the second. Whilst this is generally the case with ATS timeseries data files, SPAM and other formats do not necessarily follow this convention. There is functionality available to interpolate data on to the second built into resistics. An example of interpolation to the second can be found :doc:`here <advanced/preProcess>` 
- Magnetic channels are named |Hx|, |Hy| and |Hz|
- Electric channels are named |Ex|, |Ey|
- Resistics operates with field units. When timeseries data is read in, it is automatically converted to the following units:
    
    - Electric channels |Ex|, |Ey| are in mV/m
    - Magnetic channels |Hx|, |Hy| and |Hz| are in mV
    - Calibrated magnetic channels are in nT

The :doc:`timeseries data <conventions/timeseries>` and :doc:`calibration data <conventions/calibration>` sections cover those respective topics in more detail. Ensuring that timeseries data is appropriate read in and calibrated can be tricky when there are multiple file formats to support. If you are having any trouble with reading in timeseries data or calibrating it, please submit a request on the `GitHub repository <https://github.com/resistics/resistics>`_.

.. toctree::
    :maxdepth: 2
    :titlesonly:
    :hidden:

    conventions/timeseries.rst
    conventions/calibration.rst

