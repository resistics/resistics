.. |Ex| replace:: E\ :sub:`x`
.. |Ey| replace:: E\ :sub:`y`
.. |Hx| replace:: H\ :sub:`x`
.. |Hy| replace:: H\ :sub:`y`
.. |Hz| replace:: H\ :sub:`z`

Conventions
-----------

Resistics adopts a number of conventions which are outlined below:

- Data is assumed to be sampled on the second. Whilst this is generally the case with ATS timeseries data files, SPAM and other formats do not necessarily follow this convention. There is functionality available to interpolate data on to the second built into resistics. Examples of interpolation to the second can be found in the :doc:`time series formats <formats/timeseries>` section. 
- Magnetic channels are named |Hx|, |Hy| and |Hz|.
- Electric channels are named |Ex|, |Ey|.
- Resistics operates with field units. When timeseries data is read in, it is automatically converted to the following units:
    
    - Electric channels |Ex|, |Ey| are in mV/km
    - Magnetic channels |Hx|, |Hy| and |Hz| are in mV
    - Calibrated magnetic channels are in nT

.. note::   
 
    More specific information about the supported data formats for timeseries and calibration data are provided in the :doc:`formats <formats>` section.
