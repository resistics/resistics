Conventions
-----------

Resistics adopts a number of conventions which are outlined below:

- Resistics assumes that data is sampled on the second.
- Magnetic channels are named |Hx|, |Hy| and |Hz|
- Electric channels are named |Ex|, |Ey|
- All physical measurements are given in field units
    
    - Electric channels |Ex|, |Ey| are in mV/m
    - Magnetic channels are in V
    - Calibrated magnetic channels are in nT
- Resistics automatically identifies time data folders which are named appropriately and makes the following associations:

    - Directories of the format meas_xxx_xxx are interpreted as ATS data Directories
    - Directories 

- Resistics will automatically locate a calibration file for the data. The following calibration file conventions are used



.. |Ex| replace:: E\ :sub:`x`
.. |Ey| replace:: E\ :sub:`y`
.. |Hx| replace:: H\ :sub:`x`
.. |Hy| replace:: H\ :sub:`y`
.. |Hz| replace:: H\ :sub:`z`