.. |Ex| replace:: E\ :sub:`x`
.. |Ey| replace:: E\ :sub:`y`
.. |Hx| replace:: H\ :sub:`x`
.. |Hy| replace:: H\ :sub:`y`
.. |Hz| replace:: H\ :sub:`z`

Introduction
------------

Magnetotelluric surveying
~~~~~~~~~~~~~~~~~~~~~~~~~
Magnetotellurics (MT) is a passive geophysical method for estimating the electrical properties of the subsurface. The MT method uses variations in the natural geoelectric and geomagnetic fields as the source of energy.

.. figure:: _images/mtSetup.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    An example magnetotelluric setup

In the field, electrodes are used to measure potential difference variations  in the x direction (|Ex|) and y direction (|Ey|). Induction coils measure variations of the geomagnetic field in the x, y and z directions (|Hx|, |Hy| and |Hz| respectively). Commonly, x represents East-West and y represents North-South, though only orthongonality is really required. 

For more information, see the following references:

* Wikipedia: https://en.wikipedia.org/wiki/Magnetotellurics
* Practical Magnetotellurics (Simpson and Bahr)
* The Magnetotelluric Method (Chave and Jones)

Resistics
~~~~~~~~~
Resistics is a native Python 3 package for the processing of magnetotelluric data. The code is based on standard processing techniques. The processing methodology is as follows:

- Windowing of timeseries data
- Fourier transform of windowed timeseries data
- For a set of evaluation frequencies, calculate out relevant auto power and cross power spectra for each window
- Use auto power and cross power spectra from across all windows to estimate impedance tensor components for each evaluation frequency by using robust regression
- Decimation and repeat 

Resistics currently supports the reading in of ATS, SPAM, Phoenix and ASCII data and has an internal data format too, which is simply based on the python numpy library. However, resistics does not yet support the Phoenix calibration file format, though ASCII calibration files can be used.

Whilst other codes exist, resistics was written for the following purpose:

- Be an easy-to-use, native Python code for processing magnetotelluric data
- Allow seamless integration of different data types including: ATS, SPAM, Phoenix and ASCII
- Replicate existing functionality available in other codes but allow them to be easily extended in the future
- Allow statistics based data investigation and selection
- Incorporate signal processing methods to improve data quality
- Easy visualation and interogation
- Provide transparency and clear traceability

.. toctree::
    :maxdepth: 1
    :titlesonly:
    :glob:
    :hidden: