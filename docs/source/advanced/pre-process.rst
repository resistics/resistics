.. role:: python(code)
   :language: python

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

Pre-processing timeseries data
------------------------------

There are scenarios in which time series data must be pre-processed before transfer function calculation. Pre-processing is performed in the project environment using the :meth:`~resistics.project.projectTime.preProcess` method. There are several pre-processing options available in :meth:`~resistics.project.projectTime.preProcess` method, including:

    - Data resampling
    - Polarity reversal
    - Data interpolation    
    - Calibration 
    - Low, band and high pass filters
    - Notch filter
    - Data normalisation

Each of these are explained in more detail below.

Data resampling
~~~~~~~~~~~~~~~
Resampling time series data might be required to ensure matching sampling rates between different datasets, for example a local site and a reference site. This can be achieved through the resampling option which can save the data either to the same site or to a new site altogether.

Polarity reversal
~~~~~~~~~~~~~~~~~
This simply multiples the requested channels by -1. This is particularly useful when electric channels have been connected in the opposite way. An example is shown below of polarity reversal of e channels. 

Data interpolation
~~~~~~~~~~~~~~~~~~
Check

Calibration
~~~~~~~~~~~
The data can be pre-calibrated in a situation where calibrated time series data is required for viewing, exporting or other purposes. Calibration is demonstrated below.

Low, band and high pass filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Time series data can be low, band and high pass filtered as required. This normally useful for viewing but can be useful for exporting or other purposes.

Notch filter
~~~~~~~~~~~~
Notch filters are useful ways to remove unwanted signal in data. The most common frequency domain noise spikes which may need filtering are powerline noise or railway noise at 50 Hz and 16.6 Hz respectively. An example is given below.
