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

There are scenarios in which time series data must be pre-processed and saved before transfer function calculation. Pre-processing is performed in the project environment using the :meth:`~resistics.project.projectTime.preProcess` method. There are several pre-processing options available in :meth:`~resistics.project.projectTime.preProcess` method, including:

    - Data resampling
    - Polarity reversal
    - Data interpolation    
    - Calibration 
    - Low, band and high pass filters
    - Notch filter
    - Data normalisation

In addition, there are lower levels APIs for performing these actions for advanced users. Please see the :doc:`API doc <../api>` for more information or check the :doc:`Cookbook <../cookbook>`.

The preprocessing options are explained in more detail below. The data is the same 4096 Hz data from the tutorial and is saved in "site1". A project has already been setup following the instructions in :doc:`Project environment <../tutorial/project-environment>`. In all the following, the project has already been loaded as shown below.

.. literalinclude:: ../../../examples/advanced/preprocessResample.py
    :linenos:
    :language: python
    :lines: 1-6
    :lineno-start: 1

Data resampling
~~~~~~~~~~~~~~~
Resampling time series data might be required to ensure matching sampling rates between different datasets, for example a local site and a reference site. This can be achieved through the resampling option which can save the data either to the same site or to a new site altogether.

Below is an exampling of downsampling a 4096 Hz dataset to 1024 Hz. The output will be saved to a new site named "site1_resample". If the site does not already exist, it will be created. The name of the measurement in the new site will be:

..note::

    prepend + meas_[start_time] + postpend
    give an example

As the measurement is being out to a new site, prepend is being set "" (an empty string). By default, postpend is an empty string.

.. literalinclude:: ../../../examples/advanced/preprocessResample.py
    :linenos:
    :language: python
    :lines: 8-26
    :lineno-start: 8

These produce the following plots:

.. figure:: ../../../examples/tutorial/tutorialProject/images/statistic_coherence_view.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Coherence data plotted for evaluation frequency 32 Hz using the :meth:`~resistics.dataObjects.statisticData.StatisticData.view` method

Polarity reversal and scaling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This simply multiples the requested channels by -1. This is particularly useful when electric channels have been connected in the opposite way. An example is shown below of the polarity reversal of an electric channel. 

.. literalinclude:: ../../../examples/advanced/preprocessMath.py
    :linenos:
    :language: python
    :lines: 8-23
    :lineno-start: 8

The results of this can be seen in Figure xxxxxx.

If more scaling control is required than simply multiplying by -1, the scale option of :meth:`~resistics.project.projectTime.preProcess`. An example is provided here:

.. literalinclude:: ../../../examples/advanced/preprocessMath.py
    :linenos:
    :language: python
    :lines: 25-27
    :lineno-start: 25

This will scale the something channel by something.

Interpolation to second
~~~~~~~~~~~~~~~~~~~~~~~

Resistics assumes that data is sampled on the second. For more information on this, please see :doc:`Time series formats <../formats/timeseries>` and :doc:`Interpolation to second <../formats/timeseries/interpolation-to-second>`.

Interpolating to the second can be done in the project environment. An example follows:





Gap filling
~~~~~~~~~~~
In some situations, it is beneficial to stitch together two separate datasets to perform 

Calibration
~~~~~~~~~~~
The data can be pre-calibrated in a situation where calibrated time series data is required for viewing, exporting or other purposes. Calibration is demonstrated below.

Low, band and high pass filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Time series data can be low, band and high pass filtered as required. This normally useful for viewing but can be useful for exporting or other purposes.

Notch filter
~~~~~~~~~~~~
Notch filters are useful ways to remove unwanted signal in data. The most common frequency domain noise spikes which may need filtering are powerline noise or railway noise at 50 Hz and 16.6 Hz respectively. An example is given below.
