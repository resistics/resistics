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

Viewing timeseries data
------------------------

After time data files are in place, they can be visualised. Begin by loading the project and then using the methods in the project :mod:`~resistics.project.time` module.

.. literalinclude:: ../../../examples/tutorial/viewTime.py
    :linenos:
    :language: python
    :lines: 1-14
    :lineno-start: 1

This yields the below plot:

.. figure:: ../_static/examples/tutorial/viewTime_projtime_view.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Project time data

By default, channels |Ex|, |Ey|, |Hx|, |Hy|, |Hz| are all plotted. However, the channels to plot can be explicitly defined. Further, all sites in the project with time data in this range will be plotted. Sites to plot can be explicitly defined as a list of sites.

.. literalinclude:: ../../../examples/tutorial/viewTime.py
    :linenos:
    :language: python
    :lines: 16-26
    :lineno-start: 16

.. figure:: ../_static/examples/tutorial/viewTime_projtime_view_chans.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Project time data with sites and channels restricted

There are a number of pre-processing options that can be optionally applied to the time data. If calibration files for magnetic channels are available and appropriately placed in the project calData directory, the calibration option can be applied.

.. literalinclude:: ../../../examples/tutorial/viewTime.py
    :linenos:
    :language: python
    :lines: 28-39
    :lineno-start: 28

.. figure:: ../_static/examples/tutorial/viewTime_projtime_view_calibrate.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Project time data with magnetic fields calibrated

Low pass filters can be applied to the data as shown below:

.. literalinclude:: ../../../examples/tutorial/viewTime.py
    :linenos:
    :language: python
    :lines: 41-53
    :lineno-start: 41

.. figure:: ../_static/examples/tutorial/viewTime_projtime_view_calibrate_lpfilt.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Project time data with magnetic fields calibrated and a low pass filter applied

High pass, band pass and notch filters can also be applied to the data in a similar fashion. 

.. literalinclude:: ../../../examples/tutorial/viewTime.py
    :linenos:
    :language: python
    :lines: 55-95
    :lineno-start: 55

The channels can be individually normalised by using the normalise option. The normalisation factor here is calculated using the numpy.linalg.norm method.

.. literalinclude:: ../../../examples/tutorial/viewTime.py
    :linenos:
    :language: python
    :lines: 97-109
    :lineno-start: 97

Resistics can automatically save plots as images in the project images directory. When batching, it can often be useful to not show the plots (which tend to block the progress of the code) but rather save the plot without showing it. This can be achieved in the following way:

.. literalinclude:: ../../../examples/tutorial/viewTime.py
    :linenos:
    :language: python
    :lines: 111-123
    :lineno-start: 111

.. figure:: ../_static/examples/tutorial/viewTime_projtime_view_calibrate_bpfilt_save.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Example of a plot saved as an image

Complete example scripts
~~~~~~~~~~~~~~~~~~~~~~~~
For clarity, the complete example scripts are provided here.

.. literalinclude:: ../../../examples/tutorial/viewTime.py
    :linenos:
    :language: python


