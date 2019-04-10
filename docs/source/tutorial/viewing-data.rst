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

After time data files are in place, they can be visualised. Begin by loading the project and then using the methods in the :mod:`~resistics.project.projectTime` module.

.. literalinclude:: ../../../examples/tutorial/viewTime.py
    :linenos:
    :language: python
    :lines: 1-12
    :lineno-start: 1

This produces the below plot:

.. figure:: ../_images/projectViewTime_1.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Project time data

By default, channels |Ex|, |Ey|, |Hx|, |Hy|, |Hz| are all plotted. However, the channels to plot can be explicitly defined. Further, all sites in the project with time data in this range will be plotted. Sites to plot can be explicitly defined as a list of sites.

.. literalinclude:: ../../../examples/tutorial/viewTime.py
    :linenos:
    :language: python
    :lines: 14-21
    :lineno-start: 14

.. figure:: ../_images/projectViewTime_2.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Project time data with sites and channels restricted

There are a number of pre-processing options that can be optionally applied to the time data. If calibration files for magnetic channels are available and appropriately placed in the project calData directory, the calibration option can be applied.

.. literalinclude:: ../../../examples/tutorial/viewTime.py
    :linenos:
    :language: python
    :lines: 23-31
    :lineno-start: 23

.. figure:: ../_images/projectViewTime_3.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Project time data with magnetic fields calibrated

Low pass filters can be applied to the data as shown below:

.. literalinclude:: ../../../examples/tutorial/viewTime.py
    :linenos:
    :language: python
    :lines: 33-42
    :lineno-start: 33

.. figure:: ../_images/projectViewTime_4.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Project time data with magnetic fields calibrated and a low pass filter applied

High pass, band pass and notch filters can also be applied to the data in a similar fashion. 

.. literalinclude:: ../../../examples/tutorial/viewTime.py
    :linenos:
    :language: python
    :lines: 44-75
    :lineno-start: 44

Resistics can automatically save plots as images in the project images directory. When batching, it can often be useful to not show the plots (which tend to block the progress of the code) but rather save the plot without showing it. This can be achieved in the following way:

.. literalinclude:: ../../../examples/tutorial/viewTime.py
    :linenos:
    :language: python
    :lines: 88-99
    :lineno-start: 88

.. figure:: ../_images/projectViewTime_5.png
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


