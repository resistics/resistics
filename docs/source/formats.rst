Data formats
------------

One of the most challenging aspects of processing magnetotelluric data with third party tools is ensuring that data is correctly read in and calibration files are treated properly. This section outlines the different timeseries and calibration formats supported by resistics, how these should be used and what outcomes can be expected. 

Timeseries formats
~~~~~~~~~~~~~~~~~~

Supported timeseries formats are covered in the :doc:`timeseries data <formats/timeseries>` section, which demonstrates the various timeseries formats supported and potential pitfalls. For best results using resistics, timeseries data should be sampled on the second. For a clearer definition of what this means, please see the information about :doc:`interpolating to the second <formats/timeseries/interpolation-to-second>`.

Calibration formats
~~~~~~~~~~~~~~~~~~~

More information about calibration files is given in the :doc:`calibration data <formats/calibration>` section. Several calibration file formats are supported, though please take care to check the assumed units of the calibration data. 

Unsupported formats
~~~~~~~~~~~~~~~~~~~

If a project requires support for currently unsupported data formats the easiest way would be to convert the data to ASCII (if software is available to do this) and using the :doc:`ASCII data reader <formats/timeseries/ascii-timeseries>`.

If conversion to ASCII is not possible, please submit an issue on the `GitHub repository <https://github.com/resistics/resistics/issues>`_. 


Troubleshooting
~~~~~~~~~~~~~~~

Ensuring that timeseries data is appropriately read in and calibrated can be tricky when there are multiple file formats to support. If you are having any trouble with reading in timeseries data or calibrating it, please first look at the relevant sections for those formats. If this does not clear up any issues, submit a support request on the `GitHub repository <https://github.com/resistics/resistics/issues>`_.

.. toctree::
    :maxdepth: 2
    :titlesonly:
    :hidden:

    formats/timeseries.rst
    formats/calibration.rst