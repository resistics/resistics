Calibration data
----------------

Calibration data for magnetotelluric recordings come in many different flavours. Again, resistics aims to support the major formats, though this becomes harder when they are binary formatted. 

Currently, there are four supported calibration formats. These are:

- :doc:`Metronix <calibration/metronix-calibration>`
- :doc:`RSP <calibration/rsp-calibration>`
- :doc:`RSPX <calibration/rspx-calibration>`
- A simple :doc:`ASCII <calibration/ascii-calibration>` format for when none of the others make sense

.. note::

    Currently, resistics only supports calibration of data channels based on:
    
    - sensor type
    - sensor serial number
    - chopper information (low frequency or high frequency recording mode) 
    
    This will cover the calibration of magnetic channels. However, there is presently no easy way of doing a recording instrument calibration. Depending on the need for this feature, it can be implemented. Please get in touch if this is something that is required by visiting the :doc:`Contact <../contact>` page.


Extending calibration data
~~~~~~~~~~~~~~~~~~~~~~~~~~

Often times, calibration data does not cover the whole range of evaluation frequencies to be processed. In this case, the calibration data needs to be extended. By default, resistics will extend calibration data using simple linear interpolation. In some cases, it is better for the extension of calibration data to be performed outside of resistics. In this case, calibration extension can be switched off as a configuration option. For more details, see the :doc:`Configuration <../features/configuration>` section.

Viewing calibration data
~~~~~~~~~~~~~~~~~~~~~~~~

As calibration is another major point of pain and failure in the processing magnetotelluric data, resistics includes tools for viewing calibration data. This is good practice, to ensure that calibration files are being read in appropriately. See the sections covering each format for examples of viewing calibration data.


.. toctree::
    :maxdepth: 2
    :titlesonly:
    :glob:
    :hidden:

    calibration/metronix-calibration.rst
    calibration/rsp-calibration.rst
    calibration/rspx-calibration.rst
    calibration/ascii-calibration.rst
