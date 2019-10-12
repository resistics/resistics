ASCII Calibration
-----------------

The internal ASCII calibration format is a simple ASCII format meant to be used when no other format is available. Each file only contains one set of calibration data (no chopper on or off) and there is no provision for different chopper on and off files.  

The units of the internal format ASCII calibration files are interpreted to be:

- Frequency in Hz
- Magnitude in mV/nT
- Phase in degrees or radians depending on the "Phase unit" header

Resistics will automatically convert these units to:

- Frequency in Hz
- Magnitude in mV/nT (including any static gain)
- Phase in radians

Naming in the project environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When using the project environment, resistics automatically searches for calibration files in the calData folder. 
Internal format ASCII files should be named according to the following specification:

.. important::

    [*]IC_[SERIAL].txt
    
    where, 
    
    - SERIAL is the sensor serial number
    - [*] represents any general string

As an example, consider an induction coil with:

- sensor serial number 307

Then the file should be named:

- IC_307.TXT
- Or with any leading text, e.g. magcal_IC_307.TXT

As can be seen, there is no ability to distinguish different sensors with the same serial number or situations where chopper is on or off. For those cases, :doc:`RSP <rsp-calibration>` or :doc:`Metronix <metronix-calibration>` formats are better.

Example
~~~~~~~
The internal ASCII format calibration file will not usually be provided and will have to be created from scratch. The :class:`~resistics.calibrate.io.CalibrationIO` class provides the handy :meth:`~resistics.calibrate.io.CalibrationIO.writeInternalTemplate` method for producing a template internal ASCII format calibration file. A template can be made as follows:

.. literalinclude:: ../../../../examples/formats/calibrationInductionExample.py
    :linenos:
    :language: python
    :lines: 1-7
    :lineno-start: 1 

This produces an empty calibration file with some basic information in:

.. literalinclude:: ../../_static/examples/formats/cal/ascii.TXT
    :linenos:
    :language: text

The basic metadata can either be corrected in the file or passed as keywords to :meth:`~resistics.calibrate.io.CalibrationIO.writeInternalTemplate`. The actual calibration data needs to be copied in and should be in the units:

- Magnitude in mV/nT without static gain
- Phase in units which match the "Phase unit" header, which can either be degrees or radians

An example file with calibration data copied is provided below.

.. literalinclude:: ../../_static/examples/formats/cal/asciiWithData.TXT
    :linenos:
    :language: text

This internal format ASCII calibration file can now be read in using the :class:`~resistics.calibrate.io.CalibrationIO` class. First, the reading parameters need to be reset using the :meth:`~resistics.calibrate.io.CalibrationIO.refresh` method. Using the :meth:`~resistics.calibrate.io.CalibrationIO.read` method will return a :class:`~resistics.calibrate.data.CalibrationData` object.

.. literalinclude:: ../../../../examples/formats/calibrationInductionExample.py
    :linenos:
    :language: python
    :lines: 9-13
    :lineno-start: 9

The :meth:`~resistics.common.base.ResisticsBase.printInfo` method prints information about the calibration data.

.. literalinclude:: ../../_static/examples/formats/cal/calibrationInductionPrint.txt
    :linenos:
    :language: text
 
To view the calibration data, the :meth:`~resistics.calibrate.data.CalibrationData.view` method of :class:`~resistics.calibrate.data.CalibrationData` can be used. By passing a matplotlib figure to this, the layout of the plot can be controlled.

.. literalinclude:: ../../../../examples/formats/calibrationInductionExample.py
    :linenos:
    :language: python
    :lines: 15-22
    :lineno-start: 15

This produces the following plot:

.. figure:: ../../_static/examples/formats/cal/calibrationASCII.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Viewing the unextended calibration data from the internal format ASCII file

Complete example script
~~~~~~~~~~~~~~~~~~~~~~~
For the purposes of clarity, the example script in full.

.. literalinclude:: ../../../../examples/formats/calibrationInductionExample.py
    :linenos:
    :language: python

.. literalinclude:: ../../_static/examples/formats/cal/asciiWithData.txt
    :linenos:
    :language: text