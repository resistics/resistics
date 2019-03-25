ASCII Calibration
-----------------

Metronix calibration files are ASCII files with some metadata and calibration information for when the induction coil is operating with either chopper on or chopper off. 

The units of metronix calibration files are interpreted to be:

- Frequency in Hz
- Magnitude in mV/(nT*Hz)
- Phase in degrees


Naming in the project environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using the project environment, resistics automatically searches for calibration files in the calData folder. 
Metronix files should be named according to the following specification:

.. important::

    [SENSOR][SERIAL].txt
    
    Where SENSOR is the sensor type and SERIAL is the sensor serial number

As an example, consider an induction coil with:

- sensor type MFS06
- sensor serial number 365

Then the file should be named 


Complete example script
~~~~~~~~~~~~~~~~~~~~~~~

For the purposes of clarity, the example script in full.

.. literalinclude:: ../../../../examples/conventions/calibrationRSPExample.py
    :linenos:
    :language: python

.. literalinclude:: ../../../../examples/conventions/calData/IC_365.txt
    :linenos:
    :language: text