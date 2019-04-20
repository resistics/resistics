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

Processing ASCII data
---------------------

There are instances in which magnetotelluric time series data is better dealt with in ASCII format. ASCII format has some storage and efficiency costs, but does offer a method of last resort when either a data format is not supported or ASCII data is all that is available. To read more about the ASCII data format, see :doc:`here <../formats/timeseries/ascii-timeseries>`.

The following is an example of processing long-period ASCII data from a magnotelluric observatory, recorded by a LEMI device.

There are five data files, one for each channel:

- bxnT.ascii
- bynT.ascii
- bznT.ascii
- exmuVm.ascii
- eymuVm.ascii

The data files can be downloaded from `here <https://1drv.ms/f/s!ApKs8ZhVotKMgR6lVt3iI4cR6QUg>`_ for those interested in following along.

The magnetic fields are in nT, which is the unit required by resistics (after calibration).

The electric fields are in microvolts per metre, which is the equivalent of mV/km, again the unit required by resistics.

To begin, a new project and site is created.

.. literalinclude:: ../../../cookbook/usingAscii/createProject.py
    :linenos:
    :language: python

This is a single site, therefore, the reference time is of little significance. A dummy reference time has been set for the data. 

Under the site directory, a time series measurement directory needs to be created. This is a simple job of making a new directory, in this case named "meas". The data files should be copied into this directory.

After the project, site and measurement directory have been created, the data has to be prepared to be read in by resistics. This is achieved by using the :meth:`~resistics.ioHandlers.dataWriter.DataWriter.writeTemplateHeaderFiles` method of the class :class:`~resistics.ioHandlers.dataWriter.DataWriter`. 

.. literalinclude:: ../../../cookbook/usingAscii/asciiPrepare.py
    :linenos:
    :language: python

The :meth:`~resistics.ioHandlers.dataWriter.DataWriter.writeTemplateHeaderFiles` creates dummy header files for the ASCII data allowing resistics to read it in appropriately. The dummy files it creates are:

- chan_00.hdr
- chan_01.hdr
- chan_02.hdr
- chan_03.hdr
- chan_04.hdr
- global.hdr

For a more in depth look at what is being done here, refer to :doc:`ASCII time series <../formats/timeseries/ascii-timeseries>`, which uses the same files to demonstrate ASCII data support in resistics.

The time series is now ready to be read in. In the below code block, the project and site timelines are plotted, giving two images, and then the time series is visualised. 

.. literalinclude:: ../../../cookbook/usingAscii/runDefault.py
    :linenos:
    :language: python
    :lines: 1-15
    :lineno-start: 1

The next step is to calculate out the spectra. 

.. literalinclude:: ../../../cookbook/usingAscii/runDefault.py
    :linenos:
    :language: python
    :lines: 17-20
    :lineno-start: 17    

The easiest way to process the ascii data is using the default parameterisation, much like in :doc:`Up and running <../tutorial/up-and-running>`.

.. literalinclude:: ../../../cookbook/usingAscii/runDefault.py
    :linenos:
    :language: python
    :lines: 22-41
    :lineno-start: 22    

.. literalinclude:: ../../../cookbook/usingAscii/runDefault.py
    :linenos:
    :language: python
    :lines: 43-54
    :lineno-start: 43    



Complete example script
~~~~~~~~~~~~~~~~~~~~~~~

For the purposes of clarity, the complete example scripts are provided below.

Create the project and a site:

.. literalinclude:: ../../../cookbook/usingAscii/createProject.py
    :linenos:
    :language: python

Preparing the ascii data to be read in:

.. literalinclude:: ../../../cookbook/usingAscii/asciiPrepare.py
    :linenos:
    :language: python

Transfer function calculation with default parameters:

.. literalinclude:: ../../../cookbook/usingAscii/runDefault.py
    :linenos:
    :language: python

Run with a configuration file:

.. literalinclude:: ../../../cookbook/usingAscii/runWithConfig.py
    :linenos:
    :language: python

Run with statistics:

.. literalinclude:: ../../../cookbook/usingAscii/runWithStatistics.py
    :linenos:
    :language: python