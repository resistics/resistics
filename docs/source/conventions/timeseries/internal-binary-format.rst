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


Internal binary format
----------------------

A fair question to ask is why introduce another data format rather than use a pre-existing data format. The downsides of writing out in a pre-existing data format were the following:

- Incorrect implementation 
- No control of specification
- Makes the data no more portable

The internal format was chosen to achieve two main goals:

- A binary format that allows easy portability
- Headers written out in ascii that can be quickly checked

Therefore, numpy save was chosen as the method of writing the data. This means that the data is portable and can be opened by anyone with Python and the numpy package. 

Internal format data folders tend to look like: 

.. code-block:: text

    meas_2012-02-10_11-05-00 
    ├── global.hdr 
    ├── chan_00.hdr   
    ├── chan_00.dat 
    ├── chan_01.hdr   
    ├── chan_01.dat 
    ├── chan_02.hdr   
    ├── chan_02.dat 
    ├── chan_03.hdr   
    ├── chan_03.dat 
    ├── chan_04.hdr                        
    ├── chan_04.dat  
    └── comments.txt

The global header file contains the following information:

.. literalinclude:: ../../../../examples/conventions/timeData/atsInternal/global.hdr
    :linenos:
    :language: text

And channel headers have channel specific header information:

.. literalinclude:: ../../../../examples/conventions/timeData/atsInternal/chan_00.hdr
    :linenos:
    :language: text

.. note::

    In order for resistics to recognise an internal formatted data folder, the following have to be present:

    - Header files with extension .hdr (global and one for each channel)
    - Data files with extension .dat

.. note::

    In most cases, internally formatted data is written out from data already in field units. If the channel header scaling_applied is True, no scaling will be applied in either :meth:`~resistics.ioHandlers.dataReader.DataReader.getUnscaledSamples` or :meth:`~resistics.ioHandlers.dataReader.DataReader.getPhysicalSamples`. However, if scaling_applied is False, then :meth:`~resistics.ioHandlers.dataReader.DataReader.getPhysicalSamples` will scale the data using the ts_lsb header and divide electric channels by the electrode spacing in km.

Internally formatted binary data is usually written out with comments in a separate file. An example comments file for internally formatted data is given below.

.. literalinclude:: ../../../../examples/conventions/timeData/atsInternal/comments.txt
    :linenos:
    :language: text

The easiest method of formatting ASCII data as the internal binary format is to follow the instructions in the :doc:`ASCII timeseries <ascii-timeseries>` example.

The following will show how to read internally formatted binary data with numpy. To begin with, read an internally formatted dataset with the inbuilt :class:`~resistics.ioHandlers.dataReaderInternal.DataReaderInternal` class.

.. literalinclude:: ../../../../examples/conventions/internalReaderExamples.py
    :linenos:
    :language: python
    :lines: 1-7
    :lineno-start: 1

The :meth:`~resistics.ioHandlers.IOHandlers.printInfo` method shows information about the dataset, including various recording parameters. 

.. literalinclude:: ../../_text/printInternal.txt
    :linenos:
    :language: text

The :class:`~resistics.ioHandlers.dataReaderInternal.DataReaderInternal` class does not automatically load the data into memory. Data has to be requested, which can be done using the :meth:`~resistics.ioHandlers.dataReader.DataReader.getPhysicalSamples` or :meth:`~resistics.ioHandlers.dataReader.DataReader.getUnscaledSamples` methods if all the data is required or only a sample range. To request data using dates, the :meth:`~resistics.ioHandlers.dataReader.DataReader.getPhysicalData` or :meth:`~resistics.ioHandlers.dataReader.DataReader.getUnscaledData` methods should be used. All of these return a :class:`~resistics.dataObjects.timeData.TimeData` object. In this case, a range of samples are requested and then information about the timeseries data is printed out to the terminal.

.. literalinclude:: ../../../../examples/conventions/internalReaderExamples.py
    :linenos:
    :language: python
    :lines: 9-11
    :lineno-start: 9

.. literalinclude:: ../../_text/printInternalData.txt
    :linenos:
    :language: text

The data can be plotted by using the :meth:`~resistics.dataObjects.timeData.TimeData.view` method of :class:`~resistics.dataObjects.timeData.TimeData`. By passing a matplotlib figure, the layout of the plot can be further controlled. 

.. literalinclude:: ../../../../examples/conventions/internalReaderExamples.py
    :linenos:
    :language: python
    :lines: 13-20
    :lineno-start: 13

.. figure:: ../../../../examples/conventions/images/internalData.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Viewing internal data

To show how the internal data format can be read using numpy, first create a map between channels and the channel data files. The map is simply a Python dictionary.

.. literalinclude:: ../../../../examples/conventions/internalReaderExamples.py
    :linenos:
    :language: python
    :lines: 22-26
    :lineno-start: 22

To read in channel |Ex|, all that is required is to use the numpy fromfile method and the filename along with a specification of the data type, which is np.float32 for data in field units.

.. literalinclude:: ../../../../examples/conventions/internalReaderExamples.py
    :linenos:
    :language: python
    :lines: 28-32
    :lineno-start: 28

This method can be compared to the :class:`~resistics.ioHandlers.dataReaderInternal.DataReaderInternal` class by plotting the two on the same plot. Matplotlib can help out with this. 

.. literalinclude:: ../../../../examples/conventions/internalReaderExamples.py
    :linenos:
    :language: python
    :lines: 34-40
    :lineno-start: 34

.. figure:: ../../../../examples/conventions/images/internalData_vs_npLoad.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Internal data read in versus using numpy. There is a shift between the datasets.

As can be seen in the image, there is a shift between the two methods. This is because the get data methods of the various :class:`~resistics.ioHandlers.dataReader.DataReader` classes return data minus the average value of the data. This can be optionally turned off as in the example below.

.. literalinclude:: ../../../../examples/conventions/internalReaderExamples.py
    :linenos:
    :language: python
    :lines: 43-54
    :lineno-start: 43

Replotting the data now shows that the two are comparable.

.. figure:: ../../../../examples/conventions/images/internalDataWithAvg_vs_npLoad.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Internal data read (but without removing the average) in versus using numpy

Complete example script
~~~~~~~~~~~~~~~~~~~~~~~

For the purposes of clarity, the complete example script is shown below.

.. literalinclude:: ../../../../examples/conventions/internalReaderExamples.py
    :linenos:
    :language: python
