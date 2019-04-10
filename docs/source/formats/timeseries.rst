.. |Ex| replace:: E\ :sub:`x`
.. |Ey| replace:: E\ :sub:`y`
.. |Hx| replace:: H\ :sub:`x`
.. |Hy| replace:: H\ :sub:`y`
.. |Hz| replace:: H\ :sub:`z`

Timeseries data
---------------

One of the complexities of magnetotelluric data processing is the various different data formats that exist. Resistics tries to simplify this complexity and offer inter-operability of the various data formats. Examples are given in the following sections of reading, writing and viewing timeseries data of varying formats.

Reading
~~~~~~~

Resistics currently supports five data formats for reading:

- :doc:`ATS <timeseries/ATS-timeseries>`
- :doc:`SPAM <timeseries/SPAM-timeseries>`
- :doc:`Phoenix <timeseries/Phoenix-timeseries>`
- :doc:`ASCII <timeseries/ascii-timeseries>`
- :doc:`resistics internal binary format <timeseries/internal-binary-format>` based on numpy save

Each of these data formats has its own peculiarities, outlined in the individual sections. Resistics aims to give users a consistent output through the :class:`~resistics.ioHandlers.dataReader.DataReader` class. All data readers can return either unscaled or physically scaled samples. Whilst the meaning of unscaled samples differs from format to format, when requesting physical samples from a data reader, the data is returned in the following units:

- Electric channels |Ex|, |Ey| are in mV/km
- Magnetic channels |Hx|, |Hy| and |Hz| are in mV

.. note::

    Time data folders must have one of the following in their name to be found when using the project environment.

    - meas
    - run
    - phnx

Writing
~~~~~~~

Writing out of timeseries is supported in resistics in the following formats:

- :doc:`ASCII <timeseries/ascii-timeseries>`
- :doc:`resistics internal binary format <timeseries/internal-binary-format>`

The reason for supporting an internal binary format is two-fold. Firstly, it is based on numpy save, which allows easy reading and portability outside of the resistics environment as long as there is access to Python and numpy. Secondly, the headers are all in ASCII format, which makes it easier to read the various recording metadata in comparison to binary formatted header information. 

.. note::

    ASCII data files will be significantly larger than binary ones. ASCII is best for portability if the data needs to be used elsewhere outside of Python, but the internal format can easily be read in using Python's numpy package.

Interpolating to the second
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::
    
    For best results with resistics, timeseries data should be sampled on the second. This is not true for all timeseries formats, for example, SPAM data can be sampled off the second. Resistics includes an interpolation to second function to deal with this issue. Please read :doc:`this <timeseries/interpolation-to-second>` for more information.

Data which is not sampled on the second should be interpolated on the second to ensure best results. This can be achieved using inbuilt functionality. 

For an example of a dataset not sampled on the second, consider recording at 10 Hz, with the first sample at 0.05 seconds. Then the sample times will be:

.. code-block:: text

    0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 ...

Interpolating to the second will change the sample times to:

.. code-block:: text

    0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00 1.10 1.20 ...

Whenever operating with multiple different file formats in one project, it is best practice to ensure that all datasets are sampled on the second prior to estimating impedance tensors.

.. toctree::
    :maxdepth: 2
    :titlesonly:
    :glob:
    :hidden:

    timeseries/ATS-timeseries.rst
    timeseries/SPAM-timeseries.rst
    timeseries/Phoenix-timeseries.rst
    timeseries/ascii-timeseries.rst
    timeseries/internal-binary-format.rst
    timeseries/interpolation-to-second.rst