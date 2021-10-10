Reading data
------------

The main resistics package supports two time data formats and two calibration
data formats.

For time data:

- ASCII (including compressed ASCII, e.g. bz2)
- numpy .npy

Where possible, it is recommended to use the numpy data format for time data
as this is quicker to read from. Whilst it is a binary format, it is portable
and well supported by the numpy package.

For calibration data, resistics supports:

- Text file calibration data
- JSON calibration data

The structure of these two calibration data formats can be seen in the relevant
examples.


.. note::

    Support for other data formats is provided by the resistics-readers package.
    This includes support for Metronix ATS data, SPAM RAW data, Phoenix TS data,
    Lemi data and potentially more in the future.
