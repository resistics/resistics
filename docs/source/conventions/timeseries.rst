Timeseries data
---------------

Resistics currently supports four data formats:

- :doc:`ATS <timeseries/ATS-timeseries>`
- :doc:`SPAM <timeseries/SPAM-timeseries>`
- :doc:`Phoenix <timeseries/Phoenix-timeseries>`
- An internal :doc:`resistics <timeseries/resistics-timeseries>` data format which is based on numpy save

Each of these data formats has its own peculiarities, outlined in the individual sections. Resistics aims to simplify the complexity and give users a consistent output through the :doc:`data reader <../api/ioHandlers.dataReader>` API.

The possibility of supporting an ASCII data has been considered. However, as ASCII data files will be significantly larger than binary ones, binary ones have been favoured. If you have data which is in none of these formats, please see the instructions here on how to convert ASCII data into the internal resistics data format.

.. toctree::
    :maxdepth: 2
    :titlesonly:
    :glob:
    :hidden:

    timeseries/ATS-timeseries.rst
    timeseries/SPAM-timeseries.rst
    timeseries/Phoenix-timeseries.rst
    timeseries/resistics-timeseries.rst
    timeseries/Interpolating-to-second.rst