Features
--------

Resistics implements a number of useful features, a few of those are described in more detail in this section.

Configuration files are an optional feauture. Resistics itself uses a default configuration file to set basic processing parameters. However, users can provide their own configuration files. In fact, the same project can be processed using multiple different configurations, allowing parameters to be varied and the outcomes easily compared.

Any timeseries, frequency or statistic data written out using Resistics is written out with comments. These comments summarise the processing sequence applied to the data and the configuration used. This allows for straightforward traceability and useful transparency.

Statistics are another useful feature baked into Resistics. Statistics can be used to mask windows (remove windows from processing) but importantly, can be used to investigate and understand the raw MT time series data.

.. toctree::
    :maxdepth: 2
    :titlesonly:
    :hidden:

    features/1_config.rst
    features/2_comments.rst
    features/3_statistics.rst
    features/4_remoteStatistics.rst