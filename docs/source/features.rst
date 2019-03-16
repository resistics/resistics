Features
--------

Resistics implements a number of useful features, a few of which are described in more detail in this section. The features are intended to aid traceability and reproducibility of results, ease the general processing flow of magnetotelluric data and give more insight into magnetotelluric timeseries data. 

:doc:`Configuration files <features/configuration>` are an optional feauture. Resistics itself uses a default configuration file to set basic processing parameters. However, users can provide their own configuration files. In fact, the same project can be processed using multiple different configurations, allowing parameters to be varied and the outcomes easily compared.

Any timeseries, frequency or statistic data written out using resistics is written out with :doc:`comment files <features/comments>`. These comments summarise the processing sequence applied to the data and the configuration used. This allows for straightforward traceability and useful transparency.

:doc:`Statistics <features/statistics>` are another useful feature baked into Resistics. Statistics can be used to mask windows (remove windows from processing) but importantly, can be used to investigate and understand the raw MT timeseries data.

.. toctree::
    :maxdepth: 2
    :titlesonly:
    :hidden:

    features/configuration.rst
    features/comments.rst
    features/statistics.rst
    features/remoteStatistics.rst