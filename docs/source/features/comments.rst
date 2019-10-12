Comments
--------

To help traceability and reproducibility of processing results, time, spectra and statistic data which are written out by resistics are saved with associated comment files. Comment files detail the parameters with which the data was processed. 

.. important::

    Other types datasets, such as masks or transfer function data are not written out with comments. This is because they are calculated out from multiple datasets rather than a single chain, making the management of comments more difficult. However, this functionality may be added in the future.

An example of a comments file for SPAM timeseries data resampled and interpolated on to the second is shown below. The scaling applied to the data is detailed in the comments file as well as any other data modifications.

.. literalinclude:: ../_static/examples/features/comments/features_time_comments.txt
    :language: text

When frequency spectra data is calculated from the preprocessed timeseries data, the spectra data comments file includes the timeseries data flow too and maintains the complete flow of the data. In the following spectra data comments file, the calibration applied to the timeseries data is shown as well as the decimation and windowing.  

.. literalinclude:: ../_static/examples/features/comments/features_spec_comments.txt
    :language: text

Going one step further and calculating statistics from the spectra data adds even more information to the dataset history.

.. literalinclude:: ../_static/examples/features/comments/features_stat_comments.txt
    :language: text

The comment files associated with the timeseries, spectra and statistic data respectively help users double check and reproduce the data processing sequence.