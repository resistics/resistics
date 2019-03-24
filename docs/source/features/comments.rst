Comments
--------

To help traceability and reproducibility of processing results, most datasets which are written out by resistics are written with associated comment files. Comment files detail the parameters with which the data was processed. An example of a comments file for a spectra dataset is provided below. As can be seen, there is a significant detail about the processing sequence and the parameters used. 

Only timeseries data, spectra data and statistic data are written out with comment files. Transfer function files do not have associated comments files as they are usually calculated out from a set of data sources.

.. literalinclude:: ../_text/comments.txt
    :linenos:
    :language: text