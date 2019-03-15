Comments
========

To help traceability and reproducibility of processing results, all datasets which are written out by Resistics are written with associated comment files. Comment files detail the parameters with which the data was processed. An example of a comments file for a spectra dataset is provided below. As can be seen, there is a significant detail about the processing sequence and the parameters used. 

Only time series data, spectra data and statistic data are written out with comment files. Transfer function files do not have associated comments files as they are usually calculated out from an amalgamation of files.

.. code-block:: text

    Unscaled data 2016-03-23 02:35:00 to 2016-03-24 01:54:59.992188 read in from measurement exampleProject2\timeData\M1\meas_2016-03-23_02-35-00, samples 0 to 10751999
    Sampling frequency 128.0
    Removing gain and scaling electric channels to mV/km
    Remove zeros: False, remove nans: False, remove average: True
    ---------------------------------------------------
    Calculating project spectra
    Using configuration with name dec8_5 in configuration file ex1_04b_config.ini
    Channel Ex not calibrated
    Channel Ey not calibrated
    Channel Hx calibrated with calibration data from file exampleProject2\calData\MFS06e612.TXT
    Channel Hy calibrated with calibration data from file exampleProject2\calData\MFS06e613.TXT
    Channel Hz not calibrated
    Decimating with 8 levels and 5 frequencies per level
    Evaluation frequencies for this level 32.0, 22.62741699796952, 16.0, 11.31370849898476, 8.0
    Windowing with window size 256 samples and overlap 64 samples
    Time data decimated from 128.0 Hz to 32.0 Hz, new start time 2016-03-23 02:35:00, new end time 2016-03-24 01:54:59.968750
    Evaluation frequencies for this level 5.65685424949238, 4.0, 2.82842712474619, 2.0, 1.414213562373095
    Windowing with window size 256 samples and overlap 64 samples
    Time data decimated from 32.0 Hz to 8.0 Hz, new start time 2016-03-23 02:35:00, new end time 2016-03-24 01:54:59.875000
    Evaluation frequencies for this level 1.0, 0.7071067811865475, 0.5, 0.35355339059327373, 0.25
    Windowing with window size 256 samples and overlap 64 samples
    Time data decimated from 8.0 Hz to 1.0 Hz, new start time 2016-03-23 02:35:00, new end time 2016-03-24 01:54:59
    Evaluation frequencies for this level 0.17677669529663687, 0.125, 0.08838834764831843, 0.0625, 0.044194173824159216
    Windowing with window size 256 samples and overlap 64 samples
    Time data decimated from 1.0 Hz to 0.25 Hz, new start time 2016-03-23 02:35:00, new end time 2016-03-24 01:54:56
    Evaluation frequencies for this level 0.03125, 0.022097086912079608, 0.015625, 0.011048543456039804, 0.0078125
    Windowing with window size 256 samples and overlap 64 samples
    Time data decimated from 0.25 Hz to 0.03125 Hz, new start time 2016-03-23 02:35:00, new end time 2016-03-24 01:54:28
    Evaluation frequencies for this level 0.005524271728019902, 0.00390625, 0.002762135864009951, 0.001953125, 0.0013810679320049755
    Windowing with window size 256 samples and overlap 64 samples
    Time data decimated from 0.03125 Hz to 0.0078125 Hz, new start time 2016-03-23 02:35:00, new end time 2016-03-24 01:54:28
    Evaluation frequencies for this level 0.0009765625, 0.0006905339660024878, 0.00048828125, 0.0003452669830012439, 0.000244140625
    Windowing with window size 256 samples and overlap 64 samples
    Spectra data written out to exampleProject2\specData\M1\meas_2016-03-23_02-35-00\dec8_5 on 2019-02-23 18:03:39.284153
    ---------------------------------------------------