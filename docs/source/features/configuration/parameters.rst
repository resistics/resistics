Configuration Parameters
------------------------

A detailed explanation of parameters available for configuration is given below.

General
~~~~~~~
These parameters are general and not tied to any specific section.

.. topic:: Parameter: name

    :Type: string 
    :Default value: "default"
    :Description: A name for the configuration. This will be saved in comment files. 

Calibration
~~~~~~~~~~~
Parameters related to calibration data and calibrating of time data.

.. topic:: Parameter: extend

    :Type: bool (True or False) 
    :Default value: True
    :Description: Extrapolate out calibration data to cover a greater range of frequencies.

.. topic:: Parameter: useTheoretical

    :Type: bool (True or False) 
    :Default value: False
    :Description: Theoretical calibration for magnetic channels when a matching calibration file is not found.

Decimation
~~~~~~~~~~
Parameters to define the decimation scheme. Note that depending on the length of timeseries recording, not all decimation levels may be calculated.

.. topic:: Parameter: numlevels

    :Type: int
    :Default value: 7 (min=1, max=10)
    :Description: Number of decimation levels.

.. topic:: Parameter: minsamples

    :Type: int 
    :Default value: 100 (min=50)
    :Description: Minimum number of samples required to continue decimation.

Frequencies
~~~~~~~~~~~
Evaluation frequency related parameters. If evaluation frequencies are not explicitly supplied, then they are automatically selected internally.

.. topic:: Parameter: frequencies

    :Type: List[float] (comma separated list of floats)
    :Default value: None
    :Description: Evaluation frequencies specified as a comma separated list of floats.

.. topic:: Parameter: perlevel

    :Type: int 
    :Default value: 7 (min=1, max=10)
    :Description: Number of evaluation frequencies per decimation level.

Window
~~~~~~
timeseries windowing parameters. This defines how the timeseries data will be windowed and the overlap between windows. Resistics will automatically calculate window sizes using windowfactor and overlapfraction. If for any decimation level, the calculation results in a window size less than minwindowsize or overlap lower the minoverlapsize, the window and overlap sizes will be set to their minimum allowable values.

Window and overlap sizes can be set explicitly using the windowsizes and overlap sizes. If windowsizes are explicitly set, overlapsizes needs to be set too.

.. topic:: Parameter: minwindows

    :Type: int
    :Default value: 5 (min=1)
    :Description: Minimum windows required for a decimation level before decimation is ended.

.. topic:: Parameter: windowfactor

    :Type: int 
    :Default value: 2 (min=1)
    :Description: The factor which defines window size. The window size is calculated as: sampling frequency at decimation level / windowfactor.
    
.. topic:: Parameter: minwindowsize

    :Type: int
    :Default value: 512 (min=32)
    :Description: The minimum allowable size of a window in samples.

.. topic:: Parameter: minoverlapsize

    :Type: int 
    :Default value: 128 (min=8)
    :Description: The minimum allowable overlap size.

.. topic:: Parameter: overlapfraction

    :Type: float
    :Default value: 0.25 (min=0, max=0.5)
    :Description: The fraction of the windowsize to use as overlap size.

.. topic:: Parameter: windowsizes

    :Type: List[int] of size equal to number of decimation levels 
    :Default value: None
    :Description: Explicitly specify the window sizes as a comma separated list. 
    
.. topic:: Parameter: overlapsizes

    :Type: List[int] of size equal to number of decimation levels 
    :Default value: None
    :Description: Explicitly specify the overlap sizes as a comma separated list. 

Spectra
~~~~~~~
Parameters related to calculating timeseries fourier spectra.

.. topic:: Parameter: specdir

    :Type: string
    :Default value: "spectra"
    :Description: The spectra directory to write out to. This allows each configuration file to be related to a different run of the data.
    
.. topic:: Parameter: applywindow

    :Type: bool (True or False)
    :Default value: True
    :Description: Window the data before performing the fourier transform.
    
.. topic:: Parameter: windowfunction

    :Type: string. One of "barthann", "bartlett", "blackman", "blackmanharis", "bohman", "chebwin", "hamming", "hann", "nuttall", "parzen".
    :Default value: "hamming"
    :Description: Window function to apply before doing the fourier transform

Statistics
~~~~~~~~~~
Parameters related to calculating timeseries fourier spectra.

.. topic:: Parameter: stats

    :Type: List[str]
    :Default value: "coherence", "transferFunction"
    :Description: Comma separated list of statistics to calculate.

.. topic:: Parameter: remotestats

    :Type: List[str]
    :Default value: "RR_coherence" , "RR_transferFunction"
    :Description: Comma separated list of remote reference statistics to calculate

Solver
~~~~~~~~~~
Solution parameters

.. topic:: Parameter: intercept

    :Type: bool (True or False)
    :Default value: False
    :Description: Boolean flag for including an intercept in the least squares problem.

.. topic:: Parameter: windowfunc

    :Type: str. One of "barthann", "bartlett", "blackman", "blackmanharis", "bohman", "chebwin", "hamming", "hann", "nuttall", "parzen".
    :Default value: "hamming"
    :Description: Window function used for 


