Configuration files
-------------------

Resistics allows users to set various parameters through configuration files. The package itself comes with its default configuration. Users can change certain parameters by specifying them in a separate configuration file. Below is the default configuration:

.. literalinclude:: ../../../examples/features/resisticsDefaultConfig.ini
    :linenos:
    :language: text

Configuration files are separated into several sections which describe which part of the process the parameters affect. The parameters are detailed in the :ref:`features/configuration:Configuration parameters` section. The first thing to note is that all configuration files have a name. This is to help traceability. Configuration names are always entered into dataset comments.

A good way to begin creating a custom configuration file is to copy the default parameterisation. This can be done by using the inbuilt :meth:`~resistics.utilities.utilsConfig.copyDefaultConfig` functionality:

.. literalinclude:: ../../../examples/features/copyconfig.py
    :linenos:
    :language: python
    :lines: 1-3
    :lineno-start: 1

When providing a custom configuration file, only the settings which will be changed need to be entered. Below is an example of a user specified configuration file

.. literalinclude:: ../../../examples/tutorial/tutorialConfig.ini
    :linenos:
    :language: text

When using resistics, information about the configuraiton being used can be printed to the terminal. This will detail which parameters have been specified by the user and which ones are being defaulted. 

.. code-block:: text

    23:17:45 ConfigData: Configuration file = Default configuration
    23:17:45 ConfigData: Configuration name = default
    23:17:45 ConfigData: Flags:
    23:17:45 ConfigData: customfrequencies = False
    23:17:45 ConfigData: customwindows = False
    23:17:45 ConfigData: Configuration Parameters:
    23:17:45 ConfigData: Name = default
    23:17:45 ConfigData: Calibration:
    23:17:45 ConfigData:    extend = True
    23:17:45 ConfigData:    usetheoretical = False
    23:17:45 ConfigData:    Defaulted options = extend, usetheoretical
    23:17:45 ConfigData: Decimation:
    23:17:45 ConfigData:    numlevels = 7
    23:17:45 ConfigData:    minsamples = 100
    23:17:45 ConfigData:    Defaulted options = numlevels, minsamples
    23:17:45 ConfigData: Frequencies:
    23:17:45 ConfigData:    frequencies = []
    23:17:45 ConfigData:    perlevel = 7
    23:17:45 ConfigData:    Defaulted options = frequencies, perlevel
    23:17:45 ConfigData: Window:
    23:17:45 ConfigData:    minwindows = 5
    23:17:45 ConfigData:    windowfactor = 2.0
    23:17:45 ConfigData:    minwindowsize = 512
    23:17:45 ConfigData:    minoverlapsize = 128
    23:17:45 ConfigData:    overlapfraction = 0.25
    23:17:45 ConfigData:    windowsizes = []
    23:17:45 ConfigData:    overlapsizes = []
    23:17:45 ConfigData:    Defaulted options = minwindows, windowfactor, minwindowsize, minoverlapsize, overlapfraction, windowsizes, overlapsizes
    23:17:45 ConfigData: Spectra:
    23:17:45 ConfigData:    specdir = spectra
    23:17:45 ConfigData:    applywindow = True
    23:17:45 ConfigData:    windowfunc = hamming
    23:17:45 ConfigData:    Defaulted options = specdir, applywindow, windowfunc
    23:17:45 ConfigData: Statistics:
    23:17:45 ConfigData:    stats = ['coherence', 'transferFunction']
    23:17:45 ConfigData:    remotestats = ['RR_coherence', 'RR_transferFunction']
    23:17:45 ConfigData:    Defaulted options = stats, remotestats
    23:17:45 ConfigData: Solver:
    23:17:45 ConfigData:    intercept = False
    23:17:45 ConfigData:    boostrap = True
    23:17:45 ConfigData:    windowfunc = hamming
    23:17:45 ConfigData:    Defaulted options = intercept, boostrap, windowfunc

Configuration parameters
~~~~~~~~~~~~~~~~~~~~~~~~

A detailed explanation of parameters available for configuration is given below. 

General
^^^^^^^
These parameters are general and not tied to any specific section.

.. topic:: name

    :Type: string 
    :Default value: "default"
    :Description: A name for the configuration. This will be saved in comment files. 

Calibration
^^^^^^^^^^^
Parameters related to calibration data and calibrating of time data.

.. topic:: extend

    :Type: bool (True or False) 
    :Default value: True
    :Description: Extrapolate out calibration data to cover a greater range of frequencies.

.. topic:: usetheoretical

    :Type: bool (True or False) 
    :Default value: False
    :Description: Use a theoretical calibration for magnetic channels when a matching calibration file is not found.

Decimation
^^^^^^^^^^
Parameters to define the decimation scheme to use. Note that depending on the length of timeseries recording, not all decimation levels may be calculated.

.. topic:: numlevels

    :Type: int
    :Default value: 7 (min=1, max=10)
    :Description: Number of decimation levels.

.. topic:: minsamples

    :Type: int 
    :Default value: 100 (min=50)
    :Description: Minimum number of samples required to continue decimation.

Frequencies
^^^^^^^^^^^
Evaluation frequency related parameters. If evaluation frequencies are not explicitly supplied, then they are automatically selected internally.

.. topic:: frequencies

    :Type: List[float] (comma separated list of floats)
    :Default value: None
    :Description: Evaluation frequencies specified as a comma separated list of floats.

.. topic:: perlevel

    :Type: int 
    :Default value: 7 (min=1, max=10)
    :Description: Number of evaluation frequencies per decimation level.

Window
^^^^^^
Timeseries windowing parameters. This defines how the timeseries data will be windowed and the overlap between windows. Resistics will automatically calculate window sizes using windowfactor and overlapfraction. If for any decimation level, the calculation results in a window size less than minwindowsize or overlap lower than the minoverlapsize, the window and overlap sizes will be set to their minimum allowable values.

Window and overlap sizes can be set explicitly using the windowsizes and overlap sizes. If windowsizes are explicitly set, overlapsizes needs to be set too.

.. topic:: minwindows

    :Type: int
    :Default value: 5 (min=1)
    :Description: Minimum windows required for a decimation level before decimation is ended.

.. topic:: windowfactor

    :Type: int 
    :Default value: 2 (min=1)
    :Description: The factor which defines window size. The window size is calculated as: sampling frequency at decimation level / windowfactor.
    
.. topic:: minwindowsize

    :Type: int
    :Default value: 512 (min=32)
    :Description: The minimum allowable size of a window in samples.

.. topic:: minoverlapsize

    :Type: int 
    :Default value: 128 (min=8)
    :Description: The minimum allowable overlap size.

.. topic:: overlapfraction

    :Type: float
    :Default value: 0.25 (min=0, max=0.5)
    :Description: The fraction of the windowsize to use as overlap size.

.. topic:: windowsizes

    :Type: List[int] of size equal to number of decimation levels 
    :Default value: None
    :Description: Explicitly specify the window sizes as a comma separated list. 
    
.. topic:: overlapsizes

    :Type: List[int] of size equal to number of decimation levels 
    :Default value: None
    :Description: Explicitly specify the overlap sizes as a comma separated list. 

Spectra
^^^^^^^
Parameters related to calculating timeseries fourier spectra.

.. topic:: specdir

    :Type: string
    :Default value: "spectra"
    :Description: The spectra directory to write out to. This allows each configuration file to be related to a different run of the data.
    
.. topic:: applywindow

    :Type: bool (True or False)
    :Default value: True
    :Description: Window the data before performing the fourier transform.
    
.. topic:: windowfunction

    :Type: string. One of "barthann", "bartlett", "blackman", "blackmanharis", "bohman", "chebwin", "hamming", "hann", "nuttall", "parzen".
    :Default value: "hamming"
    :Description: Window function to apply before doing the fourier transform

Statistics
^^^^^^^^^^
Parameters related to calculating timeseries fourier spectra.

.. topic:: stats

    :Type: List[str]
    :Default value: "coherence", "transferFunction"
    :Description: Comma separated list of statistics to calculate.

.. topic:: remotestats

    :Type: List[str]
    :Default value: "RR_coherence" , "RR_transferFunction"
    :Description: Comma separated list of remote reference statistics to calculate

Solver
^^^^^^
Solution parameters

.. topic:: intercept

    :Type: bool (True or False)
    :Default value: False
    :Description: Boolean flag for including an intercept in the least squares problem.

.. topic:: windowfunc

    :Type: str. One of "barthann", "bartlett", "blackman", "blackmanharis", "bohman", "chebwin", "hamming", "hann", "nuttall", "parzen".
    :Default value: "hamming"
    :Description: Window function used for 




