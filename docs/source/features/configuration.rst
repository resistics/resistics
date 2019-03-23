Configuration files
-------------------

Resistics allows users to set various parameters through configuration files. The package itself comes with its default configuration. Users can change certain parameters by specifying them in a configuration file. Below is the default configuration:

.. literalinclude:: ../../../examples/features/resisticsDefaultConfig.ini
    :linenos:
    :language: text

Configuration files are separated into several sections which describe which part of the process the parameters affect. The parameters are detailed in the :doc:`configuration parameters <configuration/parameters>` section. The first thing to note is that all configuration files have a name. This is to help traceability. Configuration names are always entered into dataset comments.

A good way to begin creating a custom configuration file is to copy the default parameterisation. This can be done by using inbuilt functionality:

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

.. toctree::
    :maxdepth: 2
    :titlesonly:
    :glob:
    :hidden:

    configuration/parameters.rst



