Configuration files
===================

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

    21:02:15 ConfigData: Configuration Parameters:
    21:02:15 ConfigData: Name = dec8_5
    21:02:15 ConfigData: Calibration:
    21:02:15 ConfigData:    usetheoretical = False
    21:02:15 ConfigData:    Defaulted options = usetheoretical
    21:02:15 ConfigData: Decimation:
    21:02:15 ConfigData:    numlevels = 8
    21:02:15 ConfigData:    minsamples = 100
    21:02:15 ConfigData:    Defaulted options = minsamples
    21:02:15 ConfigData: Frequencies:
    21:02:15 ConfigData:    perlevel = 5
    21:02:15 ConfigData:    frequencies = []
    21:02:15 ConfigData:    Defaulted options = frequencies
    21:02:15 ConfigData: Window:
    21:02:15 ConfigData:    minwindowsize = 256
    21:02:15 ConfigData:    minoverlapsize = 64
    21:02:15 ConfigData:    minwindows = 5
    21:02:15 ConfigData:    windowfactor = 2.0
    21:02:15 ConfigData:    overlappercentage = 0.25
    21:02:15 ConfigData:    windowsizes = []
    21:02:15 ConfigData:    overlapsizes = []
    21:02:15 ConfigData:    Defaulted options = minwindows, windowfactor, overlappercentage, windowsizes, overlapsizes
    21:02:15 ConfigData: Spectra:
    21:02:15 ConfigData:    specdir = dec8_5
    21:02:15 ConfigData:    applywindow = True
    21:02:15 ConfigData:    windowfunc = hamming
    21:02:15 ConfigData:    Defaulted options = applywindow, windowfunc
    21:02:15 ConfigData: Statistics:
    21:02:15 ConfigData:    stats = ['coherence', 'transFunc']
    21:02:15 ConfigData:    remotestats = ['coherenceRR', 'transFuncRR']
    21:02:15 ConfigData:    Defaulted options = stats, remotestats
    21:02:15 ConfigData: Solver:
    21:02:15 ConfigData:    intercept = False
    21:02:15 ConfigData:    boostrap = True
    21:02:15 ConfigData:    windowfunc = hamming
    21:02:15 ConfigData:    Defaulted options = intercept, boostrap, windowfunc

.. toctree::
    :maxdepth: 2
    :titlesonly:
    :glob:
    :hidden:

    configuration/parameters.rst



