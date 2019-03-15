Configuration Parameters
========================

A detailed explanation of the configuration parameters is given below:

General
~~~~~~~
These parameters are general and not tied to any specific section

.. code-block:: text
   
   name
   type: string 
   default: "default"
   A name for the configuration. This will be saved in comment files. 


Calibration
~~~~~~~~~~~
Parameters related to calibration.

.. code-block:: text

   useTheoretical 
   type: bool (True or False)
   default: False
   Theoretical calibration for magnetic channels when a matching calibration file is not found

Decimation
~~~~~~~~~~
Parameters the determine the decimation schemes.

.. code-block:: text

   numlevels
   type: int
   default: 7 (min=1, max=10)
   Number of decimation levels

   minsamples
   type: int
   default: 100 (min=50)
   Minimum number of samples required to continue decimation

Frequencies
~~~~~~~~~~~
Evaluation frequency related parameters.

.. code-block:: text
   
   frequencies
   type: List[float]
   default: None
   Evaluation frequencies specified as a comma separated list of floats.

   perlevel
   type: int
   default: 7 (min=1, max=10)
   Frequencies per decimation level

If evaluation frequencies are not explicitly suppled, then they are automatically selected internally.

Window
~~~~~~
Time series windowing parameters.

.. code-block:: text
   
   minwindows
   type: int
   default: 5 (min=1)
   Minimum windows required for a decimation level before decimation is ended.

   windowfactor
   type: int
   default: 2 (min=1)
   The factor which defines window size

   minwindowsize
   type: int
   default: 512 (min=32)
   The minimum allowable size of a window in samples

   minoverlapsize
   type: int
   default: 128 (min=128)
   The minimum allowable overlap size

   overlapfraction
   type: float
   default: 0.25 (min=0, max=0.5)
   The fraction of the windowsize to use as overlap size

   windowsizes
   type: List[int] of size equal to number of decimation levels
   default: None
   Explicitly specify the window sizes as a comma separated list 

   overlapsizes
   type: List[int] of size equal to number of decimation levels
   default: None
   Explicitly specify the overlap sizes as a comma separated list 

Resistics will automatically calculate window sizes using windowfactor and overlapfraction. If for any decimation level, the calculation results in a window size less than minwindowsize or overlap lower the minoverlapsize, the window and overlap sizes will be set to their minimum allowable values.

Window and overlap sizes can be set explicitly using the windowsizes and overlap sizes. If windowsizes are explicitly set, overlapsizes needs to be set too.

Spectra
~~~~~~~
Parameters related to calculating time series fourier spectra.

.. code-block:: text
   
   specdir
   type: string
   default: spectra
   The spectra directory to write out to. This allows each configuration file to be related to a different run of the data.

   applywindow
   type: bool (True or False)
   default: True
   Window the data before performing the fourier transform

   windowfunction
   type: string
   default: hamming
   One of "barthann", "bartlett", "blackman", "blackmanharis", "bohman", "chebwin", "hamming", "hann", "nuttall", "parzen".

Statistics
~~~~~~~~~~
Parameters related to calculating time series fourier spectra.

.. code-block:: text
   
   stats
   type: List[str]
   default: "coherence", "transferFunction"
   Comma separated list of statistics to calculate

   remotestats
   type: List[str]
   default: "coherence", "transferFunction"
   Comma separated list of remote reference statistics to calculate

Solver
~~~~~~~~~~
Solution parameters

.. code-block:: text
   
   stats
   type: List[str]
   default: "coherence", "transferFunction"
   Comma separated list of statistics to calculate

   remotestats
   type: List[str]
   default: "coherence", "transferFunction"
   Comma separated list of remote reference statistics to calculate


