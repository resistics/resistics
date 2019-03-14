Installation
============

Python pip
~~~~~~~~~~
Resistics can be installed from using python pip as follows:

.. code-block:: bash
    
    python -m pip install resist

This is the simplest method of installing Resistics and deals with all the dependencies.

Dependencies
~~~~~~~~~~~~
Resistics is a Python3 and above only code. Python3 can be found at https://www.python.org and is available for all major operating systems.

Resistics uses a number of Python3 libraries. These are:

- numpy
- scipy
- matplotlib
- pyfftw

These can be installed using the pip package manager for Python3.

pyfftw does have an external dependency on FFTW (Fast Fourier Transform in the West) library. More information about pyfftw can be found here:
https://hgomersall.github.io/pyFFTW/

Please see installation information for each major operating system below.

Linux
~~~~~
To install magpy on Linux, please follow the below steps:

- sudo apt-get install python3
- sudo apt-get install libfftw3-dev
- python3 -m pip install --user Resistics

The final command will install the required dependencies for running Resistics in Python3.

Mac
~~~
Resistics can be installed on Mac as follows:

- Install Python3 from the python website https://www.python.org
- Install fftw by using the brew package manager, brew install fftw. For more information about brew, please visit https://brew.sh/
- python3 -m pip install --user Resistics

Windows
~~~~~~~
The easiest way to install magpy on Windows is to install it using pip.


.. toctree::
    :maxdepth: 1
    :titlesonly:
    :glob:
    :hidden: