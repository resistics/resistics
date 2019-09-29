Installation
------------

The easiest way to install resistics is to use the Python package manager. Information about the resistics package can be found at the python package index: https://pypi.org/project/resistics/. Resistics has several dependencies which are outlined :ref:`below <installation:Dependencies>`. The final option is to :ref:`install resistics from source <installation:Installing from source>`. 

Python pip
~~~~~~~~~~
Resistics is the `Python package repository <https://pypi.org/project/resistics/>`_ and can be installed from using pip as follows:

.. code-block:: bash
    
    python -m pip install --user resistics

This is the simplest method of installing resistics and deals with all the dependencies.

.. warning::

    resistics requires Python 3.6 or higher. Python can be downloaded from the `Python downloads <https://www.python.org/downloads/>`_  page and is available for all major operating systems.

Dependencies
~~~~~~~~~~~~
Resistics uses a number of Python 3 libraries. These are:

- numpy 
- scipy
- pyfftw 
- configObj
- matplotlib 

These will be installed automatically when installing resistics through the Python pip package manager.

pyfftw does have an external dependency on FFTW (Fastest Fourier Transform in the West) library. More information about pyfftw can be found here:
https://hgomersall.github.io/pyFFTW/.

Installing from source
~~~~~~~~~~~~~~~~~~~~~~
The source code for resistics can be found on `GitHub repository <https://github.com/resistics/resistics>`_. 


.. toctree::
    :maxdepth: 1
    :titlesonly:
    :glob:
    :hidden: