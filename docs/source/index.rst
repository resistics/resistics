.. Resistics documentation master file, created by
   sphinx-quickstart on Fri Dec 14 22:53:34 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. meta::
   :description: resistics, a Python 3 magnetotelluric processing package
   :keywords: magnetotellurics, electromagnetics, geophysics, statistics, robust regression

Welcome to resistics
--------------------
Resistics is a native Python 3 package for the processing of magnetotelluric (MT) data. It incorporates standard robust processing methods and adopts a modular approach to processing which allows for customisation and future improvements to be quickly adopted. 

.. figure:: _images/welcome.jpg
    :align: center
    :alt: alternate text
    :figclass: align-center

    Magnetotellurics in the rain  

Latest news
~~~~~~~~~~~
**2019-09-24:** Resistics is currently in beta mode. Updates are being made regularly. Currently the focus of updates for release 0.0.6 is:

- More complete documentation with more examples. The tutorial is complete, but more examples are coming to cover advanced methods such as pre-processing, remote reference processing, inter-operability, remote reference statistics, intersite transfer functions. 
- Speeding up of processing
- Support for Phoenix data

There is a chance that method and module names may change in the near future as they are an artefact of how the code grew up. For more information, please :doc:`contact us <contact>`.

The latest version of resistics can be downloaded by specifying 0.0.6.dev1 when downloading using pip. 

.. code-block:: bash
    
    python -m pip install resistics==0.0.6.dev1

About
~~~~~
Resistics began as a set of Python classes to help analyse noisy MT timeseries data acquired in northern Switzerland through increased use of statistics and time window based features. Since then, it has grown into a MT data processing package. The name is an amalgamation of resistivty and statistics...resistics!

Audience
~~~~~~~~
Resistics is intended for people who use magnetotelluric methods to estimate the subsurface resistivity. This may be for the purposes of furthering geological understanding, for geothermal prospecting or for other purposes.

Getting started
~~~~~~~~~~~~~~~
Read more about the magnetotelluric method and resistics in the :doc:`Introduction <introduction>`. Installation instructions are provided :doc:`here <installation>`. The quickest way to get started with resistics is to install it from pip, the python package repository, and continue through to the :doc:`tutorial <tutorial>`.

.. code-block:: bash
    
    python -m pip install --user resistics

Resistics uses a number of conventions which are described :doc:`here <conventions>`. Find out about the useful features of resistics in the :doc:`features <features>` section. Information about supported data and calibration formats is provided in the :doc:`formats <formats>` section.

The :doc:`tutorial <tutorial>` section covers the resistics project environment and basic processing of magnetotelluric data. More advanced processing, including pre-processing of timeseries data, remote reference processing and remote reference statistics are detailed in the :doc:`advanced <advanced>` section. More specialised functionality or examples will be added in the :doc:`cookbook <cookbook>` as and when it is developed. A complete API reference can be found :doc:`here <modules>`. 

A roadmap for future development can be accessed :doc:`here <roadmap>`. Useful magnetotelluric references are provided in the :doc:`Bibliography <bibliography>`.

For those interested in seeing who is contributing to the project and how resistics can be cited see the :doc:`Credits <roadmap>`. Anyone who wants to donate can do so :doc:`here <donate>`.


Open-source
~~~~~~~~~~~
Resistics is available for free under the MIT licence. The resistics source code can be found in the `GitHub repository <https://github.com/resistics/resistics>`_. Contributors are welcome. 


.. toctree::
    :maxdepth: 4
    :titlesonly:
    :glob:
    :hidden:

    Home <self>
    introduction.rst
    installation.rst
    conventions.rst 
    features.rst
    formats.rst   
    tutorial.rst
    advanced.rst
    cookbook.rst
    modules.rst
    roadmap.rst   
    bibliography.rst   
    credits.rst
    donate.rst
    contact.rst     
