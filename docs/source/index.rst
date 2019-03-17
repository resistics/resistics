.. Resistics documentation master file, created by
   sphinx-quickstart on Fri Dec 14 22:53:34 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to resistics
--------------------

Resistics is a native Python 3 package for the processing of magnetotelluric (MT) data. It incorporates standard robust processing methods and adopts a modular approach to processing which allows for customisation and future improvements to be quickly adopted. 

.. figure:: _images/welcome.jpg
    :align: center
    :alt: alternate text
    :figclass: align-center

    Magnetotellurics in the rain  

About
~~~~~
Resistics began as a set of python classes to help analyse noisy MT timeseries data acquired in northern Switzerland through increased use of statistics and time window based features. Since then, it has grown into a MT data processing package. The name is an amalgamation of resistivty and statistics...resistics!

Audience
~~~~~~~~

Resistics is intended for people who use magnetotelluric methods to estimate the subsurface resistivity. This may be for the purposes of furthering geological understanding, for geothermal prospecting or for other purposes.

Getting started
~~~~~~~~~~~~~~~

Read more about the magnetotelluric method and resistics in the :doc:`introduction <introduction>`. Installation instructions are provided :doc:`here <installation>`. The quickest way to get started with resistics is to install it from pip, the python package repository, and continue through to the :doc:`tutorial <tutorial>`.

.. code-block:: bash
    
    python -m pip install resistics

Find out about the useful features of resistics in the :doc:`features <features>` section. Resistics uses a number of conventions which are described :doc:`here <conventions>`.

The :doc:`tutorial <tutorial>` section covers the resistics project environment and basic processing of magnetotelluric data. More advanced processing, including pre-processing of timeseries data, remote reference processing and remote reference statistics are detailed in the :doc:`advanced <advanced>` section. More specialised functionality or examples will be added in the :doc:`cookbook <cookbook>` as and when it is developed. A complete API reference can be found :doc:`here <api>`. 

A roadmap for future development can be accessed :doc:`here <roadmap>`. For those interested in contributing to the project, information can be found in this section.

Useful magnetotelluric references are provided in the :doc:`References <references>` section.

Open-source
~~~~~~~~~~~

Resistics is available for free under the MIT licence. The resistics source code can be found in the `GitHub repository <https://github.com/resistics/resistics>`_. Contributors are welcome. 


Index
=====

* :ref:`modindex`

.. toctree::
    :maxdepth: 4
    :titlesonly:
    :glob:
    :hidden:

    introduction.rst
    installation.rst
    features.rst
    conventions.rst    
    tutorial.rst
    advanced.rst
    cookbook.rst
    modules.rst
    roadmap.rst        
    references.rst    
