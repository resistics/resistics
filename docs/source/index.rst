.. Resistics documentation master file, created by
   sphinx-quickstart on Fri Dec 14 22:53:34 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. meta::
   :description: resistics, a Python 3 magnetotelluric processing package
   :keywords: magnetotellurics, electromagnetics, geophysics, statistics, robust regression

Welcome to resistics
--------------------
Resistics is an open-source, native Python 3 package for the processing of magnetotelluric (MT) data. It incorporates standard robust regression methods and adopts a modular approach to processing which allows for customisation and future improvements to be quickly adopted. 

.. figure:: _static/images/welcome.jpg
    :align: center
    :alt: alternate text
    :figclass: align-center

    Magnetotellurics in the rain  

Latest news
~~~~~~~~~~~
**2020-03-15:** Resistics 0.0.6 has been released to the python package repository. No new features were added between 0.0.6rc1 and 0.0.6. There were no major changes apart from those to fix bugs and issues. Additionally, an initial set of unit tests has been added and more will be added in the future. 

.. warning::
    
    Version 0.0.6 of resistics is no longer backwards compatible with scripts written for resistics versions prior to 0.0.6.dev3. Scripts written using older versions of the package will fail due to broken imports. All documentation has been re-written to match the new structure. 

The latest version of resistics can be downloaded using pip in the normal way. 

.. code-block:: text
    
    python -m pip install --upgrade resistics

For more information on what has changed, please view the :doc:`changelog <changelog>`.

About
~~~~~
Resistics began as a set of Python classes to help analyse noisy MT timeseries data acquired in northern Switzerland through increased use of statistics and time window based features. Since then, it has grown into a MT data processing package. The name is an amalgamation of resistivty and statistics...resistics!

Whilst other codes exist, resistics was written for the following purpose:

- Freely accessible (MIT licence), open-source and cross platform
- Compatible with several :doc:`time data formats <formats/timeseries>` and corresponding :doc:`calibration data <formats/calibration>`
- :doc:`Configuration files <features/configuration>` and :doc:`dataset history <features/comments>`
- :doc:`Dataset investigation <features/statistics>`, tools for dealing with noisy MT data
- Well documented with :ref:`clear explanation of methods <tutorial:Processing theory>`, :ref:`tutorials <tutorial:Working through the tutorial>`, :doc:`advanced examples <advanced>` and a :doc:`cookbook <cookbook>` for showcasing lower level API usage.

Audience
~~~~~~~~
Resistics is intended for people who use magnetotelluric methods to estimate the subsurface resistivity. This may be for the purposes of furthering geological understanding, for geothermal prospecting or for other purposes.

Getting started
~~~~~~~~~~~~~~~
Installation instructions are provided :doc:`here <installation>`. The quickest way to get started with resistics is to install it from pip, the python package repository, and continue through to the :doc:`tutorial <tutorial>`.

.. code-block:: text
    
    python -m pip install --user resistics

Resistics uses a number of conventions which are described :doc:`here <conventions>`. Find out about the useful features of resistics in the :doc:`features <features>` section. Information about supported data and calibration formats is provided in the :doc:`formats <formats>` section.

The :doc:`tutorial <tutorial>` section covers the resistics project environment and basic processing of magnetotelluric data. More advanced processing, including pre-processing of timeseries data, remote reference processing and remote reference statistics are detailed in the :doc:`advanced <advanced>` section. Specialised functionality or examples of using lower level APIs will be added in the :doc:`cookbook <cookbook>` as and when it is developed. The :doc:`case studies <case-studies>` will cover the use of resistics to process complete field surveys. A complete API reference can be found :doc:`here <modules>`. 

A changelog and backlog for future resistics development can be accessed :doc:`here <changelog>`. Useful magnetotelluric references are provided in the :doc:`Bibliography <bibliography>`.

For those interested in seeing who is contributing to the project and how resistics can be cited see the :doc:`Credits <credits>`. Anyone who wants to donate can do so :doc:`here <donate>`.

Open-source
~~~~~~~~~~~
Resistics is available for free under the MIT licence. The resistics source code can be found in the `GitHub repository <https://github.com/resistics/resistics>`_. Contributors are welcome. 


.. toctree::
    :maxdepth: 4
    :titlesonly:
    :glob:
    :hidden:

    Home <self>
    installation.rst
    conventions.rst 
    features.rst
    formats.rst   
    tutorial.rst
    advanced.rst
    cookbook.rst
    case-studies.rst
    modules.rst
    changelog.rst   
    bibliography.rst   
    credits.rst
    donate.rst
    contact.rst     
