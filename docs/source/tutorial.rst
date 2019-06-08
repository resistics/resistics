Tutorial
--------

Magnetotelluric theory
~~~~~~~~~~~~~~~~~~~~~~
This tutorial assumes knowledge of magnetotelluric theory. However, a quick refresher is provided. 

The magnetotelluric method uses natural variations in the geoelectric and geomagnetic fields to estimate the electrical properties of the subsurface. Electrodes and induction coils measure the time varying fields. Using these measurements, magnetotellurics poses a linear, time invariant system in the frequency domain.

.. math::
    :nowrap:

    \begin{eqnarray}
    E_x & = & Z_{xx} H_x + Z_{xy} H_y \\
    E_y & = & Z_{yx} H_x + Z_{yy} H_y 
    \end{eqnarray} 

|Zxx|, |Zxy|, |Zyx| and |Zyy| are the components of the 2-D impedance tensor with magnetic channels as input and electric channels as output. The components of the impedance tensor describe volumetrically averaged electrical properites of the subsurface. The frequency for which the impedance tensor is calculated defines the  

In addition, the tipper, 

Good references for magnetotelluric theory are:

* Wikipedia: https://en.wikipedia.org/wiki/Magnetotellurics
* Practical Magnetotellurics (Simpson and Bahr)
* The Magnetotelluric Method (Chave and Jones)
* Models and Methods of Magnetotellurics (Berdichevsky and Dmitriev)

First steps with resistics
~~~~~~~~~~~~~~~~~~~~~~~~~~
To install resistics, follow the instructions in the :doc:`installation <installation>` section.

The following sections detail basic features and requirements of resistics. Resistics is a project based package, which means the first step in learning how to use the package is to set up a new project and place the time data files in the appropriate location. A dataset has been made available for new starters to practice with. This includes two recordings, one at 128Hz and the other at 4096Hz. The dataset is available here:

`Dataset link <https://1drv.ms/f/s!ApKs8ZhVotKMavU3EpQTeEwuxoc>`_

| **For more information about this dataset, see the paper:**
| `3-D analysis and interpretation of magnetotelluric data from the Aluto-Langano geothermal field, Ethiopia <https://academic.oup.com/gji/article/202/3/1923/614429>`_
| F. Samrock  A. Kuvshinov  J. Bakker  A. Jackson  S. Fisseha
| *Geophysical Journal International, Volume 202, Issue 3, September, 2015* 
| *Pages 1923–1948* 
| https://doi.org/10.1093/gji/ggv270


In most cases, only the API for the project functionality will be of interest. However, for even more advanced examples, have a look at the :doc:`advanced <advanced>` and :doc:`cookbook <cookbook>` sections, which will demonstrate more advanced processing of magnetotelluric data or usage of lower level resistics API. 

.. |Ex| replace:: E\ :sub:`x`
.. |Ey| replace:: E\ :sub:`y`
.. |Hx| replace:: H\ :sub:`x`
.. |Hy| replace:: H\ :sub:`y`
.. |Hz| replace:: H\ :sub:`z`
.. |Zxy| replace:: Z\ :sub:`xy`
.. |Zxx| replace:: Z\ :sub:`xx`
.. |Zyx| replace:: Z\ :sub:`yx`
.. |Zyy| replace:: Z\ :sub:`yy`
.. |fs| replace:: f\ :sub:`s`

.. toctree::
    :maxdepth: 2
    :titlesonly:
    :glob:
    :hidden:

    tutorial/project-environment.rst
    tutorial/viewing-data.rst
    tutorial/up-and-running.rst
    tutorial/tipper.rst
    tutorial/viewing-spectra.rst
    tutorial/multiple-spectra.rst
    tutorial/configuration-files.rst
    tutorial/statistics.rst
    tutorial/masks.rst
    tutorial/mask-processing.rst
    tutorial/recap-and-deeper-dive.rst
    tutorial/date-time-constraints.rst