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

First steps with Resistics
~~~~~~~~~~~~~~~~~~~~~~~~~~
To install Resistics, follow the instructions in the Install section.

The following sections detail basic and advanced features and requirements of Resistics. Resistics is a project based package, which means the first step in learning Resistics is to set up a new project and place the time data files in the appropriate location. A dataset has been made available for new starters to practice with. This includes two recordings, one at 128Hz and the other at 4096Hz. The dataset is available here:

Dataset link.

In most cases, only the API for the project functionality will be of interest. However, for even more advanced examples, have a look at the :doc:`cookbook <cookbook>` section, which will explicitly use the lower level Resistics API. 

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

    tutorial/projectEnvironment.rst
    tutorial/viewingData.rst
    tutorial/firstProject.rst
    tutorial/tipper.rst
    tutorial/viewingSpectra.rst
    tutorial/multipleSpectra.rst
    tutorial/configurationFiles.rst
    tutorial/statistics.rst
    tutorial/masks.rst
    tutorial/maskProcessing.rst