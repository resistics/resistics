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

Advanced
--------

This section introduces some of the more advanced features of resistics. These advanced processing methods still use the high-level resistics API. Examples of using lower level APIs can be seen in the :doc:`Cookbook <cookbook>`. For data reading, the low level API is outlined in :doc:`time series data <formats/timeseries>`. To read in various calibration file formats, see :doc:`calibration data <formats/calibration>`.

More detail is given about the advanced processing methods below and in the corresponding subsections.

Pre-processing
~~~~~~~~~~~~~~
There are several use cases requiring pre-processing of time series data. Common ones include scaling of channel data, resampling of time series data, interpolating time data on to the second and others. In the project environment, these are managed by the :meth:`~resistics.project.time.preProcess` method of the :mod:`~resistics.project.time` module.

.. toctree::
    :maxdepth: 2
    :titlesonly:
    :glob:

    advanced/pre-process.rst

Remote reference processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Remote reference processing is no longer an advanced method. However, the example included in this section demonstrates inter-operability of different data formats, remote reference processing, remote reference statistics and more, which are all discussed in further detail in the given links.

.. toctree::
    :maxdepth: 2
    :titlesonly:
    :glob:

    advanced/remote-reference.rst
    advanced/remote-reference-statistics.rst
    advanced/remote-reference-with-config.rst

Intersite processing
~~~~~~~~~~~~~~~~~~~~
Intersite processing can be useful in certain cases. For more details on intersite processing and why this is useful, see the paper,

| `The Telluric-Magnetotelluric Method <https://www.researchgate.net/publication/260919094_The_Telluric-Magnetotelluric_Method>`_
| John F. Hermance, Richard E. Thayer
| *Geophysics*
| *Volume 40 Issue 4* 
| *Pages 664-668* 
| https://doi.org/10.1190/1.1440557

and more recently,

| `Introducing inter-site phase tensors tosuppress galvanic distortion in the telluric method <https://https://earth-planets-space.springeropen.com/articles/10.1186/s40623-015-0327-7>`_
| Jenneke Bakker, Alexey Kuvshinov, Friedemann Samrock, Alexey Geraskin, Oleg Pankratov
| *Earth Planets and Space*
| *Volume 67 Article 160*
| https://doi.org/10.1186/s40623-015-0327-7

Examples are given in the below link.

.. toctree::
    :maxdepth: 2
    :titlesonly:
    :glob:

    advanced/intersite-transfer-function.rst
