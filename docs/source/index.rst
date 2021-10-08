.. resistics documentation master file, created by
   sphinx-quickstart on Thu May  6 19:32:52 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to resistics's documentation!
=====================================

Soon resistics will be upgrading to version 1.0.0. This will be a breaking
change versus version 0.0.6. Currently, the newest version is available as a
development release for those who are intersted in experimenting with its
updated feature set.

Until version 1.0.0 is released as a stable version, the existing
documentation for 0.0.6 will remain at resistics.io.

Why?
----

Resistics has been re-written from the ground up to tackle several limitations
of the previous version, namely

- Processing time
- Limited traceability
- Lack of extensibility
- Difficult to maintain

The new version of resistics aims to tackle all of these issues through better
coding practises, putting extensibility at the heart of its design and moving to
a modern deployment pipeline.

What's new?
-----------
The literal answer is everything as this is a from scratch rewrite, which has
taken some features of the previous version but combined them with new
capabilities.

For most users, notable changes are related to configuration of processing
flows and the carving out of specific data format readers into a separate
package.

Advanced users will be able to take advantage of opportunities to write their
own solvers or processors and a greater ability to customise and extend
resistics.

Other smaller changes include:

- Moving to JSON for metadata as this is a universal format
- Moving from matplotlib to plotly for plots as they are more interactive

What's missing?
---------------

The first thing to note is that time series data reader for various formats have
been removed from resistics and placed in a sister package named
resistics-readers. This is to remove any coupling of data format support to core
resistics releases. It is hoped that resistics-readers will receive more
community support as knowledge about the various data formats in the
magnetotelluric world is distributed around the community.

Statistics are another capability of resistics 0.0.6 that is missing. The
intention is to re-introduce these shortly and additionally, make it easier for
users to write their own features to extract.

Masks are also missing and these will be re-introduced with statistics.

Next steps
----------

.. toctree::
   :maxdepth: 3
   :caption: User guide:

   getting-started.rst
   custom-process.rst

.. toctree::
   :maxdepth: 3
   :caption: API reference:

   resistics.rst


Index
=====

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
