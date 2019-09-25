.. role:: python(code)
   :language: python

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

Remote reference statistics
---------------------------

Remote reference statistics are a means of understanding how adding a remote reference site to the processing of magnetotelluric data is changing the solution. There are currently five remote reference statistics. 

- "RR_coherence"
- "RR_coherenceEqn"
- "RR_absvalEqn"
- "RR_transferFunction"
- "RR_resPhase"

More information on each of these is given in :doc:`remote reference statistics <../features/remote-statistics>`.

Begin by loading the project and calculate the remote statistics using the :meth:`~resistics.project.projectStatistics.calculateRemoteStatistics` of module :mod:`~resistics.project.projectStatistics`. The :meth:`~resistics.project.projectStatistics.calculateRemoteStatistics` method will calculate the remote reference statistics for :python:`sites=["M6"]` and at sampling frequencies :python:`sampleFreqs=[128]` using the remote site "Remote". In this instance, all the remote statistics are being calculated.

Work in progress...