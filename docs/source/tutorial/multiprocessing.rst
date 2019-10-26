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
.. role:: python(code)
   :language: python

Multiprocessing
---------------

Processing magnetotelluric field surveys can be demanding if there many sites and long recordings. To help speed up processing, resistics supports multiprocessor computation using the python multiprocessing module (`read more here <https://docs.python.org/3/library/multiprocessing.html>`_). Multiprocessing can be used in the following steps:

- Spectra calculations
- Statistic calculations
- Transfer function estimation (robust regression)

For Windows users, multiprocessing requires scripts using resistics and multiprocessing to be protected using :python:`if __name__ == '__main__'`. For a more detailed explanation of why, please see the `offical python documentation <https://docs.python.org/3/library/multiprocessing.html#multiprocessing-programming>`_. What this means is shown below:



Multiprocessing is activated by specifying the :python:`ncores` keyword in the following methods:

- :meth:`~resistics.project.spectra.calculateSpectra`
- :meth:`~resistics.project.statistics.calculateStatistics`
- :meth:`~resistics.project.transfer.processProject` or :meth:`~resistics.project.transfer.processSite`

The optional argument :python:`ncores` is expected to be an integer. It is recommend to use less cores than cores available on the system. This will leave spare processing power for system tasks and for interaction with the system.

Alternatively, the ncores options can be set in configuration files, where it can be either a subparameter or a global parameter. For example:


In the above example, ncores is a subparameter of spectra, meaning only spectra calculations will take advantage of multiprocessing. The ncores option can be supplied to the following sections:

- Spectra
- Statistics
- Solver

To set a global ncores option, which will be used in any methods that support multiprocessing, the ncores option should be set in the global section as illustrated below.


The below script and configuration runs spectra, statistic and transfer function calculations on 6 cores. 


Running this script 5 times using 6 cores gives an average runtime of xxx. Running the same script, but on a single core (5 times again) gives an average runtime of xxx.



