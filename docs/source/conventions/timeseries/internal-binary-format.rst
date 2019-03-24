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


Internal binary format
----------------------

A fair question to ask is why introduce another data format rather than use a pre-existing data format. The downsides of writing out in a pre-existing data format were the following:

- Incorrect implementation 
- No control of specification
- Makes the data no more portable

The internal format was chosen to achieve two main goals:

- A binary format that allows easy portability
- Headers written out in ascii that can be quickly checked

Therefore, numpy save was chosen as the method of writing the data. This means that the data is portable and can be opened by anyone with Python and the numpy package. 

Internal format data folders contain the following files: 

.. code-block:: text

    meas_2012-02-10_11-05-00 
    ├── 059_V01_2012-02-10_11-05-00_0.xml 
    ├── 059_V01_C00_R000_TEx_BL_4096H.ats   
    ├── 059_V01_C01_R000_TEy_BL_4096H.ats 
    ├── 059_V01_C02_R000_THx_BL_4096H.ats 
    ├── 059_V01_C02_R000_THy_BL_4096H.ats              
    └── 059_V01_C02_R000_THz_BL_4096H.ats 



