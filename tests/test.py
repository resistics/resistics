from magpy.utilities.utilsConfig import loadConfig, copyDefaultConfig
from magpy.dataObjects.configData import ConfigData
import os

# path = "configTest.ini"
# copyDefaultConfig(path)

# c = loadConfig("configTest.ini")
# print(c)
# print(c.defaults)

c = loadConfig()
print(c)
print(c.defaults)
print(c["Decimation"].defaults)

cdata = ConfigData()
print(cdata)