import os
import numpy as np

# import from package
from resistics.ioHandlers.dataWriter import DataWriter


class DataWriterAscii(DataWriter):
    """Write out ascii data files

    This is simply header files and ascii data files. The header file saved is relevant only to this software and needs to be read in using DataReaderInternal.
    The header file means less processing to read the header information

	Methods
	-------
    setExtension()
        Set the data file extension
    writeDataFiles(chans, timeData)
        Write out time series data
	"""

    def setExtension(self) -> None:
        """For subclasses to set their own extension type"""
        
        self.extension = ".ascii"

    def writeDataFiles(self, chans, timeData):
        self.extension = ".ascii"
        for idx, c in enumerate(chans):
            writePath = os.path.join(
                self.getOutPath(), "chan_{:02d}{}".format(idx, self.extension)
            )
            # this could probably be made quicker - numpy savetxt maybe
            dataF = open(writePath, "w")
            size = timeData.data[c].size
            for i in range(0, size):
                dataF.write("{:9f}\n".format(timeData.data[c][i]))
            dataF.close()
