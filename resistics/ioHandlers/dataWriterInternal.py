import os
import numpy as np

# import from package
from resistics.ioHandlers.dataWriter import DataWriter


class DataWriterInternal(DataWriter):
    """Write out binary data files

    This is simply header files and binary data files. The header file saved is relevant only to this software and needs to be read in using DataReaderInternal.
    The header file means less processing to read the header information

    Methods
    -------
    writeDataFiles(chans, timeData)
        Write out time series data
    """

    def writeDataFiles(self, chans, timeData) -> None:
        for idx, c in enumerate(chans):
            writePath = os.path.join(
                self.getOutPath(), "chan_{:02d}{}".format(idx, self.extension)
            )
            dataF = open(writePath, "wb")
            dataF.write(timeData.data[c].astype(self.dtype).tobytes())
            dataF.close()
