import os
import numpy as np
from typing import List

from resistics.time.writer import TimeWriter
from resistics.time.data import TimeData

class TimeWriterInternal(TimeWriter):
    """Write out binary data files

    This is simply header files and binary data files. The header file saved is relevant only to this software and needs to be read in using DataReaderInternal.
    The header file means less processing to read the header information

    Methods
    -------
    writeDataFiles(chans, timeData)
        Write out time series data
    """

    def writeDataFiles(self, chans: List[str], timeData: TimeData) -> None:
        """Write out data files in internal format

        Parameters
        ----------
        chans : List[str]
            List of channels
        timeData : TimeData
            The time data
        """
        for idx, c in enumerate(chans):
            writePath = os.path.join(
                self.getOutPath(), "chan_{:02d}{}".format(idx, self.extension)
            )
            dataF = open(writePath, "wb")
            dataF.write(timeData[c].astype(self.dtype).tobytes())
            dataF.close()
