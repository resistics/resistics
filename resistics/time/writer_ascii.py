import os
import numpy as np
from typing import List

from resistics.time.writer import TimeWriter
from resistics.time.data import TimeData


class TimeWriterAscii(TimeWriter):
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

    def writeDataFiles(self, chans: List[str], timeData: TimeData):
        """Write the data files

        Parameters
        ----------
        chans : List[str]
            List of channels
        timeData : TimeData
            The time data to write out
        """
        self.extension = ".ascii"
        for idx, c in enumerate(chans):
            writePath = os.path.join(
                self.getOutPath(), "chan_{:02d}{}".format(idx, self.extension)
            )
            # this could probably be made quicker - numpy savetxt maybe
            dataF = open(writePath, "w")
            size = timeData[c].size
            for i in range(0, size):
                dataF.write("{:9f}\n".format(timeData[c][i]))
            dataF.close()
