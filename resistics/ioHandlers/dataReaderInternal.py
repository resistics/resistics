import os
import glob
from datetime import datetime, timedelta
import numpy as np
from typing import List, Tuple

# import from package
from resistics.ioHandlers.dataReader import DataReader
from resistics.utilities.utilsIO import checkFilepath, lineToKeyAndValue


class DataReaderInternal(DataReader):
    """Data reader for internal formatted data

    Internal formatted data is straightforward. Header information is read in from a global header file and header files for each channel, all of them ascii formatted file. Each channel has its own data file written out using numpy's binary write function and with a .dat extension.

    As raw data is not usually in the internal data format, to avoid any problems, the following workflow is suggested:

    - Get physical data from the raw data files (ATS, SPAM, Phoenix)
    - Perform any pre-processing required
    - Save as internal format, ensuring to set the scaling_applied channel header for every channel to True. This will avoid any further scaling when the data is read in again.
    
    Methods
    -------
    dataHeaders()
        Headers to read in
    readHeaders()
        Specific function for reading the headers for internal format
    lineToKeyAndValue(line)
        Separate a line into key and value with = as a delimiter
    """

    def dataHeaders(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Return the data headers in the internal file format
        
        Returns
        -------
        recordingHeaders : List[str]
            Headers with information about the recording
        globalHeaders : List[str]
            Common headers with information about the recording
        channelHeadersInput : List[str]
            Channel setup headers
        channelHeadersOutput : List[str]
            Channel recording headers      
        """

        # note, in comparison to ats headers, this also has one called scaling_applied
        recordingHeaders = ["start_time", "start_date", "stop_time", "stop_date"]
        globalHeaders = ["meas_channels", "sample_freq"]
        channelHeadersInput = ["gain_stage1", "gain_stage2", "hchopper", "echopper"]
        channelHeadersOutput = [
            "start_time",
            "start_date",
            "sample_freq",
            "num_samples",
            "ats_data_file",
            "sensor_type",
            "channel_type",
            "ts_lsb",
            "scaling_applied",
            "pos_x1",
            "pos_x2",
            "pos_y1",
            "pos_y2",
            "pos_z1",
            "pos_z2",
            "sensor_sernum",
        ]
        return (
            recordingHeaders,
            globalHeaders,
            channelHeadersInput,
            channelHeadersOutput,
        )

    def readHeader(self) -> None:
        """Read time data header file for internal format"""

        # first read the global headers
        # look in headerF for global.hdr
        if os.path.join(self.dataPath, "global.hdr") not in self.headerF:
            self.printError(
                "Global header not found. The global.hdr file is required", quitRun=True
            )
        globalF = open(os.path.join(self.dataPath, "global.hdr"))
        lines = globalF.readlines()
        globalF.close()
        # ignore the first line
        lines.pop(0)
        # now go through and get the headers
        self.headers = {}
        for l in lines:
            if l == "":
                continue
            key, val = lineToKeyAndValue(l.strip())
            self.headers[key] = val

        # now want to deal with the chan headers
        numChans = int(self.headers["meas_channels"])
        self.chanHeaders = []
        for iChan in range(0, numChans):
            chanF = open(os.path.join(self.dataPath, "chan_{:02d}.hdr".format(iChan)))
            lines = chanF.readlines()
            chanF.close()
            # remove first line and read the headers for the channel
            lines.pop(0)
            cHeader = {}
            for l in lines:
                if l == "":
                    continue
                key, val = lineToKeyAndValue(l.strip())
                cHeader[key] = val
            self.chanHeaders.append(cHeader)

        # finally, read the comments
        commentPath = os.path.join(self.dataPath, "comments.txt")
        if checkFilepath(commentPath):
            f = open(commentPath, "r")
            self.comments = f.readlines()
            for idx, comment in enumerate(self.comments):
                self.comments[idx] = comment.rstrip()
            f.close()
        else:
            self.comments = []