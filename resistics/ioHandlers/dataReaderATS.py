import os
import glob
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import numpy as np

# import from package
from resistics.ioHandlers.dataReader import DataReader


class DataReaderATS(DataReader):
    """Data reader for ATS formatted data

    For ATS files, header information is written out to an XML file. 
    
    Methods
    -------
    setParameters()
        Set data format parameters
    dataHeaders()
        Headers to read in
    readHeaders()
        Specific function for reading the headers for internal format
    lineToKeyAndValue(line)
        Separate a line into key and value with = as a delimiter

    Notes
    -----
    The end time in ATS files is actually one sample past the time of the last sample. The dataReader handles this and gives an end time corresponding to the time of the actual last sample.
    """

    def setParameters(self) -> None:
        """Set data reader parameters for ATS files"""

        self.headerF = glob.glob(os.path.join(self.dataPath, "*.xml"))
        self.dataF = glob.glob(os.path.join(self.dataPath, "*.ats"))
        self.dataByteOffset = 1024
        self.dataByteSize = 4

    def dataHeaders(self):
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

    def readHeader(self):
        """Read time data header file for internal format"""

        if len(self.headerF) > 1:
            self.printWarning(
                "More xml files than expected. Using: {}".format(self.headerF[0])
            )
        tree = ET.parse(self.headerF[0])
        root = tree.getroot()
        # get header names
        rHeaders, gHeaders, cHeadersInput, cHeadersOutput = self.dataHeaders()

        # get recording headers
        self.headers = {}
        recording = root.find("./recording")
        for rH in rHeaders:
            self.headers[rH] = recording.find(rH).text
        # get global config headers
        globalConfig = recording.find("./input/ADU07Hardware/global_config")
        for gH in gHeaders:
            self.headers[gH] = globalConfig.find(gH).text

        # get the channel headers in the input section
        self.chanHeaders = []
        for chan in root.findall(
            "./recording/input/ADU07Hardware/channel_config/channel"
        ):
            chanH = {}
            for cH in cHeadersInput:
                chanH[cH] = chan.find(cH).text
            self.chanHeaders.append(chanH)
        # get the channel headers in the output section
        outputSec = root.findall(
            "./recording/output/ProcessingTree1/output/ATSWriter/configuration/channel"
        )
        # check for old style xml file
        if len(outputSec) == 0:
            outputSec = root.findall(
                "./recording/output/ATSWriter/configuration/channel"
            )
        for chan, chanH in zip(outputSec, self.chanHeaders):
            for cH in cHeadersOutput:
                chanH[cH] = chan.find(cH).text

        # a couple of things to do: add microseconds to the times
        # remember, the actual end time is one sample back
        # if you do a calculation with the number of samples and the start time
        self.headers["start_time"] = self.headers["start_time"] + ".000000"
        self.headers["stop_time"] = self.headers["stop_time"] + ".000000"
        for chanH in self.chanHeaders:
            chanH["start_time"] = chanH["start_time"] + ".000000"

        # set the lsb applied header in chans
        # for ats files, this is not applied in the raw data files
        for idx, ch in enumerate(self.chanHeaders):
            self.chanHeaders[idx]["lsb_applied"] = False
            self.chanHeaders[idx]["ts_lsb"] = "-{}".format(
                self.chanHeaders[idx]["ts_lsb"]
            )
