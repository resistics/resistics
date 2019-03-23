import os
import glob
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict, Union, Any

# import from package
from resistics.dataObjects.timeData import TimeData
from resistics.ioHandlers.ioHandler import IOHandler
from resistics.ioHandlers.dataReader import DataReader
from resistics.utilities.utilsIO import checkAndMakeDir


class DataWriter(IOHandler):
    """Base class for data writers

    All input readers provide headers in a common format

	Attributes
	----------
    outPath : str
        The path to write to
    extension: str
        The extension to give to the data files
    dtype : data type
        Data format to write out, default is np.int32
    headers : 
        Header information to write out
    chans : List[str]
        Channels to write out
    chanMap : Dict
        Map between channel and index in channel headers
    chanHeaders : List
        Channel specific headers

	Methods
	-------
	__init__()
		Initialise writer
    getOutPath()
        Get the path to write out to
    setOutPath(path)
        Set the path to write out to
    setExtension()
        For subclasses to set their extension
    setGlobalHeadersFromKeywords(headers, keywords)
        Set the global headers 
    setChanHeadersFromKeywords(chanHeaders, keywords)
        Set channel headers
    calcStopDateTime(sampleFreq, numSamples, datetimeStart)
        Calculate time series stop time
    globalHeaderwords()
        Get a list of the global headers of interest
    chanHeaderwords()
        Get a list of the channel headers of interest
    writeTemplateHeaderFiles(chans, chanFileMap, sampleFreq, numSamples, startDate)    
        Write out a set of template header files for a case in which ASCII data without headers is to be loaded in
    writeDataset(reader, **kwargs)
        Write an existing dataset as a different format
    writeData(headers, chanHeaders, timeData, **kwargs)
        Write data based on headers, channel headers and time series data
    write(headers, chanHeaders, chanMap, timeData, **kwargs)
        Write out a dataset
    writeHeaders(headers, chans, chanMap, chanHeaders)
        Write out headers
    writeComments(comments)
        Write out comments
    writeDataFiles(chans, timeData)
        Write out time series data - not implemented in base class
    printList()
        Class status returned as list of strings   
	"""

    def __init__(self):
        self.outPath: str = ""
        # in subclasses, extension might change i.e. .ascii
        self.setExtension()
        # data type - the format of the data being written out
        self.dtype = np.int32
        # information about data being written
        self.headers: Union[Dict, None] = None
        self.chans: Union[List[str], None] = None
        self.chanMap: Union[Dict[str, int], None] = None
        self.chanHeaders: Union[List, None] = None

    def getOutPath(self) -> str:
        """Get the out path

        Parameters
        ----------
        str
            The outpath defining where data is written
        """

        return self.outPath

    def setOutPath(self, path: str) -> None:
        """Set the out path

        Parameters
        ----------
        path : str
            The new outpath defining where data is written
        """

        self.outPath = path

    def setExtension(self) -> None:
        """For subclasses to set their own extension type"""
        
        self.extension = ".dat"

    def setGlobalHeadersFromKeywords(self, headers: Dict, keywords: Dict) -> Dict:
        """Set the global headers

        Before writing out data, global headers are set. The priority order is:

        1. keywords[headername] if headername exists in keywords
        2. headers[headername] if headername exists in headers
        3. "" where the header is not defined in either keywords or headers
        
        The reason keywords takes top priority is there may be instances where the headers defined in a reader may need to be altered due to processing of time data.

        Parameters
        ----------
        headers : Dict
            Dictionary of header values
        keywords : Dict
            A dictionary of header values to overwrite those in headers
        """

        globalHeaderwords = self.globalHeaderwords()
        for gH in globalHeaderwords:
            hdrVal = ""
            if gH in headers:
                hdrVal = headers[gH]
            if gH in keywords:
                hdrVal = keywords[gH]
            headers[gH] = hdrVal
        return headers

    def setChanHeadersFromKeywords(self, chanHeaders: List, keywords: Dict) -> List:
        """Set the channel headers

        Before writing out data, channel headers are set. The priority order is:

        1. keywords[headername] if headername exists in keywords
        2. headers[headername] if headername exists in headers
        3. "" where the channel header is not defined in either keywords or headers
        
        The reason keywords takes top priority is there may be instances where the headers defined in a reader may need to be altered due to processing of time data.

        Parameters
        ----------
        chanHeaders : List
            List of channel headers
        keywords : Dict
            A dictionary of header values to overwrite those in channel headers
        """

        chanHeaderwords = self.chanHeaderwords()
        for iChan in range(0, len(chanHeaders)):
            for cH in chanHeaderwords:
                hdrVal = ""
                if cH in chanHeaders[iChan]:
                    hdrVal = chanHeaders[iChan][cH]
                if cH in keywords:
                    hdrVal = keywords[cH]
                chanHeaders[iChan][cH] = hdrVal
        return chanHeaders

    def calcStopDateTime(
        self, sampleFreq: float, numSamples: int, datetimeStart: datetime
    ) -> datetime:
        """Calculate time of last sample 
        
        Parameters
        ----------
        sampleFreq : float
            Sampling frequency in Hz of the time series data
        numSamples : int
            The number of samples in the time series data
        datetimeStart : datetime
            The time of the first sample
        """

        # calculate duration in seconds
        # numSamples - 1 because have to remove the initial sample which is taken at start time
        duration = 1.0 * (numSamples - 1) / sampleFreq
        datetimeStop = datetimeStart + timedelta(seconds=duration)
        return datetimeStop

    def globalHeaderwords(self) -> List[str]:
        """Get a list of global headerwords to write out

        Returns
        -------
        List[str]
            A list of the global header words of interest for writing out
        """

        gHeaders = [
            "sample_freq",
            "num_samples",
            "start_time",
            "start_date",
            "stop_time",
            "stop_date",
            "meas_channels",
        ]
        return gHeaders

    def chanHeaderwords(self) -> List[str]:
        """Get a list of channel headerwords to write out

        Returns
        -------
        List[str]
            A list of the global header words of interest for writing out
        """

        cHeaders = [
            "sample_freq",
            "num_samples",
            "start_time",
            "start_date",
            "stop_time",
            "stop_date",
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
            "gain_stage1",
            "gain_stage2",
            "hchopper",
            "echopper",
        ]
        return cHeaders

    def writeTemplateHeaderFiles(
        self,
        chans: List[str],
        chanFileMap: Dict,
        sampleFreq: float,
        numSamples: int,
        startDate: str,
    ):
        """Write a set of blank headers

        Blank headers might be useful for reading in ascii files where no headers are existing. By giving a few header words, many options can be set 

        Parameters
        ----------
        chans : List[str]
            List of chans (e.g. Ex, Ey, Hx, Hy, Hz)
        chanFileMap : Dict[str, str]
            Map from channel to file 
        numsamples : int
            Number of samples of data
        startdate : str
            The start date of the recording in format %Y-%m-%d %H:%M:%S
        """

        # calculate start and end datetime
        datetimeStart = datetime.strptime(startDate, "%Y-%m-%d %H:%M:%S")
        datetimeStop = self.calcStopDateTime(sampleFreq, numSamples, datetimeStart)
        # set global header words
        globalKeywords = dict()
        globalKeywords["sample_freq"] = sampleFreq
        globalKeywords["num_samples"] = numSamples
        globalKeywords["start_date"] = datetimeStart.strftime("%Y-%m-%d")
        globalKeywords["start_time"] = datetimeStart.strftime("%H:%M:%S.%f")
        globalKeywords["stop_date"] = datetimeStop.strftime("%Y-%m-%d")
        globalKeywords["stop_time"] = datetimeStop.strftime("%H:%M:%S.%f")
        globalKeywords["meas_channels"] = len(chans)

        # empty dictionary so that some headers get defaulted
        emptyDict = dict()
        # set global headers for keyword arguments
        headers = self.setGlobalHeadersFromKeywords(emptyDict, globalKeywords)
        # set channel headers for keyword arguments
        chanMap: Dict = dict()
        chanHeaders: List[Dict] = list()
        for idx, chan in enumerate(chans):
            chanMap[chan] = idx
            chanKeywords = dict(globalKeywords)
            chanKeywords["scaling_applied"] = True
            chanKeywords["ts_lsb"] = 1
            chanKeywords["gain_stage1"] = 1
            chanKeywords["gain_stage2"] = 1
            chanKeywords["hchopper"] = 0
            chanKeywords["echopper"] = 0 
            chanKeywords["pos_x1"] = 0
            chanKeywords["pos_x2"] = 1
            chanKeywords["pos_y1"] = 0
            chanKeywords["pos_y2"] = 1
            chanKeywords["pos_z1"] = 0
            chanKeywords["pos_z2"] = 1  
            chanKeywords["sensor_sernum"] = 1                     
            chanHeaders.append(chanKeywords)
        # set the chan header words
        self.setChanHeadersFromKeywords(chanHeaders, emptyDict)
        for idx, chan in enumerate(chans):   
            # amend the data file in the chan headers     
            chanHeaders[idx]["ats_data_file"] = chanFileMap[chan]
            chanHeaders[idx]["channel_type"] = chan
            for cH in chanHeaders[idx].keys():
                if chanHeaders[idx][cH] == "":
                    chanHeaders[idx][cH] = "None"
        self.writeHeaders(headers, chans, chanMap, chanHeaders, rename=False)

    def writeDataset(self, reader: DataReader, physical: bool = True, **kwargs) -> None:
        """Write out a dataset by passing a data reader

        This method is intended to transform an existing dataset into internal format

        Parameters
        ----------
        reader : DataReader
            A list of the global header words of interest for writing out
        physical : bool, optional
            An optional flag designating whether to use physical samples or not. Default is true
        """

        if self.getOutPath() == "":
            self.printError("No output filepath given", quitRun=True)
        checkAndMakeDir(self.getOutPath())
        # write using information from a reader file
        headers = reader.getHeaders()
        chanHeaders, chanMap = reader.getChanHeaders()
        # now write depending on whether scaling_applied or not
        if physical:
            # make sure dataset is written out in float and with scaling_applied header set to True
            self.dtype = np.float32
            kwargs["scaling_applied"] = True
            # write out
            self.write(
                headers, chanHeaders, chanMap, reader.getPhysicalSamples(), **kwargs
            )
        else:
            # write out unscaled samples
            self.printWarning(
                "Wrinting out of unscaled samples is not recommended due to scaling differences between the formats."
            )
            self.printWarning(
                "Dataset will be written out but problems may be encountered in the future."
            )
            self.write(
                headers, chanHeaders, chanMap, reader.getUnscaledSamples(), **kwargs
            )

    def writeData(self, headers, chanHeaders, timeData, physical: bool = True, **kwargs):
        """Write out time data 

        This method requires the user to pass global headers and chan headers explicitly.

        Parameters
        ----------
        headers : Dict
            Dictionary of headers
        chanHeaders : List
            List of channel headers
        timeData : TimeData
            Time series data to write out
        physical : bool, optional
            An optional flag designating whether the data is in field units (i.e. all scalings have been applied). This will result in the scaling_applied header being set to True. Default value for physical is True (i.e. data is assumed to be in field units).
        """

        if self.getOutPath() == "":
            self.printWarning("No output filepath given")
            return
        # make the directory
        checkAndMakeDir(self.getOutPath())
        # calculate our own cMap
        chanMap = {}
        for iChan in range(0, len(chanHeaders)):
            chanType = chanHeaders[iChan]["channel_type"]
            chanMap[chanType] = iChan
        # check if in physical units
        if physical:
            kwargs["scaling_applied"] = True
            self.dtype = np.float32
        # write the data
        self.write(headers, chanHeaders, chanMap, timeData, **kwargs)

    def write(
        self,
        headers: Dict,
        chanHeaders: List,
        chanMap: Dict,
        timeData: TimeData,
        **kwargs
    ):
        """Write out the header file

        Parameters
        ----------
        headers : Dict
            Dictionary of headers
        chanHeaders : List
            List of channel headers
        chanMap : Dict
            Maps channel to index for chanHeaders    
        timeData : TimeData
            Time series data as TimeData object        
        """

        # set global headers for keyword arguments
        headers = self.setGlobalHeadersFromKeywords(headers, kwargs)
        # set channel headers for keyword arguments
        chanHeaders = self.setChanHeadersFromKeywords(chanHeaders, kwargs)

        # now overwrite the options by checking the TimeData object
        # number of samples and sample frequency
        # Current method favours the time data object
        chans = sorted(list(timeData.chans))
        dataSizes = []
        for c in chans:
            dataSizes.append(timeData.data[c].size)
        if min(dataSizes) != max(dataSizes):
            self.printWarning(
                "Channels do not have the same number of samples: {} - {}".format(
                    ", ".join(chans), ", ".join(dataSizes)
                )
            )
            self.printWarning("Only the smallest number of samples will be written out")
        numSamples = min(dataSizes)
        if headers["num_samples"] != numSamples:
            self.printWarning(
                "Number of samples {} in headers does not match number of samples in TimeData object {}. TimeData info will be used.".format(
                    headers["num_samples"], numSamples
                )
            )
            headers["num_samples"] = numSamples
        timeData.numSamples = numSamples
        # sample freq
        if headers["sample_freq"] != timeData.sampleFreq:
            self.printWarning(
                "Sample frequency of {} Hz in headers does not match {} Hz in TimeData object".format(
                    headers["sample_freq"], timeData.sampleFreq
                )
            )
            self.printWarning("Sample frequency in TimeData object will be used")
            headers["sample_freq"] = timeData.sampleFreq

        # deal with start and end time and create datetime objects
        # the start time does not change on resampling, only the end time
        datetimeStart = datetime.strptime(
            "{} {}".format(headers["start_date"], headers["start_time"]),
            "%Y-%m-%d %H:%M:%S.%f",
        )
        datetimeStop = datetime.strptime(
            "{} {}".format(headers["stop_date"], headers["stop_time"]),
            "%Y-%m-%d %H:%M:%S.%f",
        )
        # now let's compare to the time data
        if datetimeStart != timeData.startTime:
            self.printWarning(
                "Start in headers {} does not match that in TimeData object {}. TimeData start time will be used".format(
                    datetimeStart, timeData.startTime
                )
            )
            datetimeStart = timeData.startTime
        if datetimeStop != timeData.stopTime:
            self.printWarning(
                "Stop in headers {} does not match that in TimeData object {}. TimeData stop time will be used".format(
                    datetimeStop, timeData.stopTime
                )
            )
            datetimeStop = timeData.stopTime
        # now recalculate datetime using the number of samples and compare again
        datetimeRecalc = self.calcStopDateTime(
            timeData.sampleFreq, numSamples, datetimeStart
        )
        if datetimeRecalc != datetimeStop:
            self.printWarning(
                "Note, discrepancy between stop time in given headers and those calculated from data"
            )
            self.printWarning(
                "Causes of this might be resampling or interpolation processes and the limiting of data"
            )
            self.printWarning(
                "If no resampling, interpolation or limiting of data has been performed, please check all times"
            )
            self.printWarning(
                "Stop time {} calculated from data will be used instead of that in data {}".format(
                    datetimeRecalc, datetimeStop
                )
            )
            datetimeStop = datetimeRecalc
        headers["start_date"] = datetimeStart.strftime("%Y-%m-%d")
        headers["start_time"] = datetimeStart.strftime("%H:%M:%S.%f")
        headers["stop_date"] = datetimeStop.strftime("%Y-%m-%d")
        headers["stop_time"] = datetimeStop.strftime("%H:%M:%S.%f")

        # now update all the chan headers and limit data to numSamples
        for c in chans:
            timeData.data[c] = timeData.data[c][:numSamples]
            cIndex = chanMap[c]
            chanHeaders[cIndex]["num_samples"] = headers["num_samples"]
            chanHeaders[cIndex]["sample_freq"] = headers["sample_freq"]
            chanHeaders[cIndex]["start_date"] = headers["start_date"]
            chanHeaders[cIndex]["start_time"] = headers["start_time"]
            chanHeaders[cIndex]["stop_date"] = headers["stop_date"]
            chanHeaders[cIndex]["stop_time"] = headers["stop_time"]

        # finally, check the number of measurement channels
        headers["meas_channels"] = len(chans)

        # now write out the headers and save to class variables
        self.writeHeaders(headers, chans, chanMap, chanHeaders)
        self.headers = headers
        self.chans = chans
        self.chanMap = chanMap
        self.chanHeaders = chanHeaders
        # write out comment file
        self.writeComments(timeData.comments)
        # write out the data files
        self.writeDataFiles(chans, timeData)

    def writeHeaders(
        self,
        headers: Dict[str, Any],
        chans: List[str],
        chanMap: Dict[str, int],
        chanHeaders: List[Dict],
        rename: bool = True,
    ) -> bool:
        """Write out the header file

        Parameters
        ----------
        headers : Dict
            Dictionary of headers
        chans : List[str]
            Channels as a list of strings
        chanMap : Dict
            Maps channel to index for chanHeaders
        chanHeaders : List
            List of channel headers
        rename : bool, optional
            Rename the output ats_data_files. Default is True and this is the case when writing out data which has been read in from a different source with pre-existing headers. However, if creating template header files, then set this to False.
        """

        # write out the global headers
        f = open(os.path.join(self.getOutPath(), "global.hdr"), "w")
        f.write("HEADER = GLOBAL\n")
        globalHeaderwords = self.globalHeaderwords()
        for gH in globalHeaderwords:
            f.write("{} = {}\n".format(gH, headers[gH]))
        f.close()

        # write out the channel headers
        chanHeaderwords = self.chanHeaderwords()
        for idx, c in enumerate(chans):
            cf = open(
                os.path.join(self.getOutPath(), "chan_{:02d}.hdr".format(idx)), "w"
            )
            cf.write("HEADER = CHANNEL\n")
            # use the chanMap to get the index of the chanHeaders list
            cIndex = chanMap[c]
            # change the data file if necessary
            if rename:
                chanHeaders[cIndex]["ats_data_file"] = "chan_{:02d}{}".format(
                    idx, self.extension
                )
            # write out all the header words
            for cH in chanHeaderwords:
                cf.write("{} = {}\n".format(cH, chanHeaders[cIndex][cH]))
            cf.close()
        return True

    def writeComments(self, comments: List[str]) -> None:
        """Write out a comments file

        Parameters
        ----------
        comments : List[str]
            List of strings with data comments
        """

        f = open(os.path.join(self.getOutPath(), "comments.txt"), "w")
        for c in comments:
            f.write("{}\n".format(c))
        # add another comment about writing
        f.write("Dataset written to {} on {}".format(self.getOutPath(), datetime.now()))
        f.close()

    def writeDataFiles(self, chans, timeData) -> None:
        """Write out data files"""

        raise NotImplementedError(
            "Write data files not implemented in base class. Only child classes should ever be instantiated."
        )

    def printList(self) -> List[str]:
        """Class information as a list of strings

        Returns
        -------
        out : List[str]
            List of strings with information
        """

        textLst = []
        textLst.append("Output file path for data = {}".format(self.getOutPath()))
        # if it exists, print out the headers
        if self.headers:
            textLst.append("Global Headers")
            textLst.append(self.headers)
        # if exists, print out a list of chans
        if self.chans:
            textLst.append("Channels found:")
            textLst.append(self.chans)
        # if exists, print out the chanMap
        if self.chanMap:
            textLst.append("Channel Map")
            textLst.append(self.chanMap)
        # if it exists, print out the chanHeaders
        if self.chanHeaders:
            textLst.append("Channel Headers")
            for c in self.chans:
                textLst.append(c)
                textLst.append(self.chanHeaders[self.chanMap[c]])
        return textLst
