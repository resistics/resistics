import os
import glob
from datetime import datetime, timedelta
import numpy as np
from typing import List, Tuple

from resistics.time.data import TimeData
from resistics.time.reader_internal import TimeReaderInternal
from resistics.time.clean import removeZerosSingle, removeNansSingle


class TimeReaderAscii(TimeReaderInternal):
    """Data reader for ascii formatted data

    The ASCII data reader reads ascii data files and internally formatted header files. No further scaling is applied to the data values in either getUnscaledSamples or getPhysicalSamples. All the data is assumed to be in the correct units.

    In fact, if the data does not have to be calibrated, the units could be anything as long as they are internally consistent.

    Methods
    -------
    setParameters()
        Set data format parameters    
    getUnscaledSamples()
        Get the raw unscaled samples from an ascii file
    getPhysicalSamples()
        Get data in field units. Note: no further scaling is applied in this function, ascii data is assumed to be in field units
    """

    def setParameters(self) -> None:
        """Set data reader parameters

        This will vary for the different data formats. By default, setup for the internal data format.
        """
        self.headerF = glob.glob(os.path.join(self.dataPath, "*.hdr"))
        self.dataF = glob.glob(os.path.join(self.dataPath, "*.ascii"))
        self.dataByteOffset = 0
        self.dataByteSize = 4

    def getUnscaledSamples(self, **kwargs) -> TimeData:
        """Get raw data from ascii data file

        This function simply reads the lines which match the samples to be read
        
        Parameters
        ----------
        chans : List[str], optional
            List of channels to return if not all are required
        startSample : int, optional
            First sample to return
        endSample : int, optional
            Last sample to return

        Returns
        -------
        TimeData
            Time data object
        """
        # initialise chans, startSample and endSample with the whole dataset
        options = self.parseGetDataKeywords(kwargs)
        # get samples - this is inclusive
        dSamples = options["endSample"] - options["startSample"] + 1

        # loop through chans and get data
        data = {}
        for chan in options["chans"]:
            # check to make sure channel exists
            self.checkChan(chan)
            # get data file
            dFile = os.path.join(self.dataPath, self.getChanDataFile(chan))
            # read the lines
            dataChan = np.zeros(shape=(dSamples), dtype=np.float32)
            with open(dFile) as dF:
                for il, line in enumerate(dF):
                    if il > options["endSample"]:
                        break
                    if il >= options["startSample"] and il <= options["endSample"]:
                        dIndex = il - options["startSample"]
                        dataChan[dIndex] = float(line.strip())
            # set the data
            data[chan] = dataChan

        # get data start and stop time
        startTime, stopTime = self.sample2time(
            options["startSample"], options["endSample"]
        )
        # dataset comments
        comments = []
        comments.append(
            "Unscaled data {} to {} read in from measurement {}, samples {} to {}".format(
                startTime,
                stopTime,
                self.dataPath,
                options["startSample"],
                options["endSample"],
            )
        )
        comments.append("Sampling frequency {}".format(self.getSampleFreq()))
        if len(self.comments) > 0:
            comments = self.comments + comments
        return TimeData(
            sampleFreq=self.getSampleFreq(),
            startTime=startTime,
            stopTime=stopTime,
            data=data,
            comments=comments,
        )

    def getPhysicalSamples(self, **kwargs):
        """Get ascii data scaled to physical values
        
        Warnings
        --------
        No scaling happens in getPhysicalSamples. Ascii data is assumed to be properly scaled to mV for magnetic channels and mV/km for electric channels (i.e. field units)

        Parameters
        ----------
        chans : List[str]
            List of channels to return if not all are required
        startSample : int
            First sample to return
        endSample : int
            Last sample to return
        remaverage : bool
            Remove average from the data
        remzeros : bool
            Remove zeroes from the data
        remnans: bool
            Remove NanNs from the data

        Returns
        -------
        TimeData
            Time data object
        """
        options = self.parseGetDataKeywords(kwargs)
        timeData = self.getUnscaledSamples(
            chans=options["chans"],
            startSample=options["startSample"],
            endSample=options["endSample"],
        )

        # no further scaling applied to ascii data
        for chan in options["chans"]:
            # if remove zeros - False by default
            if options["remzeros"]:
                timeData.data[chan] = removeZerosSingle(timeData.data[chan])
            # if remove nans - False by default
            if options["remnans"]:
                timeData.data[chan] = removeNansSingle(timeData.data[chan])
            # remove the average from the data - True by default
            # do this after all scaling and removing nans and zeros
            if options["remaverage"]:
                timeData.data[chan] = timeData.data[chan] - np.average(
                    timeData.data[chan]
                )

        timeData.addComment(
            "Remove zeros: {}, remove nans: {}, remove average: {}".format(
                options["remzeros"], options["remnans"], options["remaverage"]
            )
        )
        return timeData
