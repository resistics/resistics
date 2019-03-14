import os
import sys
import itertools
import glob
from typing import Union

# import the different readers
from resistics.ioHandlers.dataReader import DataReader
from resistics.ioHandlers.dataReaderATS import DataReaderATS
from resistics.ioHandlers.dataReaderSpam import DataReaderSPAM
from resistics.ioHandlers.dataReaderInternal import DataReaderInternal
from resistics.ioHandlers.dataReaderPhoenix import DataReaderPhoenix


def getDataReader(datapath: str) -> Union[DataReader, bool]:
    """Get the data reader for a time data directory (and format)
    
    Parameters
    ----------
    datapath : str
        Path to data directory

    Returns
    -------
    dataReader : DataReader (DataReaderATS, DataReaderSPAM, DataReaderInternal, DataReaderPhonix)
        Data reader object
    """
    
    # check the file endings in the datapath
    # internal format
    headerF = glob.glob(os.path.join(datapath, "*.hdr"))
    headerF = headerF + glob.glob(os.path.join(datapath, "*.HDR"))
    if len(headerF) > 0:  # then internal format
        return DataReaderInternal(datapath)
    # ATS data files
    headerF = glob.glob(os.path.join(datapath, "*.xml"))
    headerF = headerF + glob.glob(os.path.join(datapath, "*.XML"))
    if len(headerF) > 0:  # then ats format
        return DataReaderATS(datapath)
    # SPAM data files
    headerF = glob.glob(os.path.join(datapath, "*.xtrx"))
    headerF = headerF + glob.glob(os.path.join(datapath, "*.XTRX"))
    if len(headerF) > 0:  # then xtrx format
        return DataReaderSPAM(datapath)
    headerF = glob.glob(os.path.join(datapath, "*.xtr"))
    headerF = headerF + glob.glob(os.path.join(datapath, "*.XTR"))
    if len(headerF) > 0:  # then xtr format
        return DataReaderSPAM(datapath)
    # Phoenix data files
    headerF = glob.glob(os.path.join(datapath, "*.tbl"))
    headerF = headerF + glob.glob(os.path.join(datapath, "*.TBL"))
    if len(headerF) > 0:  # then xtr format
        return DataReaderPhoenix(datapath)
    # if nothing found, return false
    return False
