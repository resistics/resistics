import os
from typing import Union

from resistics.time.reader import TimeReader


def getTimeReader(datapath: str) -> Union[TimeReader, bool]:
    """Get the data reader for a time data directory (and format)

    Format is determined through a mix of header and data file extensions. Not perfect, but it saves the user from having to specify each time
    
    Parameters
    ----------
    datapath : str
        Path to data directory

    Returns
    -------
    dataReader : DataReader (DataReaderATS, DataReaderSPAM, DataReaderInternal, DataReaderPhonix, DataReaderLemiB423, DataReaderLemiB423E)
        Data reader object
    """
    import glob

    # Lemi B423 data files
    headerF = glob.glob(os.path.join(datapath, "*.h423"))
    if len(headerF) > 0:  # then lemi b423 format
        from resistics.time.reader_lemib423 import TimeReaderLemiB423

        return TimeReaderLemiB423(datapath)

    # Lemi B423E data files
    headerF = glob.glob(os.path.join(datapath, "*.h423E"))
    if len(headerF) > 0:
        from resistics.time.reader_lemib423e import TimeReaderLemiB423E

        return TimeReaderLemiB423E(datapath)

    # ATS data files
    headerF = glob.glob(os.path.join(datapath, "*.xml"))
    headerF = headerF + glob.glob(os.path.join(datapath, "*.XML"))
    if len(headerF) > 0:  # then ats format
        from resistics.time.reader_ats import TimeReaderATS

        return TimeReaderATS(datapath)

    # SPAM data files with XTRX headers
    headerF = glob.glob(os.path.join(datapath, "*.xtrx"))
    headerF = headerF + glob.glob(os.path.join(datapath, "*.XTRX"))
    if len(headerF) > 0:
        from resistics.time.reader_spam import TimeReaderSPAM

        return TimeReaderSPAM(datapath)

    # SPAM data files with XTR headers
    headerF = glob.glob(os.path.join(datapath, "*.xtr"))
    headerF = headerF + glob.glob(os.path.join(datapath, "*.XTR"))
    if len(headerF) > 0:
        from resistics.time.reader_spam import TimeReaderSPAM

        return TimeReaderSPAM(datapath)

    # Phoenix data files
    headerF = glob.glob(os.path.join(datapath, "*.tbl"))
    headerF = headerF + glob.glob(os.path.join(datapath, "*.TBL"))
    if len(headerF) > 0:
        from resistics.time.reader_phoenix import TimeReaderPhoenix

        return TimeReaderPhoenix(datapath)

    # internal header format
    headerF = glob.glob(os.path.join(datapath, "*.hdr"))
    headerF = headerF + glob.glob(os.path.join(datapath, "*.HDR"))
    if len(headerF) > 0:
        dataF = glob.glob(os.path.join(datapath, "*.ascii"))
        if len(dataF) > 0:
            from resistics.time.reader_ascii import TimeReaderAscii

            return TimeReaderAscii(datapath)

        # otherwise assume internal format
        from resistics.time.reader_internal import TimeReaderInternal

        return TimeReaderInternal(datapath)

    # if nothing found, return false
    return False
