"""Test resistics.common.io"""


def test_dir_formats() -> None:
    from resistics.common.io import data_dir_formats

    assert data_dir_formats() == ["meas", "run", "phnx", "lemi"]


def test_dir_contents() -> None:
    from resistics.common.io import dir_contents
    
    assert False


# def getFilesInDirectory(path: str) -> List:
#     """Get files in directory

#     Excludes hidden files

#     Parameters
#     ----------
#     path : str
#         Parent directory path

#     Returns
#     -------
#     files : list
#         List of files excluding hidden files
#     """
#     _, files = getDirectoryContents(path)
#     return files


# def getDirsInDirectory(path: str):
#     """Get subdirectories in directory

#     Excludes hidden files

#     Parameters
#     ----------
#     path : str
#         Parent directory path

#     Returns
#     -------
#     dirs : list
#         List of subdirectories
#     """
#     dirs, _ = getDirectoryContents(path)
#     return dirs


def test_removeHiddenFiles() -> None:
    from resistics.common.io import removeHiddenFiles

    files = ["hello", "gutentag", ".bonjour", "hola.hermano"]
    assert removeHiddenFiles(files) == ["hello", "gutentag", "hola.hermano"]


# def getDataDirsInDirectory(path: str) -> List:
#     """Get subdirectories in directory

#     This uses known data formats as defined in getDataDirectoryFormats

#     Parameters
#     ----------
#     path : str
#         Parent directory path

#     Returns
#     -------
#     dirs : list
#         List of directories containing time data
#     """
#     dirs = getDirsInDirectory(path)
#     dirsData = []
#     formats = getDataDirectoryFormats()
#     for d in dirs:
#         for f in formats:
#             if f in d:
#                 dirsData.append(d)
#     return dirsData


# def checkDirExistence(path: str) -> bool:
#     """Check if directory exists

#     ..todo::

#         Should check that it is actually a directory

#     Parameters
#     ----------
#     path : str
#         Path to check

#     Returns
#     -------
#     out : bool
#         True if directory exists
#     """
#     if not os.path.exists(path):
#         return False
#     return True


# def makeDir(path: str) -> None:
#     """Make directory

#     Parameters
#     ----------
#     path : str
#         Directory path to make
#     """
#     os.makedirs(path)


# def checkAndMakeDir(path: str) -> None:
#     """Check if directory exists and make if not

#     Parameters
#     ----------
#     path : str
#         Directory path to make
#     """
#     if not checkDirExistence(path):
#         makeDir(path)


# def checkFilepath(path: str) -> bool:
#     """Check if file exists

#     TODO: Should check that it is actually a file

#     Parameters
#     ----------
#     path : str
#         Filepath to check

#     Returns
#     -------
#     out : bool
#         True if file exists
#     """
#     if not os.path.exists(path):
#         generalPrint(
#             "utilsio::checkFilepath", "File path {} could not be found.".format(path)
#         )
#         return False
#     return True


def test_fileFormatSampleFreq() -> None:
    from resistics.common.io import fileFormatSampleFreq

    assert fileFormatSampleFreq(128) == "128_000"
    assert fileFormatSampleFreq(4096.1567) == "4096_157"


def test_lineToKeyAndValue() -> None:
    from resistics.common.io import lineToKeyAndValue

    teststr = "test=hello"
    key, val = lineToKeyAndValue(teststr)
    assert key == "test"
    assert val == "hello"

    teststr = "test:hello"
    key, val = lineToKeyAndValue(teststr, delim=":")
    assert key == "test"
    assert val == "hello"
