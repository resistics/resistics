"""
Common resistics functions and classes used throughout the package
"""
from loguru import logger
from typing import List, Tuple, Union, Dict, Set, Any, Collection, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from datetime import datetime
import numpy as np

from resistics.sampling import RSDateTime, datetime_to_string


def get_version() -> str:
    """Get the version of resistics"""
    import resistics

    return resistics.__version__


def is_file(file_path: Path) -> bool:
    """
    Check if a path exists and points to a file

    Parameters
    ----------
    file_path : Path
        The path to check

    Returns
    -------
    bool
        True if it exists and is a file, False otherwise
    """
    if not file_path.exists():
        logger.warning(f"File path {file_path} does not exist")
        return False
    if not file_path.is_file():
        logger.warning(f"File path {file_path} is not a file")
        return False
    return True


def assert_file(file_path: Path) -> None:
    """
    Require that a file exists

    Parameters
    ----------
    file_path : Path
        The path to check

    Raises
    ------
    FileNotFoundError
        If the path does not exist
    NotFileError
        If the path is not a file
    """
    from resistics.errors import NotFileError

    if not file_path.exists():
        raise FileNotFoundError(f"Path {file_path} not found")
    if not file_path.is_file():
        raise NotFileError(file_path)


def is_dir(dir_path: Path) -> bool:
    """
    Check if a path exists and points to a directory

    Parameters
    ----------
    dir_path : Path
        The path to check

    Returns
    -------
    bool
        True if it exists and is a directory, False otherwise
    """
    if not dir_path.exists():
        logger.warning(f"Directory path {dir_path} does not exist")
        return False
    if not dir_path.is_dir():
        logger.warning(f"Directory path {dir_path} is not a directory")
        return False
    return True


def assert_dir(dir_path: Path) -> None:
    """
    Require that a path is a directory

    Parameters
    ----------
    dir_path : Path
        Path to check

    Raises
    ------
    FileNotFoundError
        If the path does not exist
    NotDirectoryError
        If the path is not a directory
    """
    from resistics.errors import NotDirectoryError

    if not dir_path.exists():
        raise FileNotFoundError(f"Path {dir_path} does not exist")
    if not dir_path.is_dir():
        raise NotDirectoryError(dir_path)


def dir_contents(dir_path: Path) -> Tuple[List[Path], List[Path]]:
    """
    Get contents of directory

    Includes both files and directories

    Parameters
    ----------
    dir_path : Path
        Parent directory path

    Returns
    -------
    dirs : list
        List of directories
    files : list
        List of files excluding hidden files

    Raises
    ------
    PathNotFoundError
        Path does not exist
    NotDirectoryError
        Path is not a directory
    """
    from resistics.errors import PathNotFoundError, NotDirectoryError

    if not dir_path.exists():
        raise PathNotFoundError(dir_path)
    if not dir_path.is_dir():
        raise NotDirectoryError(dir_path)

    dirs = []
    files = []
    for obj in dir_path.iterdir():
        if obj.is_file():
            files.append(obj)
        elif obj.is_dir():
            dirs.append(obj)
    return dirs, files


def dir_files(dir_path: Path) -> List[Path]:
    """
    Get files in directory

    Excludes hidden files

    Parameters
    ----------
    dir_path : Path
        Parent directory path

    Returns
    -------
    files : list
        List of files excluding hidden files
    """
    _, files = dir_contents(dir_path)
    return files


def dir_subdirs(dir_path: Path) -> List[Path]:
    """
    Get subdirectories in directory

    Excludes hidden files

    Parameters
    ----------
    dir_path : Path
        Parent directory path

    Returns
    -------
    dirs : list
        List of subdirectories
    """
    dirs, _ = dir_contents(dir_path)
    return dirs


def data_dir_names() -> List[str]:
    """
    Get list of data directory names

    Returns
    -------
    List[str]
        A list of allowable data directory names
    """
    return ["meas", "run", "phnx", "lemi"]


def electric_chans() -> List[str]:
    """
    List of acceptable electric channels

    Returns
    -------
    List[str]
        List of acceptable electric channels
    """
    return ["Ex", "Ey", "E1", "E2", "E3", "E4"]


def is_electric(chan: str) -> bool:
    """
    Check if a channel is electric

    Parameters
    ----------
    chan : str
        Channel name

    Returns
    -------
    bool
        True if channel is electric

    Examples
    --------
    >>> from resistics.common import is_electric
    >>> is_electric("Ex")
    True
    >>> is_electric("Hx")
    False
    """
    if chan in electric_chans():
        return True
    return False


def magnetic_chans() -> List[str]:
    """
    List of acceptable magnetic channels

    Returns
    -------
    List[str]
        List of acceptable magnetic channels
    """
    return ["Hx", "Hy", "Hz", "Bx", "By", "Bz"]


def is_magnetic(chan: str) -> bool:
    """
    Check if channel is magnetic

    Parameters
    ----------
    chan : str
        Channel name

    Returns
    -------
    bool
        True if channel is magnetic

    Examples
    --------
    >>> from resistics.common import is_magnetic
    >>> is_magnetic("Ex")
    False
    >>> is_magnetic("Hx")
    True
    """
    if chan in magnetic_chans():
        return True
    return False


def to_resistics_chan(chan: str) -> str:
    """
    Convert channels to ensure consistency

    Parameters
    ----------
    chan : str
        Channel name

    Returns
    -------
    str
        Converted channel name

    Examples
    --------
    >>> from resistics.common import to_resistics_chan
    >>> to_resistics_chan("Bx")
    'Hx'
    >>> to_resistics_chan("Ex")
    'Ex'
    """
    standard_chans = ["Hx", "Hy", "Hz", "Ex", "Ey"]
    if chan in standard_chans:
        return chan
    elif chan == "Bx":
        return "Hx"
    elif chan == "By":
        return "Hy"
    elif chan == "Bz":
        return "Hz"
    else:
        return chan


def check_chan(chan: str, chans: Collection[str]) -> bool:
    """
    Check a channel exists and raise a KeyError if not

    Parameters
    ----------
    chan : str
        The channel to check
    chans : Collection[str]
        A collection of channels to check against

    Returns
    -------
    bool
        True if all checks passed

    Raises
    ------
    ChannelNotFoundError
        If the channel is not found in the channel list
    """
    from resistics.errors import ChannelNotFoundError

    if chan not in chans:
        logger.error(f"Channel {chan} not in channel list {chans}")
        raise ChannelNotFoundError(chan, chans)
    return True


def fs_to_string(fs: float) -> str:
    """
    Convert sampling frequency into a string for filenames

    Parameters
    ----------
    fs : float
        The sampling frequency

    Returns
    -------
    str
        Sample frequency converted to string for the purposes of a filename

    Examples
    --------
    >>> from resistics.common import fs_to_string
    >>> fs_to_string(512.0)
    '512_000000'
    """
    return (f"{fs:.6f}").replace(".", "_")


def lines_to_array(lines: List[str]) -> np.ndarray:
    """
    Format lines of numbers into a multi-dimensional array

    This is most likely of use when reading in files with lines of numbers

    Parameters
    ----------
    lines : List[str]
        A list of strings, most commonly read from a file

    Returns
    -------
    np.ndarray
        Lines formatted as a mutli-dimensional numeric array

    Examples
    --------
    >>> from resistics.common import lines_to_array
    >>> lines_to_array(["3 4", "4 5", "5 4"]) # doctest: +NORMALIZE_WHITESPACE
    array([[3., 4.],
            [4., 5.],
            [5., 4.]])
    """
    return np.array([x.split() for x in lines], dtype=float)


def array_to_string(
    data: np.ndarray, sep: str = ", ", precision: int = 8, scientific: bool = False
) -> str:
    """
    Convert an array to a string for logging or printing

    Parameters
    ----------
    data : np.ndarray
        The array
    sep : str, optional
        The separator to use, by default ", "
    precision : int, optional
        Number of decimal places, by default 8. Ignored for integers.
    scientific : bool, optional
        Flag for formatting floats as scientific, by default False

    Returns
    -------
    str
        String representation of array

    Examples
    --------
    >>> import numpy as np
    >>> from resistics.common import array_to_string
    >>> data = np.array([1,2,3,4,5])
    >>> array_to_string(data)
    '1, 2, 3, 4, 5'
    >>> data = np.array([1,2,3,4,5], dtype=np.float32)
    >>> array_to_string(data)
    '1.00000000, 2.00000000, 3.00000000, 4.00000000, 5.00000000'
    >>> array_to_string(data, precision=3, scientific=True)
    '1.000e+00, 2.000e+00, 3.000e+00, 4.000e+00, 5.000e+00'
    """
    style: str = "e" if scientific else "f"
    output_str = np.array2string(
        data,
        separator=sep,
        formatter={"float_kind": lambda x: f"{x:.{precision}{style}}"},
    )
    return output_str.lstrip("[").rstrip("]")


def list_to_string(lst: List[Any]) -> str:
    """
    Convert a list to a comma separated string

    Parameters
    ----------
    lst : List[Any]
        Input list to convert to a string

    Returns
    -------
    str
        Output string

    Examples
    --------
    >>> from resistics.common import list_to_string
    >>> list_to_string(["a", "b", "c"])
    'a, b, c'
    >>> list_to_string([1,2,3])
    '1, 2, 3'
    """
    output_str = ""
    for value in lst:
        output_str += f"{value}, "
    return output_str.strip().rstrip(",")


def list_to_ranges(data: Union[List, Set]) -> str:
    """
    Convert a list of numbers to a list of ranges

    Parameters
    ----------
    data : Union[List, Set]
        List or set of integers

    Returns
    -------
    str
        Formatted output string

    Examples
    --------
    >>> from resistics.common import list_to_ranges
    >>> data = [1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21, 24, 26, 40, 45, 48, 49]
    >>> list_to_ranges(data)
    '1-4:1,6-12:2,15-24:3,26,40,45,48,49'
    """
    lst = sorted(list(data))
    n = len(lst)

    def formatter(start, stop, step):
        return f"{start}-{stop}:{step}"

    result = []
    scan = 0
    while n - scan > 2:
        step = lst[scan + 1] - lst[scan]
        if lst[scan + 2] - lst[scan + 1] != step:
            result.append(str(lst[scan]))
            scan += 1
            continue

        for jj in range(scan + 2, n - 1):
            if lst[jj + 1] - lst[jj] != step:
                result.append(formatter(lst[scan], lst[jj], step))
                scan = jj + 1
                break
        else:
            result.append(formatter(lst[scan], lst[-1], step))
            return ",".join(result)

    for jj in range(scan, n):
        result.append(str(lst[jj]))

    return ",".join(result)


class ResisticsBase(object):
    """
    Resistics base class

    Parent class to ensure consistency of common methods
    """

    def __repr__(self) -> str:
        """Return a string of class information"""
        return self.to_string()

    def __str__(self) -> str:
        """Return a string of class information"""
        return self.to_string()

    def type_to_string(self) -> str:
        """Get the class type as a string"""
        return str(self.__class__)

    def to_string(self) -> str:
        """Class details as a string"""
        return self.type_to_string()

    def summary(self, symbol: str = "-") -> None:
        """Print a summary of class details"""
        name = str(self.__class__)
        length = len(name) + 10
        print("##" + 3 * symbol + "Begin Summary" + ((length - 18) * symbol))
        print(self.to_string())
        print("##" + 3 * symbol + "End summary" + (length - 16) * symbol)


class ResisticsData(ResisticsBase):
    """
    Base class for a resistics data object
    """

    def __init__(self):
        pass


class ResisticsModel(BaseModel):
    """Base resistics model"""

    def __str__(self) -> str:
        return self.to_string()

    def to_string(self) -> str:
        """Class info as string"""
        import json
        import yaml

        json_dict = json.loads(self.json())
        return yaml.dump(json_dict, indent=4, sort_keys=False)

    def summary(self) -> None:
        """Print a summary of the class"""
        import json
        from prettyprinter import cpprint

        cpprint(json.loads(self.json()))

    class Config:
        """pydantic configuration information"""

        json_encoders = {RSDateTime: datetime_to_string}


class Metadata(ResisticsModel):
    """Base class for metadata"""

    pass


class Record(ResisticsModel):
    """
    Class to hold a process record

    Parameters
    ----------
    time_utc : datetime
        The UTC time when the process ran
    time_local : datetime
        The local time when the process ran
    creator : str
        The name of the record creator
    parameters : Dict[str, Any]
        The parameters of the process
    messages : List[str]
        Any messages in the process
    record_type : str
        The process type
    """

    time_local: datetime = Field(default_factory=datetime.now)
    time_utc: datetime = Field(default_factory=datetime.utcnow)
    creator: str
    parameters: Dict[str, Any]
    messages: List[str]
    record_type: str


class History(ResisticsModel):
    """
    Class for storing processing history

    Parameters
    ----------
    records : List[Record], optional
        List of records, by default []

    Examples
    --------
    >>> from resistics.testing import record_example1, record_example2
    >>> from resistics.common import History
    >>> record1 = record_example1()
    >>> record2 = record_example2()
    >>> history = History(records=[record1, record2])
    >>> history.summary()
    {
        'records': [
            {
                'time_local': '...',
                'time_utc': '...',
                'creator': 'example1',
                'parameters': {'a': 5, 'b': -7.0},
                'messages': ['Message 1', 'Message 2'],
                'record_type': 'process'
            },
            {
                'time_local': '...',
                'time_utc': '...',
                'creator': 'example2',
                'parameters': {'a': 'parzen', 'b': -21},
                'messages': ['Message 5', 'Message 6'],
                'record_type': 'process'
            }
        ]
    }
    """

    records: List[Record] = []

    def add_record(self, record: Record):
        """
        Add a process record to the list

        Parameters
        ----------
        record : Record
            The record to add
        """
        self.records.append(record)


def get_record(
    process_name: str,
    parameters: Dict[str, Any],
    messages: Union[str, List[str]],
    record_type: str = "process",
    time_utc: Optional[datetime] = None,
    time_local: Optional[datetime] = None,
) -> Record:
    """
    Get a process record

    Parameters
    ----------
    process_name : str
        The name of the process
    parameters : Dict[str, Any]
        The parameters as a dictionary
    messages : Union[str, List[str]]
        The messages as either a single str or a list of strings
    record_type : str, optional
        The type of record, by default "process"
    time_utc : Optional[datetime], optional
        UTC time to attach to the record, by default None. If None, will default
        to UTC now
    time_local : Optional[datetime], optional
        Local time to attach to the record, by default None. If None, will
        defult to local now

    Returns
    -------
    Record
        The process record

    Examples
    --------
    >>> from resistics.common import get_record
    >>> record = get_record("example", {"a": 5, "b": -7.0}, messages="a message")
    >>> record.creator
    'example'
    >>> record.parameters
    {'a': 5, 'b': -7.0}
    >>> record.messages
    ['a message']
    >>> record.record_type
    'process'
    >>> record.time_utc
    datetime.datetime(...)
    >>> record.time_local
    datetime.datetime(...)
    """
    if isinstance(messages, str):
        messages = [messages]
    if time_utc is None:
        time_utc = datetime.utcnow()
    if time_local is None:
        time_local = datetime.now()
    return Record(
        time_utc=time_utc,
        time_local=time_local,
        creator=process_name,
        parameters=parameters,
        messages=messages,
        record_type=record_type,
    )


def get_history(record: Record, history: Optional[History] = None) -> History:
    """
    Get a new History instance or add a record to a copy of an existing one

    This method always makes a deepcopy of an input history to avoid any
    unplanned modifications to the inputs.

    Parameters
    ----------
    record : Record
        The record
    history : Optional[History], optional
        A history to add to, by default None

    Returns
    -------
    History
        History with the record added

    Examples
    --------
    Get a new History with a single Record

    >>> from resistics.common import get_history
    >>> from resistics.testing import record_example1, record_example2
    >>> record1 = record_example1()
    >>> history = get_history(record1)
    >>> history.summary()
    {
        'records': [
            {
                'time_local': '...',
                'time_utc': '...',
                'creator': 'example1',
                'parameters': {'a': 5, 'b': -7.0},
                'messages': ['Message 1', 'Message 2'],
                'record_type': 'process'
            }
        ]
    }

    Alternatively, add to an existing History. This will make a copy of the
    original history. If a copy is not needed, the add_record method of history
    can be used.

    >>> record2 = record_example2()
    >>> history = get_history(record2, history)
    >>> history.summary()
    {
        'records': [
            {
                'time_local': '...',
                'time_utc': '...',
                'creator': 'example1',
                'parameters': {'a': 5, 'b': -7.0},
                'messages': ['Message 1', 'Message 2'],
                'record_type': 'process'
            },
            {
                'time_local': '...',
                'time_utc': '...',
                'creator': 'example2',
                'parameters': {'a': 'parzen', 'b': -21},
                'messages': ['Message 5', 'Message 6'],
                'record_type': 'process'
            }
        ]
    }
    """
    if history is None:
        return History(records=[record])
    history = History(**history.dict())
    history.add_record(record)
    return history


def histories_to_parameters(histories: List[History]) -> Dict[str, Any]:
    """
    Convert histories to a dictionary of parameters

    Parameters
    ----------
    histories : List[History]
        List of process histories

    Returns
    -------
    Dict[str, Any]
        Parameter dictionary
    """
    parameters = {
        f"Dataset {idx}": history.dict() for idx, history in enumerate(histories)
    }
    return parameters


class ResisticsProcess(BaseModel):
    """
    Base class for resistics processes

    Resistics processes perform operations on data (including read and write
    operations). Each time a ResisticsProcess child class is run, it should add
    a process record to the dataset

    The execution loop for a reistics process is:

    .. code-block::

        processor.check()
        processor.run()
    """

    @property
    def name(self) -> str:
        """
        Return the processor class name

        Returns
        -------
        str
            The processor (class) name
        """
        return self.__class__.__name__

    def parameters(self) -> Dict[str, Any]:
        """
        Return any process parameters

        These parameters are expected to be primatives and should be sufficient
        to reinitialise the process and re-run the data. The base class assumes
        all class variables meet this description.

        Returns
        -------
        Dict[str, Any]
            Dictionary of parameters
        """
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("__") and not callable(key)
        }

    def check(self) -> bool:
        """
        Check to ensure the processor can run

        Returns
        -------
        bool
            A boolean flag to confirm whether the processor can run
        """
        return True

    def run(self):
        """
        Run the process

        Raises
        ------
        NotImplementedError
            Needs to be implemented in child classes
        """
        raise NotImplementedError("Child processor classes must have a run method")

    def _get_record(self, messages: Union[str, List[str]]) -> Record:
        """
        Get the record for the processor

        Parameters
        ----------
        messages : Union[str, List[str]]
            The messages to add for the processor

        Returns
        -------
        Record
            A record
        """
        return get_record(self.name, self.parameters(), messages)


class JSONFile(ResisticsModel):
    """
    Model for writing out resistics JSON files

    Examples
    --------
    >>> from resistics.common import JSONFile
    >>> from resistics.testing import time_metadata_simple
    >>> metadata = time_metadata_simple()
    >>> test = JSONFile(metadata=metadata)
    >>> test.summary()
    {
        'created_by': 'resistics',
        'created_on_local': '2021-05-10T20:33:30.584333',
        'created_on_utc': '2021-05-10T19:33:30.584333',
        'version': '0.0.7.dev1',
        'metadata': {
            'fs': 10.0,
            'n_chans': 2,
            'n_samples': 11,
            'chans': ['chan1', 'chan2'],
            'chans_metadata': {
                'chan1': {
                    'data_files': ['example1.ascii'],
                    'sensor': '',
                    'serial': '',
                    'gain1': 1,
                    'gain2': 1,
                    'scaling': 1,
                    'hchopper': False,
                    'echopper': False,
                    'dx': 1,
                    'dy': 1,
                    'dz': 1,
                    'sensor_calibration_file': '',
                    'instrument_calibration_file': ''
                },
                'chan2': {
                    'data_files': ['example2.ascii'],
                    'sensor': 'MFS',
                    'serial': '',
                    'gain1': 1,
                    'gain2': 1,
                    'scaling': 1,
                    'hchopper': False,
                    'echopper': False,
                    'dx': 1,
                    'dy': 1,
                    'dz': 1,
                    'sensor_calibration_file': '',
                    'instrument_calibration_file': ''
                }
            },
            'first_time': '2021-01-01 00:00:00.000000_000000_000000_000000',
            'last_time': '2021-01-01 00:00:01.000000_000000_000000_000000',
            'system': '',
            'wgs84_latitude': -999.0,
            'wgs84_longitude': -999.0,
            'easting': -999.0,
            'northing': -999.0,
            'elevation': -999.0,
            'history': None
        }
    }
    """

    created_by: str = "resistics"
    created_on_local: datetime = Field(default_factory=datetime.now)
    created_on_utc: datetime = Field(default_factory=datetime.utcnow)
    version: Optional[str] = Field(default_factory=get_version)
    metadata: Optional[Metadata] = None

    def write(self, json_path: Path):
        """
        Write out JSON metadata file

        Parameters
        ----------
        json_path : Path
            Path to write JSON file
        """
        with json_path.open("w") as f:
            f.write(self.json())
