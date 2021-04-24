"""
Common resistics functions and classes used throughout the package
"""
from logging import getLogger
from typing import Type, List, Tuple, Union, Dict, Set, Any
from typing import Collection, Iterator, Callable, Optional
from pathlib import Path
import numpy as np
import pandas as pd

from resistics.sampling import RSDateTime
from resistics.sampling import datetime_to_string, to_datetime

logger = getLogger(__name__)


# alias a metadata specification
Spec = Dict[str, Dict[str, Any]]


serialize_fncs: Dict[Type[Any], Callable] = {
    str: (lambda x: x),
    float: (lambda x: x),
    int: (lambda x: x),
    bool: (lambda x: str(x)),
    RSDateTime: (lambda x: datetime_to_string(x)),
    pd.Timestamp: (lambda x: x.isoformat()),
    pd.Timedelta: (lambda x: x.total_seconds()),
    list: (lambda x: list_to_string(x)),
}


deserialize_fncs: Dict[Type[Any], Callable] = {
    str: (lambda x: str(x)),
    float: (lambda x: float(x)),
    int: (lambda x: int(x)),
    bool: (lambda x: x.lower() == "true"),
    RSDateTime: (lambda x: to_datetime(x)),
    pd.Timestamp: (lambda x: pd.Timestamp(x)),
    pd.Timedelta: (lambda x: pd.Timedelta(x, "s")),
    list: (lambda x: [y.strip() for y in x.split(",")]),
}


def serialize(x: Any) -> Any:
    """
    Serialize a variable

    Parameters
    ----------
    x : Any
        Variable

    Returns
    -------
    Any
        Serialized

    Raises
    ------
    SerializationError
        If type not supported
    """
    from resistics.errors import SerializationError

    if type(x) not in serialize_fncs:
        raise SerializationError(x)
    fnc = serialize_fncs[type(x)]
    return fnc(x)


def deserialize(x: Any, expected_type: Type[Any]) -> Any:
    """
    Derialize a variable

    Parameters
    ----------
    x : Any
        Variable
    expected_type : Type[Any]
        The expected type for the variable

    Returns
    -------
    Any
        The derialized variable

    Raises
    ------
    DeserializationError
        If expected type not supported
    """
    from resistics.errors import DeserializationError

    if expected_type not in deserialize_fncs:
        raise DeserializationError(x, expected_type)
    fnc = deserialize_fncs[expected_type]
    return fnc(x)


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


def format_value(value: Any, format_type: Type[Any]) -> Any:
    """
    Format a value

    Parameters
    ----------
    value : Any
        The value to format
    format_type : Type
        The type to format to

    Returns
    -------
    Any
        The formatted value

    Raises
    ------
    TypeError
        If unable to convert the value

    Examples
    --------
    >>> from resistics.common import format_value
    >>> format_value("5", int)
    5
    """
    if isinstance(value, format_type):
        return value
    try:
        value = deserialize(value, format_type)
    except Exception:
        raise TypeError(f"Unable to convert {value} to type {format_type}")
    return value


def format_dict(in_dict: Dict[str, Any], specs: Spec) -> Dict[str, Any]:
    """
    Format the values in a dictionary

    .. warning::

        If a key is not present in in_dict and is present in the specifications
        but without a default, a KeyError will be raised.

    Parameters
    ----------
    in_dict : Dict[str, Any]
        The dictionary to format
    specs : Spec
        Dictionary mapping key to key specifications type and default value

    Returns
    -------
    Dict[str, Any]
        Dictionary with values formatted

    Raises
    ------
    KeyError
        If a key in the specifications has no default and does not exist in the
        in_dict

    Examples
    --------
    An example where a default is provided for a missing header

    >>> from resistics.common import format_dict
    >>> in_dict = {"a": "12", "b": "something"}
    >>> specs = {"a": {"type": int, "default": 1}, "c": {"type": float, "default": -2.3}}
    >>> format_dict(in_dict, specs)
    {'a': 12, 'b': 'something', 'c': -2.3}
    """
    for key, spec in specs.items():
        if key in in_dict:
            in_dict[key] = format_value(in_dict[key], spec["type"])
        else:
            if spec["default"] is not None:
                in_dict[key] = spec["default"]
            else:
                raise KeyError(f"Required key {key} not found in input dictionary")
    return in_dict


class ResisticsBase(object):
    """
    Resistics base class

    Parent class to ensure consistency of common methods
    """

    def __repr__(self) -> str:
        """
        Return a string of class information

        Returns
        -------
        str
            String representation of class
        """
        return self.to_string()

    def __str__(self) -> str:
        """
        Return a string of class information

        Returns
        -------
        str
            String representation of class
        """
        return self.to_string()

    def type_to_string(self) -> str:
        """
        Get the class type

        Returns
        -------
        str
            Class type as string
        """
        return str(self.__class__)

    def to_string(self) -> str:
        """
        Get a string representation of the class

        Returns
        -------
        str
            String representation of class
        """
        return self.type_to_string()

    def summary(self, symbol: str = "-") -> None:
        """
        Print a summary of the class

        Parameters
        ----------
        symbol : str, optional
            The symbol to use in the summary prints, by default "-"
        """
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


class ProcessRecord(ResisticsBase):
    """
    Class to hold a process record
    """

    def __init__(self, record_dict: Dict[str, Any]) -> None:
        """
        Initialise a process record with a dictionary

        The keywords process, parameters and messages are required

        Parameters
        ----------
        record_dict : Dict[str, Any]
            A dictionary with information for the process

        Raises
        ------
        KeyError
            If any of the required keywords are missing
        """
        required = ["process", "parameters", "messages", "entry_type"]
        for req in required:
            if req not in record_dict:
                raise KeyError(f"Provided dictionary missing required key {req}")
        self._record = dict(record_dict)

    def copy(self) -> "ProcessRecord":
        """
        Get a copy of the ProcessRecord

        Returns
        -------
        ProcessRecord
            A copy of the process record
        """
        return ProcessRecord(self._record)

    def to_dict(self) -> Dict[str, Any]:
        """
        Return the record as a dictionary

        Returns
        -------
        Dict[str, Any]
            Record of process as a dictionary
        """
        return dict(self._record)

    def to_string(self) -> str:
        """
        Represent the process record as a string

        Returns
        -------
        str
            Process record as a string
        """
        import yaml

        return yaml.dump(self._record, indent=4, sort_keys=False).rstrip().rstrip("\n")


def get_process_record(
    name: str,
    parameters: Dict[str, Any],
    messages: Union[str, List[str]],
    process_type: str = "process",
) -> ProcessRecord:
    """
    Get a process record

    Parameters
    ----------
    name : str
        The name of the process
    parameters : Dict[str, Any]
        The parameters as a dictionary
    messages : Union[str, List[str]]
        The messages as either a single str or a list of strings
    process_type : str
        The type of process

    Returns
    -------
    ProcessRecord
        The process record
    """
    if isinstance(messages, str):
        messages = [messages]
    return ProcessRecord(
        {
            "run_on_local": pd.Timestamp.now(tz=None).isoformat(),
            "run_on_utc": pd.Timestamp.utcnow().isoformat(),
            "process": name,
            "parameters": parameters,
            "messages": messages,
            "entry_type": process_type,
        }
    )


class ProcessHistory(ResisticsBase):
    """
    Class for storing processing history
    """

    def __init__(self, records: Optional[List[ProcessRecord]] = None):
        """
        Initialise either an empty process history or with a list of records

        Parameters
        ----------
        records : Optional[List[ProcessRecord]], optional
            List of process records, by default None
        """
        if records is None:
            self._records = []
        else:
            self._records = records

    def add_record(self, record: ProcessRecord):
        """
        Add a process record to the list

        Parameters
        ----------
        record : ProcessRecord
            The record to add
        """
        self._records.append(record)

    def copy(self) -> "ProcessHistory":
        """
        Get a copy of the ProcessHistory

        Returns
        -------
        ProcessHistory
            A copy of the ProcessHistory
        """
        return ProcessHistory([x.copy() for x in self._records])

    def to_dict(self) -> Dict[str, ProcessRecord]:
        """
        Convert to a dictionary of process records

        Returns
        -------
        Dict[str, ProcessRecord]
            The dictionary of process records
        """
        history = {}
        for idx, record in enumerate(self._records):
            history[f"process_{idx}"] = record.to_dict()
        return history

    def to_string(self) -> str:
        """
        Get the process history as a string

        Returns
        -------
        str
            Process history as a string
        """
        import yaml

        outstr = f"{self.type_to_string()}\n"
        outstr += yaml.dump(self.to_dict(), indent=4, sort_keys=False)
        return outstr.rstrip().rstrip("\n")


def histories_to_parameters(histories: List[ProcessHistory]) -> Dict[str, Any]:
    """
    Convert histories to a dictionary of parameters

    Parameters
    ----------
    histories : List[ProcessHistory]
        List of process histories

    Returns
    -------
    Dict[str, Any]
        Parameter dictionary
    """
    parameters = {
        f"Dataset {idx}": history.to_dict() for idx, history in enumerate(histories)
    }
    return parameters


class ResisticsProcess(ResisticsBase):
    """
    Base class for resistics processes

    Resistics processes perform operations on data (including read and write
    operations). Each time a ResisticsProcess child class is run, it should add
    a process record to the dataset

    The execution loop for a reistics process is:

    .. code-block::

        processor.check()
        processor.prepare()
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

    def prepare(self):
        """
        Any preparation logic should be placed in the prepare method of a child
        class

        Where no prepare logic is required, this method does not have to be
        implemented in child classes
        """
        pass

    def run(self):
        """
        Run the process

        Raises
        ------
        NotImplementedError
            Needs to be implemented in child classes
        """
        raise NotImplementedError("Child processor classes must have a run method")

    def _get_process_record(self, messages: Union[str, List[str]]) -> ProcessRecord:
        """
        Get the process record for the processor

        Parameters
        ----------
        messages : Union[str, List[str]]
            The messages to add for the processor

        Returns
        -------
        ProcessRecord
            A process record
        """
        return get_process_record(self.name, self.parameters(), messages)


class Metadata(ResisticsBase):
    """
    A class for holding metadata for various data types

    Internally, metadata is essentially a dictionary and has keys and values.
    Given an appropriate set of specifications, it can ensure metadata values
    have the appropriate type and will raise an error if they do not.

    Examples
    --------
    Initialising metadata

    >>> from resistics.common import Metadata
    >>> metadata_dict = {"a": "12", "b": "something"}
    >>> metadata = Metadata(metadata_dict)
    >>> metadata.to_dict()
    {'a': '12', 'b': 'something', 'describes': 'unknown'}
    >>> metadata["a"]
    '12'
    >>> metadata["a"] = 15
    >>> metadata.to_dict()
    {'a': 15, 'b': 'something', 'describes': 'unknown'}

    If specifications are provided, updating a metadata value will be checked
    against the specifications

    >>> metadata_dict = {"a": "12", "b": "something"}
    >>> spec = {"a": {"type": int, "default": 0}}
    >>> metadata = Metadata(metadata_dict, spec)
    >>> metadata.to_dict()
    {'a': 12, 'b': 'something', 'describes': 'unknown'}
    >>> metadata["a"] = "try to set to string"
    Traceback (most recent call last):
    ...
    TypeError: Unable to convert try to set to string to type <class 'int'>
    >>> metadata["b"] = 12
    >>> metadata.to_dict()
    {'a': 12, 'b': 12, 'describes': 'unknown'}
    """

    def __init__(
        self,
        metadata: Dict[str, Any],
        specs: Optional[Spec] = None,
    ) -> None:
        """
        Initialise metadata

        Providing a specifications dictionary will automatically format metadata
        values and insert default values for missing metadata keys.

        Metadata makes a copy of the metadata to ensure changing it somewhere
        else will leave it unaltered here. However, specifications are not
        copied as these are assumed to remain constant in a run.

        It is suggested to pass describes as a key with a single word value to
        make it clear what the metadata describes. If no 'describes' key is
        passed, one will be added with the value 'unknown'

        Parameters
        ----------
        metadata : Dict[str, Any]
            Metadata dictionary
        specs : Optional[Specification], optional
            Mapping of metadata key to a dictionary with type and default
            information, by default None. If no specifications are provided, no
            checking is done of header value type and no defaults are added for
            missing metadata.
        """
        import copy

        self._metadata = copy.deepcopy(metadata)
        self._specs = specs
        if self._specs is not None:
            self._metadata = format_dict(self._metadata, self._specs)
        # add a tag if one doesn't exist
        if "describes" not in self._metadata:
            self._metadata["describes"] = "unknown"

    def __iter__(self) -> Iterator:
        """Iterator over metadata keys"""
        return self._metadata.__iter__()

    def __getitem__(self, key: str) -> Any:
        """
        Get a metadata value

        Parameters
        ----------
        key : str
            The metadata key

        Returns
        -------
        Any
            The value

        Raises
        ------
        MetadataKeyNotFound
            If key does not exist
        """
        from resistics.errors import MetadataKeyNotFound

        if key not in self._metadata:
            raise MetadataKeyNotFound(key, self.keys())
        return self._metadata[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set the value of a metadata key

        Type checking will be performed against a specification if available for
        the key. If no specification is available for the key, the value will be
        added without checking.

        Parameters
        ----------
        key : str
            The key
        value : Any
            The value
        """
        if key not in self._metadata:
            logger.info(f"Adding new key {key} with value {value}")
        if self._specs is not None and key in self._specs:
            spec_type = self._specs[key]["type"]
            logger.info(f"Specifications for this header, require type {spec_type}")
            value = format_value(value, spec_type)
        self._metadata[key] = value

    def keys(self, describes: bool = True) -> List[str]:
        """
        Get the keys in the metadata

        Parameters
        ----------
        describes : bool, optional
            Flag for including the describes entry, by default True

        Returns
        -------
        List[str]
            List of keys
        """
        keys = list(self._metadata.keys())
        if not describes:
            keys.remove("describes")
        return keys

    def copy(self) -> "Metadata":
        """
        Copy the metadata

        Returns
        -------
        Metadata
            A copy of the metadata
        """
        return Metadata(self.to_dict(), self._specs)

    def to_dict(self) -> Dict[str, Any]:
        """
        Get headers as a dictionary

        Returns
        -------
        Dict
            Get a dictonary of the header key to header value
        """
        import copy

        return copy.deepcopy(self._metadata)

    def to_series(self) -> pd.Series:
        """
        Get the headers as a pandas Series

        Returns
        -------
        pd.Series
            The headers as a pandas Series with headers as indices and values as
            the values
        """
        return pd.Series(self._metadata)

    def to_string(self) -> str:
        """
        Get headers as string

        Returns
        -------
        str
            String representation of headers
        """
        outstr = f"{self.type_to_string()}\n"
        return outstr + str(self._metadata)

    def serialize(self) -> Dict[str, Any]:
        """
        Serialize metadata values to allow writing out as a json file

        .. warning::

            This does not perform a full serialization but simply enough to
            allow writing out with json

        Returns
        -------
        Dict[str, Any]
            Serialized dictionary
        """
        return {key: serialize(value) for key, value in self._metadata.items()}


def metadata_from_specs(
    specs: Spec, overwrite: Optional[Dict[str, Any]] = None
) -> Metadata:
    """
    Get a new Metadata instance given a set of key specifications
    (e.g. time, calibration) and optional override values.

    Parameters
    ----------
    specs : Spec
        Key specifications
    overwrite : Optional[Dict[str, Any]], optional
        Override values for particular keys which will be used instead of
        defaults in the specification, by default None

    Returns
    -------
    Metadata
        A metadata instance
    """
    metadata_dict = {key: spec["default"] for key, spec in specs.items()}
    if overwrite is not None:
        for key, value in overwrite.items():
            metadata_dict[key] = value
    return Metadata(metadata_dict, specs)


class MetadataGroup(ResisticsBase):
    """Object for holding a group of metadata

    This can be used for time data or other types of datasets which have
    multiple Metadata. For example, time data has metadata which is
    common to all channels (e.g. latitude, longitude) and then channel specific
    metadata.

    Each Metadata instance in the group is called an entry.

    Examples
    --------
    >>> from resistics.common import MetadataGroup
    >>> group = {}
    >>> group["common"] = {"fs":512, "n_samples": 512000}
    >>> group["Ex"] = {"sensor": "MFS", "serial": 100}
    >>> group["Ey"] = {"sensor": "Phnx", "serial": 20}
    >>> metadata_grp = MetadataGroup(group)
    >>> metadata_grp.summary()
    ##---Begin Summary--------------------------------
    <class 'resistics.common.MetadataGroup'>
    Entry 'common' metadata
            <class 'resistics.common.Metadata'>
            {'fs': 512, 'n_samples': 512000, 'describes': 'unknown'}
    Entry 'Ex' metadata
            <class 'resistics.common.Metadata'>
            {'sensor': 'MFS', 'serial': 100, 'describes': 'unknown'}
    Entry 'Ey' metadata
            <class 'resistics.common.Metadata'>
            {'sensor': 'Phnx', 'serial': 20, 'describes': 'unknown'}
    ##---End summary----------------------------------
    >>> metadata_grp["common", "fs"]
    512
    >>> metadata_grp["Ex", "sensor"]
    'MFS'
    >>> metadata_grp["common", "fs"] = 128
    >>> metadata_grp["Ex", "sensor"] = "this is a test"
    >>> metadata_grp.summary()
    ##---Begin Summary--------------------------------
    <class 'resistics.common.MetadataGroup'>
    Entry 'common' metadata
            <class 'resistics.common.Metadata'>
            {'fs': 128, 'n_samples': 512000, 'describes': 'unknown'}
    Entry 'Ex' metadata
            <class 'resistics.common.Metadata'>
            {'sensor': 'this is a test', 'serial': 100, 'describes': 'unknown'}
    Entry 'Ey' metadata
            <class 'resistics.common.Metadata'>
            {'sensor': 'Phnx', 'serial': 20, 'describes': 'unknown'}
    ##---End summary----------------------------------
    """

    def __init__(
        self,
        group: Union[Dict[str, Metadata], Dict[str, Dict[str, Any]]],
        specs: Optional[Spec] = None,
    ) -> None:
        """
        Initialise with a group and a single specification

        The same specification will be used for all entries in the group. Where
        different specifications are required, more can be added using the
        add_entry method

        Parameters
        ----------
        group : Union[Dict[str, Metadata], Dict[str, Dict[str, Any]]]
            Dictionary mapping to Metadata or to dictionaries to be converted to
            Metadata
        specs : Optional[Spec], optional
            The specifications, by default None
        """
        self._group: Dict[str, Metadata] = {}
        self.add_entries(group, specs)

    def __iter__(self) -> Iterator:
        """Iterator over MetadataGroup entries"""
        return self._group.__iter__()

    def entries(
        self, describes: Optional[Union[str, Collection[str]]] = None
    ) -> List[str]:
        """
        Get a list of entries in the group

        Parameters
        ----------
        describes : Optional[Union[str, Collection[str]]], optional
            Restrict to entries which describe the same thing, by default None.
            For example, pull out entries that describe channels.

        Returns
        -------
        List[str]
            List of entries
        """
        entries = self._group.keys()
        if describes is None:
            return list(entries)
        if isinstance(describes, str):
            return [x for x in entries if self._group[x]["describes"] == describes]
        return [x for x in entries if self._group[x]["describes"] in describes]

    def __getitem__(self, args: Union[str, Tuple[str, str]]) -> Any:
        """
        Get a either a Metadata or a key value from a Metadata

        Parameters
        ----------
        args : Union[str, Tuple[str, str]]
            The arguments. Argument one must be the name of an entry. Argument
            two is optional and a key in the entry Metadata

        Returns
        -------
        Any
            Metadata if only an entry is passed, other the value of a metadata
            entry key

        Raises
        ------
        ValueError
            If the arguments have been incorrectly specified
        """
        if isinstance(args, str):
            return self.get_entry(args)
        elif isinstance(args, tuple) and len(args) == 2:
            return self.get_entry(args[0])[args[1]]
        else:
            raise ValueError(f"Arguments {args} have been incorrectly specified")

    def get_entry(self, entry: str) -> Metadata:
        """
        Get a metadata entry

        Parameters
        ----------
        entry : str
            Entry name

        Returns
        -------
        Metadata
            A metdata

        Raises
        ------
        MetadataEntryNotFound
            If the entry is not part of the MetadataGroup
        """
        from resistics.errors import MetadataEntryNotFound

        if entry not in self._group:
            raise MetadataEntryNotFound(entry, self.entries())
        return self._group[entry]

    def __setitem__(
        self, args: Union[str, Tuple[str, str]], value: Union[Metadata, Any]
    ) -> None:
        """
        Set either an entry or the value of a key in an entry

        Parameters
        ----------
        args : Union[str, Tuple[str, str]]
            The arguments. One argument must be the name of an entry. The second
            is optional and should be the name of a key in Metadata.
        value : Union[Metadata, Any]
            If only an entry is passed, this should be a Metadata. Otherwise,
            when setting the value of a Metadata key, this should be a value.

        Raises
        ------
        ValueError
            If the arguments have been incorrectly specified
        """
        if isinstance(args, str):
            self.set_entry(args, value)
        elif len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], str):
            self.get_entry(args[0])[args[1]] = value
        else:
            raise ValueError(f"Arguments {args} have been incorrectly specified")

    def set_entry(self, entry: str, metadata: Metadata) -> None:
        """
        Set an entry to a new metadata

        Note that setting entry Metadata makes a copy so if it is changed
        elsewhere, those changes will not be reflected in the MetadataGroup

        Parameters
        ----------
        entry : str
            The entry
        metadata : Metadata
            The metdata to set the entry to

        Raises
        ------
        MetadataEntryNotFound
            If the entry is not found. This method cannot be used for adding new
            entries. The add_entry method should be used to add new entries.
        """
        from resistics.errors import MetadataEntryNotFound

        if entry not in self._group:
            raise MetadataEntryNotFound(
                entry,
                self.entries(),
                "Entry does not exist. Use add_entry to add a new entry",
            )
        self._group[entry] = metadata.copy()

    def add_entries(
        self,
        group: Union[Dict[str, Metadata], Dict[str, Dict[str, Any]]],
        specs: Optional[Spec] = None,
    ) -> None:
        """
        Add entries to the MetadataGroup

        If an entry is provided as Metadata, it is added without modification.
        If an entry is provided as a dictionary, the specification will be
        applied if it is passed.

        Parameters
        ----------
        group : Union[Dict[str, Metadata], Dict[str, Dict[str, Any]]]
            Group of metadata
        specs : Optional[Spec], optional
            Specifications which will be applied to any entry passed as a
            dictionary, by default None
        """
        for entry, entry_metadata in group.items():
            if isinstance(entry_metadata, Metadata):
                self.add_entry(entry, entry_metadata)
            elif isinstance(entry, dict) and specs is not None:
                self.add_entry(entry, Metadata(entry_metadata, specs))
            else:
                self.add_entry(entry, Metadata(entry_metadata))

    def add_entry(self, entry: str, entry_metadata: Metadata) -> None:
        """
        Add an entry to the MetadataGroup

        Parameters
        ----------
        entry : str
            The name of the entry
        entry_metadata : Metadata
            The entry metadata

        Raises
        ------
        MetadataEntryAlreadyExists
            If the entry already exists. In this case, set entry should be used
            instead
        """
        from resistics.errors import MetadataEntryAlreadyExists

        if entry in self._group:
            raise MetadataEntryAlreadyExists(
                entry, self.entries(), "Use set_entry to change an existing entry"
            )
        self._group[entry] = entry_metadata.copy()

    def copy(self) -> "MetadataGroup":
        """
        Get a copy of the MetadataGroup

        Returns
        -------
        MetadataGroup
            A copy
        """
        return MetadataGroup(
            {
                entry: entry_metadata.copy()
                for entry, entry_metadata in self._group.items()
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Get the MetadataGroup as a dictionary

        Returns
        -------
        Dict[str, Any]
            MetadataGroup as dictionary
        """
        return {
            entry: entry_metadata.to_dict()
            for entry, entry_metadata in self._group.items()
        }

    def to_string(self) -> str:
        """
        Get string representation of MetadataGroup

        Returns
        -------
        str
            Class information as a string
        """
        outstr = f"{self.type_to_string()}\n"
        for entry, entry_metadata in self._group.items():
            outstr += f"Entry '{entry}' metadata\n"
            outstr += "\t" + entry_metadata.to_string().replace("\n", "\n\t")
            outstr += "\n"
        return outstr.rstrip("\n")

    def serialize(self) -> Dict[str, Any]:
        """
        Get the MetadataGroup as a serialized dictionary

        .. warning::

            This does not perform a full serialization but simply enough to
            allow writing out with json

        Returns
        -------
        Dict[str, Any]
            MetadataGroup converted to a serialized dictionary that can be saved
        """
        return {
            entry: entry_metadata.serialize()
            for entry, entry_metadata in self._group.items()
        }


def metadata_group_from_specs(
    specs_grp: Dict[str, Spec],
    overwrite: Optional[Dict[str, Dict[str, Any]]] = None,
) -> MetadataGroup:
    """
    Create a new MetadataGroup from a group of specifications

    Parameters
    ----------
    specs_grp : Dict[str, Spec]
        Group of specifications
    overwrite : Optional[Dict[str, Dict[str, Any]]], optional
        Any override values, by default None

    Returns
    -------
    MetadataGroup
        The new MetadataGroup
    """
    metadata = {}
    for entry, specs in specs_grp.items():
        if overwrite is not None and entry in overwrite:
            metadata[entry] = metadata_from_specs(specs, overwrite[entry])
        else:
            metadata[entry] = metadata_from_specs(specs)
    return MetadataGroup(metadata)


def metadata_to_json(metadata: Union[Metadata, MetadataGroup], json_path: Path) -> None:
    """
    Write a metadata to a file

    Parameters
    ----------
    metadata : Union[Metadata, MetadataGroup]
        A Metadata or MetadataGroup instance
    json_path : Path
        The path to write to
    """
    import resistics
    import pandas as pd
    import json

    out_dict = {
        "created_by": "resistics",
        "created_on_local": serialize(pd.Timestamp.now(tz=None)),
        "created_on_utc": serialize(pd.Timestamp.utcnow()),
        "version": resistics.__version__,
        "type": metadata.type_to_string(),
        "content": metadata.serialize(),
    }
    with json_path.open("w") as f:
        json.dump(out_dict, f)


def json_to_metadata(json_path: Path) -> Union[Metadata, MetadataGroup]:
    """
    Read metadata from a json file

    Parameters
    ----------
    json_path : Path
        The JSON file

    Returns
    -------
    Union[Metadata, MetadataGroup]
        Returns Metadata or MetadataGroup depending on contents of file

    Raises
    ------
    MetadataReadError
        If JSON file is unrecognised header type
    """
    import json
    from resistics.errors import MetadataReadError

    with json_path.open("r") as f:
        json_data = json.load(f)
    metadata_type = json_data["type"]
    content = json_data["content"]
    if "MetadataGroup" in metadata_type:
        logger.info(f"Found MetadataGroup in file {json_path}")
        return MetadataGroup(content)
    elif "Metadata" in metadata_type:
        logger.info(f"Found Metadata in file {json_path}")
        return Metadata(content)
    else:
        logger.error(f"Unrecognised type of metadata in {json_path}")
        raise MetadataReadError(json_path)


def history_to_json(history: ProcessHistory, json_path: Path) -> None:
    """
    Save ProcessHistory as a json file

    Parameters
    ----------
    history : ProcessHistory
        The process history
    json_path : Path
        Path to write JSON data to
    """
    import resistics
    import pandas as pd
    import json

    out_dict = {
        "created_by": "resistics",
        "created_on_local": serialize(pd.Timestamp.now(tz=None)),
        "created_on_utc": serialize(pd.Timestamp.utcnow()),
        "version": resistics.__version__,
        "type": history.type_to_string(),
        "content": history.to_dict(),
    }
    with json_path.open("w") as f:
        json.dump(out_dict, f)


def json_to_history(yaml_path: Path) -> ProcessHistory:
    return ProcessHistory()
