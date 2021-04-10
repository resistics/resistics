"""
Common resistics functions and classes used throughout the package
"""
from logging import getLogger
from typing import Type, List, Tuple, Union, Dict, Set, Any
from typing import Collection, Iterator, Callable, Optional
from pathlib import Path
import numpy as np
import pandas as pd

logger = getLogger(__name__)


serialize_fncs: Dict[Type[Any], Callable] = {
    str: (lambda x: x),
    float: (lambda x: x),
    int: (lambda x: x),
    bool: (lambda x: str(x)),
    pd.Timestamp: (lambda x: x.isoformat()),
    pd.Timedelta: (lambda x: x.total_seconds()),
    list: (lambda x: ", ".join(x)),
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
    """
    if type(x) not in serialize_fncs:
        raise ValueError(f"Unable to serialize x with type {type(x)}")
    fnc = serialize_fncs[type(x)]
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
        logger.warning(f"File path is not a file")
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
        logger.warning(f"Directory path is not a file")
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
        raise FileNotFoundError("Path does not exist")
    if not dir_path.is_dir():
        raise NotDirectoryError(dir_path)


def dir_contents(dir_path: Path) -> Tuple[List[Path], List[Path]]:
    """
    Get contents of directory

    Includes both files and directories

    Parameters
    ----------
    path : Path
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
    path : str
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
    path : str
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
    .. doctest::

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
    .. doctest::

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
    .. doctest::

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
    .. doctest::

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
    .. doctest::

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
    .. doctest::

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
    float_formatter = lambda x: f"{x:.{precision}{style}}"
    output_str = np.array2string(
        data, separator=sep, formatter={"float_kind": float_formatter}
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
    .. doctest::

        >>> from resistics.common import list_to_string
        >>> list_to_string(["a", "b", "c"])
        'a, b, c'
        >>> list_to_string([1,2,3])
        '1, 2, 3'
    """
    output_str = ""
    for val in lst:
        output_str += f"{val}, "
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
    .. doctest::

        >>> from resistics.common import list_to_ranges
        >>> data = [1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21, 24, 26, 35, 40, 45]
        >>> list_to_ranges(data)
        '1-4:1,6-12:2,15-24:3,26,35-45:5'
    """
    lst = list(data) if isinstance(data, set) else data
    lst = sorted(lst)
    n = len(lst)
    formatter = lambda start, stop, step: f"{start}-{stop}:{step}"

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

    if n - scan == 1:
        result.append(str(lst[scan]))
    elif n - scan == 2:
        result.append(",".join(map(str, lst[scan:])))

    return ",".join(result)


def format_value(val: Any, format_type: Type) -> Any:
    """
    Format a value

    Parameters
    ----------
    val : Any
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
    .. doctest::

        >>> from resistics.common import format_value
        >>> format_value("5", int)
        5
    """
    if isinstance(val, format_type):
        return val
    try:
        val = format_type(val)
    except:
        raise TypeError(f"Unable to convert {val} to type {format_type}")
    return val


def format_dict(
    in_dict: Dict[str, Any], specs: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Format the values in a dictionary

    .. warning::

        If a key is not present in in_dict and is present in the specifications but without a default, a KeyError will be raised.

    Parameters
    ----------
    in_dict : Dict[str, Any]
        The dictionary to format
    specs : Dict[str, Dict[str, Any]]
        Dictionary mapping key to key specifications type and default value

    Returns
    -------
    Dict[str, Any]
        Dictionary with values formatted

    Raises
    ------
    KeyError
        If a key in the specifications has no default and does not exist in the in_dict

    Examples
    --------
    An example where a default is provided for a missing header

    .. doctest::

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

    def summary(self, symbol_start: str = "*", symbol_end = "-") -> None:
        """
        Print a summary of the class

        Parameters
        ----------
        symbol : str, optional
            The symbol to use in the summary prints, by default "-"
        """
        name = str(self.__class__)
        length = len(name) + 10
        print(5 * symbol_start + "Start summary" + ((length - 18) * symbol_start))
        print(5 * symbol_start + name + 5 * symbol_start)
        print(self.to_string())
        print(5 * symbol_end + "End summary" + (length - 16) * symbol_end)


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
            "run_on_local": pd.Timestamp.now(None).isoformat(),
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

    def to_record(self, record) -> ProcessRecord:
        """
        Convert the process history to a record, useful for when merging histories
        """
        raise NotImplementedError("Still to be implemented")

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

    Resistics processes perform operations on data (including read and write operations)
    Each time a ResisticsProcess child class is run, it will add a process record to the dataset

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

        These parameters are expected to be primatives and should be sufficient to reinitialise the process and re-run the data. The base class assumes all class variables meet this description. This Shoul

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
        Any preparation logic should be placed in the prepare method of a child class

        Where no prepare logic is required, this method does not have to be implemented in child classes
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


class Headers(ResisticsBase):
    """
    A class for header data

    Examples
    --------
    .. doctest::

        >>> from resistics.common import Headers
        >>> headers_dict = {"a": "12", "b": "something"}
        >>> headers = Headers(headers_dict)
        >>> headers.to_dict()
        {'a': '12', 'b': 'something'}
        >>> headers.summary()
        -----------Summary---------------
        <class 'resistics.common.Headers'>
        {'a': '12', 'b': 'something'}
        ---------------------------------
        >>> headers["a"]
        '12'
        >>> headers["a"] = 15
        >>> headers.summary()
        -----------Summary---------------
        <class 'resistics.common.Headers'>
        {'a': 15, 'b': 'something'}
        ---------------------------------

    If specifications are provided, updating a header value will be checked against the specifications

    .. doctest::

        >>> from resistics.common import Headers
        >>> headers_dict = {"a": "12", "b": "something"}
        >>> spec = {"a": {"type": int, "default": 0}}
        >>> headers = Headers(headers_dict, spec)
        >>> headers.summary()
        -----------Summary---------------
        <class 'resistics.common.Headers'>
        {'a': 12, 'b': 'something'}
        ---------------------------------
        >>> headers["a"] = "try to set to string"
        Traceback (most recent call last):
        ...
        TypeError: Unable to convert try to set to string to type <class 'int'>
        >>> headers["b"] = 12
        >>> headers.summary()
        -----------Summary---------------
        <class 'resistics.common.Headers'>
        {'a': 12, 'b': 12}
        ---------------------------------
    """

    def __init__(
        self,
        headers: Dict[str, Any],
        specs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """
        Initialise headers

        Providing a specifications dictionary will automatically format header values and insert default values for missing header keys.

        Parameters
        ----------
        headers : Dict[str, Any]
            Header dictionary
        specs : Optional[Dict[str, Dict[str, Any]]], optional
            Mapping of header value to a dictionary with type and default information, by default None. If no specifications are provided, no checking is done of header value type and no defaults are added for missing headers.
        """
        self._headers = dict(headers)
        self._specs = specs
        if self._specs is not None:
            self._headers = format_dict(self._headers, self._specs)

    def __getitem__(self, header: str) -> Any:
        """
        Get a header value

        Parameters
        ----------
        header : str
            The header

        Returns
        -------
        Any
            The value

        Raises
        ------
        HeaderNotFoundError
            If key does not exist
        """
        from resistics.errors import HeaderNotFoundError

        if header not in self._headers:
            logger.error(f"{header} does not exist in headers")
            raise HeaderNotFoundError(header, self.keys())
        return self._headers[header]

    def __setitem__(self, header: str, val: Any) -> None:
        """
        Set a header value

        Type checking will be performed against a specification if available for the header. If no specification available for the header, the value will be added without checking

        Parameters
        ----------
        header : str
            The header
        val : Any
            The value to set the header to
        """
        if header not in self._headers:
            logger.info(f"Adding new header {header} with value {val}")
        if self._specs is not None and header in self._specs:
            spec_type = self._specs[header]["type"]
            logger.info(f"Specifications for this header, required type {spec_type}")
            val = format_value(val, spec_type)
        self._headers[header] = val

    def __iter__(self) -> Iterator:
        """Get an iterator over the headers"""
        return self._headers.__iter__()

    def keys(self) -> List[str]:
        """
        Get a list of headers

        Returns
        -------
        List[str]
            A list of headers
        """
        return list(self._headers.keys())

    def to_dict(self) -> Dict[str, Any]:
        """
        Get headers as a dictionary

        Returns
        -------
        Dict
            Get a dictonary of the header key to header value
        """
        return dict(self._headers)

    def to_series(self) -> pd.Series:
        """
        Get the headers as a pandas Series

        Returns
        -------
        pd.Series
            The headers as a pandas Series with headers as indices and values as the values
        """
        return pd.Series(self._headers)

    def to_string(self) -> str:
        """
        Get headers as string

        Returns
        -------
        str
            String representation of headers
        """
        outstr = f"{self.type_to_string()}\n"
        return outstr + str(self._headers)

    def serialize(self) -> Dict[str, Any]:
        """
        Serialize some of header entries to allow writing out as a json file

        Returns
        -------
        Dict[str, Any]
            Serialized dictionary
        """
        output = {}
        for header, val in self._headers.items():
            header_type = type(val)
            if header_type in serialize_fncs:
                output[header] = serialize_fncs[header_type](val)
            else:
                output[header] = val
        return output


def template_headers(
    specs: Dict[str, Dict[str, Any]], vals: Optional[Dict[str, Any]] = None
) -> Headers:
    """
    Get a template Headers instance given a set of header specifications (e.g. time, calibration) and optional override values.

    Parameters
    ----------
    specs : Dict[str, Dict[str, Any]]
        Header specifications
    vals : Optional[Dict[str, Any]], optional
        Values for particular headers which will be used instead of defaults in the specification, by default None

    Returns
    -------
    Headers
        A headers instance
    """
    headers_dict = {}
    for key, spec in specs.items():
        headers_dict[key] = spec["default"]
        if vals is not None and key in vals:
            headers_dict[key] = vals[key]
    return Headers(headers_dict, specs)


class DatasetHeaders(ResisticsBase):
    """
    Object for holding dataset headers, with some global dataset headers that apply to all channels and then some channel specific headers

    Examples
    --------
    .. doctest::

        >>> from resistics.common import DatasetHeaders
        >>> dataset = {"fs":512, "n_samples": 512000}
        >>> channel = {"Ex": {"sensor": "MFS", "serial": 100}, "Ey": {"sensor": "Phnx", "serial": 20}}
        >>> headers = DatasetHeaders(dataset, channel)
        >>> headers.summary() # doctest: +NORMALIZE_WHITESPACE
        -----------Summary---------------
        <class 'resistics.common.DatasetHeaders'>
        Dataset headers
                <class 'resistics.common.Headers'>
                {'fs': 512, 'n_samples': 512000}
        Channel Ex headers
                <class 'resistics.common.Headers'>
                {'sensor': 'MFS', 'serial': 100}
        Channel Ey headers
                <class 'resistics.common.Headers'>
                {'sensor': 'Phnx', 'serial': 20}
        ---------------------------------
        >>> headers["fs"]
        512
        >>> headers["Ex", "sensor"]
        'MFS'
        >>> headers["fs"] = 128
        >>> headers["Ex", "sensor"] = "this is a test"
        >>> headers.summary() # doctest: +NORMALIZE_WHITESPACE
        -----------Summary---------------
        <class 'resistics.common.DatasetHeaders'>
        Dataset headers
                <class 'resistics.common.Headers'>
                {'fs': 128, 'n_samples': 512000}
        Channel Ex headers
                <class 'resistics.common.Headers'>
                {'sensor': 'this is a test', 'serial': 100}
        Channel Ey headers
                <class 'resistics.common.Headers'>
                {'sensor': 'Phnx', 'serial': 20}
        ---------------------------------
    """

    def __init__(
        self,
        dataset_headers: Dict[str, Any],
        chan_headers: Dict[str, Dict[str, Any]],
        dataset_specs: Optional[Dict[str, Dict[str, Any]]] = None,
        chan_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """
        Initialise dataset headers

        Parameters
        ----------
        dataset_headers : Dict[str, Any]
            The overall dataset headers as a dictionary
        chan_headers : Dict[str, Dict[str, Any]]
            The individual channel headers as a dictionary of dictionaries
        dataset_specs : Optional[Dict[str, Dict[str, Any]]], optional
            Specifications for the dataset headers, by default None
        chan_specs : Optional[Dict[str, Dict[str, Any]]], optional
            Specifications mapping for channel headers, by default None
        """
        self._dataset_headers = Headers(dataset_headers, dataset_specs)
        self._chan_headers = dict()
        for chan, chan_header in chan_headers.items():
            self._chan_headers[chan] = Headers(chan_header, chan_specs)
        self.chans = list(self._chan_headers.keys())

    def dataset_keys(self) -> List[str]:
        """
        Get dataset header keys

        Returns
        -------
        List[str]
            Dataset header keys
        """
        return self._dataset_headers.keys()

    def chan_keys(self, chan: str) -> List[str]:
        """
        Get the header keys for a channel

        Parameters
        ----------
        chan : str
            The channel

        Returns
        -------
        List[str]
            A list of header keys
        """
        return self._chan_headers[chan].keys()

    def __getitem__(self, args: Union[str, Tuple[str, str]]) -> Any:
        """
        Get a dataset or channel header

        If one argument is provided, this is considred to be a dataset header.
        If two arguments are provided, this is considered to be the channel in argument 0 and the header in argument 1.

        Parameters
        ----------
        args : Union[Tuple[str], Tuple[str, str]]
            Single header for a dataset header, i.e. [header_name] or [chan, header_name] for a channel header

        Returns
        -------
        Any
            Header value

        Raises
        ------
        ValueError
            If the number of arguments is greater than two
        """
        if isinstance(args, str):
            return self.dataset(args)
        elif isinstance(args, tuple) and len(args) == 2:
            return self.chan(args[0], args[1])
        if len(args) > 2:
            raise ValueError("Arguments are have been incorrectly specified")

    def __setitem__(self, key: Union[str, Tuple[str, str]], val: Any) -> None:
        """
        Set a header for datasets or specific channels

        If two arguments are provided, this will attempt to set the value of a dataset header.
        If three arguments are provided, this will attempt to set the value of a channel (argument 0) header (argument 1).

        Parameters
        ----------
        key : Union[str, Tuple[str, str]]
            The header for a dataset header or a Tuple [str, str] for channel headers, the first string specifying the channel and the second specifyig the channel header
        val : Any
            The value to set the header to

        Raises
        ------
        ValueError
            If less than two or more than three arguments are received
        """
        if isinstance(key, str):
            return self.set_dataset(key, val)
        elif isinstance(key, tuple) and len(key) == 2:
            chan, header = key
            return self.set_chan(chan, header, val)
        else:
            raise ValueError("Arguments are have been incorrectly specified")

    def dataset(self, header: str) -> Any:
        """
        Get a dataset header

        Parameters
        ----------
        header : str
            The header name

        Returns
        -------
        Any
            The header value

        Raises
        ------
        KeyError
            If header does not exist
        """
        if header not in self._dataset_headers:
            raise KeyError(f"Unknown header {header}")
        return self._dataset_headers[header]

    def set_dataset(self, header: str, val: Any) -> None:
        """
        Set a dataset header value

        Parameters
        ----------
        header : str
            The header name
        val : Any
            The new value

        Raises
        ------
        KeyError
            If header does not exist
        """
        if header not in self._dataset_headers:
            raise KeyError(f"Unknown header {header}")
        self._dataset_headers[header] = val

    def chan(self, chan: str, header: str) -> Any:
        """
        Get a channel header

        Parameters
        ----------
        chan : str
            The channel
        header : str
            The header name

        Returns
        -------
        Any
            The header value

        Raises
        ------
        KeyError
            If the channel header does not exist
        """
        check_chan(chan, self.chans)
        if header not in self._chan_headers[chan]:
            raise KeyError(f"Header {header} not found for channel {chan}")
        return self._chan_headers[chan][header]

    def set_chan(self, chan: str, header: str, val: Any) -> None:
        """
        Set a channel header value

        Parameters
        ----------
        chan : str
            The channel
        header : str
            The header name
        val : Any
            The header value
        Raises
        ------
        KeyError
            If the channel header does not exist
        """
        check_chan(chan, self.chans)
        if header not in self._chan_headers[chan]:
            raise KeyError(f"Header {header} not found for channel {chan}")
        self._chan_headers[chan][header] = val

    def copy(self) -> "DatasetHeaders":
        """
        Get a copy of the dataset headers

        Returns
        -------
        DatasetHeaders
            A copy of the dataset headers
        """
        headers_dict = self.to_dict()
        return DatasetHeaders(headers_dict["dataset"], headers_dict["channel"])

    def to_dict(self) -> Dict[str, Any]:
        """
        Get the dataset headers as a dictionary

        Returns
        -------
        Dict[str, Any]
            Dataset headers as dictionary
        """
        header_dict = {"dataset": self._dataset_headers.to_dict()}
        header_dict["channel"] = {}
        for chan, chan_header in self._chan_headers.items():
            header_dict["channel"][chan] = chan_header.to_dict()
        return header_dict

    def to_string(self) -> str:
        """
        Get string representation of dataset headers

        Returns
        -------
        str
            Class information as a string
        """
        outstr = f"{self.type_to_string()}\n"
        outstr += "Dataset headers\n"
        header_string = self._dataset_headers.to_string()
        header_string = header_string.replace("\n", "\n\t")
        outstr += f"\t{header_string}\n"
        for chan in self._chan_headers:
            outstr += f"Channel {chan} headers\n"
            header_string = self._chan_headers[chan].to_string()
            header_string = header_string.replace("\n", "\n\t")
            outstr += f"\t{header_string}\n"
        outstr = outstr.rstrip("\n")
        return outstr

    def serialize(self) -> Dict[str, Any]:
        """
        Get the dataset headers as a serialized dictionary

        Returns
        -------
        Dict[str, Any]
            Dataset headers as dictionary
        """
        header_dict = {"dataset": self._dataset_headers.serialize()}
        header_dict["channel"] = {}
        for chan, chan_header in self._chan_headers.items():
            header_dict["channel"][chan] = chan_header.serialize()
        return header_dict


def template_dataset_headers(
    chans: List[str],
    dataset_specs: Dict[str, Dict[str, Any]],
    chan_specs: Dict[str, Dict[str, Any]],
    dataset_vals: Optional[Dict[str, Any]] = None,
    chan_vals: Optional[Dict[str, Dict[str, Any]]] = None,
) -> DatasetHeaders:
    """
    Get a template DatasetHeaders

    Parameters
    ----------
    chans : List[str]
        Channels in the dataset
    dataset_specs : Dict[str, Dict[str, Any]]
        Specifications for the dataset header
    chan_specs : Dict[str, Dict[str, Any]]
        Specifications for the channel headers
    dataset_vals : Optional[Dict[str, Any]], optional
        Any known values for the dataset headers, by default None
    chan_vals : Optional[Dict[str, Dict[str, Any]]], optional
        Any known values for the channel headers, by default None

    Returns
    -------
    DatasetHeaders
        The DatasetHeader
    """
    dataset_headers = template_headers(dataset_specs, dataset_vals).to_dict()
    chan_headers = {}
    for chan in chans:
        vals = None
        if chan_vals is not None and chan in chan_vals:
            vals = chan_vals[chan]
        chan_headers[chan] = template_headers(chan_specs, vals).to_dict()
    return DatasetHeaders(dataset_headers, chan_headers, dataset_specs, chan_specs)


def headers_to_json(headers: Union[Headers, DatasetHeaders], json_path: Path) -> None:
    """
    Write a header to a file

    Parameters
    ----------
    headers : Union[Header, DatasetHeader]
        A header or dataset header
    json_path : Path
        The path to write to
    """
    import resistics
    import pandas as pd
    import json

    out_dict = {
        "created_by": "resistics",
        "created_on_local": serialize_fncs[pd.Timestamp](pd.Timestamp.now(tz=None)),
        "created_on_utc": serialize_fncs[pd.Timestamp](pd.Timestamp.utcnow()),
        "version": resistics.__version__,
        "type": headers.type_to_string(),
        "content": headers.serialize(),
    }
    with json_path.open("w") as f:
        json.dump(out_dict, f)


def json_to_headers(json_path: Path) -> Union[Headers, DatasetHeaders]:
    """
    Read headers from a json file

    Parameters
    ----------
    json_path : Path
        The JSON file

    Returns
    -------
    Union[Headers, DatasetHeaders]
        Returns Headers or DatasetHeaders depending on type of headers in the file

    Raises
    ------
    ValueError
        If JSON file is unrecognised header type
    """
    import json

    with json_path.open("r") as f:
        json_data = json.load(f)
    header_type = json_data["type"]
    content = json_data["content"]
    if "DatasetHeaders" in header_type:
        logger.info(f"Found DatasetHeaders in file {json_path}")
        dataset = content["dataset"]
        channel = content["channel"]
        return DatasetHeaders(dataset, channel)
    elif "Headers" in header_type:
        return Headers(content)
    else:
        logger.error(f"Unrecognised type of headers in {json_path}")
        raise ValueError(f"Unable to read headers in {json_path}")


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
        "created_on_local": serialize_fncs[pd.Timestamp](pd.Timestamp.now(tz=None)),
        "created_on_utc": serialize_fncs[pd.Timestamp](pd.Timestamp.utcnow()),
        "version": resistics.__version__,
        "type": history.type_to_string(),
        "content": history.to_dict(),
    }
    with json_path.open("w") as f:
        json.dump(out_dict, f)


def json_to_history(yaml_path: Path) -> ProcessHistory:
    return ProcessHistory()