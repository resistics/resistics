"""
Common resistics functions and classes used throughout the package
"""
from loguru import logger
from typing import List, Tuple, Union, Dict
from typing import Any, Collection, Optional, Type
from pathlib import Path
from pydantic import BaseModel, Field, validator
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


def any_electric(chans: List[str]) -> bool:
    """
    Return boolean if any channels in list are electric

    Parameters
    ----------
    chans : List[str]
        List of channels

    Returns
    -------
    bool
        True if any electric

    Examples
    --------
    List with no electric channels should evaluate to False

    >>> from resistics.common import any_electric
    >>> chans = ["Hx", "Hy", "Hz"]
    >>> any_electric(chans)
    False

    Now with one electric channel

    >>> chans = ["Ex", "Hy", "Hz"]
    >>> any_electric(chans)
    True
    """
    return np.any([is_electric(x) for x in chans])


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


def any_magnetic(chans: List[str]) -> bool:
    """
    Return boolean if any channels in list are magnetic

    Parameters
    ----------
    chans : List[str]
        List of channels

    Returns
    -------
    bool
        True if any magnetic

    Examples
    --------
    List with no magnetic channels should evaluate to False

    >>> from resistics.common import any_magnetic
    >>> chans = ["Ex", "Ey", "Ez"]
    >>> any_magnetic(chans)
    False

    Now with one magnetic channel

    >>> chans = ["Ex", "Ey", "Hz"]
    >>> any_magnetic(chans)
    True
    """
    return np.any([is_magnetic(x) for x in chans])


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


class ResisticsFile(ResisticsModel):
    """Required information for writing out a resistics file"""

    created_on_local: datetime = Field(default_factory=datetime.now)
    created_on_utc: datetime = Field(default_factory=datetime.utcnow)
    version: Optional[str] = Field(default_factory=get_version)


class Metadata(ResisticsModel):
    """Parent class for metadata"""

    @validator("n_chans", check_fields=False, always=True)
    def validate_n_chans(cls, value: Union[None, int], values: Dict[str, Any]) -> int:
        """Initialise number of channels"""
        if value is None:
            return len(values["chans"])
        return value


class WriteableMetadata(Metadata):
    """Base class for writeable metadata"""

    file_info: Optional[ResisticsFile] = None
    """Information about a file, relevant if writing out or reading back in"""

    def write(self, json_path: Path):
        """
        Write out JSON metadata file

        Parameters
        ----------
        json_path : Path
            Path to write JSON file
        """
        self.file_info = ResisticsFile()
        with json_path.open("w") as f:
            f.write(self.json())


class Record(ResisticsModel):
    """
    Class to hold a record

    A record holds information about a process that was run. It is intended to
    track processes applied to data, allowing a process history to be saved
    along with any datasets.

    Examples
    --------
    A simple example of creating a process record

    >>> from resistics.common import Record
    >>> messages = ["message 1", "message 2"]
    >>> record = Record(
    ...     creator={"name": "example", "parameter1": 15},
    ...     messages=messages,
    ...     record_type="example"
    ... )
    >>> record.summary()
    {
        'time_local': '...',
        'time_utc': '...',
        'creator': {'name': 'example', 'parameter1': 15},
        'messages': ['message 1', 'message 2'],
        'record_type': 'example'
    }
    """

    time_local: datetime = Field(default_factory=datetime.now)
    """The local time when the process ran"""
    time_utc: datetime = Field(default_factory=datetime.utcnow)
    """The UTC time when the process ran"""
    creator: Dict[str, Any]
    """The creator and its parameters as a dictionary"""
    messages: List[str]
    """Any messages in the record"""
    record_type: str
    """The record type"""


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
                'creator': {
                    'name': 'example1',
                    'a': 5,
                    'b': -7.0
                },
                'messages': ['Message 1', 'Message 2'],
                'record_type': 'process'
            },
            {
                'time_local': '...',
                'time_utc': '...',
                'creator': {
                    'name': 'example2',
                    'a': 'parzen',
                    'b': -21
                },
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
    creator: Dict[str, Any],
    messages: Union[str, List[str]],
    record_type: str = "process",
    time_utc: Optional[datetime] = None,
    time_local: Optional[datetime] = None,
) -> Record:
    """
    Get a process record

    Parameters
    ----------
    creator : Dict[str, Any]
        The creator and its parameters as a dictionary
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
    >>> record = get_record(
    ...     creator={"name": "example", "a": 5, "b": -7.0},
    ...     messages="a message"
    ... )
    >>> record.creator
    {'name': 'example', 'a': 5, 'b': -7.0}
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
        creator=creator,
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
                'creator': {
                    'name': 'example1',
                    'a': 5,
                    'b': -7.0
                },
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
                'creator': {
                    'name': 'example1',
                    'a': 5,
                    'b': -7.0
                },
                'messages': ['Message 1', 'Message 2'],
                'record_type': 'process'
            },
            {
                'time_local': '...',
                'time_utc': '...',
                'creator': {
                    'name': 'example2',
                    'a': 'parzen',
                    'b': -21
                },
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


class ResisticsProcess(ResisticsModel):
    """
    Base class for resistics processes

    Resistics processes perform operations on data (including read and write
    operations). Each time a ResisticsProcess child class is run, it should add
    a process record to the dataset
    """

    _types: Dict[str, type] = {}
    name: Optional[str]

    def __init_subclass__(cls) -> None:
        """
        Used to automatically register child processors in `_types`

        When a resistics process is imported, it is added to the base
        ResisticsProcess _types variable. Later, this dictionary of class types
        can be used to initialise processes from a dictonary.

        The intention of this method is to support initialising processes from
        JSON files.
        """
        cls._types[cls.__name__] = cls

    @classmethod
    def __get_validators__(cls):
        """Get the validators that will be used by pydantic"""
        yield cls.validate

    @classmethod
    def validate(
        cls, value: Union["ResisticsProcess", Dict[str, Any]]
    ) -> "ResisticsProcess":
        """
        Validate a ResisticsProcess in another pydantic class

        Parameters
        ----------
        value : Union[ResisticsProcess, Dict[str, Any]]
            A ResisticsProcess child class or a dictionary

        Returns
        -------
        ResisticsProcess
            A ResisticsProcess child class

        Raises
        ------
        ValueError
            If the value is neither a ResisticsProcess or a dictionary
        KeyError
            If name is not in the dictionary
        ValueError
            If initialising from dictionary fails

        Examples
        --------
        The following example will show how a generic ResisticsProcess child
        class can be instantiated from ResisticsProcess using a dictionary,
        which might be read in from a JSON configuration file.

        >>> from resistics.common import ResisticsProcess
        >>> from resistics.decimate import DecimationSetup
        >>> process = {"name": 'DecimationSetup', "n_levels": 8, "per_level": 5, "min_samples": 256, "div_factor": 2, "eval_freqs": None}
        >>> ResisticsProcess(**process)
        ResisticsProcess(name='DecimationSetup')

        This is not what was expected. To get the right result, the class
        validate method needs to be used. This is done automatically by
        pydantic.

        >>> ResisticsProcess.validate(process)
        DecimationSetup(name='DecimationSetup', n_levels=8, per_level=5, min_samples=256, div_factor=2, eval_freqs=None)

        That's better. Note that errors will be raised if the dictionary is not
        formatted as expected.

        >>> process = {"n_levels": 8, "per_level": 5, "min_samples": 256, "div_factor": 2, "eval_freqs": None}
        >>> ResisticsProcess.validate(process)
        Traceback (most recent call last):
        ...
        KeyError: 'No name provided for initialisation of process'

        This functionality is most useful in the resistics configurations which
        can be saved as JSON files. The default configuration uses the default
        parameterisation of DecimationSetup.

        >>> from resistics.letsgo import Configuration
        >>> config = Configuration(name="example1")
        >>> config.dec_setup
        DecimationSetup(name='DecimationSetup', n_levels=8, per_level=5, min_samples=256, div_factor=2, eval_freqs=None)

        Now create another configuration with a different setup by passing a
        dictionary. In practise, this dictionary will most likely be read in
        from a configuration file.

        >>> setup = DecimationSetup(n_levels=4, per_level=3)
        >>> test_dict = setup.dict()
        >>> test_dict
        {'name': 'DecimationSetup', 'n_levels': 4, 'per_level': 3, 'min_samples': 256, 'div_factor': 2, 'eval_freqs': None}
        >>> config2 = Configuration(name="example2", dec_setup=test_dict)
        >>> config2.dec_setup
        DecimationSetup(name='DecimationSetup', n_levels=4, per_level=3, min_samples=256, div_factor=2, eval_freqs=None)

        This method allows the saving of a configuration with custom processors
        in a JSON file which can be loaded and used again.
        """
        if isinstance(value, ResisticsProcess):
            return value
        if not isinstance(value, dict):
            raise ValueError(
                "ResisticsProcess unable to initialise from type {type(value)}"
            )
        if "name" not in value:
            raise KeyError("No name provided for initialisation of process")
        name = value.pop("name")
        try:
            return cls._types[name](**value)
        except Exception:
            raise ValueError(f"Unable to initialise {name} from dictionary")

    @validator("name", always=True)
    def validate_name(cls, value: Union[str, None]) -> str:
        """Inialise the name attribute of the resistics process"""
        if value is None:
            return cls.__name__
        return value

    def parameters(self) -> Dict[str, Any]:
        """
        Return any process parameters incuding the process name

        These parameters are expected to be primatives and should be sufficient
        to reinitialise the process and re-run the data. The base class assumes
        all class variables meet this description.

        Returns
        -------
        Dict[str, Any]
            Dictionary of parameters
        """
        import json

        return json.loads(self.json())

    def run(self, *args: Any):
        """
        Run the process

        Parameters
        ----------
        args : Any
            The parameters for the process, should be detailed for child
            processors

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
        return get_record(self.parameters(), messages)


class ResisticsWriter(ResisticsProcess):
    """
    Parent process for data writers

    Parameters
    ----------
    overwrite : bool, optional
        Boolean flag for overwriting the existing data, by default False
    """

    overwrite: bool = True

    def _check_dir(self, dir_path: Path) -> bool:
        """Check the output directory"""
        if dir_path.exists() and not self.overwrite:
            logger.error(f"Write path {dir_path} exists and overwrite is False")
            return False
        if dir_path.exists():
            logger.warning(f"Overwriting existing directory {dir_path}")
        if not dir_path.exists():
            logger.info(f"Directory {dir_path} not found. Creating including parents.")
            dir_path.mkdir(parents=True)
        return True

    def _get_record(self, dir_path: Path, data_type: Type):
        """Get a process record for the writer"""
        return super()._get_record([f"Writing out {data_type.__name__} to {dir_path}"])


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
