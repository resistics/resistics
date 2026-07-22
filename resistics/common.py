"""Common resistics functions and classes used throughout the package"""

from collections.abc import Callable, Collection, Mapping
from datetime import UTC, datetime
from enum import StrEnum
from inspect import signature
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator

from resistics.sampling import RSDateTime, datetime_to_string


def json_fallback(value: Any) -> Any:
    """Fallback serializer for values Pydantic v2 cannot encode directly.

    :param value: Value that the normal Pydantic serializer could not encode.
    :return: A JSON-compatible timestamp, callable name, or string.
    """
    if isinstance(value, RSDateTime):
        return datetime_to_string(value)
    if isinstance(value, Callable):
        return getattr(value, "__name__", str(value))
    return str(value)


def _summary_lines(value: Any, level: int = 0) -> list[str]:
    """Render JSON-compatible model data with stable four-space nesting.

    :param value: Value decoded from a model's JSON representation.
    :param level: Current nesting level.

    :return: Rendered output lines without a trailing newline.
    """
    indentation = " " * (4 * level)
    if isinstance(value, dict):
        if not value:
            return ["{}"]
        if level == 1 and len(indentation) + len(repr(value)) <= 88:
            return [repr(value)]
        output = ["{"]
        items = list(value.items())
        for index, (key, item) in enumerate(items):
            rendered = _summary_lines(item, level + 1)
            child_indent = " " * (4 * (level + 1))
            output.append(f"{child_indent}{key!r}: {rendered[0]}")
            output.extend(rendered[1:])
            if index < len(items) - 1:
                output[-1] += ","
        output.append(f"{indentation}}}")
        return output
    if isinstance(value, list):
        is_simple = all(not isinstance(item, (dict, list)) for item in value)
        if not value or (is_simple and len(indentation) + len(repr(value)) <= 88):
            return [repr(value)]
        output = ["["]
        for index, item in enumerate(value):
            rendered = _summary_lines(item, level + 1)
            child_indent = " " * (4 * (level + 1))
            output.append(f"{child_indent}{rendered[0]}")
            output.extend(rendered[1:])
            if index < len(value) - 1:
                output[-1] += ","
        output.append(f"{indentation}]")
        return output
    return [repr(value)]


ELECTRIC_CHANS = ["Ex", "Ey", "E1", "E2", "E3", "E4"]
MAGNETIC_CHANS = ["Hx", "Hy", "Hz", "Bx", "By", "Bz"]


class ProcessingCancelled(Exception):
    """Raised when a processing operation observes a cancellation request."""


class ProcessingProgressState(StrEnum):
    """Lifecycle state for a structured processing progress event.

    **Attributes**

    - **started :** — Work has begun.
    - **advanced :** — One or more work units have completed.
    - **completed :** — All work units completed successfully.
    - **cancelled :** — A cancellation request stopped the work.
    - **failed :** — Work stopped because an operation failed.
    """

    started = "started"
    advanced = "advanced"
    completed = "completed"
    cancelled = "cancelled"
    failed = "failed"


class ProcessingProgressEvent(BaseModel):
    """Serializable progress emitted by a processing operation.

    **Attributes**

    - **model_config** — Frozen Pydantic configuration rejecting unknown fields.
    - **state** — Current lifecycle state.
    - **task** — Stable task identifier suitable for programmatic consumers.
    - **current** — Number of completed work units.
    - **total** — Total work units when known.
    - **message** — Human-readable progress description.
    - **stage_id** — Owning flow stage when executed through a flow.
    - **node_id** — Owning flow node when executed through a flow.
    - **process** — Qualified process class when executed through a flow.
    - **error** — Failure detail for failed events.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    state: ProcessingProgressState
    task: str
    current: int = Field(ge=0)
    total: int | None = Field(default=None, ge=0)
    message: str
    stage_id: str | None = None
    node_id: str | None = None
    process: str | None = None
    error: str | None = None

    @model_validator(mode="after")
    def validate_current_not_after_total(self) -> "ProcessingProgressEvent":
        """Reject progress beyond a known total.

        :return: Validated progress event.

        :raises ValueError: If completed work exceeds the known total.
        """
        if self.total is not None and self.current > self.total:
            raise ValueError("current progress cannot exceed total progress")
        return self


ProcessingProgressCallback = Callable[[ProcessingProgressEvent], None]
CancellationCallback = Callable[[], bool]


def validate_output_label(value: str) -> str:
    """Validate one output-label path component used for derived artifacts.

    :param value: Candidate label, including any surrounding whitespace.
    :return: The stripped, safe path component.
    :raises ValueError: If the label is empty, special, or contains a path or
        null separator.
    """
    value = value.strip()
    if (
        not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError("output_label must be a non-empty single path component")
    return value


def get_version() -> str:
    """Get the installed Resistics version.

    :return: Package version string.
    """
    import resistics

    return resistics.__version__


def is_file(file_path: Path) -> bool:
    """Check if a path exists and points to a file

    :param file_path: The path to check

    :return: True if it exists and is a file, False otherwise
    """
    if not file_path.exists():
        logger.warning(f"File path {file_path} does not exist")
        return False
    if not file_path.is_file():
        logger.warning(f"File path {file_path} is not a file")
        return False
    return True


def assert_file(file_path: Path) -> None:
    """Require that a file exists

    :param file_path: The path to check

    :raises FileNotFoundError: If the path does not exist
    :raises NotFileError: If the path is not a file
    """
    from resistics.errors import NotFileError

    if not file_path.exists():
        raise FileNotFoundError(f"Path {file_path} not found")
    if not file_path.is_file():
        raise NotFileError(file_path)


def save_compressed_arrays(file_path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Persist named NumPy arrays in one compressed archive.

    :param file_path: Archive path, with or without the ``.npz`` suffix.
    :param arrays: Named arrays to persist.
    """
    np.savez_compressed(
        file_path,
        **arrays,  # pyrefly: ignore[bad-argument-type]
    )


def is_dir(dir_path: Path) -> bool:
    """Check if a path exists and points to a directory

    :param dir_path: The path to check

    :return: True if it exists and is a directory, False otherwise
    """
    if not dir_path.exists():
        logger.warning(f"Directory path {dir_path} does not exist")
        return False
    if not dir_path.is_dir():
        logger.warning(f"Directory path {dir_path} is not a directory")
        return False
    return True


def assert_dir(dir_path: Path) -> None:
    """Require that a path is a directory

    :param dir_path: Path to check

    :raises FileNotFoundError: If the path does not exist
    :raises NotDirectoryError: If the path is not a directory
    """
    from resistics.errors import NotDirectoryError

    if not dir_path.exists():
        raise FileNotFoundError(f"Path {dir_path} does not exist")
    if not dir_path.is_dir():
        raise NotDirectoryError(dir_path)


def dir_contents(dir_path: Path) -> tuple[list[Path], list[Path]]:
    """Get contents of directory

    Includes both files and directories

    :param dir_path: Parent directory path

    :return:
        - **dirs** — List of directories
        - **files** — List of files excluding hidden files

    :raises PathNotFoundError: Path does not exist
    :raises NotDirectoryError: Path is not a directory
    """
    from resistics.errors import NotDirectoryError, PathNotFoundError

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


def dir_files(dir_path: Path) -> list[Path]:
    """Get files in directory

    Excludes hidden files

    :param dir_path: Parent directory path

    :return: **files** — List of files excluding hidden files
    """
    _, files = dir_contents(dir_path)
    return files


def dir_subdirs(dir_path: Path) -> list[Path]:
    """Get subdirectories in directory

    Excludes hidden files

    :param dir_path: Parent directory path

    :return: **dirs** — List of subdirectories
    """
    dirs, _ = dir_contents(dir_path)
    return dirs


def known_chan(chan: str) -> bool:
    """Check whether resistics is familiar with a channel name

    :param chan: The channel name

    :return: True if it is a resistics known channel, false otherwise

    **Examples**

    ```{doctest}
    >>> from resistics.common import known_chan
    >>> known_chan("Ex")
    True
    >>> known_chan("Hy")
    True
    >>> known_chan("cat")
    False

    ```
    """
    return chan in ELECTRIC_CHANS or chan in MAGNETIC_CHANS


def is_electric(chan: str) -> bool:
    """Check if a channel is electric

    :param chan: Channel name

    :return: True if channel is electric

    **Examples**

    ```{doctest}
    >>> from resistics.common import is_electric
    >>> is_electric("Ex")
    True
    >>> is_electric("Hx")
    False

    ```
    """
    return chan in ELECTRIC_CHANS


def is_magnetic(chan: str) -> bool:
    """Check if channel is magnetic

    :param chan: Channel name

    :return: True if channel is magnetic

    **Examples**

    ```{doctest}
    >>> from resistics.common import is_magnetic
    >>> is_magnetic("Ex")
    False
    >>> is_magnetic("Hx")
    True

    ```
    """
    return chan in MAGNETIC_CHANS


def get_chan_type(chan: str) -> str:
    """Get the channel type from the channel name

    :param chan: The name of the channel

    :return: The channel type

    :raises ValueError: If the channel is not known to resistics

    **Examples**

    ```{doctest}
    >>> from resistics.common import get_chan_type
    >>> get_chan_type("Ex")
    'electric'
    >>> get_chan_type("Hz")
    'magnetic'
    >>> get_chan_type("abc")
    Traceback (most recent call last):
    ...
    ValueError: Channel abc not recognised as either electric or magnetic

    ```
    """
    if is_electric(chan):
        return "electric"
    if is_magnetic(chan):
        return "magnetic"
    raise ValueError(f"Channel {chan} not recognised as either electric or magnetic")


def check_chan(chan: str, chans: Collection[str]) -> bool:
    """Check a channel exists and raise a KeyError if not

    :param chan: The channel to check
    :param chans: A collection of channels to check against

    :return: True if all checks passed

    :raises ChannelNotFoundError: If the channel is not found in the channel list
    """
    from resistics.errors import ChannelNotFoundError

    if chan not in chans:
        logger.error(f"Channel {chan} not in channel list {chans}")
        raise ChannelNotFoundError(chan, chans)
    return True


def fs_to_string(fs: float) -> str:
    """Convert sampling frequency into a string for filenames

    :param fs: The sampling frequency

    :return: Sample frequency converted to string for the purposes of a filename

    **Examples**

    ```{doctest}
    >>> from resistics.common import fs_to_string
    >>> fs_to_string(512.0)
    '512_000000'

    ```
    """
    return (f"{fs:.6f}").replace(".", "_")


def array_to_string(
    data: np.ndarray, sep: str = ", ", precision: int = 8, scientific: bool = False
) -> str:
    """Convert an array to a string for logging or printing

    :param data: The array
    :param sep: The separator to use, by default ", "
    :param precision: Number of decimal places, by default 8. Ignored for integers.
    :param scientific: Flag for formatting floats as scientific, by default False

    :return: String representation of array

    **Examples**

    ```{doctest}
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

    ```
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

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={RSDateTime: datetime_to_string},
        validate_default=True,
    )

    def __str__(self) -> str:
        return self.to_string()

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Dump model data, preserving subclass fields for process registries.

        :param *args: Positional arguments forwarded to Pydantic.
        :param **kwargs: Keyword arguments forwarded to Pydantic.
        :return: Serialized model data.
        """
        kwargs.setdefault("serialize_as_any", True)
        return super().model_dump(*args, **kwargs)

    def model_dump_json(self, *args: Any, **kwargs: Any) -> str:
        """Dump model JSON, preserving subclass fields for process registries.

        :param *args: Positional arguments forwarded to Pydantic.
        :param **kwargs: Keyword arguments forwarded to Pydantic.
        :return: Serialized model JSON.
        """
        kwargs.setdefault("serialize_as_any", True)
        kwargs.setdefault("fallback", json_fallback)
        return super().model_dump_json(*args, **kwargs)

    def dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Provide the legacy dictionary serialization name.

        :param *args: Positional arguments forwarded to ``model_dump``.
        :param **kwargs: Keyword arguments forwarded to ``model_dump``.
        :return: Serialized model data using Pydantic v2 semantics.
        """
        return self.model_dump(*args, **kwargs)

    def json(self, *args: Any, **kwargs: Any) -> str:
        """Provide the legacy JSON serialization name.

        :param *args: Positional arguments forwarded to ``model_dump_json``.
        :param **kwargs: Keyword arguments forwarded to ``model_dump_json``.
        :return: Serialized model JSON using Pydantic v2 semantics.
        """
        return self.model_dump_json(*args, **kwargs)

    def to_string(self) -> str:
        """Render model data as readable YAML.

        :return: YAML representation preserving field order.
        """
        import json

        import yaml

        json_dict = json.loads(self.model_dump_json())
        return yaml.dump(json_dict, indent=4, sort_keys=False)

    def summary(self) -> None:
        """Print a summary of the class"""
        import json

        print("\n".join(_summary_lines(json.loads(self.model_dump_json()))))


class ResisticsFile(ResisticsModel):
    """Required information for writing out a resistics file"""

    created_on_local: datetime = Field(default_factory=datetime.now)
    created_on_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))
    version: str | None = Field(default_factory=get_version)


class Metadata(ResisticsModel):
    """Parent class for metadata"""

    @model_validator(mode="after")
    def validate_n_chans(self) -> "Metadata":
        """Initialise the channel count when the model defines channels.

        :return: Validated metadata with a derived channel count.
        """
        values = self.__dict__
        if values.get("n_chans") == 0:
            values["n_chans"] = len(values["chans"])
        return self


class WriteableMetadata(Metadata):
    """Base class for writeable metadata"""

    file_info: ResisticsFile | None = None
    """Information about a file, relevant if writing out or reading back in"""

    def write(self, json_path: Path):
        """Write out JSON metadata file

        :param json_path: Path to write JSON file
        """
        self.file_info = ResisticsFile()
        with json_path.open("w") as f:
            f.write(self.model_dump_json())


class Record(ResisticsModel):
    """Class to hold a record

    A record holds information about a process that was run. It is intended to
    track processes applied to data, allowing a process history to be saved
    along with any datasets.

    **Examples**

    A simple example of creating a process record

    ```{doctest}
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

    ```
    """

    time_local: datetime = Field(default_factory=datetime.now)
    """The local time when the process ran"""
    time_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))
    """The UTC time when the process ran"""
    creator: dict[str, Any]
    """The creator and its parameters as a dictionary"""
    messages: list[str]
    """Any messages in the record"""
    record_type: str
    """The record type"""


class History(ResisticsModel):
    """Class for storing processing history

    :param records: List of records, by default []

    **Examples**

    ```{doctest}
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

    ```
    """

    records: list[Record] = Field(default_factory=list)

    def add_record(self, record: Record):
        """Add a process record to the list

        :param record: The record to add
        """
        self.records.append(record)


def get_record(
    creator: dict[str, Any],
    messages: str | list[str],
    record_type: str = "process",
    time_utc: datetime | None = None,
    time_local: datetime | None = None,
) -> Record:
    """Get a process record

    :param creator: The creator and its parameters as a dictionary
    :param messages: The messages as either a single str or a list of strings
    :param record_type: The type of record, by default "process"
    :param time_utc: UTC time to attach to the record, by default None. If None, will default
        to UTC now
    :param time_local: Local time to attach to the record, by default None. If None, will
        defult to local now

    :return: The process record

    **Examples**

    ```{doctest}
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

    ```
    """
    if isinstance(messages, str):
        messages = [messages]
    if time_utc is None:
        time_utc = datetime.now(UTC)
    if time_local is None:
        time_local = datetime.now()
    return Record(
        time_utc=time_utc,
        time_local=time_local,
        creator=creator,
        messages=messages,
        record_type=record_type,
    )


def get_history(record: Record, history: History | None = None) -> History:
    """Get a new History instance or add a record to a copy of an existing one

    This method always makes a deepcopy of an input history to avoid any
    unplanned modifications to the inputs.

    :param record: The record
    :param history: A history to add to, by default None

    :return: History with the record added

    **Examples**

    Get a new History with a single Record

    ```{doctest}
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

    ```

    Alternatively, add to an existing History. This will make a copy of the
    original history. If a copy is not needed, the add_record method of history
    can be used.

    ```{doctest}
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

    ```
    """
    if history is None:
        return History(records=[record])
    history = History(**history.model_dump())
    history.add_record(record)
    return history


class ResisticsProcess(ResisticsModel):
    """Base class for resistics processes

    Resistics processes perform operations on data (including read and write
    operations). Each time a ResisticsProcess child class is run, it should add
    a process record to the dataset
    """

    input_types: ClassVar[dict[str, str]] = {}
    output_type: ClassVar[str | None] = None
    runtime_requirements: ClassVar[list[str]] = []
    include_in_default_parameters: ClassVar[bool] = False
    name: str = ""

    @model_validator(mode="after")
    def validate_name(self) -> "ResisticsProcess":
        """Initialise an omitted process name from its concrete class.

        :return: Validated process with a stable name.
        """
        if not self.name:
            self.name = self.__class__.__name__
        return self

    def parameters(self) -> dict[str, Any]:
        """Return any process parameters incuding the process name

        These parameters are expected to be primatives and should be sufficient
        to reinitialise the process and re-run the data. The base class assumes
        all class variables meet this description.

        :return: Dictionary of parameters
        """
        import json

        return json.loads(self.model_dump_json())

    def execute(self, inputs: dict[str, Any], context: Any) -> Any:
        """Execute this process as a flow node.

        The default preserves existing numerical ``run`` methods. Readers,
        writers, and selectors that require project or batch context override
        this method in their owning modules.

        :param inputs: Named arguments for the process ``run`` method.
        :param context: Flow execution context. Structured progress and cancellation
            callbacks are forwarded when the ``run`` method accepts them.

        :return: The value returned by the process ``run`` method.

        :raises NotImplementedError: If the process implements neither ``run`` nor its own ``execute``.
        """
        run = getattr(self, "run", None)
        if not callable(run):
            raise NotImplementedError("Process must implement run() or execute()")
        run_inputs = dict(inputs)
        if isinstance(context, Mapping):
            parameters = signature(run).parameters
            callback_keys = {
                "progress_callback": "_resistics_progress_callback",
                "cancellation_callback": "_resistics_cancellation_callback",
            }
            for parameter, context_key in callback_keys.items():
                if parameter in parameters and context_key in context:
                    run_inputs[parameter] = context[context_key]
        return run(**run_inputs)

    def _get_record(self, messages: str | list[str]) -> Record:
        """Get the record for the processor

        :param messages: The messages to add for the processor

        :return: A record
        """
        return get_record(self.parameters(), messages)


class ResisticsBase:
    """Resistics base class

    Parent class to ensure consistency of common methods
    """

    def __repr__(self) -> str:
        """Return a string of class information.

        :return: Human-readable class details.
        """
        return self.to_string()

    def __str__(self) -> str:
        """Return a string of class information.

        :return: Human-readable class details.
        """
        return self.to_string()

    def type_to_string(self) -> str:
        """Get the class type as a string.

        :return: Qualified runtime class representation.
        """
        return str(self.__class__)

    def to_string(self) -> str:
        """Render class details as a string.

        :return: Human-readable class details.
        """
        return self.type_to_string()

    def summary(self, symbol: str = "-") -> None:
        """Print a delimited summary of class details.

        :param symbol: Character used to draw the summary delimiters.
        """
        name = str(self.__class__)
        length = len(name) + 10
        print("##" + 3 * symbol + "Begin Summary" + ((length - 18) * symbol))
        print(self.to_string())
        print("##" + 3 * symbol + "End summary" + (length - 16) * symbol)


class ResisticsData(ResisticsBase):
    """Base class for a resistics data object"""

    pass


class ResisticsWriter(ResisticsProcess):
    """Parent process for data writers

    :param overwrite: Boolean flag for overwriting the existing data, by default False
    """

    overwrite: bool = True

    def run(self, dir_path: Path, data: ResisticsData) -> None:
        """Write a data object to a directory in a concrete writer.

        :param dir_path: Destination directory.
        :param data: Data object to persist.
        :raises NotImplementedError: Always; concrete writers implement this
            operation.
        """
        raise NotImplementedError("To be implemented in child writers")

    def _check_dir(self, dir_path: Path) -> bool:
        """Prepare an output directory according to overwrite policy.

        :param dir_path: Destination directory to validate or create.
        :return: Whether writing may proceed.
        """
        if dir_path.exists() and not self.overwrite:
            logger.error(f"Write path {dir_path} exists and overwrite is False")
            return False
        if dir_path.exists():
            logger.warning(f"Overwriting existing directory {dir_path}")
        if not dir_path.exists():
            logger.info(f"Directory {dir_path} not found. Creating including parents.")
            dir_path.mkdir(parents=True)
        return True

    def _get_writer_record(self, dir_path: Path, data_type: type):
        """Get a process record for the writer.

        :param dir_path: Destination directory.
        :param data_type: Concrete data type being written.

        :return: Writer process record.
        """
        return super()._get_record([f"Writing out {data_type.__name__} to {dir_path}"])
