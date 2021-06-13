"""
Module defining transfer functions
"""
from typing import List, Optional, Dict, Any, Union
from pydantic import validator
import numpy as np

from resistics.common import Metadata


class Component(Metadata):
    """
    Data class for a single component in a Transfer function

    Example
    -------
    >>> from resistics.transfunc import Component
    >>> component = Component(real=[1, 2, 3, 4, 5], imag=[-5, -4, -3, -2 , -1])
    >>> component.get_value(0)
    (1-5j)
    >>> component.to_numpy()
    array([1.-5.j, 2.-4.j, 3.-3.j, 4.-2.j, 5.-1.j])
    """

    real: List[float]
    """The real part of the component"""
    imag: List[float]
    """The complex part of the component"""

    def get_value(self, eval_idx: int) -> complex:
        """Get the value for an evaluation frequency"""
        return self.real[eval_idx] + 1j * self.imag[eval_idx]

    def to_numpy(self) -> np.ndarray:
        """Get the component as a numpy complex array"""
        return np.array(self.real) + 1j * np.array(self.imag)


def get_component_key(out_chan: str, in_chan: str) -> str:
    """
    Get key for out channel and in channel combination in the solution

    Parameters
    ----------
    out_chan : str
        The output channel
    in_chan : str
        The input channel

    Returns
    -------
    str
        The component key

    Examples
    --------
    >>> from resistics.regression import get_component_key
    >>> get_component_key("Ex", "Hy")
    'ExHy'
    """
    return f"{out_chan}{in_chan}"


class ResisticsTransferFunction(Metadata):
    """
    This is the resistics parent transfer function. It should not be used
    directly but implements the logic for automatic re-initialization of the
    correct transfer function from a JSON file or dictionary.

    .. warning::

        Users inheriting to define a custom transfer function should inherit
        from TransferFunction instead.

    See Also
    --------
    TransferFunction : The class to use for a generic transfer function
    ImpedanceTensor: The class to use for the impedance tensor
    Tipper: The class to use for the tipper
    """

    _types: Dict[str, type] = {}
    """Store types which will help automatic instantiation"""
    name: Optional[str] = None
    """The name of the transfer function, this will be set automatically"""

    def __init_subclass__(cls) -> None:
        """
        Used to automatically register child transfer functions in `_types`

        When a ResisticsTransferFunction child class is imported, it is added
        to the base ResisticsTransferFunction _types variable. Later, this
        dictionary of class types can be used to initialise a specific
        child transfer function from a dictonary as long as that specific child
        transfer fuction has already been imported and it is called from a
        pydantic class that will validate the inputs.

        The intention of this method is to support initialising transfer
        functions from JSON files.

        This is a similar approach to ResisticsProcess.
        """
        cls._types[cls.__name__] = cls

    @classmethod
    def __get_validators__(cls):
        """Get the validators that will be used by pydantic"""
        yield cls.validate

    @classmethod
    def validate(
        cls, value: Union["ResisticsTransferFunction", Dict[str, Any]]
    ) -> "ResisticsTransferFunction":
        """
        Validate a ResisticsTransferFunction in another pydantic class which
        will use the validate function in instantiation

        Parameters
        ----------
        value : Union[ResisticsTransferFunction, Dict[str, Any]]
            A ResisticsTransferFunction child class or a dictionary

        Returns
        -------
        ResisticsTransferFunction
            A ResisticsTransferFunction child class

        Raises
        ------
        ValueError
            If the value is neither a ResisticsTransferFunction or a dictionary
        KeyError
            If name is not in the dictionary
        ValueError
            If initialising from dictionary fails

        Examples
        --------
        The following example will show how a child ResisticsTransferFunction
        class can be instantiated using a dictionary and the parent
        ResisticsTransferFunction (but only as long as that child class has been
        imported).

        >>> from resistics.transfunc import ResisticsTransferFunction

        Show known ResisticsTransferFunction types in built into resistics

        >>> for entry in ResisticsTransferFunction._types.items():
        ...     print(entry)
        ('TransferFunction', <class 'resistics.transfunc.TransferFunction'>)
        ('ImpedanceTensor', <class 'resistics.transfunc.ImpedanceTensor'>)
        ('Tipper', <class 'resistics.transfunc.Tipper'>)

        Now let's initialise an ImpedanceTensor from the base
        ResisticsTransferFunction and a dictionary.

        >>> mytf = {"name": "ImpedanceTensor", "cross_chans": ["Ex", "Ey"]}
        >>> test = ResisticsTransferFunction(**mytf)
        >>> test.summary()
        {'name': 'ImpedanceTensor'}

        This is not quite what we were expecting. The reason is because the
        validate method of the class was not used. This method is used by
        pydantic.

        >>> mytf = {"name": "ImpedanceTensor", "cross_chans": ["Ex", "Ey"]}
        >>> test = ResisticsTransferFunction.validate(mytf)
        >>> test.summary()
        {
            'name': 'impedancetensor',
            'out_chans': ['Ex', 'Ey'],
            'in_chans': ['Hx', 'Hy'],
            'cross_chans': ['Ex', 'Ey'],
            'n_out': 2,
            'n_in': 2,
            'n_cross': 2
        }

        That's more like it. This will raise errors if an unknown type of
        ResisticsTransferFunction is received.

        >>> mytf = {"name": "NewTF", "cross_chans": ["Ex", "Ey"]}
        >>> test = ResisticsTransferFunction.validate(mytf)
        Traceback (most recent call last):
        ...
        ValueError: Unable to initialise NewTF from dictionary

        Or if the dictionary does not have a name key

        >>> mytf = {"cross_chans": ["Ex", "Ey"]}
        >>> test = ResisticsTransferFunction.validate(mytf)
        Traceback (most recent call last):
        ...
        KeyError: 'No name provided for initialisation of transfer function'

        Unexpected inputs will also raise an error

        >>> test = ResisticsTransferFunction.validate(5)
        Traceback (most recent call last):
        ...
        ValueError: ResisticsTransferFunction unable to initialise from <class 'int'>
        """
        if isinstance(value, ResisticsTransferFunction):
            return value
        if not isinstance(value, dict):
            raise ValueError(
                f"ResisticsTransferFunction unable to initialise from {type(value)}"
            )
        if "name" not in value:
            raise KeyError("No name provided for initialisation of transfer function")
        name = value.pop("name")
        try:
            return cls._types[name](**value)
        except Exception:
            raise ValueError(f"Unable to initialise {name} from dictionary")

    @validator("name", always=True)
    def validate_name(cls, value: Union[str, None]) -> str:
        """Inialise the name attribute of the transfer function"""
        if value is None:
            return cls.__name__
        return value


class TransferFunction(ResisticsTransferFunction):
    """
    Define a generic transfer function

    This class is a describes generic transfer function, including:

    - The output channels for the transfer function
    - The input channels for the transfer function
    - The cross channels for the transfer function

    The cross channels are the channels that will be used to calculate out the
    cross powers for the regression.

    This general parent class has no plotting function. However, child classes
    may have a plotting function as different transfer functions may need
    different types of plots.

    .. note::

        Users interested in writing a custom transfer function should inherit
        from this generic Transfer function

    See Also
    --------
    ImpandanceTensor : Transfer function for the MT impedance tensor
    Tipper : Transfer function for the MT tipper

    Examples
    --------
    A standard magnetotelluric transfer function

    >>> from resistics.transfunc import TransferFunction
    >>> tf = TransferFunction(name="impedancetensor", out_chans=["Ex", "Ey"], in_chans=["Hx", "Hy"])
    >>> print(tf.to_string())
    | Ex | = | Ex_Hx Ex_Hy | | Hx |
    | Ey |   | Ey_Hx Ey_Hy | | Hy |

    The magnetotelluric tipper

    >>> tf = TransferFunction(name="tipper", out_chans=["Hz"], in_chans=["Hx", "Hy"])
    >>> print(tf.to_string())
    | Hz | = | Hz_Hx Hz_Hy | | Hx |
                             | Hy |

    Combining the impedance tensor and the tipper

    >>> tf = TransferFunction(name="combined", out_chans=["Ex", "Ey"], in_chans=["Hx", "Hy", "Hz"])
    >>> print(tf.to_string())
    | Ex |   | Ex_Hx Ex_Hy Ex_Hz | | Hx |
    | Ey | = | Ey_Hx Ey_Hy Ey_Hz | | Hy |
                                   | Hz |

    And a generic example

    >>> tf = TransferFunction(name="example", out_chans=["bye", "see you", "ciao"], in_chans=["hello", "hi_there"])
    >>> print(tf.to_string())
    | bye      |   | bye_hello         bye_hi_there      | | hello    |
    | see you  | = | see you_hello     see you_hi_there  | | hi_there |
    | ciao     |   | ciao_hello        ciao_hi_there     |
    """

    out_chans: List[str]
    """The output channels"""
    in_chans: List[str]
    """The input channels"""
    cross_chans: Optional[List[str]] = None
    """The channels to use for calculating the cross spectra"""
    n_out: Optional[int] = None
    """The number of output channels"""
    n_in: Optional[int] = None
    """The number of input channels"""
    n_cross: Optional[int] = None
    """The number of cross power channels"""

    @validator("cross_chans", always=True)
    def validate_cross_chans(
        cls, value: Union[None, List[str]], values: Dict[str, Any]
    ) -> List[str]:
        """Validate cross spectra channels"""
        if value is None:
            return values["in_chans"]
        return value

    @validator("n_out", always=True)
    def validate_n_out(cls, value: Union[None, int], values: Dict[str, Any]) -> int:
        """Validate number of output channels"""
        if value is None:
            return len(values["out_chans"])
        return value

    @validator("n_in", always=True)
    def validate_n_in(cls, value: Union[None, int], values: Dict[str, Any]) -> int:
        """Validate number of input channels"""
        if value is None:
            return len(values["in_chans"])
        return value

    @validator("n_cross", always=True)
    def validate_n_cross(cls, value: Union[None, int], values: Dict[str, Any]) -> int:
        """Validate number of cross channels"""
        if value is None:
            return len(values["cross_chans"])
        return value

    def n_eqns_per_output(self) -> int:
        """Get the number of equations per output"""
        return len(self.cross_chans)

    def n_regressors(self) -> int:
        """Get the number of regressors"""
        return self.n_in

    def to_string(self):
        """Get the transfer function as as string"""
        n_lines = max(len(self.in_chans), len(self.out_chans))
        lens = [len(x) for x in self.in_chans] + [len(x) for x in self.out_chans]
        max_len = max(lens)
        line_equals = (n_lines - 1) // 2
        outstr = ""
        for il in range(n_lines):
            out_chan = self._out_chan_string(il, max_len)
            in_chan = self._in_chan_string(il, max_len)
            tensor = self._tensor_string(il, max_len)
            eq = "=" if il == line_equals else " "
            outstr += f"{out_chan} {eq} {tensor} {in_chan}\n"
        return outstr.rstrip("\n")

    def _out_chan_string(self, il: int, max_len: int) -> str:
        """Get the out channels string"""
        if il >= self.n_out:
            empty_len = max_len + 4
            return f"{'':{empty_len}s}"
        return f"| { self.out_chans[il]:{max_len}s} |"

    def _in_chan_string(self, il: int, max_len: int) -> str:
        """Get the in channel string"""
        if il >= self.n_in:
            return ""
        return f"| { self.in_chans[il]:{max_len}s} |"

    def _tensor_string(self, il: int, max_len: int) -> str:
        """Get the tensor string"""
        if il >= self.n_out:
            element_len = ((max_len * 2 + 1) + 1) * self.n_in + 3
            return f"{'':{element_len}s}"
        elements = "| "
        for chan in self.in_chans:
            component = f"{self.out_chans[il]}_{chan}"
            elements += f"{component:{2*max_len + 1}s} "
        elements += "|"
        return elements


class ImpedanceTensor(TransferFunction):
    """
    Standard magnetotelluric impedance tensor

    Examples
    --------
    >>> from resistics.transfunc import ImpedanceTensor
    >>> tf = ImpedanceTensor()
    >>> print(tf.to_string())
    | Ex | = | Ex_Hx Ex_Hy | | Hx |
    | Ey |   | Ey_Hx Ey_Hy | | Hy |
    """

    out_chans: List[str] = ["Ex", "Ey"]
    in_chans: List[str] = ["Hx", "Hy"]


class Tipper(TransferFunction):
    """
    Magnetotelluric tipper

    Examples
    --------
    >>> from resistics.transfunc import Tipper
    >>> tf = Tipper()
    >>> print(tf.to_string())
    | Hz | = | Hz_Hx Hz_Hy | | Hx |
                             | Hy |
    """

    out_chans: List[str] = ["Hz"]
    in_chans: List[str] = ["Hx", "Hy"]
