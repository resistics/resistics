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


class TransferFunction(Metadata):
    """
    Define the transfer function

    This class has few methods and is a simple way of defining the transfer
    input and output channels for which to calculate the transfer function

    Examples
    --------
    A standard magnetotelluric transfer function

    >>> from resistics.transfunc import TransferFunction
    >>> tf = TransferFunction(out_chans=["Ex", "Ey"], in_chans=["Hx", "Hy"])
    >>> print(tf.to_string())
    | Ex | = | Ex_Hx Ex_Hy | | Hx |
    | Ey |   | Ey_Hx Ey_Hy | | Hy |

    Additionally including the Hz component

    >>> tf = TransferFunction(out_chans=["Ex", "Ey"], in_chans=["Hx", "Hy", "Hz"])
    >>> print(tf.to_string())
    | Ex |   | Ex_Hx Ex_Hy Ex_Hz | | Hx |
    | Ey | = | Ey_Hx Ey_Hy Ey_Hz | | Hy |
                                   | Hz |

    The magnetotelluric tipper

    >>> tf = TransferFunction(out_chans=["Hz"], in_chans=["Hx", "Hy"])
    >>> print(tf.to_string())
    | Hz | = | Hz_Hx Hz_Hy | | Hx |
                             | Hy |

    And a generic example

    >>> tf = TransferFunction(out_chans=["bye", "see you", "ciao"], in_chans=["hello", "hi_there"])
    >>> print(tf.to_string())
    | bye      |   | bye_hello         bye_hi_there      | | hello    |
    | see you  | = | see you_hello     see you_hi_there  | | hi_there |
    | ciao     |   | ciao_hello        ciao_hi_there     |
    """

    out_chans: List[str]
    in_chans: List[str]
    cross_chans: Optional[List[str]] = None
    n_out: Optional[int] = None
    n_in: Optional[int] = None
    n_cross: Optional[int] = None

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
