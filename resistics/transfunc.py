"""Module defining transfer functions"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Annotated, Any, ClassVar

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydantic import Field, model_validator
from pydantic_core import core_schema

from resistics.common import Metadata


class Component(Metadata):
    """Data class for a single component in a Transfer function

    **Examples**

    ```{doctest}
    >>> from resistics.transfunc import Component
    >>> component = Component(real=[1, 2, 3, 4, 5], imag=[-5, -4, -3, -2 , -1])
    >>> component.get_value(0)
    (1-5j)
    >>> component.to_numpy()
    array([1.-5.j, 2.-4.j, 3.-3.j, 4.-2.j, 5.-1.j])

    ```
    """

    real: list[float]
    """The real part of the component"""
    imag: list[float]
    """The complex part of the component"""

    def get_value(self, eval_idx: int) -> complex:
        """Get the complex value at one evaluation index.

        :param eval_idx: Evaluation-frequency index.
        :return: Complex component value.
        """
        return self.real[eval_idx] + 1j * self.imag[eval_idx]

    def to_numpy(self) -> np.ndarray:
        """Get the complete component as a complex NumPy array.

        :return: Complex values with real and imaginary parts combined.
        """
        return np.array(self.real) + 1j * np.array(self.imag)


def get_component_key(out_chan: str, in_chan: str) -> str:
    """Get key for out channel and in channel combination in the solution

    :param out_chan: The output channel
    :param in_chan: The input channel

    :return: The component key

    **Examples**

    ```{doctest}
    >>> from resistics.regression import get_component_key
    >>> get_component_key("Ex", "Hy")
    'ExHy'

    ```
    """
    return f"{out_chan}{in_chan}"


class TransferFunction(Metadata):
    """Define a generic transfer function

    This class is a describes generic transfer function, including:

    - The output channels for the transfer function
    - The input channels for the transfer function
    - The cross channels for the transfer function

    The cross channels are the channels that will be used to calculate out the
    cross powers for the regression.

    This generic parent class has no implemented plotting function. However,
    child classes may have a plotting function as different transfer functions
    may need different types of plots.

    ```{note}
    Users interested in writing a custom transfer function should inherit
    from this generic Transfer function
    ```
    **Attributes**

    - **name** — Registered transfer-function model name.
    - **variation** — Short label identifying this variation.
    - **out_chans** — Output channels.
    - **in_chans** — Input channels.
    - **cross_chans** — Channels used to calculate cross spectra.
    - **n_out** — Number of output channels.
    - **n_in** — Number of input channels.
    - **n_cross** — Number of cross-power channels.

    **See Also**

    ImpandanceTensor : Transfer function for the MT impedance tensor
    Tipper : Transfer function for the MT tipper

    **Examples**

    A generic example

    ```{doctest}
    >>> from resistics.transfunc import TransferFunction
    >>> tf = TransferFunction(variation="example", out_chans=["bye", "see you", "ciao"], in_chans=["hello", "hi_there"])
    >>> print(tf.to_string())  # doctest: +NORMALIZE_WHITESPACE
    | bye      |   | bye_hello         bye_hi_there      | | hello    |
    | see you  | = | see you_hello     see you_hi_there  | | hi_there |
    | ciao     |   | ciao_hello        ciao_hi_there     |

    ```

    Combining the impedance tensor and the tipper into one TransferFunction

    ```{doctest}
    >>> tf = TransferFunction(variation="combined", out_chans=["Ex", "Ey"], in_chans=["Hx", "Hy", "Hz"])
    >>> print(tf.to_string())
    | Ex |   | Ex_Hx Ex_Hy Ex_Hz | | Hx |
    | Ey | = | Ey_Hx Ey_Hy Ey_Hz | | Hy |
                                   | Hz |

    ```
    """

    _types: ClassVar[dict[str, type[TransferFunction]]] = {}
    """Store types which will help automatic instantiation"""
    name: str = ""
    """The name of the transfer function, this will be set automatically"""
    variation: Annotated[str, Field(max_length=16)] = "generic"
    """A short additional bit of information about this variation"""
    out_chans: list[str]
    """The output channels"""
    in_chans: list[str]
    """The input channels"""
    cross_chans: list[str] = Field(default_factory=list)
    """The channels to use for calculating the cross spectra"""
    n_out: int = 0
    """The number of output channels"""
    n_in: int = 0
    """The number of input channels"""
    n_cross: int = 0
    """The number of cross power channels"""

    def __init_subclass__(cls) -> None:
        """Used to automatically register child transfer functions in `_types`

        When a TransferFunction child class is imported, it is added to the base
        TransferFunction _types variable. Later, this dictionary of class types
        can be used to initialise a specific child transfer function from a
        dictonary as long as that specific child transfer fuction has already
        been imported and it is called from a pydantic class that will validate
        the inputs.

        The intention of this method is to support initialising transfer
        functions from JSON files. This is a similar approach to
        ResisticsProcess.
        """
        cls._types[cls.__name__] = cls

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: Any
    ) -> core_schema.CoreSchema:
        """Build the Pydantic schema for registered transfer functions.

        :param source_type: Source annotation supplied by Pydantic.
        :param handler: Pydantic schema-generation callback.
        :return: Core schema dispatching dictionaries before normal validation.
        """
        return core_schema.no_info_before_validator_function(
            cls.validate_model_input, handler(source_type)
        )

    @classmethod
    def validate_model_input(cls, value: Any) -> Any:
        """Resolve registered transfer-function dictionaries in Pydantic.

        :param value: Existing transfer function, dictionary, or unrelated value.
        :return: A dispatched transfer function or the unchanged input.
        """
        if isinstance(value, TransferFunction):
            return value
        if isinstance(value, dict) and "name" in value:
            return cls.validate(value)
        return value

    @classmethod
    def __get_validators__(
        cls,
    ) -> Iterator[Callable[[TransferFunction | dict[str, Any]], TransferFunction]]:
        """Yield the compatibility transfer-function validator.

        :yield: Registered transfer-function validator.
        """
        yield cls.validate

    @classmethod
    def validate(cls, value: TransferFunction | dict[str, Any]) -> TransferFunction:
        """Validate a TransferFunction

        :param value: A TransferFunction child class or a dictionary

        :return: A TransferFunction or TransferFunction child class

        :raises ValueError: If the input is invalid, its registered name is unknown,
            or child-model initialization fails.
        :raises KeyError: If name is not in the dictionary

        **Examples**

        The following example will show how a child TransferFunction class
        can be instantiated using a dictionary and the parent TransferFunction
        (but only as long as that child class has been imported).

        ```{doctest}
        >>> from resistics.transfunc import TransferFunction

        ```

        Show known TransferFunction types in built into resistics

        ```{doctest}
        >>> for entry in TransferFunction._types.items():
        ...     print(entry)
        ('ImpedanceTensor', <class 'resistics.transfunc.ImpedanceTensor'>)
        ('Tipper', <class 'resistics.transfunc.Tipper'>)

        ```

        Now let's initialise an ImpedanceTensor from the base TransferFunction
        and a dictionary.

        ```{doctest}
        >>> mytf = {"name": "ImpedanceTensor", "variation": "ecross", "cross_chans": ["Ex", "Ey"]}
        >>> test = TransferFunction(**mytf) # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        pydantic_core._pydantic_core.ValidationError: ...

        ```

        The generic TransferFunction does not dispatch to a child class during
        direct base-class construction. To get this to work, instead use the
        validate class method. This is the class method used by pydantic for
        fields typed as TransferFunction.

        ```{doctest}
        >>> mytf = {"name": "ImpedanceTensor", "variation": "ecross", "cross_chans": ["Ex", "Ey"]}
        >>> test = TransferFunction.validate(mytf)
        >>> test.summary()
        {
            'name': 'ImpedanceTensor',
            'variation': 'ecross',
            'out_chans': ['ex', 'ey'],
            'in_chans': ['hx', 'hy'],
            'cross_chans': ['Ex', 'Ey'],
            'n_out': 2,
            'n_in': 2,
            'n_cross': 2
        }

        ```

        That's more like it. Unknown transfer function names are rejected so
        malformed or unavailable model types cannot leak dictionaries through
        a field declared as a TransferFunction.

        ```{doctest}
        >>> mytf = {"name": "NewTF", "cross_chans": ["Ex", "Ey"]}
        >>> test = TransferFunction.validate(mytf)
        Traceback (most recent call last):
        ...
        ValueError: Unknown transfer function 'NewTF'

        ```

        Or if the dictionary does not have a name key

        ```{doctest}
        >>> mytf = {"cross_chans": ["Ex", "Ey"]}
        >>> test = TransferFunction.validate(mytf)
        Traceback (most recent call last):
        ...
        KeyError: 'No name provided for initialisation of TransferFunction'

        ```

        Unexpected inputs will also raise an error

        ```{doctest}
        >>> test = TransferFunction.validate(5)
        Traceback (most recent call last):
        ...
        ValueError: TransferFunction unable to initialise from <class 'int'>

        ```
        """
        if isinstance(value, TransferFunction):
            return value
        if not isinstance(value, dict):
            raise ValueError(
                f"TransferFunction unable to initialise from {type(value)}"
            )
        if "name" not in value:
            raise KeyError("No name provided for initialisation of TransferFunction")
        data = dict(value)
        name = data.pop("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Transfer function name must be a non-empty string")
        if name == "TransferFunction":
            return TransferFunction(**data)
        model_type = cls._types.get(name)
        if model_type is None:
            raise ValueError(f"Unknown transfer function '{name}'")
        try:
            return model_type(**data)
        except Exception:
            raise ValueError(f"Unable to initialise {name} from dictionary") from None

    @model_validator(mode="after")
    def validate_contract(self) -> TransferFunction:
        """Resolve and validate all channel-derived fields.

        :return: The validated transfer function with concrete derived fields.

        :raises ValueError: If a supplied dimension disagrees with its channel list.
        """
        if not self.name:
            self.name = self.__class__.__name__
        if "cross_chans" not in self.model_fields_set:
            self.cross_chans = list(self.in_chans)

        dimensions = {
            "n_out": len(self.out_chans),
            "n_in": len(self.in_chans),
            "n_cross": len(self.cross_chans),
        }
        for field, expected in dimensions.items():
            if field in self.model_fields_set and getattr(self, field) != expected:
                channels_field = field.removeprefix("n_") + "_chans"
                raise ValueError(f"{field} must equal len({channels_field})")
            setattr(self, field, expected)
        return self

    def n_eqns_per_output(self) -> int:
        """Get the number of equations per output.

        :return: Number of configured cross channels.
        """
        return len(self.cross_chans)

    def n_regressors(self) -> int:
        """Get the number of regressors.

        :return: Number of configured input channels.
        """
        return self.n_in

    def solution_components(self) -> list[str]:
        """Get the components of the solution based on the input and output
        channels

        :return: The solution components

        **Examples**

        ```{doctest}
        >>> from resistics.transfunc import TransferFunction
        >>> tf = TransferFunction(
        ...     variation="a",
        ...     in_chans=["a", "b", "c"],
        ...     out_chans=["x", "y"]
        ... )
        >>> tf.solution_components()
        ['xa', 'xb', 'xc', 'ya', 'yb', 'yc']

        ```
        """
        return [
            f"{out_chan}{in_chan}"
            for out_chan in self.out_chans
            for in_chan in self.in_chans
        ]

    def to_string(self) -> str:
        """Render the transfer-function equation layout.

        :return: Multiline matrix-style representation.
        """
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
        """Render the output-channel cell for one line.

        :param il: Zero-based output line.
        :param max_len: Shared channel-name field width.
        :return: Padded output-channel cell or whitespace.
        """
        if il >= self.n_out:
            empty_len = max_len + 4
            return f"{'':{empty_len}s}"
        return f"| {self.out_chans[il]:{max_len}s} |"

    def _in_chan_string(self, il: int, max_len: int) -> str:
        """Render the input-channel cell for one line.

        :param il: Zero-based input line.
        :param max_len: Shared channel-name field width.
        :return: Padded input-channel cell or an empty string.
        """
        if il >= self.n_in:
            return ""
        return f"| {self.in_chans[il]:{max_len}s} |"

    def _tensor_string(self, il: int, max_len: int) -> str:
        """Render the solution-component cells for one line.

        :param il: Zero-based output line.
        :param max_len: Shared channel-name field width.
        :return: Padded component row or whitespace.
        """
        if il >= self.n_out:
            element_len = ((max_len * 2 + 1) + 1) * self.n_in + 3
            return f"{'':{element_len}s}"
        elements = "| "
        for chan in self.in_chans:
            component = f"{self.out_chans[il]}_{chan}"
            elements += f"{component:{2 * max_len + 1}s} "
        elements += "|"
        return elements


class ImpedanceTensor(TransferFunction):
    """Standard magnetotelluric impedance tensor

    **Notes**

    Information about data units

    - Magnetic permeability in nT . m / A
    - Electric (E) data is in mV/m
    - Magnetic (H) data is in nT
    - Z = E/H is in mV / m . nT
    - Units of resistance = Ohm = V / A

    **Attributes**

    - **variation** — Short label identifying this impedance-tensor variation.
    - **out_chans** — Electric output channels.
    - **in_chans** — Magnetic input channels.

    **Examples**

    ```{doctest}
    >>> from resistics.transfunc import ImpedanceTensor
    >>> tf = ImpedanceTensor()
    >>> print(tf.to_string())
    | ex | = | ex_hx ex_hy | | hx |
    | ey |   | ey_hx ey_hy | | hy |

    ```
    """

    variation: Annotated[str, Field(max_length=16)] = "default"
    out_chans: list[str] = ["ex", "ey"]
    in_chans: list[str] = ["hx", "hy"]

    @staticmethod
    def get_resistivity(periods: np.ndarray, component: Component) -> np.ndarray:
        """Get apparent resistivity for a component

        :param periods: The periods of the component
        :param component: The component values

        :return: Apparent resistivity
        """
        squared = np.power(np.absolute(component.to_numpy()), 2)
        return 0.2 * periods * squared

    @staticmethod
    def get_phase(key: str, component: Component) -> np.ndarray:
        """Get the phase for the component

        ```{note}
        Components exhx and exhy are wrapped around in [0,90]
        ```
        :param key: The component name
        :param component: The component values

        :return: The phase values
        """
        phase = np.angle(component.to_numpy())
        # unwrap into specific quadrant and convert to degrees
        phase = np.unwrap(phase) * 180 / np.pi
        if key.lower() in {"exhx", "exhy"}:
            phase = np.mod(phase, 360) - 180
        return phase

    @staticmethod
    def get_fig(
        x_lim: list[float] | None = None,
        res_lim: list[float] | None = None,
        phs_lim: list[float] | None = None,
    ) -> go.Figure:
        """Get a figure for plotting the ImpedanceTensor

        :param x_lim: The x limits, to be provided as powers of 10, by default None. For
            example, for 0.001, use -3
        :param res_lim: The y limits for resistivity, to be provided as powers of 10, by
            default None. For example, for 1000, use 3
        :param phs_lim: The phase limits, by default None

        :return: Plotly figure
        """
        from resistics.plot import PLOTLY_MARGIN, PLOTLY_TEMPLATE

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=["Apparent resistivity", "Phase"],
        )
        # apparent resistivity axes
        fig.update_xaxes(type="log", showticklabels=True, row=1, col=1)
        fig.update_yaxes(title_text="App. resistivity (Ohm m)", row=1, col=1)
        fig.update_yaxes(type="log", row=1, col=1)
        if x_lim is not None:
            fig.update_xaxes(range=x_lim, row=1, col=1)
        if res_lim is not None:
            fig.update_yaxes(range=res_lim, row=1, col=1)
        # phase axes
        fig.update_xaxes(title_text="Period (s)", type="log", row=2, col=1)
        fig.update_xaxes(showticklabels=True, row=2, col=1)
        # fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
        fig.update_yaxes(title_text="Phase (degrees)", row=2, col=1)
        if phs_lim is not None:
            fig.update_yaxes(range=phs_lim, row=2, col=1)
        # update the layout
        fig.update_layout(template=PLOTLY_TEMPLATE, margin=dict(PLOTLY_MARGIN))
        return fig

    @staticmethod
    def plot(
        freqs: list[float],
        components: dict[str, Component],
        fig: go.Figure | None = None,
        to_plot: list[str] | None = None,
        legend: str = "Impedance tensor",
        x_lim: list[float] | None = None,
        res_lim: list[float] | None = None,
        phs_lim: list[float] | None = None,
        symbol: str | None = "circle",
    ) -> go.Figure:
        """Plot the Impedance tensor

        :param freqs: The frequencies where the impedance tensor components have been
            calculated
        :param components: The component data
        :param fig: Figure to add to, by default None
        :param to_plot: The components to plot, by default all of the components of the
            impedance tensor
        :param legend: Legend prefix for the components, by default "Impedance tensor"
        :param x_lim: The x limits, to be provided as powers of 10, by default None. For
            example, for 0.001, use -3. Only used when a figure is not provided.
        :param res_lim: The y limits for resistivity, to be provided as powers of 10, by
            default None. For example, for 1000, use 3. Only used when a figure
            is not provided.
        :param phs_lim: The phase limits, by default None. Only used when a figure is not
            provided.
        :param symbol: The marker symbol to use, by default "circle"

        :return: [description]
        """
        if fig is None:
            fig = ImpedanceTensor.get_fig(x_lim=x_lim, res_lim=res_lim, phs_lim=phs_lim)
        if to_plot is None:
            components_by_casefold = {
                component.casefold(): component for component in components
            }
            to_plot = [
                components_by_casefold[component]
                for component in ("exhy", "eyhx", "exhx", "eyhy")
                if component in components_by_casefold
            ]

        periods = np.reciprocal(freqs)
        colors = {
            "exhx": "orange",
            "eyhy": "green",
            "exhy": "red",
            "eyhx": "blue",
        }
        for comp in to_plot:
            res = ImpedanceTensor.get_resistivity(periods, components[comp])
            phs = ImpedanceTensor.get_phase(comp, components[comp])
            comp_legend = f"{legend} - {comp}"
            color = colors[comp.casefold()]
            scatter = go.Scatter(
                x=periods,
                y=res,
                mode="lines+markers",
                marker={"color": color, "symbol": symbol},
                line={"color": color},
                name=comp_legend,
                legendgroup=comp_legend,
            )
            fig.add_trace(scatter, row=1, col=1)
            scatter = go.Scatter(
                x=periods,
                y=phs,
                mode="lines+markers",
                marker={"color": color, "symbol": symbol},
                line={"color": color},
                name=comp_legend,
                legendgroup=comp_legend,
                showlegend=False,
            )
            fig.add_trace(scatter, row=2, col=1)
        return fig


class Tipper(TransferFunction):
    """Magnetotelluric tipper

    The tipper components are Tx = HzHx and Ty = HzHy

    The tipper length is sqrt(Re(Tx)^2 + Re(Ty)^2)

    The tipper angle is arctan (Re(Ty)/Re(Tx))

    **Attributes**

    - **variation** — Short label identifying this tipper variation.
    - **out_chans** — Vertical magnetic output channel.
    - **in_chans** — Horizontal magnetic input channels.

    **Notes**

    Information about units

    - Tipper T = H/H is dimensionless

    **Examples**

    ```{doctest}
    >>> from resistics.transfunc import Tipper
    >>> tf = Tipper()
    >>> print(tf.to_string())
    | Hz | = | Hz_Hx Hz_Hy | | Hx |
                             | Hy |

    ```
    """

    variation: Annotated[str, Field(max_length=16)] = "default"
    out_chans: list[str] = ["Hz"]
    in_chans: list[str] = ["Hx", "Hy"]

    def get_length(self, components: dict[str, Component]) -> np.ndarray:
        """Calculate real tipper-vector length.

        :param components: ``HzHx`` and ``HzHy`` solution components.
        :return: Length at each evaluation frequency.
        """
        txRe = components["HzHx"].real
        tyRe = components["HzHy"].real
        return np.sqrt(np.power(txRe, 2) + np.power(tyRe, 2))

    def get_real_angle(self, components: dict[str, Component]) -> np.ndarray:
        """Calculate the real tipper-vector angle.

        :param components: ``HzHx`` and ``HzHy`` solution components.
        :return: Angle in degrees at each evaluation frequency.
        """
        txRe = np.array(components["HzHx"].real)
        tyRe = np.array(components["HzHy"].real)
        return np.arctan(tyRe / txRe) * 180 / np.pi

    def get_imag_angle(self, components: dict[str, Component]) -> np.ndarray:
        """Calculate the imaginary tipper-vector angle.

        :param components: ``HzHx`` and ``HzHy`` solution components.
        :return: Angle in degrees at each evaluation frequency.
        """
        txIm = np.array(components["HzHx"].imag)
        tyIm = np.array(components["HzHy"].imag)
        return np.arctan(tyIm / txIm) * 180 / np.pi

    def plot(
        self,
        freqs: list[float],
        components: dict[str, Component],
        x_lim: list[float] | None = None,
        len_lim: list[float] | None = None,
        ang_lim: list[float] | None = None,
    ) -> go.Figure:
        """Plot the impedance tensor

        ```{warning}
        This probably needs further checking and verification
        ```
        :param freqs: The x axis frequencies
        :param components: The component data
        :param x_lim: The x limits, to be provided as powers of 10, by default None. For
            example, for 0.001, use -3
        :param len_lim: The y limits for tipper length, to be provided as powers of 10, by
            default None. For example, for 1000, use 3
        :param ang_lim: The angle limits, by default None

        :return: Plotly figure
        """
        import warnings

        from plotly.subplots import make_subplots

        warnings.warn("Plotting of tippers needs further verification", stacklevel=2)

        periods = np.reciprocal(freqs)
        if x_lim is None:
            x_lim = [-3, 5]
        if len_lim is None:
            len_lim = [-2, 6]
        if ang_lim is None:
            ang_lim = [-10, 100]

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=["Length", "Angles"],
        )
        fig.update_layout(width=1000, autosize=True)
        # x axes
        fig.update_xaxes(title_text="Period (s)", type="log", range=x_lim, row=1, col=1)
        fig.update_xaxes(showticklabels=True, row=1, col=1)
        fig.update_xaxes(title_text="Period (s)", type="log", range=x_lim, row=2, col=1)
        fig.update_xaxes(showticklabels=True, row=2, col=1)
        # y axes
        fig.update_yaxes(title_text="Tipper length", row=1, col=1)
        # fig.update_yaxes(type="log", row=1, col=1)
        # fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
        fig.update_yaxes(title_text="Angle (degrees)", row=2, col=1)
        # plot the tipper length
        scatter = go.Scatter(
            x=periods,
            y=self.get_length(components),
            mode="lines+markers",
            marker={"color": "red"},
            line={"color": "red"},
            name="Tipper length",
        )
        fig.add_trace(scatter, row=1, col=1)
        # plot the real angle
        scatter = go.Scatter(
            x=periods,
            y=self.get_real_angle(components),
            mode="lines+markers",
            marker={"color": "green"},
            line={"color": "green"},
            name="Real angle",
        )
        fig.add_trace(scatter, row=2, col=1)
        # plot the imag angle
        scatter = go.Scatter(
            x=periods,
            y=self.get_imag_angle(components),
            mode="lines+markers",
            marker={"color": "blue"},
            line={"color": "blue"},
            name="Imag angle",
        )
        fig.add_trace(scatter, row=2, col=1)
        return fig
