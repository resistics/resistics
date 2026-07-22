"""Representative contracts used only to evaluate static API generation."""

from typing import overload

from pydantic import BaseModel, Field

__all__ = [
    "BaseContract",
    "DerivedContract",
    "Identifier",
    "PrototypeModel",
    "choose",
]

type Identifier = str


class BaseContract:
    """Base class used to verify inherited-member discovery.

    The source-link and cross-reference checks use Python's
    {py:class}`pathlib.Path` as an intersphinx target.
    """

    def inherited(self, value: int) -> str:
        """Render an integer through an inherited method.

        :param value: Value to render.
        :return: The decimal representation.
        """
        return str(value)


class DerivedContract(BaseContract):
    """Derived class linking to {py:meth}`BaseContract.inherited`."""

    def direct(self, value: str, /, *, suffix: str = "") -> str:
        """Return a value with an optional keyword-only suffix.

        :param value: Value to render.
        :param suffix: Text appended to the value.
        :return: The combined text.
        """
        return f"{value}{suffix}"


class PrototypeModel(BaseModel):
    """Small Pydantic model exercising MyST docstring features.

    The internal reference resolves to {py:func}`choose`, while
    {py:class}`pathlib.Path` is resolved through Python intersphinx.

    :param display_name: Public label accepted through a Pydantic alias.
    :param count: Non-negative number of items.
    :raises ValueError: If ``count`` is negative.

    ```{doctest}
    >>> from prototype_api import PrototypeModel
    >>> model = PrototypeModel(display_name="prototype", count=2)
    >>> (model.name, model.count)
    ('prototype', 2)
    ```

    ```{plot}
    :include-source: true

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    axis.set_title("MyST plot directive")
    ```
    """

    name: str = Field(alias="display_name")
    count: int = Field(default=1, ge=0)


@overload
def choose(value: int) -> int: ...


@overload
def choose(value: str) -> str: ...


def choose(value: int | str) -> int | str:
    """Return a value while preserving its overload-specific type.

    :param value: Integer or text input.
    :return: The unchanged value.
    """
    return value


class ExcludedContract:
    """Public-looking object deliberately excluded by ``__all__``."""
