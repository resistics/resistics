import pytest
from pydantic import ValidationError

from resistics.transfunc import ImpedanceTensor, TransferFunction


def test_transfer_function_resolves_channels_and_dimensions():
    """Derived transfer-function fields should be concrete after validation."""
    transfer_function = TransferFunction(out_chans=["Ex"], in_chans=["Hx", "Hy"])

    assert transfer_function.name == "TransferFunction"
    assert transfer_function.cross_chans == ["Hx", "Hy"]
    assert transfer_function.n_out == 1
    assert transfer_function.n_in == 2
    assert transfer_function.n_cross == 2


def test_transfer_function_dispatches_registered_model():
    """Serialized registered transfer functions should restore their model type."""
    transfer_function = TransferFunction.validate(
        {
            "name": "ImpedanceTensor",
            "variation": "electric-cross",
            "cross_chans": ["Ex", "Ey"],
        }
    )

    assert isinstance(transfer_function, ImpedanceTensor)
    assert transfer_function.n_out == 2
    assert transfer_function.n_in == 2
    assert transfer_function.n_cross == 2


def test_transfer_function_rejects_unknown_serialized_model():
    """Unknown serialized model names should not leak untyped dictionaries."""
    with pytest.raises(ValueError, match="Unknown transfer function 'NewTF'"):
        TransferFunction.validate(
            {
                "name": "NewTF",
                "out_chans": ["Ex"],
                "in_chans": ["Hx"],
            }
        )


def test_transfer_function_rejects_inconsistent_dimensions():
    """Serialized dimensions must agree with their channel lists."""
    with pytest.raises(ValidationError, match=r"n_in must equal len\(in_chans\)"):
        TransferFunction(out_chans=["Ex"], in_chans=["Hx", "Hy"], n_in=1)


def test_transfer_function_rejects_long_variation():
    """Variation labels remain intentionally short in all model variants."""
    with pytest.raises(ValidationError):
        TransferFunction(
            variation="a variation longer than sixteen characters",
            out_chans=["Ex"],
            in_chans=["Hx"],
        )
