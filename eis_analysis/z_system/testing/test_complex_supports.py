# Suggested placement: in your test_complex_data.py or similar test file

import numpy as np
import pytest

from testing.generators import check_result_type_and_print
from eis_analysis.z_system.complexer import Complexer
from eis_analysis.z_system.imped_parsing import BaseParser, ItemTransforms


@pytest.fixture
def complexer_test_arrays():
    """Fixture providing test arrays for Complexer tests."""
    return {
        "single": np.array([1 + 2j]),
        "multi": np.array([1 + 2j, 3 - 4j, -1 + 0.5j]),
        "real_only": np.array([1.0, 2.0, 3.0]),
        "imag_only": np.array([1j, -2j, 3j]),
        "2d_cartesian": np.array([[1, 2], [3, -4], [-1, 0.5]]),  # (real, imag)
    }


@pytest.mark.parametrize(
    "array_key, operation, operand, expected_func, description",
    [
        # Clarifying comment: test __add__ with another Complexer
        ("multi", "__add__", "multi", lambda a, b: a + b, "addition with another Complexer"),
        # Clarifying comment: test __sub__ with scalar
        ("multi", "__sub__", 2, lambda a, b: a - b, "subtraction with scalar"),
        # Clarifying comment: test __mul__ with scalar
        ("multi", "__mul__", 2, lambda a, b: a * b, "multiplication with scalar"),
        # Clarifying comment: test __truediv__ with scalar
        ("multi", "__truediv__", 2, lambda a, b: a / b, "division with scalar"),
        # Clarifying comment: test __neg__
        ("multi", "__neg__", None, lambda a, _: -a, "negation"),
        # Clarifying comment: test __abs__
        ("multi", "__abs__", None, lambda a, _: np.abs(a), "absolute value"),
        # Clarifying comment: test __eq__ with itself
        ("multi", "__eq__", "multi", lambda a, b: np.array_equal(a, b), "equality with itself"),
        # Clarifying comment: test __getitem__ with index
        ("multi", "__getitem__", 1, lambda a, idx: a[idx], "getitem by index"),
        # Clarifying comment: test .real property
        ("multi", "real", None, lambda a, _: a.real, "real part"),
        # Clarifying comment: test .imag property
        ("multi", "imag", None, lambda a, _: a.imag, "imaginary part"),
        # Clarifying comment: test .mag property
        ("multi", "mag", None, lambda a, _: np.abs(a), "magnitude"),
        # Clarifying comment: test .phase property
        ("multi", "phase", None, lambda a, _: np.angle(a, deg=True), "phase (degrees)"),
        # Clarifying comment: test .slope property
        ("multi", "slope", None, lambda a, _: np.tan(np.angle(a, deg=False)), "tangent of phase"),
        # Clarifying comment: test 2D cartesian input
        ("2d_cartesian", None, None, lambda a, _: a[:, 0] + 1j * a[:, 1], "2D cartesian input"),
    ],
)
def test_complexer_operations(
    complexer_test_arrays, array_key, operation, operand, expected_func, description
):
    """
    Test Complexer class for arithmetic, properties, and indexing.

    Clarifying comments:
    - Each test checks a specific operator, property, or input handling.
    - The expected_func lambda computes the expected result for comparison.
    """
    arr = complexer_test_arrays[array_key]
    c = Complexer(arr)
    if operation == "__add__":
        other = Complexer(complexer_test_arrays[operand]) if isinstance(operand, str) else operand
        result = c + other
        check_result_type_and_print(result, Complexer, description)
        np.testing.assert_allclose(
            result.array,
            expected_func(
                arr, complexer_test_arrays[operand] if isinstance(operand, str) else operand
            ),
        )
    elif operation == "__sub__":
        result = c - operand
        check_result_type_and_print(result, Complexer, description)
        np.testing.assert_allclose(result.array, expected_func(arr, operand))
    elif operation == "__mul__":
        result = c * operand
        check_result_type_and_print(result, Complexer, description)
        np.testing.assert_allclose(result.array, expected_func(arr, operand))
    elif operation == "__truediv__":
        result = c / operand
        check_result_type_and_print(result, Complexer, description)
        np.testing.assert_allclose(result.array, expected_func(arr, operand))
    elif operation == "__neg__":
        result = -c
        check_result_type_and_print(result, Complexer, description)
        np.testing.assert_allclose(result.array, expected_func(arr, None))
    elif operation == "__abs__":
        result = abs(c)
        check_result_type_and_print(result, Complexer, description)
        np.testing.assert_allclose(result.array, expected_func(arr, None))
    elif operation == "__eq__":
        other = Complexer(complexer_test_arrays[operand]) if isinstance(operand, str) else operand
        result = c == other
        check_result_type_and_print(result, bool, description)
        assert result == expected_func(
            arr, complexer_test_arrays[operand] if isinstance(operand, str) else operand
        )
    elif operation == "__getitem__":
        result = c[operand]
        check_result_type_and_print(result, type(arr[operand]), description)
        assert np.all(result == expected_func(arr, operand))
    elif operation == "real":
        result = c.real
        check_result_type_and_print(result, np.ndarray, description)
        np.testing.assert_allclose(result, expected_func(arr, None))
    elif operation == "imag":
        result = c.imag
        check_result_type_and_print(result, np.ndarray, description)
        np.testing.assert_allclose(result, expected_func(arr, None))
    elif operation == "mag":
        result = c.mag
        check_result_type_and_print(result, np.ndarray, description)
        np.testing.assert_allclose(result, expected_func(arr, None))
    elif operation == "phase":
        result = c.phase
        check_result_type_and_print(result, np.ndarray, description)
        np.testing.assert_allclose(result, expected_func(arr, None))
    elif operation == "slope":
        result = c.slope
        check_result_type_and_print(result, np.ndarray, description)
        np.testing.assert_allclose(result, expected_func(arr, None))
    elif array_key == "2d_cartesian":
        # Clarifying comment: test 2D cartesian input is converted to complex array
        result = c.array
        check_result_type_and_print(result, np.ndarray, description)
        np.testing.assert_allclose(result, expected_func(arr, None))


class DummyParser(BaseParser, form_str="_form"):
    def __init__(self):
        self.value = np.array([1, 2, 3])

    def square_form(self, x):
        return np.array(x) ** 2

    def cube_form(self, x):
        return np.array(x) ** 3


@pytest.fixture
def dummy_parser():
    return DummyParser()


def test_add_valid_forms_and_getitem(dummy_parser):
    """
    Clarifying comment: tests that _add_valid_forms discovers and registers transformation methods,
    and that __getitem__ applies them correctly.
    """
    DummyParser._add_valid_forms()
    # Should register 'square' and 'cube' forms
    assert "square" in DummyParser._valid_forms
    assert "cube" in DummyParser._valid_forms

    # __getitem__ with direct attribute
    dummy_parser.value = np.array([2, 3])
    assert np.all(dummy_parser["value"] == np.array([2, 3]))

    # __getitem__ with transformation
    result = dummy_parser._parse_and_transform("square(value)")
    check_result_type_and_print(result, np.ndarray, "square transformation")
    assert np.all(result == np.array([4, 9]))

    # __getitem__ with operator
    result = dummy_parser._parse_and_transform("square(value) + cube(value)")
    check_result_type_and_print(result, np.ndarray, "operator in transformation")
    assert np.all(result == np.array([4 + 8, 9 + 27]))

    # Clarifying comment: test error on invalid form
    with pytest.raises(ValueError):
        dummy_parser._parse_and_transform("unknown(value)")


class DummyTransforms(ItemTransforms):
    def __init__(self):
        super().__init__()
        self.default_x = "dummy_arr"
        self.dummy_arr = np.linspace(0, 1, 5)


@pytest.fixture
def dummy_transforms():
    return DummyTransforms()


def test_smoothed_and_derivative_forms(dummy_transforms):
    """
    Clarifying comment: tests smoothed_form and derivative_form with default and custom kwargs.
    """
    arr = np.linspace(0, 10, 5)
    # Smoothed (should be close to original for linear data)
    result = dummy_transforms.smoothed_form(arr)
    check_result_type_and_print(result, np.ndarray, "smoothed_form")
    assert np.allclose(result, arr, atol=1e-1)

    # Derivative (should be close to constant for linear data)
    result = dummy_transforms.derivative_form(arr)
    check_result_type_and_print(result, np.ndarray, "derivative_form")
    assert np.allclose(result, np.full_like(arr, arr[1] - arr[0], dtype=float), atol=1e-1)


def test_form_kwargs_setters(dummy_transforms):
    """
    Clarifying comment: tests that form_kwargs, savgol_kwargs, interp_kwargs, norm_kwargs can be set and updated.
    """
    dummy_transforms.savgol_kwargs = {"window_length": 7}
    assert dummy_transforms.savgol_kwargs["window_length"] == 7

    dummy_transforms.interp_kwargs = {"axis": 1}
    assert dummy_transforms.interp_kwargs["axis"] == 1

    dummy_transforms.norm_kwargs = {"normalize_to": "max"}
    assert dummy_transforms.norm_kwargs["normalize_to"] == "max"


def test_valid_forms_aliasing(dummy_transforms):
    """
    Clarifying comment: tests that all aliases for smoothed_form are registered and work.
    """
    arr = np.arange(5)
    for alias in ["S", "ƒₛₘ", "sm", "smooth", "smoothed"]:
        result = getattr(dummy_transforms, dummy_transforms._valid_forms[alias])(arr)
        check_result_type_and_print(result, np.ndarray, f"alias {alias} for smoothed_form")
        assert np.allclose(result, arr, atol=1e-1)
