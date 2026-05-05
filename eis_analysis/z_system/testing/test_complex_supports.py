# Suggested placement: in your test_complex_data.py or similar test file


from itertools import product

import numpy as np
import pytest

from testing.helpers import buffered_print
from testing.generators import check_result_type_and_print
from eis_analysis.z_system.complexer import Complexer, parse_z_array
from eis_analysis.z_system.imped_parsing import BaseParser
from eis_analysis.z_system.testing.helpers import (
    get_data_df,
    get_data_form,
    make_timing_bins,
    percent_mismatch,
    explain_parsed_mismatch,
)

c_bins, time_class, c_info = make_timing_bins()
np.set_printoptions(precision=3, threshold=20, linewidth=100)  # suppress=True, linewidth=100)


def run_single_case(
    rc_ckt,
    sign,
    form,
    noise_opt,
    freq_opt,
    extra_opt,
    etype,
    data_form="Z",
    sequential=False,
    verbose=False,
):
    """Core test logic, reusable by both parametrized and aggregate tests."""
    data = get_data_form(rc_ckt, noise_opt, data_form=data_form)

    ref_data = np.repeat(data, 2) if sequential else data
    ref_freq = np.repeat(rc_ckt.freq, 2) if sequential else rc_ckt.freq

    df = get_data_df(
        data, rc_ckt.freq, form, freq_opt, extra_opt, sign, etype, seqential=sequential
    )

    parsed, freq = parse_z_array(df.to_numpy(), eval_polar=True, sign=sign)

    # buffered_print("Initial input:\n", df)
    # buffered_print("Result array:\n", np.array2string(parsed))
    # buffered_print("Result freq:\n", np.array2string(freq) if freq is not None else "None")

    assert isinstance(parsed, np.ndarray), "type(parsed) incorrect"
    assert np.iscomplexobj(parsed), "parsed is not complex"

    # ref_data = np.hstack([data, data[::-1]]) if sequential else data
    # ref_freq = np.hstack([rc_ckt.freq, rc_ckt.freq[::-1]]) if sequential else rc_ckt.freq
    # ref_data = np.repeat(data, 2) if sequential else data
    # ref_freq = np.repeat(rc_ckt.freq, 2) if sequential else rc_ckt.freq

    if not np.allclose(parsed, ref_data):
        msmatch = percent_mismatch(parsed, ref_data)
        msg = explain_parsed_mismatch(df, parsed, perc=f"{msmatch}%", print_arr=True)
        raise AssertionError(msg)

    if freq_opt:
        assert freq is not None, "No freq returned"
        if not np.allclose(freq, ref_freq):
            msmatch = percent_mismatch(freq, ref_freq)
            msg = explain_parsed_mismatch(
                df, freq, prefix="freq from", perc=f"{msmatch}%", print_arr=True
            )
            raise AssertionError(msg)
    else:
        if freq is not None:
            msmatch = "100%"
            msg = explain_parsed_mismatch(
                df, freq, prefix="freq should be None but is from", perc=msmatch
            )
            raise AssertionError(msg)


# %% ---- Parsing Test Section ----


@pytest.mark.parametrize(
    "value, expected_freq",
    [
        ([], None),  # empty
        (3.0, None),  # scalar
        (5 + 6j, None),  # Complex scalar
    ],
)
def test_parse_z_array_scalar_and_empty(value, expected_freq):
    data, freq = parse_z_array(value, eval_polar=True)
    assert isinstance(data, np.ndarray)
    assert np.iscomplexobj(data)
    assert freq is expected_freq


def test_parse_z_array_all_variations(rc_ckt, subtests):
    """
    Exhaustively test all combinations, grouped by DATA_FORM.
    No external params list; all variation logic is inline.
    """
    ETYPE_OPTS = ["a", "b", "c", "-a", "-b", "-c"]
    for data_form in ["Z", "Y", "M", "e_r"]:
        for form in ["complex", "1j*rect", "rect", "polar"]:
            # for seq in [False, True]:
            for noise_opt in [False, True]:
                with subtests.test(data_form=data_form, form=form, noise=noise_opt):
                    for sign in [1, -1]:
                        for freq_opt in [False, True]:
                            for extra in [0, 1, 2]:
                                etypes = ETYPE_OPTS if extra == 1 else [""]
                                for etype in etypes:
                                    try:
                                        run_single_case(
                                            rc_ckt,
                                            sign=sign,
                                            form=form,
                                            noise_opt=noise_opt,
                                            freq_opt=freq_opt,
                                            extra_opt=extra,
                                            etype=etype,
                                            data_form=data_form,
                                            verbose=False,
                                            # sequential=seq,
                                        )
                                    except AssertionError:
                                        raise


# %% ---- Complexer Section ----
def test_complexer_operations(complexer_test_arrays, subtests):
    # Shared array for most tests
    arr_multi: np.ndarray = complexer_test_arrays["multi"]
    c_multi = Complexer(arr_multi.copy())

    # 1. addition with another Complexer
    with subtests.test("addition with another Complexer"):
        result = c_multi + Complexer(arr_multi.copy())
        np.testing.assert_allclose(result.array, 2 * arr_multi)
        np.testing.assert_allclose(c_multi.array, arr_multi)

    # 2. subtraction with scalar
    with subtests.test("subtraction with scalar"):
        result = c_multi - 2
        np.testing.assert_allclose(result.array, arr_multi - 2)
        np.testing.assert_allclose(c_multi.array, arr_multi)

    # 3. multiplication with scalar
    with subtests.test("multiplication with scalar"):
        result = c_multi * 2
        np.testing.assert_allclose(result.array, arr_multi * 2)
        np.testing.assert_allclose(c_multi.array, arr_multi)
    # 4. division with scalar
    with subtests.test("division with scalar"):
        result = c_multi / 2
        np.testing.assert_allclose(result.array, arr_multi / 2)
        np.testing.assert_allclose(c_multi.array, arr_multi)

    # 5. negation
    with subtests.test("negation"):
        result = -c_multi
        np.testing.assert_allclose(result.array, -arr_multi)
        np.testing.assert_allclose(c_multi.array, arr_multi)

    # 6. absolute value
    with subtests.test("absolute value"):
        result = abs(c_multi)
        np.testing.assert_allclose(result.array, np.abs(arr_multi))
        np.testing.assert_allclose(c_multi.array, arr_multi)

    # 7. equality with itself
    with subtests.test("equality with itself"):
        result = c_multi == Complexer(arr_multi.copy())
        assert result == np.array_equal(arr_multi, arr_multi)
        np.testing.assert_allclose(c_multi.array, arr_multi)

    # 8. getitem by index
    with subtests.test("getitem by index"):
        assert np.all(c_multi[1] == arr_multi[1])
        np.testing.assert_allclose(c_multi.array, arr_multi)

    # 9. real part
    with subtests.test("real part"):
        np.testing.assert_allclose(c_multi.real, arr_multi.real)
        np.testing.assert_allclose(c_multi.array, arr_multi)

    # 10. imaginary part
    with subtests.test("imaginary part"):
        np.testing.assert_allclose(c_multi.imag, arr_multi.imag)
        np.testing.assert_allclose(c_multi.array, arr_multi)

    # 11. magnitude
    with subtests.test("magnitude"):
        np.testing.assert_allclose(c_multi.mag, np.abs(arr_multi))
        np.testing.assert_allclose(c_multi.array, arr_multi)

    # 12. phase (degrees)
    with subtests.test("phase (degrees)"):
        np.testing.assert_allclose(c_multi.phase, np.angle(arr_multi, deg=True))
        np.testing.assert_allclose(c_multi.array, arr_multi)

    # 13. tangent of phase
    with subtests.test("tangent of phase"):
        np.testing.assert_allclose(c_multi.slope, np.tan(np.angle(arr_multi, deg=False)))
        np.testing.assert_allclose(c_multi.array, arr_multi)

    # 14. 2D cartesian input (the only case with a different array)
    with subtests.test("2D cartesian input"):
        arr_2d: np.ndarray = complexer_test_arrays["2d_cartesian"]
        c_2d = Complexer(arr_2d.copy())
        expected = arr_2d[:, 0] + 1j * arr_2d[:, 1]
        np.testing.assert_allclose(c_2d.array, expected)


# %% ---- BaseParser Section ----
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


# %% ---- ItemTransforms Section ----
def test_item_transform_log_forms(dummy_transforms, subtests):
    """
    Clarifying comment: tests smoothed_form and derivative_form with default and custom kwargs.
    """
    # prec = 3
    arr = np.linspace(-5, 10, 16, dtype=float)
    c_real = 10**arr
    c_imag = 2 ** arr[::-1] - 1.1

    reals = {"real": c_real, "inv_real": 1 / c_real, "neg_real": -1 * c_real}
    imags = {"imag": c_imag, "inv_imag": 1 / c_imag, "neg_imag": -1 * c_imag}

    with subtests.test("ln_form with default base"):
        result = dummy_transforms.ln_form(np.e**arr)  # ln(e^x) should return x
        np.testing.assert_allclose(result, arr)
        buffered_print("Initial array:\n", np.array2string(arr))
        buffered_print("Result array:\n", np.array2string(result))

    with subtests.test("ln_form with inverted input"):
        result = dummy_transforms.ln_form(-np.e**arr)  # ln(e^x) should return x
        np.testing.assert_allclose(result, -arr)
        buffered_print("Initial array:\n", np.array2string(-arr))
        buffered_print("Result array:\n", np.array2string(result))

    with subtests.test("log10_form with default base"):
        result = dummy_transforms.log10_form(10**arr)  # log10(10^x) should return x
        np.testing.assert_allclose(result, arr)
        buffered_print("Initial array:\n", np.array2string(arr))
        buffered_print("Result array:\n", np.array2string(result))

    with subtests.test("log10_form with inverted input"):
        result = dummy_transforms.log10_form(-(10**arr))  # log10(10^x) should return x
        np.testing.assert_allclose(result, -arr)
        buffered_print("Initial array:\n", np.array2string(-arr))
        buffered_print("Result array:\n", np.array2string(result))

    for base in [2, 10, np.e]:
        with subtests.test("ln_form with normal input", base=round(base, 3)):
            result = dummy_transforms.ln_form(base**arr, base=base)
            np.testing.assert_allclose(result, arr)
            buffered_print("Initial array:\n", np.array2string(arr))
            buffered_print("Result array:\n", np.array2string(result))

        with subtests.test("ln_form with varied complex inputs", base=round(base, 3)):
            for real, imag in product(reals.keys(), imags.keys()):
                c_arr = reals[real] + 1j * imags[imag]
                result = base ** dummy_transforms.ln_form(c_arr, base=base)
                buffered_print("Initial array:\n", np.array2string(c_arr))
                buffered_print("Result array:\n", np.array2string(result))
                np.testing.assert_allclose(result, c_arr)

                c_arr = imags[imag] + 1j * reals[real]
                result = base ** dummy_transforms.ln_form(c_arr, base=base)
                buffered_print("Initial array:\n", np.array2string(c_arr))
                buffered_print("Result array:\n", np.array2string(result))
                np.testing.assert_allclose(result, c_arr)


def test_smoothed_and_derivative_forms(dummy_transforms):
    """
    Clarifying comment: tests smoothed_form and derivative_form with default and custom kwargs.
    """
    arr = np.linspace(0, 10, 5)
    # Smoothed (should be close to original for linear data)
    result = dummy_transforms.smoothed_form(arr)
    check_result_type_and_print(result, np.ndarray, "smoothed_form")
    np.testing.assert_allclose(result, arr, atol=1e-1)

    # Derivative (should be close to constant for linear data)
    result = dummy_transforms.derivative_form(arr)
    check_result_type_and_print(result, np.ndarray, "derivative_form")
    np.testing.assert_allclose(result, np.full_like(arr, arr[1] - arr[0], dtype=float), atol=1e-1)


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
        np.testing.assert_allclose(result, arr, atol=1e-1)


# def case_id(sign, form, noise, freq, extra, etype):
#     s_str = "-" if sign == -1 else "+"
#     parts = [f"{s_str}{form}"]
#     parts.append("w/ noise" if noise else "no noise")
#     parts.append("w/ freq" if freq else "no freq")
#     if extra == 1:
#         if len(etype) >= 2:
#             parts.append(f"w/ -1{etype[-1]} extra")
#         else:
#             parts.append(f"w/ 1{etype} extra")
#     else:
#         parts.append(f"w/ {extra} extra" if extra else "no extra")
#     return " | ".join(parts)

# @pytest.fixture
# def rc_ckt():
#     """Fixture providing frequency and impedance data from RCCircuit."""
#     return RCCircuit(freq=(-4, 7, 100), true_values=[24, 1e9, 1e-11], noise=0.001)


# def test_parse_z_array_all_variations(rc_ckt: RCCircuit | None):
#     error_cases = []
#     if rc_ckt is None:
#         rc_ckt = RCCircuit(freq=(-4, 7, 101), true_values=[24, 1e9, 1e-11], noise=0.01)
#     for f in DATA_FORMS:
#         for p in params:
#             try:
#                 run_single_case(rc_ckt, *p.values, data_form=f, verbose=False)
#             except AssertionError as exc:
#                 error_cases.append(f"{f}[{p.id}] {exc}")
#     if error_cases:
#         new_msg = (
#             f"{len(error_cases)} Failures in test_parse_z_array_all_variations:\n"
#             + "\n".join(error_cases)
#         )
#         raise AssertionError(new_msg)
# DATA_FORMS = ["Z", "Y", "M", "e_r"]
# SIGN_OPTS = [1, -1]
# FORMS = ["complex", "1j*rect", "rect", "polar"]
# BOOL_OPTS = [False, True]
# EXTRA_OPTS = [0, 1, 2]


# params = []
# for parts in itertools.product(SIGN_OPTS, FORMS, BOOL_OPTS, BOOL_OPTS, EXTRA_OPTS):
#     if parts[4] == 1:
#         for etype in ETYPE_OPTS:
#             eparts = parts + (etype,)
#             params.append(pytest.param(*eparts, id=case_id(*eparts)))
#     else:
#         eparts = parts + ("",)
#         params.append(pytest.param(*eparts, id=case_id(*eparts)))


# def test_parse_z_array_all_variations0(rc_ckt, subtests):
#     for f in DATA_FORMS:
#         with subtests.test(msg=f):
#             for parts in itertools.product(SIGN_OPTS, FORMS, BOOL_OPTS, BOOL_OPTS, EXTRA_OPTS):
#                 # with subtests.test(msg=f"{f} | {p.id}"):
#                 if parts[4] == 1:
#                     for etype in ETYPE_OPTS:
#                         eparts = parts + (etype,)
#                         try:
#                             run_single_case(rc_ckt, *eparts, data_form=f, verbose=False)
#                         except AssertionError:
#                             raise  # Re-raise to let subtests handle it
#                 else:
#                     eparts = parts + ("",)
#                     try:
#                         run_single_case(rc_ckt, *eparts, data_form=f, verbose=False)
#                     except AssertionError:
#                         raise  # Re-raise to let subtests handle it


# def test_parse_z_array_all_variations_old(rc_ckt):
#     error_cases = []
#     for f in DATA_FORMS:
#         for p in params:
#             try:
#                 run_single_case(rc_ckt, *p.values, data_form=f, verbose=False)
#             except AssertionError as exc:
#                 error_cases.append(f"{f}[{p.id}] {exc}")
#     if error_cases:
#         new_msg = (
#             f"{len(error_cases)} Failures in test_parse_z_array_all_variations:\n"
#             + "\n".join(error_cases)
#         )
#         raise AssertionError(new_msg)


# @pytest.fixture
# def complexer_test_arrays():
#     """Fixture providing test arrays for Complexer tests."""
#     return {
#         "single": np.array([1 + 2j]),
#         "multi": np.array([1 + 2j, 3 - 4j, -1 + 0.5j]),
#         "real_only": np.array([1.0, 2.0, 3.0]),
#         "imag_only": np.array([1j, -2j, 3j]),
#         "2d_cartesian": np.array([[1, 2], [3, -4], [-1, 0.5]]),  # (real, imag)
#     }


# @pytest.mark.parametrize(
#     "array_key, operation, operand, expected_func, description",
#     [
#         # Clarifying comment: test __add__ with another Complexer
#         ("multi", "__add__", "multi", lambda a, b: a + b, "addition with another Complexer"),
#         # Clarifying comment: test __sub__ with scalar
#         ("multi", "__sub__", 2, lambda a, b: a - b, "subtraction with scalar"),
#         # Clarifying comment: test __mul__ with scalar
#         ("multi", "__mul__", 2, lambda a, b: a * b, "multiplication with scalar"),
#         # Clarifying comment: test __truediv__ with scalar
#         ("multi", "__truediv__", 2, lambda a, b: a / b, "division with scalar"),
#         # Clarifying comment: test __neg__
#         ("multi", "__neg__", None, lambda a, _: -a, "negation"),
#         # Clarifying comment: test __abs__
#         ("multi", "__abs__", None, lambda a, _: np.abs(a), "absolute value"),
#         # Clarifying comment: test __eq__ with itself
#         ("multi", "__eq__", "multi", lambda a, b: np.array_equal(a, b), "equality with itself"),
#         # Clarifying comment: test __getitem__ with index
#         ("multi", "__getitem__", 1, lambda a, idx: a[idx], "getitem by index"),
#         # Clarifying comment: test .real property
#         ("multi", "real", None, lambda a, _: a.real, "real part"),
#         # Clarifying comment: test .imag property
#         ("multi", "imag", None, lambda a, _: a.imag, "imaginary part"),
#         # Clarifying comment: test .mag property
#         ("multi", "mag", None, lambda a, _: np.abs(a), "magnitude"),
#         # Clarifying comment: test .phase property
#         ("multi", "phase", None, lambda a, _: np.angle(a, deg=True), "phase (degrees)"),
#         # Clarifying comment: test .slope property
#         ("multi", "slope", None, lambda a, _: np.tan(np.angle(a, deg=False)), "tangent of phase"),
#         # Clarifying comment: test 2D cartesian input
#         ("2d_cartesian", None, None, lambda a, _: a[:, 0] + 1j * a[:, 1], "2D cartesian input"),
#     ],
# )
# def test_complexer_operations(
#     complexer_test_arrays, array_key, operation, operand, expected_func, description
# ):
#     """
#     Test Complexer class for arithmetic, properties, and indexing.

#     Clarifying comments:
#     - Each test checks a specific operator, property, or input handling.
#     - The expected_func lambda computes the expected result for comparison.
#     """
#     arr = complexer_test_arrays[array_key]
#     c = Complexer(arr)
#     if operation == "__add__":
#         other = Complexer(complexer_test_arrays[operand]) if isinstance(operand, str) else operand
#         result = c + other
#         check_result_type_and_print(result, Complexer, description)
#         np.testing.assert_allclose(
#             result.array,
#             expected_func(
#                 arr, complexer_test_arrays[operand] if isinstance(operand, str) else operand
#             ),
#         )
#     elif operation == "__sub__":
#         result = c - operand
#         check_result_type_and_print(result, Complexer, description)
#         np.testing.assert_allclose(result.array, expected_func(arr, operand))
#     elif operation == "__mul__":
#         result = c * operand
#         check_result_type_and_print(result, Complexer, description)
#         np.testing.assert_allclose(result.array, expected_func(arr, operand))
#     elif operation == "__truediv__":
#         result = c / operand
#         check_result_type_and_print(result, Complexer, description)
#         np.testing.assert_allclose(result.array, expected_func(arr, operand))
#     elif operation == "__neg__":
#         result = -c
#         check_result_type_and_print(result, Complexer, description)
#         np.testing.assert_allclose(result.array, expected_func(arr, None))
#     elif operation == "__abs__":
#         result = abs(c)
#         check_result_type_and_print(result, Complexer, description)
#         np.testing.assert_allclose(result.array, expected_func(arr, None))
#     elif operation == "__eq__":
#         other = Complexer(complexer_test_arrays[operand]) if isinstance(operand, str) else operand
#         result = c == other
#         check_result_type_and_print(result, bool, description)
#         assert result == expected_func(
#             arr, complexer_test_arrays[operand] if isinstance(operand, str) else operand
#         )
#     elif operation == "__getitem__":
#         result = c[operand]
#         check_result_type_and_print(result, type(arr[operand]), description)
#         assert np.all(result == expected_func(arr, operand))
#     elif operation == "real":
#         result = c.real
#         check_result_type_and_print(result, np.ndarray, description)
#         np.testing.assert_allclose(result, expected_func(arr, None))
#     elif operation == "imag":
#         result = c.imag
#         check_result_type_and_print(result, np.ndarray, description)
#         np.testing.assert_allclose(result, expected_func(arr, None))
#     elif operation == "mag":
#         result = c.mag
#         check_result_type_and_print(result, np.ndarray, description)
#         np.testing.assert_allclose(result, expected_func(arr, None))
#     elif operation == "phase":
#         result = c.phase
#         check_result_type_and_print(result, np.ndarray, description)
#         np.testing.assert_allclose(result, expected_func(arr, None))
#     elif operation == "slope":
#         result = c.slope
#         check_result_type_and_print(result, np.ndarray, description)
#         np.testing.assert_allclose(result, expected_func(arr, None))
#     elif array_key == "2d_cartesian":
#         # Clarifying comment: test 2D cartesian input is converted to complex array
#         result = c.array
#         check_result_type_and_print(result, np.ndarray, description)
#         np.testing.assert_allclose(result, expected_func(arr, None))


# class DummyTransforms(ItemTransforms):
#     def __init__(self):
#         super().__init__()
#         self.default_x = "dummy_arr"
#         self.dummy_arr = np.linspace(0, 1, 5)


# @pytest.fixture
# def dummy_transforms():
#     return DummyTransforms()
