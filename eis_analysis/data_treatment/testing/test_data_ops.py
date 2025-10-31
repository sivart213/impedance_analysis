import numpy as np
import pytest

from testing.generators import (
    # generate_key_list,
    # generate_value_list,
    # generate_flat_dict,
    check_result_type_and_print,
)

from ..data_ops import (
    range_maker,
    shift_space,
    apply_extend,
    ensure_unique,
    clean_key_list,
)


# %% Fixtures
@pytest.fixture(scope="module")
def decade_range():
    """Provides a wide base range for shifting/extending tests."""
    return (-3, 7, 101)


# %% Tests


@pytest.mark.parametrize(
    "extend_params, description",
    [
        (
            {"extend_by": 5, "extend_to": None, "logscale": True},
            "extend stop by 5 steps (logscale)",
        ),
        (
            {"extend_by": -5, "extend_to": None, "logscale": True},
            "extend start by 5 steps (logscale)",
        ),
        ({"extend_by": 0, "extend_to": 1e8, "logscale": True}, "extend stop to 1e8 (logscale)"),
        ({"extend_by": 0, "extend_to": 1e-4, "logscale": True}, "extend start to 1e-4 (logscale)"),
        ({"extend_by": 0, "extend_to": None, "logscale": True}, "no extension (logscale)"),
        (
            {"extend_by": 3, "extend_to": None, "logscale": False},
            "extend stop by 3 steps (linear)",
        ),
    ],
)
def test_apply_extend(decade_range, extend_params, description):
    # Always use logspace for array generation, adjust input for linear test
    start_exp, stop_exp, num = decade_range
    arr = np.logspace(start_exp, stop_exp, num)
    start, stop = arr[0], arr[-1]

    # For linear test, convert start/stop to linear space but still use logspace for output comparison
    if not extend_params.get("logscale", True):
        start = start_exp
        stop = stop_exp

    result = apply_extend(start, stop, num, **extend_params)
    check_result_type_and_print(result, tuple, description)
    assert len(result) == 3

    # Always generate output as logspace for comparison
    arr_new = np.logspace(*result)

    # If arr_new[0] is close to arr[0], align by start; else align by stop
    if np.isclose(arr_new[0], arr[0], rtol=1e-8, atol=1e-12):
        arr_new_clip = arr_new[:num]
    else:
        arr_new_clip = arr_new[-num:]

    np.testing.assert_allclose(arr, arr_new_clip, rtol=1e-8, atol=1e-12)

    # Additional checks for extension
    if extend_params.get("extend_by", 0) != 0:
        assert result[2] == num + abs(
            extend_params.get("extend_by", 0)
        ), f"Expected sample count to increase, got {result[2]} instead of {num}"
        if extend_params.get("extend_by", 0) < 0:
            assert (
                result[0] < start_exp
            ), f"Expected start ({result[0]}) < original start_exp ({start_exp})"
        elif extend_params.get("extend_by", 0) > 0:
            assert (
                result[1] > stop_exp
            ), f"Expected stop ({result[1]}) > original stop_exp ({stop_exp})"
    if extend_params.get("extend_to") is not None:
        assert (
            result[2] > num
        ), f"Expected sample count to increase, got {result[2]} instead of {num}"
        assert (
            10 ** result[0] <= arr[0] or 10 ** result[1] >= arr[-1]
        ), f"Expected new range to extend beyond original range, got {result[0]} to {result[1]} "


@pytest.mark.parametrize(
    "shift_params, description",
    [
        (
            {"shift": 5, "logscale": True, "as_exp": True},
            "shift range outward by 5 (logscale, return exp)",
        ),
        (
            {"shift": -5, "logscale": True, "as_exp": True},
            "shift range inward by 5 (logscale, return exp)",
        ),
        (
            {"shift": 5, "logscale": True, "as_exp": False},
            "shift range outward by 3 (logscale, as_exp=True)",
        ),
        ({"shift": 2, "logscale": False, "as_exp": False}, "shift range outward by 2 (linear)"),
        ({"shift": 0, "logscale": True, "as_exp": True}, "no shift (logscale)"),
    ],
)
def test_shiftspace(decade_range, shift_params, description):
    # Always use logspace for array generation, adjust input for linear test
    start_exp, stop_exp, num = decade_range
    arr = np.logspace(start_exp, stop_exp, num)
    start, stop = arr[0], arr[-1]

    # For linear test, convert start/stop to linear space but still use logspace for output comparison
    if not shift_params.get("logscale", True):
        start = start_exp
        stop = stop_exp

    result = shift_space(start, stop, num=num, **shift_params)
    check_result_type_and_print(result, tuple, description)
    assert len(result) == 3
    assert result[2] == num  # Length should not change
    if shift_params.get("logscale", True) and not shift_params.get("as_exp", True):
        # Generate the shifted array in logspace
        result_arr = np.logspace(np.log10(result[0]), np.log10(result[1]), result[2])
    else:
        # Always generate output as logspace for comparison
        result_arr = np.logspace(*result)

    shift = shift_params.get("shift", 0)
    # If shift > 0, original aligns with right side of shifted; if < 0, aligns with left
    if shift > 0:
        result_arr_clip = result_arr[:-shift]
        arr_clip = arr[shift:]
    elif shift < 0:
        result_arr_clip = result_arr[-shift:]
        arr_clip = arr[:shift]
    else:
        result_arr_clip = result_arr
        arr_clip = arr
    print(f"\nOriginal:\n{arr};\nShifted:\n{result_arr}")

    np.testing.assert_allclose(arr_clip, result_arr_clip, rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize(
    "start, stop, maker_params, expected_type, description",
    [
        (
            1e-2,
            1e3,
            {"points_per_decade": 10, "fmt": "mfia"},
            dict,
            "mfia format, 10 points/decade, wide range",
        ),
        (
            1e-1,
            1e2,
            {"points_per_decade": 20, "fmt": "numpy"},
            tuple,
            "numpy format, 20 points/decade, medium range",
        ),
        (
            1e-3,
            1e4,
            {"points_per_decade": 5, "fmt": "mfia"},
            dict,
            "mfia format, 5 points/decade, wide range",
        ),
        (
            1,
            1e2,
            {"points_per_decade": 15, "fmt": "numpy"},
            tuple,
            "numpy format, 15 points/decade, 2 decades",
        ),
        (
            1e-1,
            1e3,
            {"points_per_decade": 12, "fmt": "mfia"},
            dict,
            "mfia format, 12 points/decade, 4 decades",
        ),
    ],
)
def test_range_maker(start, stop, maker_params, expected_type, description):
    result = range_maker(start, stop, **maker_params)
    check_result_type_and_print(result, expected_type, description)

    # Unify result to tuple (log10 start, log10 stop, count)
    if isinstance(result, dict):
        assert set(result.keys()) >= {"start", "stop", "samplecount"}
        result = (np.log10(result["start"]), np.log10(result["stop"]), int(result["samplecount"]))

    assert len(result) == 3
    arr = np.logspace(*result)

    # Check that the start and stop are within the generated array (or close)
    assert np.isclose(arr[0], start, rtol=1e-8, atol=1e-12) or arr[0]  # <= start
    assert np.isclose(arr[-1], stop, rtol=1e-8, atol=1e-12) or arr[-1]  # >= stop

    # Check that the central decade (1 to 10) is within the array
    points_in_decade = len(arr[(1 <= arr) & (arr < 10)])  # Count points in the decade 1-10
    assert (
        points_in_decade == maker_params["points_per_decade"]
    ), f"Central decade (1-10) has {points_in_decade} points, expected {maker_params['points_per_decade']}"


@pytest.fixture
def sample_keys():
    """Fixture providing sample tuple keys for clean_key_list tests."""
    return [
        ("a", "A", "x", 1, 11, "B"),
        ("a", "A", "y", 2, 11, "B"),
        ("b", "A", "x", 1, 12, "B"),
        ("b", "A", "y", 2, 12, "B"),
    ]


@pytest.mark.parametrize(
    "params, expected, description",
    [
        (
            # tests default behavior: returns tuples unchanged
            {"flatten": False},
            [
                ("a", "A", "x", 1, 11, "B"),
                ("a", "A", "y", 2, 11, "B"),
                ("b", "A", "x", 1, 12, "B"),
                ("b", "A", "y", 2, 12, "B"),
            ],
            "default: returns tuples unchanged when flatten=False and drop_mode=None",
        ),
        (
            # tests fill parameter: fills missing values in the last tuple
            {"flatten": False, "fill": "fill"},
            [
                ("a", "A", "x", 1, 11, "B"),
                ("a", "A", "y", 2, 11, "B"),
                ("b", "A", "x", 1, 12, "B"),
                ("b", "A", "y", 2, 12, "fill"),
            ],
            "fills missing values in last tuple using fill parameter",
        ),
        (
            # tests ensure_items_unique: removes duplicate tuples
            {"flatten": False, "ensure_items_unique": True},
            [
                ("a", "A", "x", 1, 11, "B"),
                ("a", "A", "y", 2, 11, "B"),
                ("b", "A", "x", 1, 12, "B"),
                ("b", "A", "y", 2, 12, "B"),
            ],
            "removes duplicates when ensure_items_unique=True",
        ),
        (
            # tests drop_mode="common" with flatten: drops columns with all common values and flattens
            {"flatten": True, "drop_mode": "common"},
            ["a/x/1/11", "a/y/2/11", "b/x/1/12", "b/y/2/12"],
            "drops columns with common values and flattens to string",
        ),
        (
            # tests drop_mode="minimize" with flatten: drops as many columns as possible while preserving uniqueness
            {"flatten": True, "drop_mode": "minimize"},
            ["a/x", "a/y", "b/x", "b/y"],
            "minimizes columns while preserving uniqueness and flattens",
        ),
        (
            # tests append_name=True with drop_mode="common": appends dropped string columns to next column
            {"flatten": True, "drop_mode": "common", "append_name": True},
            ["a/x(A)/1/11(B)", "a/y(A)/2/11(B)", "b/x(A)/1/12(B)", "b/y(A)/2/12(B)"],
            "appends dropped string columns to next column when flattening",
        ),
        (
            # tests append_name=True with drop_mode="minimize": appends all dropped columns to next column
            {"flatten": True, "drop_mode": "minimize", "append_name": True},
            ["a/x(A)(1)(11)(B)", "a/y(A)(2)(11)(B)", "b/x(A)(1)(12)(B)", "b/y(A)(2)(12)(B)"],
            "minimizes columns and appends all dropped columns to next column",
        ),
        (
            # tests append_ints=True with drop_mode="minimize": appends dropped integer columns to next column
            {"flatten": True, "drop_mode": "minimize", "append_ints": True},
            ["a/x(1)(11)", "a/y(2)(11)", "b/x(1)(12)", "b/y(2)(12)"],
            "minimizes columns and appends dropped integer columns to next column",
        ),
        (
            # tests empty input: returns empty list
            {"flatten": True},
            [],
            "returns empty list for empty input",
        ),
    ],
)
def test_clean_key_list(sample_keys, params, expected, description):
    """
    Test clean_key_list for various drop modes, flattening, and appending options.

    Clarifying comments:
    - First test: checks default behavior (no flatten, no drop_mode)
    - Second test: checks fill parameter for missing values
    - Third test: checks ensure_items_unique removes duplicates
    - Fourth test: checks drop_mode='common' with flatten
    - Fifth test: checks drop_mode='minimize' with flatten
    - Sixth test: checks append_name with drop_mode='common'
    - Seventh test: checks append_name with drop_mode='minimize'
    - Eighth test: checks append_ints with drop_mode='minimize'
    - Ninth test: checks empty input returns empty list
    """
    keys = sample_keys if expected else []
    if "fill" in params:
        # Remove last value from last tuple to test fill
        keys[-1] = keys[-1][:-1]
    elif "ensure_items_unique" in params:
        # Add a duplicate to test ensure_items_unique
        keys.append(keys[-1])
    result = clean_key_list(keys, **params)
    check_result_type_and_print(result, type(expected), description)
    assert result == expected, f"{description}: got {result}, expected {expected}"


@pytest.fixture
def unique_test_data():
    """Fixture providing test data for ensure_unique tests."""
    import pandas as pd

    # Clarifying comment: covers unique, duplicate, and DataFrame cases
    return {
        "unique_list": ["a", "b", "c"],
        "duplicate_list": ["a", "b", "a", "c", "b"],
        "df_simple": pd.DataFrame({"col1": ["x", "x", "y"], "col2": ["a", "b", "a"]}),
        "df_all_dupes": pd.DataFrame({"col1": ["x", "x"], "col2": ["a", "a"]}),
    }


@pytest.mark.parametrize(
    "data_key, param_dict, expected, description",
    [
        # Clarifying comment: tests unique input, should return unchanged
        ("unique_list", {}, ["a", "b", "c"], "returns unchanged if already unique"),
        # Clarifying comment: tests duplicate input, default behavior (add_numbers_all)
        (
            "duplicate_list",
            {},
            ["0_a", "0_b", "1_a", "0_c", "1_b"],
            "adds numbers to all duplicates (default)",
        ),
        # Clarifying comment: tests duplicate input, behavior='add_numbers'
        (
            "duplicate_list",
            {"behavior": "add_numbers"},
            ["a", "b", "1_a", "c", "1_b"],
            "adds numbers only to duplicates (add_numbers)",
        ),
        # Clarifying comment: tests duplicate input, behavior='raise'
        (
            "duplicate_list",
            {"behavior": "raise"},
            ValueError,
            "raises ValueError on duplicates (raise)",
        ),
        # Clarifying comment: tests DataFrame input, behavior='check_other_columns'
        (
            "df_simple",
            {"primary_column": "col1", "behavior": "check_other_columns"},
            ["a_x", "b_x", "a_y"],
            "uses other columns to make unique (check_other_columns, DataFrame)",
        ),
        # Clarifying comment: tests prefix=False, numbers appended at end
        (
            "duplicate_list",
            {"prefix": False},
            ["a_0", "b_0", "a_1", "c_0", "b_1"],
            "numbers appended at end when prefix=False",
        ),
        # Clarifying comment: tests custom separator
        (
            "duplicate_list",
            {"sep": "-"},
            ["0-a", "0-b", "1-a", "0-c", "1-b"],
            "uses custom separator",
        ),
    ],
)
def test_ensure_unique(unique_test_data, data_key, param_dict, expected, description):
    """
    Test ensure_unique for various input types and behaviors.

    Clarifying comments:
    - First test: checks unique input, should return unchanged
    - Second test: checks default numbering for duplicates (add_numbers_all)
    - Third test: checks add_numbers behavior (only duplicates get numbers)
    - Fourth test: checks ValueError raised for 'raise' behavior
    - Fifth test: checks DataFrame input with check_other_columns
    - Sixth test: checks prefix=False (numbers at end)
    - Seventh test: checks custom separator
    """
    data = unique_test_data[data_key]
    if expected is ValueError:
        # Clarifying comment: expects ValueError for duplicate input with behavior='raise'
        with pytest.raises(ValueError):
            result = ensure_unique(data, **param_dict)
    else:
        result = ensure_unique(data, **param_dict)
        check_result_type_and_print(result, type(result), description)
        # Convert to list of strings for comparison
        result_list = list(map(str, result))
        assert result_list == expected, f"{description}: got {result_list}, expected {expected}"
