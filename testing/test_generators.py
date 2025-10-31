import numpy as np
import pytest

from testing.generators import (
    char_sets,
    generate_key_list,
    generate_value_list,
    check_result_type_and_print,
)


# %% Special test for the factory fixture
# --- Key generation tests ---
@pytest.mark.skip(reason="Background test, only run when specifically requested")
@pytest.mark.parametrize(
    "num_items, str_kwargs, expected_keys, description",
    [
        # base tests
        (0, {}, (), "zero items"),
        (2, {"levels": 1, "sep": "-"}, ("a", "b"), "1 level, sep irrelevant"),
        (5, {}, ("a0/a1", "a0/b1"), "default: 2 levels, slash sep, alpha lowercase"),
        # Level tests
        (20, {"levels": 4}, ("a0/a1/a2/a3",), "4 levels, levels < num_items"),
        (4, {"levels": 4}, ("a0/a1/a2/a3",), "4 levels, levels == num_items"),
        (2, {"levels": 4}, ("a0/a1/a2/a3",), "4 levels, levels > num_items"),
        # Separator tests
        (5, {"sep": None}, (("a0", "a1"),), "tuple keys"),
        (5, {"sep": "_"}, ("a0_a1",), "underscore sep"),
        # Character set tests
        (5, {"charset": "lower"}, ("a0/a1",), "ascii lowercase"),
        (5, {"charset": "upper"}, ("A0/A1",), "ascii uppercase"),
        (5, {"charset": "coord"}, ("x0/x1",), "coordinate chars"),
        (5, {"charset": "greek lower"}, ("α0/α1",), "greek lower"),
        (5, {"charset": "upper x2"}, ("AA0/AA1",), "uppercase x2"),
        (150, {"charset": ["coord", "lower"]}, ("x0/x1", "x0/a1"), "coord & lowercase"),
        # Pool tests
        (5, {"pool": "greek"}, ("alpha0/alpha1",), "greek name pool"),
        (5, {"pool": list(char_sets["uppercase"])}, ("A0/A1",), "flat pool list, ignores charset"),
        (
            25,
            {"pool": list(char_sets["uppercase"])[:3]},
            ("A0/A1", "A0/a1"),
            "flat pool list, num_items > pool size => default charset appears",
        ),
        (
            5,
            {"pool": [list(char_sets["coord"]), list(char_sets["lowercase"])]},
            ("x/a",),
            "composite pool list",
        ),
        (
            5,
            {"pool": [list(char_sets["coord"]), list(char_sets["lowercase"])], "add_index": True},
            ("x0/a1",),
            "composite pool list with add_index=True",
        ),
        (
            5,
            {"levels": 3, "pool": [list(char_sets["coord"]), list(char_sets["lowercase"])]},
            ("x0/a1/x2",),
            "composite pool list with repeated pool",
        ),
        (
            5,
            {
                "levels": 3,
                "pool": [list(char_sets["coord"]), list(char_sets["lowercase"])],
                "add_index": False,
            },
            ("x/a/x",),
            "composite pool list with repeated pool and add_index=False",
        ),
    ],
)
def test_generate_key_list(num_items, str_kwargs, expected_keys, description):
    """Test key generation for various charsets, levels, and separators."""
    result = generate_key_list(num_items, **str_kwargs)
    check_result_type_and_print(result, list, description)
    assert len(result) == num_items
    if expected_keys is not None and result:
        for key in expected_keys:
            assert key in result, f"Expected key '{key}' not found in generated result: {result}"
    # Check uniqueness
    assert len(set(result)) == len(result), f"Keys are not unique: {result}"


# --- Value generation tests ---
@pytest.mark.skip(reason="Background test, only run when specifically requested")
@pytest.mark.parametrize(
    "num_items, value_kwargs, expected_shape, description",
    [
        (0, {}, 0, "zero items"),
        (5, {}, 5, "default: float values"),
        (5, {"values": 3}, (3,), "5 1D lists (uniform)"),
        (5, {"values": (3,)}, (3,), "5 1D lists (uniform)"),
        (5, {"values": (2, 4)}, (2, 4), "5 2D lists (uniform)"),
        (5, {"values": (2, 3, 2)}, (2, 3, 2), "5 3D lists (uniform)"),
        (5, {"randomize": 0.5}, (4,), "5 float values (randomize has no effect)"),
        (20, {"values": (4,), "randomize": 0.5}, (4,), "20 1D lists (randomized)"),
        (20, {"values": (2, 4), "randomize": 0.3}, (2, 4), "20 2D lists (randomized)"),
        (5, {"values": [1, 2, 3]}, 3, "value list, limited by provided values (3)"),
        (5, {"values": [1, 2], "ensure_item_count": True}, 5, "value list, forced to 5"),
        (5, {"values": [1, 2, 3, 4, 5, 6, 7]}, 5, "value list, limited by num_items (5)"),
        (5, {"dtype": int}, 5, "5 int values"),
        (5, {"values": (3,), "dtype": int}, (3,), "5 1D lists of ints"),
        (5, {"values": (2, 4), "dtype": int}, (2, 4), "5 2D lists of ints"),
    ],
)
def test_generate_value_list(num_items, value_kwargs, expected_shape, description):
    """Test value generation for various shapes and randomization."""
    result = generate_value_list(num_items, **value_kwargs)
    check_result_type_and_print(result, list, description)

    if isinstance(expected_shape, int):
        assert (
            len(result) == expected_shape
        ), f"Expected {expected_shape} items, got {len(result)}: {result}"
        # Check type
        if isinstance(value_kwargs.get("values"), (list)):
            dtype = type(value_kwargs.get("values", [0])[0])
        else:
            dtype = value_kwargs.get("dtype", float)
        for v in result:
            assert isinstance(v, dtype)
    else:
        assert len(result) == num_items, f"Expected {num_items} items, got {len(result)}: {result}"
        # Expect a list of arrays/lists with the given shape
        rand = value_kwargs.get("randomize", 0)
        dtype = value_kwargs.get("dtype", float)
        print(f"Value shape: {expected_shape}, dtype: {dtype}")
        max_val = num_items * np.prod(expected_shape) * (1 + rand)
        sizes = []
        for v in result:
            arr = np.array(v)
            sizes.append(arr.shape)
            assert np.all(arr <= max_val), f"Value exceeds max_val: {arr}"
            assert arr.dtype == np.dtype(dtype), f"Expected dtype {dtype}, got {arr.dtype}"

        sizes = np.array(sizes)

        if rand > 0:
            expected = np.array(expected_shape)
            matches = np.sum([np.all(s == expected) for s in sizes])
            # Allow for randomness, but at least (1 - randomize - 0.15) fraction should match
            print(
                f"Randomized: {100 * (1 - matches / len(sizes)):.1f}% | "
                f"Shape min: {tuple( np.min(sizes, axis=0))}, max: {tuple(np.max(sizes, axis=0))}"
            )

            assert matches >= int(
                (1 - rand - 0.15) * len(sizes)
            ), f"Too few arrays of expected shape {expected_shape}: {matches}/{len(sizes)}"
        else:
            for s in sizes:
                assert tuple(s) == expected_shape, f"Expected shape {expected_shape}, got {s}"
