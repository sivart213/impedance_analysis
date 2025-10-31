import re
import random
import string
import itertools
from typing import Any

import numpy as np
import pandas as pd


# %% Helper functions
def build_expected_dict(
    data: dict,
    columns: list,
    attrs: dict | None = None,
    fill_value: float = 0,
    all_attrs: bool = False,
) -> dict:
    """
    Build a dictionary suitable for DataFrame construction from data and columns.
    Handles DataFrames, Series, arrays, and scalars.

    Parameters:
        data (dict): The input data dictionary.
        columns (list): The list of columns to include.
        attrs (dict, optional): Attributes to set on the DataFrame(s).
        fill_value (float, optional): Value to fill missing entries in the DataFrame.
        all_attrs (bool, optional): If True, use all DataFrame attrs; otherwise, only other attrs.

    Returns:
        dict: Dictionary with "combined_data" DataFrame and any DataFrame columns as values.
    """
    if attrs is None:
        attrs = build_expected_attrs(data, columns)
    else:
        attrs = attrs.copy()
    df_dict: dict = {}
    arr_dict: dict = {}
    for key in columns:
        if key not in data:
            continue
        if isinstance(data[key], pd.DataFrame):
            df_dict[key] = data[key].copy()
            df_dict[key].attrs.update(attrs.copy())
        else:
            arr_dict[key] = pd.Series(np.asarray(data[key]).flatten())
    df = pd.DataFrame(arr_dict).fillna(fill_value)
    df.attrs.update(attrs.copy())
    res_dict: dict = {"combined_data": df, **df_dict}
    if df_dict:
        _, other_attrs, all_df_test_dict = compile_df_test_dict(data, list(df_dict.keys()))
        add_attrs = all_df_test_dict if all_attrs else other_attrs
        for value in res_dict.values():
            value.attrs.update(add_attrs)

    return res_dict


def build_expected_df(
    data: dict,
    columns: list,
    attrs: dict | None = None,
    as_multi: bool = False,
    fill_value: float = 0,
) -> pd.DataFrame:
    """
    Helper to build an expected DataFrame from a dict of data.

    Parameters:
        data (dict): The input data dictionary.
        columns (list): The list of columns to include.
        attrs (dict, optional): Attributes to set on the DataFrame.
        as_multi (bool, optional): If True, use MultiIndex for columns.
        fill_value (float, optional): Value to fill missing entries in the DataFrame.

    Returns:
        pd.DataFrame: The constructed DataFrame with specified columns and attributes.
    """
    if attrs is None:
        attrs = build_expected_attrs(data, columns)
    else:
        attrs = attrs.copy()

    res_dict = build_expected_dict(data, columns, attrs, fill_value, all_attrs=True)

    if not res_dict:
        df = pd.DataFrame()
        df.attrs.update(attrs.copy())
    elif len(res_dict) == 1:
        df = res_dict["combined_data"]
    elif as_multi:
        df = pd.concat(res_dict, axis=1).fillna(fill_value)
        df.attrs.update(res_dict["combined_data"].attrs.copy())
    else:
        df = pd.concat(list(res_dict.values()), axis=1).fillna(fill_value)
        df.attrs.update(res_dict["combined_data"].attrs.copy())

    return df


def build_expected_attrs(data, columns=None, attrs=None, top_key=""):
    """
    Build expected attrs dict from data, excluding keys in columns.
    - Arrays become lists of base types.
    - Arrays of length 1 become the scalar value.
    - DataFrames are split into lists with their columns as keys.
    - Nested dicts are flattened into attrs (recursively).
    """

    def to_base(val):
        if isinstance(val, np.generic):
            return val.item()
        return val

    if columns is None:
        columns = []
    if attrs is None:
        attrs = {}

    for key, value in data.items():
        if key in columns:
            continue
        key = f"{top_key}_{key}" if key in attrs else key
        # DataFrame: convert to dict of lists, flatten into attrs
        if isinstance(value, pd.DataFrame):
            nested = build_expected_attrs(
                value.dropna(how="all").to_dict(orient="list"), [], attrs, key
            )
            attrs.update(nested)

        # Array-like: convert to Series, then list of base types
        elif isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
            vals = pd.Series(list(value)).tolist()
            vals = [to_base(x) for x in vals]
            if len(vals) == 1:
                attrs[key] = vals[0]
            else:
                attrs[key] = vals
        # Dict: flatten recursively
        elif isinstance(value, dict):
            nested = build_expected_attrs(value, [], attrs, key)
            attrs.update(nested)
        # Everything else: store as base type
        else:
            attrs[key] = to_base(value)
    return attrs


def compile_df_test_dict(data, keys=None):
    """
    Returns three dicts of attrs:
    - valid_df_test_dict: {key+'_attrs': list(df.attrs.items())} for used DataFrames
    - other_df_test_dict: {key+'_attrs': list(df.attrs.items())} for DataFrames in attrs
    - all_df_test_dict: both combined
    """
    if keys is None:
        keys = []
    valid_df_test_dict = {}
    other_df_test_dict = {}
    all_df_test_dict = {}

    for key, value in data.items():
        if isinstance(value, pd.DataFrame) and value.attrs:
            all_df_test_dict[f"{key}_attrs"] = list(value.attrs.items())
            if key in keys:
                valid_df_test_dict[f"{key}_attrs"] = list(value.attrs.items())
            else:
                other_df_test_dict[f"{key}_attrs"] = list(value.attrs.items())

    return valid_df_test_dict, other_df_test_dict, all_df_test_dict


def check_result_type_and_print(
    result: dict | pd.DataFrame | None | Any,
    expected_type: type | None = None,
    description: str = "",
    print_mode: str = "",
    **kwargs,
) -> None:
    """
    Helper to print test description, assert result type, and print result info.

    Parameters:
        result (any): The result to check.
        expected_type (type or None): The expected type of the result.
        description (str, optional): Description of the test.
        print_mode (str, optional): '', 'dtypes', 'attrs', 'all'
        attrs (dict, optional): Attributes to print alongside the result.
        kwargs (dict, optional): Additional keyword arguments for the test.
    """

    kwargs_str = " (" + ", ".join(f"{k}={v}" for k, v in kwargs.items()) + ")" if kwargs else ""
    print_mode = print_mode.lower()
    # FLAT, REPETITIVE, PRINT-ASSERT-PRINT
    if expected_type is None:
        print(f"\nTest {description}{kwargs_str}:\nResult is None")
        assert result is None, f"Expected None, got {type(result)}"
        return

    if result is None or not isinstance(expected_type, type):
        raise ValueError(f"Invalid expected_type: {expected_type} of type {type(expected_type)}")

    if "df" in print_mode and "dtype" in print_mode and expected_type is dict:
        # dict of DataFrames, requesting dtype info
        print(f"\nTest {description}{kwargs_str}:")
        assert isinstance(result, dict), f"Expected {expected_type}, got {type(result)}"
        for key, value in result.items():
            print(f"\n{key} dtypes:\n{value.dtypes}")
            assert isinstance(value, pd.DataFrame), f"Expected {expected_type}, got {type(value)}"
            if "attr" in print_mode:
                print(f"\nAttrs dtypes: {dict_value_types(value.attrs)}")

    elif "df" in print_mode and expected_type is dict:
        # Indicates parent result is a dict of df and printing should be within a for loop
        print(f"\nTest {description}{kwargs_str}:")
        assert isinstance(result, dict), f"Expected {expected_type}, got {type(result)}"
        for key, value in result.items():
            print(f"\n{key}:\n{value}")
            assert isinstance(value, pd.DataFrame), f"Expected {expected_type}, got {type(value)}"
            if "attr" in print_mode:
                print(f"\nAttrs: {value.attrs}")

    elif ("only" in print_mode or "none" in print_mode) and expected_type is dict:
        # Indicates parent result is a dict of df and printing should be within a for loop
        print(f"\nTest {description}{kwargs_str}:")
        assert isinstance(result, dict), f"Expected {expected_type}, got {type(result)}"

    elif "dict" in print_mode and "dtype" in print_mode and expected_type is pd.DataFrame:
        # Indicates parent result is a dict and print is within a for loop
        print(f"\n`{description}`\n")
        assert isinstance(
            result, pd.DataFrame
        ), f"Expected {expected_type.__name__}, got {type(result)}"
        print(f"DataFrame dtypes:\n{result.dtypes}")
        if "attr" in print_mode:
            print(f"\nAttrs dtypes: {dict_value_types(result.attrs)}")

    elif "dict" in print_mode and expected_type is pd.DataFrame:
        # Indicates parent result is a dict and print is within a for loop
        print(f"\n`{description}`\nDataFrame\n{result}")
        assert isinstance(
            result, pd.DataFrame
        ), f"Expected {expected_type.__name__}, got {type(result)}"
        if "attr" in print_mode:
            print(f"\nAttrs: {result.attrs}")

    elif "dtype" in print_mode and expected_type is pd.DataFrame:
        # Basic request of dtypes for a DataFrame
        print(f"\nTest {description}{kwargs_str}:\n")
        assert isinstance(
            result, pd.DataFrame
        ), f"Expected {expected_type.__name__}, got {type(result)}"
        print(f"DataFrame dtypes:\n{result.dtypes}")
        if "attr" in print_mode:
            print(f"\nAttrs dtypes: {dict_value_types(result.attrs)}")

    elif expected_type is pd.DataFrame:
        # Base case, print DataFrame and its attrs
        print(f"\nTest {description}{kwargs_str}:\n")
        assert isinstance(
            result, pd.DataFrame
        ), f"Expected {expected_type.__name__}, got {type(result)}"
        print(f"DataFrame:\n{result}")
        if "attr" in print_mode:
            print(f"\nAttrs: {result.attrs}")

    elif expected_type is dict:
        # Unkown dict print request
        print(f"\nTest {description}{kwargs_str}:\n")
        assert isinstance(result, dict), f"Expected {expected_type.__name__}, got {type(result)}"
        print(f"Dict:\n{result}")
        if "attr" in print_mode:
            print(f"\nAttrs: {result.get('attrs', {})}")

    else:
        print(f"\nTest {description}{kwargs_str}:\n")
        assert isinstance(
            result, expected_type
        ), f"Expected {expected_type.__name__}, got {type(result)}"
        print(f"{expected_type.__name__}:\n{result}")
        if "attr" in print_mode:
            if hasattr(result, "attrs"):
                print(f"\nAttrs: {result.attrs}")  # type: ignore[attr-defined]
            elif hasattr(result, "get"):
                print(f"\nAttrs: {result.get('attrs', {})}")
            else:
                try:
                    if hasattr(result, "index") and "attrs" in result:
                        ind = result.index("attrs")  # type: ignore[attr-defined]
                        print(f"\nAttrs: {result[ind]}")
                except (AttributeError, TypeError, IndexError):
                    print("\nAttrs: {}")
                    # print(f"\nAttrs: {result.attrs if hasattr(result, 'attrs') else {}}")


def dict_value_types(attr_dict: dict) -> dict:
    """
    Return a dict mapping each key to the type name(s) found in its value.
    If the value is a list, returns a string like 'List[int, float]'.
    Otherwise, returns the type name as a string.
    """
    result = {}
    for key, value in attr_dict.items():
        if isinstance(value, dict):
            # Recursively process nested dicts
            result[key] = dict_value_types(value)
        elif hasattr(value, "__iter__") and not isinstance(value, str):
            type_names = sorted({type(item).__name__ for item in value})
            iter_type = type(value).__name__
            if iter_type in ["list", "tuple", "set"]:
                iter_type = iter_type.title()
            result[key] = f"{iter_type}[{', '.join(type_names)}]"
        else:
            result[key] = type(value).__name__
    return result


def assert_df_and_attrs_equal(
    result: pd.DataFrame,
    reference: pd.DataFrame,
    check_column_order: bool = False,
    check_values: bool = False,
    check_dtypes: bool = False,
    check_attr_val_order: bool = True,
) -> None:
    """
    Assert that result DataFrame matches reference in columns, attrs, and optionally values/dtypes.

    Parameters:
        result (pd.DataFrame): The DataFrame to check.
        reference (pd.DataFrame): The reference DataFrame.
        check_column_order (bool): If True, columns must match order; else, just keys.
        check_values (bool): If True, check all values for equality.
        check_dtypes (bool): If True, check dtypes for all columns.
        check_attr_val_order (bool): If False, sort values in attrs before comparison.
    """
    # Check the DataFrame itself
    if check_column_order and check_values:
        pd.testing.assert_frame_equal(result, reference)
    elif check_values:
        assert set(result.columns) == set(
            reference.columns
        ), f"Column keys mismatch:\nResult: {set(result.columns)}\nRef: {set(reference.columns)}"
        pd.testing.assert_frame_equal(result[reference.columns], reference)
    else:
        assert list(result.columns) == list(
            reference.columns
        ), f"Column order mismatch:\nResult: {list(result.columns)}\nRef: {list(reference.columns)}"
        if check_dtypes:
            pd.testing.assert_series_equal(result.dtypes, reference.dtypes)

    # Attrs (order-insensitive for keys, and for list values)
    assert set(result.attrs.keys()) == set(
        reference.attrs.keys()
    ), f"Attr keys mismatch:\nResult: {set(result.attrs.keys())}\nRef: {set(reference.attrs.keys())}"
    for attr_key in reference.attrs:
        result_value = result.attrs[attr_key]
        reference_value = reference.attrs[attr_key]
        if check_dtypes:
            if isinstance(reference_value, list):
                # Both should be lists of built-in types
                assert isinstance(result_value, list), f"Attr '{attr_key}' is not a list in result"

                result_types = [type(item) for item in result_value]
                reference_types = [type(item) for item in reference_value]
                assert (
                    result_types == reference_types
                ), f"Attr '{attr_key}' element types mismatch: {result_types} vs {reference_types}"
            else:
                assert type(result_value) is type(
                    reference_value
                ), f"Attr '{attr_key}' type mismatch: {type(result_value)} vs {type(reference_value)}"
        if check_values and check_attr_val_order:
            # Order-agnostic comparison for list/array-like attrs
            np.testing.assert_array_equal(
                np.asarray(result_value),
                np.asarray(reference_value),
                err_msg=f"Attr '{attr_key}' value mismatch",
            )
        elif check_values:
            # Order-agnostic comparison for list/array-like attrs
            np.testing.assert_array_equal(
                np.sort(np.asarray(result_value)),
                np.sort(np.asarray(reference_value)),
                err_msg=f"Attr '{attr_key}' value mismatch",
            )


# %% Predefined example values and types
# fmt: off
example_types = {
    "types": {  # All actual example values are nested here
        "string": "hello world",
        "integer": 42,
        "float": 3.14159,
        "boolean_true": True,
        "boolean_false": False,
        "none_type": None,
        "complex_number": 1 + 2j,
        "NaN": float("nan"),
        "inf": float("inf"),
        "int_bool_true": 1,  # True as int
        "int_bool_false": 0,  # False as int
        "single_element_list": [100],
        "multi_element_list": [1, 2, 3],
        "single_element_tuple": (200,),
        "multi_element_tuple": (4, 5, 6),
        "single_element_set": {300},
        "multi_element_set": {7, 8, 9},
        "dict": {"foo": "bar", "baz": 7},
        "empty_list": [],
        "empty_tuple": (),
        "empty_set": set(),
        "empty_dict": {},
        "empty_string": "",
        "single_char_string": "x",
        "unicode_string": "café",
        "escaped_string": "line1\nline2\tend",
        "bytes": b"bytes",
        "bytearray": bytearray(b"bytes"),
        "frozen_set": frozenset([1, 2, 3]),
        "range": range(3),
        "slice": slice(1, 10, 2),
    },
    # Groupings: lists of keys into the above dict
    "numbers": ["integer", "float", "complex_number", "NaN", "inf", "int_bool_true", "int_bool_false"],
    "strings": ["string", "empty_string", "single_char_string", "unicode_string", "escaped_string"],
    "collections": [
        "single_element_list", "multi_element_list", "single_element_tuple", "multi_element_tuple",
        "single_element_set", "multi_element_set", "dict", "empty_list", "empty_tuple",
        "empty_set", "empty_dict", "bytes", "bytearray", "frozen_set", "range"
    ],
    "empty": ["empty_list", "empty_tuple", "empty_set", "empty_dict", "empty_string"],
    "booleans": ["boolean_true", "boolean_false", "int_bool_true", "int_bool_false"],
    "single_element": ["single_element_list", "single_element_tuple", "single_element_set"],
    "special": ["none_type", "NaN", "inf", "slice"],
}

char_sets = {
    "lowercase": string.ascii_lowercase,
    "uppercase": string.ascii_uppercase,
    "ascii": string.ascii_letters,
    "coord": "xyzrstuvwqp",
    "greek_lowercase": "αβγδεζηθικλμνξοπρστυφχψω",
    "greek_uppercase": "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ",
    "greek": "αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ",
}
name_sets = {
    "greek": ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota",
            "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau",
            "upsilon", "phi", "chi", "psi", "omega"],
}
# fmt: on


# %% Generator functions
def slice_iter(iterable, n, total):
    """Yield n items from iterable, then skip m items."""
    m = total - n
    if n <= 0 or m <= 0:
        yield from iterable
        return
    it = iter(iterable)
    while True:
        yield from itertools.islice(it, n)
        next_chunk = list(itertools.islice(it, m))
        if len(next_chunk) < m:
            break


def pool_gen(pool, min_len, default_pool, suffix: str | object = "", suffix_sep: str = ""):
    """Yield from pool, extending with default_pool as needed."""
    suffix_str = suffix_sep + str(suffix)
    yield from (
        str(item) + suffix_str
        for item in itertools.islice(itertools.chain(pool, default_pool), min_len)
    )


def levels_gen(pool, min_pool_size, default_pool, add_index: bool = True, suffix_sep: str = ""):
    """Generate an infinite pool of keys, cycling through the provided pool."""
    if pool is None:
        pool = default_pool
    if not isinstance(pool[0], (list, tuple)):
        pool = [pool]
    pool_chain = itertools.cycle(pool)

    if add_index:
        for i, p in enumerate(pool_chain):
            yield list(pool_gen(p, min_pool_size, default_pool, i, suffix_sep))
    else:
        for p in pool_chain:
            yield list(pool_gen(p, min_pool_size, default_pool))


def value_gen(num_items, dtype=float):
    """Yield from pool, extending with default_pool as needed."""
    if not isinstance(dtype, type):
        dtype = float
    yield from (
        dtype(v + random.randrange(10) / 10)
        for v in random.sample(range(int(num_items)), k=int(num_items))
    )


def make_dimensional_values(shape: tuple[int, ...], numbers, rand_frac=0.0):
    """
    Generate a uniform list (or list of lists, etc.) of the given shape.
    If rand_frac > 0, each dimension will vary by up to ±rand_frac * dim, but dims are fixed within a call.
    """
    # Randomize each dimension ONCE
    dims = []
    for dim in shape:
        if rand_frac > 0 and random.random() < rand_frac:
            max_dev = max(1, int(dim * rand_frac))
            dim = max(2, dim + random.randint(-max_dev, max_dev))
        dims.append(dim)

    def build(level):
        if level == len(dims) - 1:
            return [next(numbers) for _ in range(dims[level])]
        else:
            return [build(level + 1) for _ in range(dims[level])]

    return build(0)


def make_char_list(character_set: str | tuple | list) -> list[str]:
    """
    Parses character_set to produce a list of characters/strings for use as keys.
    Supports:
      - "lower", "upper", "letters" for ascii sets (only one per string)
      - "coord" for common coordinate characters (only one per string)
      - "x<#>" (no space allowed, e.g., "upperx2") to repeat each character <#> times
      - tuple/list of strings: recursively concatenates results
      - Any other string: treated as literal characters (duplicates removed)
    """
    # Recursive handling for tuple/list
    if isinstance(character_set, (tuple, list)):
        out = []
        for part in character_set:
            out.extend(make_char_list(part))
        return out

    # Handle string inputs
    if isinstance(character_set, str):
        # If the source is a known character set, return it directly
        if character_set in char_sets.values():
            return list(character_set)

        # Otherwise, parse the string for known patterns
        s = character_set.strip().lower()
        if ("greek" in s or "unicode" in s) and "lower" in s:
            chars = list(char_sets["greek_lowercase"])
        elif ("greek" in s or "unicode" in s) and "upper" in s:
            chars = list(char_sets["greek_uppercase"])
        elif "greek" in s or "unicode" in s:
            chars = list(char_sets["greek"])
        elif "lower" in s:
            chars = list(char_sets["lowercase"])
        elif "upper" in s:
            chars = list(char_sets["uppercase"])
        elif "letters" in s or "ascii" in s:
            chars = list(char_sets["ascii"])
        elif "coord" in s:
            chars = list(char_sets["coord"])
        else:
            # If no known set matched, treat as literal and remove duplicates
            seen = set()
            chars = [c for c in character_set.strip() if not (c in seen or seen.add(c))]

        # Handle x<#> pattern for repetition (no space allowed)
        match = re.search(r"x(\d+)", s)
        if match and chars:
            n = int(match.group(1))
            chars = [c * n for c in chars]
        return chars
    else:
        raise ValueError(f"Unsupported character_set type: {type(character_set)}")


def generate_key_list(
    num_items: int,
    levels: int = 2,
    sep: str | object = "/",
    pool: list | str | None = None,
    charset: str | tuple | list = string.ascii_lowercase,
    base_item_count: int = 0,
    num_items_is_multiple: bool = False,
    **kwargs: Any,
):
    """
    Generate a list of string or tuple keys for a flat dictionary.

    Parameters:
        num_items (int): The number of keys to generate.
        levels (int, optional): The number of hierarchical levels in the keys. Defaults to 2.
        sep (str or object, optional): Separator to use between levels. If None, returns tuple keys.
        pool (list, str, or None, optional): Pool of names or characters to use for key generation.
            If a string, can refer to a named set (e.g., "greek"). If a list, used directly.
            If None, uses the charset.
        charset (str, tuple, or list, optional): Character set to use for key generation if pool is not provided.
            Can be a string (e.g., "lower", "upper", "coord", "greek lower"), a tuple/list of such strings,
            or a literal string of characters.

    Returns:
        list[str] or list[tuple]: List of generated keys as strings (if sep is a string) or tuples (if sep is None).

    Notes:
        - If levels == 1, keys are generated as flat strings.
        - If sep is None, keys are returned as tuples representing each level.
        - If the pool or charset is too small for the requested number of items, keys are extended as needed.
        - Useful for generating synthetic or test data with structured keys.
    """
    levels = max(1, int(levels))
    if num_items <= 0:
        return []

    # If pool is a str, assume it refers to a name set
    if isinstance(pool, str):
        pool = name_sets.get(pool.lower(), None)
    elif (
        isinstance(pool, list) and isinstance(pool[0], (list, tuple)) and len(pool) >= levels
    ) or levels == 1:
        kwargs["add_index"] = kwargs.get("add_index", False)
    elif isinstance(pool, list) and not pool:
        pool = None

    req_pool_size = int(np.ceil(num_items ** (1 / levels)))
    if base_item_count > 0:
        pool_size = max(req_pool_size, base_item_count) + max(0, req_pool_size - base_item_count)
    else:
        pool_size = max(req_pool_size, 1) + 1
        base_item_count = 0
    base_pool = make_char_list(charset)
    level_pools = list(itertools.islice(levels_gen(pool, pool_size, base_pool, **kwargs), levels))
    keys = list(slice_iter(itertools.product(*level_pools), base_item_count, pool_size))[
        :num_items
    ]

    if isinstance(sep, str):
        keys = [sep.join(k) for k in keys]
    return keys


def generate_value_list(
    num_items: int,
    values: list | tuple | int | None = None,
    randomize: float = 0.0,
    dtype: type = float,
    ensure_item_count: bool = False,
):
    """
    Generate a list of values for a flat dictionary.

    Parameters:
        num_items (int): The number of values to generate.
        values (list, tuple, int, or None, optional): Specifies the shape or content of the values.
            - If None, generates random floats.
            - If int, generates lists of that length for each item.
            - If tuple of up to 3 ints, generates multi-dimensional lists of the given shape.
            - If list, uses the provided list (repeats or truncates as needed).
        randomize (float, optional): If >0, introduces random variation in the shape of generated lists.
            The value is the fraction of items to randomize (0.0 = no randomization).
        dtype (type, optional): Data type for generated values (default: float).
        ensure_item_count (bool, optional): If True and values is a list shorter than num_items,
            repeats/truncates the list to match num_items.

    Returns:
        list: List of generated values, which may be scalars, lists, or nested lists depending on input.

    Notes:
        - If values is a tuple, generates lists of the specified shape for each item.
        - If randomize > 0, the length of each dimension may vary randomly for each item.
        - Useful for generating synthetic or test data with controlled structure and randomness.
    """
    if num_items <= 0:
        return []

    if isinstance(values, int):
        values = (values,)

    if isinstance(values, tuple) and len(values) <= 3 and all(isinstance(v, int) for v in values):
        all_values = value_gen(num_items * np.prod(values) * (1 + randomize), dtype=dtype)
        return [make_dimensional_values(values, all_values, randomize) for _ in range(num_items)]
    elif isinstance(values, list):  # and len(values) >= num_items:
        if ensure_item_count and len(values) < num_items:
            return values * (num_items // len(values)) + values[: num_items % len(values)]
        return values[:num_items]
    else:
        return list(value_gen(num_items, dtype=dtype))


def generate_flat_dict(
    num_items: int = 4,
    levels: int = 2,
    sep: str | object = "/",
    pool: list | str | None = None,
    charset: str | tuple | list = string.ascii_lowercase,
    values: list | tuple | int | None = None,
    randomize: float = 0.0,
    dtype: type = float,
    ensure_item_count: bool = False,
    set_vals_to: Any = None,
    **kwargs,
) -> dict[Any, Any]:
    """
    Factory fixture to create a flat dict with customizable separator, depth, number of items, and values.
    Passes through all keyword arguments to generate_key_list and generate_value_list.
    """
    keys = generate_key_list(
        num_items=num_items,
        levels=levels,
        sep=sep,
        pool=pool,
        charset=charset,
        **kwargs,
    )

    if set_vals_to is not None:
        return dict(zip(keys, [set_vals_to] * num_items))

    vals = generate_value_list(
        num_items=num_items,
        values=values,
        randomize=randomize,
        dtype=dtype,
        ensure_item_count=ensure_item_count,
    )
    return dict(zip(keys, vals))


def generate_diverse_dict(
    levels=2,
    num_items=5,
    with_dataframes=False,
    shape=(3, 3),
    num_empty=0,
    **kwargs,
) -> dict[Any, Any]:
    """
    Returns a dictionary with non-standard items for dict_to_df tests.
    Uses generate_key_list and generate_value_list to create:
    - Flat dicts
    - Nested dicts
    - Dicts of dicts
    - Dicts with DataFrames as values
    - Dicts with empty or short arrays
    - Dicts with mixed types (some empty, some short, some valid)
    """
    if num_items <= 0:
        return {}

    if num_empty < 0:
        num_empty = 0
    num_empty, num_items = min(num_empty, num_items), max(num_empty, num_items)

    if not shape:
        flat_dict = generate_flat_dict(num_items=num_items, set_vals_to={})
    elif len(shape) == 2:
        flat_dict = generate_flat_dict(
            num_items=num_items, levels=levels, values=(shape[1], shape[0]), **kwargs
        )
        for key, val in flat_dict.items():
            keys = generate_key_list(num_items=len(val), levels=1, charset="upper x2")
            flat_dict[key] = dict(zip(keys, val))
    else:
        flat_dict = generate_flat_dict(num_items=num_items, levels=levels, values=shape, **kwargs)

    if num_empty > 0:
        # convert the last num_empty items to empty lists
        empty_keys = list(flat_dict.keys())[-num_empty:]
        for key in empty_keys:
            flat_dict[key] = []

    # Dict with DataFrames as values
    if with_dataframes:
        # res_dict = {}
        for key, val in flat_dict.items():
            flat_dict[key] = pd.DataFrame(val)

    return flat_dict


def generate_df(
    num_cols: int = 5,
    length: list | tuple | int | None = (40,),
    col_names: list | str | None = "greek",
    randomize: float = 0.0,
    dtype: type = float,
    ensure_item_count: bool = False,
    set_vals_to: Any = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Factory fixture to create a flat dict with customizable separator, depth, number of items, and values.
    Passes through all keyword arguments to generate_key_list and generate_value_list.
    """
    if isinstance(length, int):
        length = (length,)
    kwargs.setdefault("levels", 1)
    return pd.DataFrame(
        generate_flat_dict(
            num_items=num_cols,
            pool=col_names,
            values=length,
            randomize=randomize,
            dtype=dtype,
            ensure_item_count=ensure_item_count,
            set_vals_to=set_vals_to,
            **kwargs,
        )
    )
