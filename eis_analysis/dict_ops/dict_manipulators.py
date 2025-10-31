# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
from copy import deepcopy
from typing import Any, TypeVar, cast
from collections import defaultdict
from collections.abc import Callable

import numpy as np

try:
    from ..string_ops import compile_search_patterns
    from ..utils.decorators import handle_subdicts
except ImportError:
    from eis_analysis.string_ops import compile_search_patterns
    from eis_analysis.utils.decorators import handle_subdicts

T = TypeVar("T")


def nested_defaultdict():
    return defaultdict(nested_defaultdict)


def safe_deepcopy(obj: T) -> T:
    """Recursively perform a safe deepcopy for nested dictionaries, avoiding TypeError on unpicklable objects."""
    try:
        return deepcopy(obj)
    except TypeError:
        if isinstance(obj, dict):
            return cast(T, {key: safe_deepcopy(value) for key, value in obj.items()})
        elif isinstance(obj, (list, set, tuple)):
            return cast(T, type(obj)(safe_deepcopy(item) for item in obj))
        elif hasattr(obj, "__dict__"):
            new_obj = obj.__class__()
            new_obj.__dict__.update(
                {key: safe_deepcopy(value) for key, value in obj.__dict__.items()}
            )
            return cast(T, new_obj)
        return obj  # Return the object reference if it cannot be deepcopied


def nest_dict(data: dict, sep: str = "/") -> dict:
    """
    Splits concatenated keys in a dictionary into nested dictionaries using the specified separator.
    """
    if not isinstance(data, dict):
        return data

    def _insert(d, keys, value):
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value

    # Convert defaultdicts to dicts recursively
    def _to_dict(d):
        if isinstance(d, (defaultdict, dict)):
            return {k: _to_dict(v) for k, v in d.items()}
        return d

    result = nested_defaultdict()
    for key, val in data.items():
        if isinstance(key, tuple) or not isinstance(sep, str):
            keys = [k for k in key if k]
        else:
            keys = [k for k in key.split(sep) if k]
        _insert(result, keys, val)

    return _to_dict(result)


def flatten_dict(
    arg: dict,
    parent_key: str | tuple = "",
    sep: str | object = "/",
) -> dict:
    """
    Flattens a nested dictionary into a single dictionary, combining keys with a specified separator
    or as a tuple if sep is not a string.

    Parameters:
    - arg (dict): The nested dictionary to flatten.
    - parent_key (str or tuple, optional): The base key for the current level. Default is an empty string.
    - sep (str or any, optional): The separator to use for combining keys. If not a string, keys are returned as tuples.

    Returns:
    - dict: The flattened dictionary.
    """
    if not isinstance(arg, dict):
        return arg

    def combine_keys(*args):
        """
        Combines multiple keys into a tuple, flattening nested tuples/lists and filtering out invalid entries.
        """
        flat_args = []
        for arg in args:
            if isinstance(arg, (tuple, list)):
                flat_args.extend(tuple(arg))  # Flatten nested tuples/lists
            else:
                flat_args.append(arg)  # Add single keys directly
        # Filter out invalid entries (e.g., None or non-string types)
        return tuple([f_arg for f_arg in flat_args if isinstance(f_arg, str) and f_arg])

    items = []
    for k, v in arg.items():
        if isinstance(sep, str):
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
        else:
            new_key = combine_keys(parent_key, k)

        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def update_dict(base_dict: dict, updater_dict: dict) -> None:
    """Recursively update the base dictionary with values from the update dictionary."""
    if not base_dict or not updater_dict:
        return

    for key, value in updater_dict.items():
        if isinstance(value, dict) and key in base_dict:
            update_dict(base_dict[key], value)
        else:
            base_dict[key] = value
        # return new_dict


def filter_dict(base_dict: dict, filtering_dict: dict) -> dict:
    """Recursively filter the base dictionary to keep only values present in the filter dictionary."""
    if not filtering_dict:
        return base_dict

    new_dict = {}
    for key, value in filtering_dict.items():
        if key in base_dict:
            if isinstance(value, dict) and isinstance(base_dict[key], dict):
                new_dict[key] = filter_dict(base_dict[key], value)
            else:
                new_dict[key] = base_dict[key]

    return new_dict


def check_dict(to_check_dict: dict, base_dict: dict) -> dict:
    """Recursively nest to_check_dict within base_dict if keys are not found at the top level."""
    if not to_check_dict:
        return to_check_dict

    # Check if any keys of to_check_dict are in base_dict
    keys_in_base = any(key in base_dict for key in to_check_dict)

    if keys_in_base:
        return to_check_dict
    else:
        # If no keys of to_check_dict are in base_dict, recurse through values of base_dict that are dicts
        for key, value in base_dict.items():
            if isinstance(value, dict):
                nested_dict = check_dict(to_check_dict, value)
                if nested_dict:
                    return {key: nested_dict}
        return {}


def dict_level_ops(data: dict, operation: Callable, level: int = 1) -> dict:
    """
    Applies a specified operation to a nested dictionary at a given level.

    This function recursively traverses a nested dictionary and applies a specified
    operation to the data at the specified level. If the level is greater than 1,
    it continues to traverse deeper into the dictionary.

    Parameters:
    data (dict): The nested dictionary to process.
    operation (callable): The operation to apply to the data. Must be a callable function.
    level (int, optional): The level at which to apply the operation. Default is 1.

    Returns:
    dict or any: The dictionary with the operation applied at the specified level,
                 or the result of the operation if the level is reached.
    """
    if not callable(operation):
        return data

    if level > 1 and isinstance(data, dict):
        return {k: dict_level_ops(v, operation, level - 1) for k, v in data.items()}
    else:
        return operation(data)
    # return data


def truncate_dict_levels(
    arg: dict,
    max_levels: int = 2,
    sep: str | object = "/",
    merge_at: str = "outer",  # "inner" (default) or "outer"
) -> dict:
    """
    Limits the nesting of a dictionary to a specified number of levels.
    Flattens the dictionary to tuple keys, then re-nests up to max_levels.
    The remaining keys are merged at the specified level.

    Parameters:
        arg (dict): The nested dictionary to limit.
        max_levels (int): Maximum allowed nesting levels.
        sep (str or object): Separator for final keys if merging occurs.
        merge_at (str): Where to merge excess levels: "inner"  merges deepest keys,
                        "outer" (default) merges outermost keys.

    Returns:
        dict: The dictionary with limited nesting.
    """
    flat = flatten_dict(arg, sep=None)  # tuple keys

    if not flat:
        return arg

    if max_levels <= 1:
        return flat

    # If already within max_levels, just re-nest and return
    max_key_len = max(len(k) if isinstance(k, tuple) else 1 for k in flat.keys())
    if max_key_len <= max_levels:
        # Use nest_dict to re-nest, passing tuple keys directly
        return nest_dict(flat)

    idx = int(max_levels - 1)

    if isinstance(sep, str):
        if str(merge_at).lower() == "inner":
            modified = {(*k[:idx], sep.join(k[idx:])): v for k, v in flat.items()}
        else:  # "outer"
            modified = {(sep.join(k[:-idx]), *k[-idx:]): v for k, v in flat.items()}
    else:
        if str(merge_at).lower() == "inner":
            modified = {(*k[:idx], k[idx:]): v for k, v in flat.items()}
        else:  # "outer"
            modified = {(k[:-idx], *k[-idx:]): v for k, v in flat.items()}
    return nest_dict(modified)


def flip_dict_levels(arg: dict, levels: int = 2) -> dict:
    """
    Flips the first `levels` of a nested dictionary.
    For levels=2, this is equivalent to the classic "flip" of two-level dicts.

    Parameters:
        arg (dict): The nested dictionary to flip.
        levels (int): Number of levels to flip (default 2). If 0, reverses all levels.

    Returns:
        dict: The dictionary with flipped levels of nesting.
    """
    flat = flatten_dict(arg, sep=None)

    if not flat:
        return arg

    if not isinstance(levels, (float, int)) or int(levels) < 1:
        levels = 0
    levels = int(levels)

    flipped = {}
    for key, value in flat.items():
        new_key = key
        if isinstance(key, tuple):
            if levels == 0 or levels > len(key):
                new_key = tuple(reversed(key))
            else:
                new_key = tuple(reversed(key[:levels])) + key[levels:]
        flipped[new_key] = value

    return nest_dict(flipped)


def push_non_dict_items(d: Any) -> Any:
    """
    Ensure that only the deepest level of a multi-level nested dictionary has non-dict items.
    If a level has both dicts and non-dicts, add the non-dict items to all dicts and continue the process.
    """
    if not isinstance(d, dict):
        return d

    # Separate dict and non-dict items
    dict_items = {k: v for k, v in d.items() if isinstance(v, dict)}
    non_dict_items = {k: v for k, v in d.items() if not isinstance(v, dict)}

    # Recursively process each sub-dict
    if dict_items:
        for k, v in dict_items.items():
            v.update(non_dict_items)
            dict_items[k] = push_non_dict_items(v)

        return dict_items
    return non_dict_items


def merge_unique_sub_dicts(data: dict, keep_keys=None) -> dict:
    """
    Analyzes a dictionary and merges sub-dictionaries if all sub-dict keys are unique.

    Parameters:
    data (dict): The dictionary to analyze and merge.

    Returns:
    dict: The merged dictionary if all sub-dict keys are unique, otherwise the original dictionary structure.
    """
    if not isinstance(data, dict) or not all((isinstance(a, dict) for a in data.values())):
        return data
    if not isinstance(keep_keys, (tuple, list, np.ndarray)):
        keep_keys = [keep_keys]

    # Collect all keys from sub-dictionaries
    unique_keys = set()
    all_keys = []
    for key, value in data.items():
        if isinstance(value, dict):
            unique_keys.update(value.keys())
            all_keys = all_keys + list(value.keys())

    # Check if all keys are unique
    # if len(unique_keys) == sum(len(value) for value in data.values() if isinstance(value, dict)):
    if len(unique_keys) == len(all_keys) and not any(k in keep_keys for k in data.keys()):
        # Merge sub-dictionaries
        merged_dict = {}
        for key, value in data.items():
            if isinstance(value, dict):
                merged_dict.update(value)
            else:
                merged_dict[key] = value
        if all((isinstance(a, dict) for a in merged_dict.values())):
            return merge_unique_sub_dicts(merged_dict, keep_keys)
        else:
            return merged_dict
    else:
        # Recursively analyze sub-dictionaries
        return {key: merge_unique_sub_dicts(value, keep_keys) for key, value in data.items()}


@handle_subdicts
def separate_dict(
    data: dict,
    search_terms: list,
    reject_terms: list | None = None,
    keys: list | None = None,
) -> dict[Any, Any]:
    """
    Separates a dictionary into multiple dictionaries based on search terms.

    Parameters:
    - data (dict): The original dictionary to be separated.
    - search_terms (list): A list of search terms (strings or tuples of strings).
    - reject_terms (list): A list of common reject terms (strings).
    - keys (list): Optional list of keys for the result dictionary.

    Returns:
    - dict: A dictionary containing the desired groupings and residuals.
    """
    # Determine keys for the result dictionary
    if keys is None:
        keys = [str(term) for term in search_terms]
    keys.append("residuals")

    patterns = compile_search_patterns(search_terms, reject_terms)

    # Initialize dictionaries for each search term
    grouped_dicts = [defaultdict(dict) for _ in search_terms]
    residuals = {}

    # Iterate over dictionary items
    for key, value in data.items():
        matched = False
        for i, pattern in enumerate(patterns):
            if pattern.search(key):
                grouped_dicts[i][key] = value
                matched = True
                break
        if not matched:
            residuals[key] = value

    # Construct the result dictionary
    result = {key: dict(grouped_dict) for key, grouped_dict in zip(keys, grouped_dicts)}
    result["residuals"] = residuals

    return result


# ARCHIVE


# def merge_single_key(data: dict) -> dict:
#     """
#     Flattens a nested dictionary by recursively merging single-key dictionaries.

#     This function processes a nested dictionary and recursively merges any dictionaries
#     that contain only a single key, effectively flattening the structure.

#     Parameters:
#     data (dict): The nested dictionary to flatten.

#     Returns:
#     dict or any: The flattened dictionary, or the original value if the input is not a dictionary.
#     """
#     if isinstance(data, dict):
#         if len(data) <= 1:
#             return merge_single_key(list(data.values())[0])
#         else:
#             return {k: merge_single_key(v) for k, v in data.items()}
#     else:
#         return data
