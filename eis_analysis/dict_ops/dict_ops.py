# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""

import re
import numpy as np
import pandas as pd

from collections import defaultdict

from ..string_ops import (
    common_substring,
    str_in_list,
    slugify,
    re_not,
    compile_search_patterns,
)
from ..utils.decorators import handle_subdicts

def dict_level_ops(data, operation, level=1):
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
    return data


def rename_from_subset(arg, name="name", func=None):
    """
    Recursively renames keys in a nested dictionary based on a subset of values.

    This function processes a nested dictionary and renames its keys based on a subset
    of values determined by a provided function. If no function is provided, it defaults
    to checking if the values are numeric strings. The renaming is applied uniformly
    across all levels of the dictionary.

    Parameters:
    arg (dict): The nested dictionary to process.
    name (str, optional): The default name to use for renaming keys. Default is "name".
    func (callable, optional): A function to determine the subset of values to use for renaming.
                               If None, defaults to checking if values are numeric strings.

    Returns:
    dict: The dictionary with renamed keys based on the subset of values.
    """
    if not isinstance(arg, dict):
        return arg

    if func is None:
        func = lambda x: str(x).isnumeric()

    # renaming is all or nothing
    if all(isinstance(v, dict) for v in arg.values()):
        res_dict = {k: rename_from_subset(v, name, func) for k, v in arg.items()}
        names = []
        if all(isinstance(v, dict) for v in res_dict.values()):
            for val in res_dict.values():
                sub_names = list(
                    np.unique([v for v in val.values() if isinstance(v, str)])
                )
                if len(sub_names) != 0:
                    if len(sub_names) == 1:
                        sub_name = str(sub_names[0])

                    else:
                        sub_name, resids = common_substring(sub_names, sep=" ")
                        sub_name = sub_name + " " + max(resids, key=len)

                    names.append(slugify(sub_name, True, " "))

        if len(names) == 0:
            return {**arg, **res_dict}

    elif any(isinstance(v, pd.DataFrame) for v in arg.values()):
        names = []
        for val in arg.values():
            if isinstance(val, pd.DataFrame) and name in val.attrs.keys():
                names.append(slugify(val.attrs[name], True, " "))
            else:
                names.append(None)
    else:
        names = ""
        if name in arg.keys():
            names = arg[name]
        elif str_in_list(name, arg.keys()) in arg.keys():
            names = str_in_list(name, arg.keys())
        else:
            return arg

        if isinstance(names, str):
            return slugify(names, True, " ")
        else:
            return arg

    if not any(names):
        return arg

    is_names = [n for n in names if n is not None]

    res = {}
    n_keys = list(arg.keys())
    nvals = list(arg.values())

    all_unique = len(is_names) == len(list(np.unique(is_names)))

    for n in range(len(arg)):
        if names[n] is not None:
            if func(n_keys[n]):
                if all_unique:
                    res[names[n]] = nvals[n]
                else:
                    res[n_keys[n] + "_" + names[n]] = nvals[n]
            else:
                res[n_keys[n]] = names[n]
        else:
            res[n_keys[n]] = nvals[n]

    return res


def flip_dict_levels(arg):
    """
    Flips the levels of a nested dictionary.

    This function takes a nested dictionary where each value is itself a dictionary,
    and flips the levels of nesting. The keys of the inner dictionaries become the
    outer keys, and the outer keys become the inner keys.

    Parameters:
    arg (dict): The nested dictionary to flip.

    Returns:
    dict: The dictionary with flipped levels of nesting.
    """
    if isinstance(arg, dict) and all((isinstance(a, dict) for a in arg.values())):
        res = {}
        for k1, v1 in arg.items():
            for k2, v2 in v1.items():
                res = {**{k2: {}}, **res}
                res[k2] = {**res[k2], **{k1: v2}}
        return res
    return arg


def dict_key_sep(data, sep="/"):
    """
    Separates concatenated keys in a dictionary using a specified separator.

    This function processes a dictionary where keys may be concatenated with a separator
    (e.g., "key1/key2"). It splits these keys into nested dictionaries based on the
    specified separator.

    Parameters:
    data (dict): The dictionary with concatenated keys to process.
    sep (str, optional): The separator used to concatenate keys. Default is "/".

    Returns:
    dict: The dictionary with keys separated into nested dictionaries.
    """
    if not isinstance(data, dict):
        return data
    blank = {}
    for key, val in data.items():
        pre = ""
        if key[0] == sep:
            pre = sep
        keys = key.split(sep)
        subkey = pre + sep.join(keys[2:])
        if len(subkey) > 1:
            if keys[1] not in list(blank.keys()):
                blank[keys[1]] = {}
            blank[keys[1]][subkey] = val
        else:
            try:
                blank[keys[1]] = val
            except IndexError:
                blank[key] = val
    return {k: dict_key_sep(v) for k, v in blank.items()}


def merge_single_key(data):
    """
    Flattens a nested dictionary by recursively merging single-key dictionaries.

    This function processes a nested dictionary and recursively merges any dictionaries
    that contain only a single key, effectively flattening the structure.

    Parameters:
    data (dict): The nested dictionary to flatten.

    Returns:
    dict or any: The flattened dictionary, or the original value if the input is not a dictionary.
    """
    if isinstance(data, dict):
        if len(data) <= 1:
            return merge_single_key(list(data.values())[0])
        else:
            return {k: merge_single_key(v) for k, v in data.items()}
    else:
        return data


def flatten_dict(d, parent_key='', sep='/'):
    """
    Flattens a nested dictionary into a single dictionary, combining keys with a specified separator.

    Parameters:
    - d (dict): The nested dictionary to flatten.
    - parent_key (str, optional): The base key for the current level. Default is an empty string.
    - sep (str, optional): The separator to use for combining keys. Default is "/".

    Returns:
    - dict: The flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


@handle_subdicts
def separate_dict(data, search_terms, reject_terms=None, keys=None):
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

    search_terms = [
        (
            re_not(term.replace("not ", "").strip())
            if term.startswith("not ")
            else term.strip()
        )
        for term in search_terms
    ]

    if isinstance(reject_terms, (tuple, list)):
        reject_terms = [
            (
                re_not(term.replace("not ", "").strip())
                if term.startswith("not ")
                else term.strip()
            )
            for term in reject_terms
        ]
        search_terms = [
            (
                [*t, *reject_terms]
                if isinstance(t, (tuple, list))
                else [t, *reject_terms]
            )
            for t in search_terms
        ]

    # Compile regex patterns for search terms
    patterns = [re.compile(compile_search_patterns(term)) for term in search_terms]

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


def recursive_concat(data, drop_singles=False):
    """
    Recursively concatenates all dictionaries into one pd.MultiIndex DataFrame,
    assuming the lowest level contains DataFrames.

    Parameters:
    data (dict): The dictionary to concatenate.
    parent_key (str, optional): The base key for the current level. Default is an empty string.

    Returns:
    pd.DataFrame: The concatenated DataFrame with a MultiIndex.
    """

    if isinstance(data, dict):
        frames = []
        keys = []
        for key, value in data.items():
            if isinstance(value, dict):
                frames.append(recursive_concat(value, drop_singles))
            elif isinstance(value, (pd.Series, pd.DataFrame)):
                frames.append(value)
            else:
                raise ValueError(
                    "The lowest level of the dictionary must contain DataFrames."
                )
            keys.append(key)
            # frames[-1].attrs = combine_attrs(attrs, frames[-1].attrs)

        if drop_singles and len(frames) == 1:
            res = frames[0].rename(
                columns={
                    k: k + " (" + list(data.keys())[0] + ")"
                    for k in frames[0].columns.levels[0]
                }
            )
            return res

        res = pd.concat(frames, keys=keys, axis=1)
        for n, frame in enumerate(frames):
            frame.attrs["df_names"] = (
                str(res.attrs["df_names"]) + "; " + keys[n]
                if "df_names" in res.attrs.keys()
                else keys[n]
            )
            for k, v in frame.attrs.items():
                res.attrs[k] = (
                    str(res.attrs[k]) + "; " + str(v)
                    if k in res.attrs.keys() and res.attrs[k] != v
                    else v
                )

        names = ["key" + str(n + 1) for n in range(len(res.columns.names) - 1)]
        names.append("cols")

        return res.rename_axis(columns=names)
    else:
        raise ValueError("The lowest level of the dictionary must contain DataFrames.")


def merge_unique_sub_dicts(data, keep_keys=None):
    """
    Analyzes a dictionary and merges sub-dictionaries if all sub-dict keys are unique.

    Parameters:
    data (dict): The dictionary to analyze and merge.

    Returns:
    dict: The merged dictionary if all sub-dict keys are unique, otherwise the original dictionary structure.
    """
    if not isinstance(data, dict) or not all(
        (isinstance(a, dict) for a in data.values())
    ):
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
    if len(unique_keys) == len(all_keys) and not any(
        k in keep_keys for k in data.keys()
    ):
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
        return {
            key: merge_unique_sub_dicts(value, keep_keys) for key, value in data.items()
        }


def dict_df(data, single=True):
    """
    Recursively converts a nested dictionary into a Pandas DataFrame.

    This function processes a nested dictionary where the values can be lists, tuples,
    numpy arrays, or other dictionaries. It converts these structures into DataFrames,
    handling nested dictionaries recursively.

    Parameters:
    data (dict): The nested dictionary to convert.
    single (bool): If True, converts only the first level of nested dictionaries into DataFrames.
                   If False, converts all levels of nested dictionaries.

    Returns:
    pd.DataFrame or dict: The resulting DataFrame if the dictionary can be fully converted,
                          otherwise the partially converted dictionary.
    """
    # TODO: Evaluate if this function is still necessary
    if not isinstance(data, dict):
        return data
    try:
        for key, val in data.items():
            if not isinstance(val, dict):
                continue
            if all([isinstance(v, dict) for v in val.values()]):
                data[key] = dict_df(val, single)
            else:
                vals = {
                    k: v
                    for k, v in val.items()
                    if not isinstance(v, dict)
                    and isinstance(v, (np.ndarray, list, tuple))
                }
                vals_items = {
                    k: v
                    for k, v in val.items()
                    if not isinstance(v, (np.ndarray, list, tuple, dict))
                }
                vals_dicts = {
                    k: dict_df(v, single) for k, v in val.items() if isinstance(v, dict)
                }
                if single and len(vals) > 0:
                    vlen = max([len(v) for v in vals.values()])
                    data[key] = pd.DataFrame(
                        {str(kk): vv for kk, vv in vals.items() if len(vv) == vlen}
                    )
                else:
                    tmp_new = {
                        str(len(v)): pd.DataFrame(
                            {kk: vv for kk, vv in vals.items() if len(vv) == len(v)}
                        )
                        for k, v in vals.items()
                    }
                    if len(vals_items) > 0:
                        attrs = pd.Series(vals_items)
                    if len(tmp_new) + len(vals_dicts) == 0:
                        data[key] = attrs
                    else:
                        data[key] = {
                            **tmp_new,
                            **vals_dicts,
                            **{"attrs": attrs},
                        }
    except (AttributeError, ValueError) as e:
        print(e)
        return data

    try:
        dlens = [v.size for v in data.values()]
        if all([d == max(dlens) for d in dlens]):
            return pd.DataFrame(data)
        else:
            return data
    except (AttributeError, ValueError) as e:
        print(e)
        return data


def dict_to_df(arg, columns=None, attrs=None):
    """
    Converts a nested dictionary into a Pandas DataFrame, handling attributes and column specifications.

    This function processes a nested dictionary where the values can be lists, tuples,
    numpy arrays, or other dictionaries. It recursively converts these structures into DataFrames,
    handling nested dictionaries and attaching additional attributes.

    Parameters:
    arg (dict): The nested dictionary to convert.
    columns (dict or list, optional): Specifies which keys to include as columns in the resulting DataFrame.
                                      Can be a dictionary for nested structures or a list for flat dictionaries.
    attrs (dict, optional): Additional attributes to attach to the resulting DataFrame.

    Returns:
    pd.DataFrame or dict: The resulting DataFrame if the dictionary can be fully converted,
                          otherwise the partially converted dictionary.
    """
    if not isinstance(arg, dict):
        return arg

    # Sanitize imports
    if attrs is None:
        attrs = {}

    # parse attrs
    if all((isinstance(a, dict) for a in arg.values())):
        attrs = {**{k: {} for k in arg.keys()}, **attrs}
        if isinstance(columns, dict) and any(k in arg.keys() for k in columns.keys()):
            columns = {**{k: {} for k in arg.keys()}, **columns}
        else:
            columns = {k: columns for k in arg.keys()}
        res = {k: dict_to_df(v, columns[k], attrs[k]) for k, v in arg.items()}
        return res

    sort_func = True
    if isinstance(columns, dict) and len(columns) == 1:
        columns = list(columns.values())[0]
    elif columns is None or isinstance(columns, dict):
        columns = list(arg.keys())
        sort_func = False

    unique, counts = np.unique([
            len(v)
            for v in arg.values()
            if isinstance(v, (tuple, list, np.ndarray, set, dict))
        ], return_counts=True)

    target_len = int(unique[np.argmax(counts)])

    attrs = {
        **attrs,
        **{
            k: v[0]
            for k, v in arg.items()
            if isinstance(v, (list, np.ndarray)) and len(v) == 1
        },
    }

    unified_args = {}
    for key, val in arg.items():
        if (
            key in columns
            and isinstance(val, (list, np.ndarray))
            and len(val) == target_len
        ):
            unified_args[key] = val
        elif isinstance(val, dict):
            attrs = {
                **attrs,
                **{
                    k: v
                    for k, v in val.items()
                    if (isinstance(v, (int, float)) and v != 0)
                    or (isinstance(v, (str, bytes)))
                },
            }
        elif (isinstance(val, (int, float)) and val != 0) or isinstance(
            val, (str, bytes)
        ):
            attrs[key] = val

    cols = [k for k in columns if k in unified_args.keys()]
    df = pd.DataFrame(
        {
            k: v
            for k, v in arg.items()
            if k in cols and isinstance(v, (list, np.ndarray)) and len(v) == target_len
        },
        columns=cols,
        dtype=float,
    )
    df.attrs = attrs

    if sort_func:
        df = df.sort_values(by=cols[0]).reset_index(drop=True)
    df = df.dropna(how="all").fillna(0)

    return df.loc[(df != 0).any(axis=1)]
