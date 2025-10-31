# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
import re
from typing import Literal
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

try:
    from ..string_ops import safe_eval, find_common_str
    from ..data_treatment import (
        ensure_unique,
        clean_key_list,
        evaluate_1D_array,
        evaluate_nD_array,
    )
    from .dict_manipulators import flatten_dict
except ImportError:
    from eis_analysis.string_ops import safe_eval, find_common_str
    from eis_analysis.data_treatment import (
        ensure_unique,
        clean_key_list,
        evaluate_1D_array,
        evaluate_nD_array,
    )
    from eis_analysis.dict_ops.dict_manipulators import flatten_dict


def rename_from_internal_df(arg: dict, level: int = 0, name: str = "name") -> dict:
    """
    Recursively renames keys in a nested dictionary based on the names found in internal DataFrames.

    Parameters:
    arg (dict): The nested dictionary to process.
    level (int, optional): The level of the dictionary to rename. Default is 0.
                           If -1, rename the key that contains the data.
    name (str, optional): The attribute name to use for renaming keys. Default is "name".

    Returns:
    dict: The dictionary with renamed keys based on the names found in internal DataFrames.
    """
    if not isinstance(arg, dict):
        return arg

    def parse_names(arg_in, n_key="name"):
        """Parse names from the internal DataFrames."""
        names = []
        if isinstance(arg_in, pd.DataFrame) and n_key in arg_in.attrs.keys():
            # names.append(slugify(arg_in.attrs[n_key], True, " ")) # Warning: untested change!!!
            names.append(re.sub(r"[\s-]+", " ", re.sub(r"[^\w\s-]", "", n_key)).strip("-_"))
        elif isinstance(arg_in, dict):
            for val in arg_in.values():
                names.extend(parse_names(val))

        return names

    if level == 0 or (level < 0 and not all(isinstance(v, dict) for v in arg.values())):
        names = []

        for key, val in arg.items():
            possible_names = list(np.unique(parse_names(val))) or [key]
            if len(possible_names) == 1:
                name = possible_names[0]
            else:
                name, resids = find_common_str(*possible_names, sep=" ")
                name = name + " " + max(resids, key=len)
            names.append(name)

        names = ensure_unique(names, prefix=False)
        return {k: v for k, v in zip(names, arg.values())}
    else:
        res = {}
        for key, value in arg.items():
            if isinstance(value, dict):
                res[key] = rename_from_internal_df(value, level - 1, name)
        return res


def recursive_concat(
    data: dict,
    non_df_control: str = "raise",
    key_drop_mode: Literal["common", "minimize", None] = None,
) -> pd.DataFrame:
    """
    Recursively concatenates all dictionaries into one pd.MultiIndex DataFrame,
    assuming the lowest level contains DataFrames.

    Parameters:
    data (dict): The dictionary to concatenate.
    parent_key (str, optional): The base key for the current level. Default is an empty string.

    Returns:
    pd.DataFrame: The concatenated DataFrame with a MultiIndex.
    """
    flat = flatten_dict(data, sep=None)
    if not flat:
        raise ValueError("Input dictionary is empty or invalid.")

    def default_update(base, new, p_key):
        """Update where in the DF.attrs key is above the DF name"""
        for k, v in new.items():
            base[k][p_key] = v

    # Square
    keys = clean_key_list(list(flat.keys()), drop_mode=key_drop_mode)

    # Ensure all values are DataFrames or Series
    frames = []
    attrs = defaultdict(dict)
    for key, val in flat.items():
        if isinstance(val, pd.Series):
            frames.append(val.to_frame())
            default_update(attrs, val.attrs, "/".join(map(str, key)))
        elif isinstance(val, pd.DataFrame):
            frames.append(val)
            default_update(attrs, val.attrs, "/".join(map(str, key)))
        elif "ignore" in non_df_control.lower():
            # If non_df_control is set to ignore, skip non-DataFrame values
            continue
        else:
            raise ValueError(
                "The lowest level of the dictionary must contain DataFrames or Series."
            )

    res = pd.concat(frames, keys=keys, axis=1)
    # Set column names
    res.columns.names = ["key" + str(n + 1) for n in range(res.columns.nlevels - 1)] + ["cols"]

    # Merge attrs
    # Need to insert cleanup step
    res.attrs.update(cleanup_dict(attrs, flatten_dict=False, drop_dict_duplicates=True))
    res.attrs["df_names"] = ["/".join(map(str, k)) for k in keys]
    res.attrs["df_name_keys"] = [k + (slice(None),) for k in keys]
    res.attrs["py_id"] = id(res)
    return res


def parse_dict_of_datasets(
    arg: dict,
    attrs: dict | None = None,
    min_len: int = 3,
    **kwargs,
) -> dict:
    """
    Converts a nested dictionary into a Pandas DataFrame, handling attributes and column specifications.

    This function processes a nested dictionary where the values can be lists, tuples,
    numpy arrays, or other dictionaries. It recursively converts these structures into DataFrames,
    handling nested dictionaries and attaching additional attributes.

    Parameters:
    arg (dict): The nested dictionary to convert.
    attrs (dict, optional): Additional attributes to attach to the resulting DataFrame.
    min_len (int, optional): Minimum length of data to consider for conversion. Default is 3.

    Returns:
    pd.DataFrame or dict: The resulting DataFrame if the dictionary can be fully converted,
                          otherwise the partially converted dictionary.

    Notes:
    - Recursion requires uniformity in the data structure, interpreting differences as the data location.
    """
    if not isinstance(arg, dict) or not arg:
        return arg

    # Initialize `attrs` if it is not provided
    if attrs is None:
        attrs = {}

    # Get the type of the first value in the dictionary
    first_type = type(next(iter(arg.values())))

    # only `dict` or `pd.DataFrame` are allowed, set `first_type` to `False` to bypass the check
    if first_type not in (dict, pd.DataFrame):
        first_type = False

    # --- If all values in arg are of the same type (dict or pd.DataFrame), process them ---
    if first_type and all((isinstance(a, first_type) for a in arg.values())):
        res = {}

        for key, value in arg.items():
            # Get attributes for the current key, defaulting to an empty dict
            current_attrs = attrs.get(key, {})
            # Recursively process nested dictionaries
            if not isinstance(value, pd.DataFrame):
                value = parse_dict_of_datasets(value, current_attrs, min_len, **kwargs)

            # Handle DataFrames
            if isinstance(value, pd.DataFrame):
                if value.empty or (min_len and len(value) < min_len):
                    value = {}
                else:
                    # Clean up attributes for DataFrames
                    value.attrs = _cleanup_attrs({**current_attrs, **value.attrs}, id(value), key)

            # Add non-None results to the output dictionary
            if (isinstance(value, dict) and value) or (
                isinstance(value, pd.DataFrame) and not value.empty
            ):
                res[key] = value

        # Return the processed result if it contains valid results
        return res if res else {}

    # --- Otherwise treat as a data dictionary ---
    # sanitize kwargs prior to passing to dict_to_df
    valid_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k in ["selection_mode", "prevent_dict_return", "prevent_none_return"]
    }
    # Flat case: convert this dict to DataFrame
    return dict_to_df(arg, attrs, min_len, **valid_kwargs)


def dict_to_df(
    arg,
    attrs=None,
    min_len=3,
    selection_mode="most_common",  # or "all_valid", "all_arrays", "max_valid", "max_arrays"
    prevent_dict_return=True,
    prevent_none_return=False,
):
    """
    Convert a flat dictionary to a Pandas DataFrame, extracting array-like values as columns
    and collecting scalar or string values as DataFrame attributes.

    Parameters:
        arg (dict): The flat dictionary to convert.
        attrs (dict or None): Additional attributes to attach to the DataFrame.
        min_len (int): Minimum number of rows required for the DataFrame.
        selection_mode (str): Mode for selecting which array-like values become columns.
            Allowed values:
                - "most_common": Make columns with the most common array length.
                - "all_valid": Use all arrays with length >= min_len.
                - "all_arrays": Use all arrays with length > 1 (min_len is ignored).
                - "max_valid": Use only the arrays with the max length available, when also >= min_len.
                - "max_arrays": Use only the arrays with the max length available, regardless of min_len.
        prevent_dict_return (bool): If True, always returns a DataFrame, not a dict of DataFrames.
        prevent_none_return (bool): If True, returns a DataFrame from attributes if no columns are found.

    Returns:
        pd.DataFrame or None: The resulting DataFrame with attributes, or None if conversion is not possible.
    """
    # --- Selection mode pre-parsing ---
    if not isinstance(selection_mode, str):
        selection_mode = "most_common"

    selection_mode = selection_mode.lower()

    if "array" in selection_mode:
        min_len = 0
        selection_mode = selection_mode.split("_")[0] + "_valid"

    # --- Helper functions ---
    def _is_short(val):
        """Check if the value meets the length requirement."""
        return min_len and val < min_len

    def _cleanup_dict(
        val: dict,
        top_key: str = "",
        d_in: dict | None = None,
    ) -> dict:
        d = d_in if isinstance(d_in, dict) else {}

        for k, v in val.items():
            re_key = top_key + "_" + str(k) if k in d else str(k)
            if (isinstance(v, str) or not hasattr(v, "__iter__")) and v:
                d[re_key] = v
            elif isinstance(v, (list, np.ndarray, tuple, set, pd.Series)):
                if not hasattr(v, "__getitem__"):
                    v = list(v)
                v = (
                    pd.Series(np.asarray(v).flatten())
                    .dropna(ignore_index=True)
                    .drop_duplicates(ignore_index=True)
                )
                if v.empty:
                    continue
                elif len(v) == 1:
                    d[re_key] = v.iloc[0]
                else:
                    d[re_key] = v.tolist()
            elif isinstance(v, dict):
                d.update(_cleanup_dict(v, re_key, d))
        return d

    def _residual_arrays(*rejects):
        """
        Efficiently process non-target-length arrays from any number of candidate dicts.
        Each input is a defaultdict(dict) mapping length -> {key: arr}.
        Returns a dict of attributes: {key: value or list}.
        """
        attr = {}
        for d in rejects:  # iter the tuple
            for subdict in d.values():  # iter the dicts of a given length
                for key, arr in subdict.items():  # iter the items for that length
                    # arr is a pd.Series
                    if arr.nunique(dropna=False) == 1:
                        attr[key] = arr.iloc[0]
                    else:
                        attr[key] = arr.tolist()
        return attr

    def _eval_dfs(df_dict, attr=None):
        """Evaluate the DataFrames in the dictionary and return a single DataFrame or None."""
        if attr is None:
            attr = {}
        if attr:
            attr = {k: v.item() if isinstance(v, np.generic) else v for k, v in attr.items()}

        if df_dict:
            res = df_dict.copy()
            if prevent_dict_return:
                # df_attrs = {str(k) + "_attrs": list(v.attrs.items()) for k, v in df_dict.items() if v.attrs}
                res = pd.concat(df_dict, axis=1)
                res = res.loc[((res != 0) & ~res.isna()).any(axis=1)]

                num_cols = res.select_dtypes(include="number").columns
                res[num_cols] = res[num_cols].fillna(0)
                res.attrs.update(
                    {
                        str(k) + "_attrs": list(v.attrs.items())
                        for k, v in df_dict.items()
                        if v.attrs
                    }
                )
                # res.attrs.update({**{k: v for k, v in df_attrs.items() if v}, **attr})
                # res.attrs = attr
                if (
                    isinstance(res.columns, pd.MultiIndex)
                    and len(res.columns) == res.columns.get_level_values(-1).nunique()
                ):
                    res.columns = res.columns.get_level_values(-1)
                res = {"a": res}

            for df in res.values():
                df.attrs.update(attr)

            if len(res) == 1:
                res = list(res.values())[0]
            return res
        else:
            if prevent_none_return and attr:
                return pd.DataFrame(
                    {k: [v] if isinstance(v, list) else v for k, v in attr.items()},
                    index=[0],
                )
            return None

    cleanup_kwargs = dict(
        flatten_dict=True,
        flatten_arr=True,
        dropna=True,
        drop_duplicates=True,
    )
    # --- Main pass: gather candidates for columns, attributes, and lengths ---
    candidates = defaultdict(dict)
    short_candidates = defaultdict(dict)
    df_candidates = {}
    attr_candidates = cleanup_dict(dict(attrs), **cleanup_kwargs) if attrs else {}
    # attr_candidates = _cleanup_dict(dict(attrs)) if attrs else {}
    lengths = []
    orig_keys = list(arg.keys())  # For final column order

    for key, val in arg.items():
        key_str = str(key)
        # --- DataFrame handling ---
        if isinstance(val, pd.DataFrame):

            # revised version
            if 1 in val.shape:
                # returned value will be a series in this case
                val = evaluate_nD_array(val, operations="reshape")
            else:
                # is 2D dataframe,
                reduced = evaluate_nD_array(
                    val, operations=["drop_null"], null_values=[0], drop_on=0
                )
                if reduced.empty:
                    continue

                # if df wouldn't be rejected add,
                if not _is_short(len(reduced)):
                    df_candidates[key_str] = val
                    continue
                else:
                    # convert to dict
                    if val.attrs:
                        attr_candidates.update({f"{key_str}_attrs": list(val.attrs.items())})
                    val = evaluate_nD_array(
                        val, operations=["drop_na"], drop_on="both", as_type=pd.DataFrame
                    ).to_dict(orient="list")

        # --- Series with named index (treated as dict) ---
        if isinstance(val, pd.Series) and not pd.api.types.is_integer_dtype(val.index):
            val = val.to_dict()

        # --- Scalars and strings ---
        if (not hasattr(val, "__iter__") or isinstance(val, (str, bytes))) and val:  # type: ignore[assignment]
            attr_candidates[key_str] = val.item() if isinstance(val, np.generic) else val

        # --- Dicts: extract scalar/string or list-like items as attributes ---
        elif isinstance(val, dict):
            attr_candidates.update(
                cleanup_dict(
                    val, parent_key=key_str, parent_dict=attr_candidates, **cleanup_kwargs
                )
            )
        # --- General iterables: convert to Series, handle as column candidates ---
        elif hasattr(val, "__iter__"):
            if not hasattr(val, "__getitem__"):
                val = list(val)

            val = (
                val.reset_index(drop=True)
                if isinstance(val, pd.Series)
                else pd.Series(np.asarray(val).flatten())
            )
            val_len = len(val)
            if val_len == 1 and min_len != 1:
                if not pd.isna(val.iloc[0]) and val.iloc[0]:
                    attr_candidates[key_str] = val.iloc[0]
                continue

            lengths.append(val_len)
            if not _is_short(val_len):
                candidates[val_len][key_str] = val
            else:
                short_candidates[val_len][key_str] = val

    # --- Initial lengths check ---
    # If no lengths found, then there are no array-like values
    if not lengths:
        return _eval_dfs(df_candidates, attr_candidates)

    # If the max is short, then no arrays are long enough
    if not candidates:
        attr_candidates.update(_residual_arrays(candidates, short_candidates))
        return _eval_dfs(df_candidates, attr_candidates)

    # --- Selection mode logic ---
    # - Note: Selection mode convention -
    # Modes ending with "_valid" respect the min_len argument.
    # Modes ending with "_arrays" override min_len to 0 before parsing.
    # Conceptually this changes the available candidates when applying the selection mode.
    if selection_mode == "most_common":
        target_length = int(Counter(lengths).most_common(1)[0][0])
        if target_length not in candidates:
            # if true then target_length is also too short
            attr_candidates.update(_residual_arrays(candidates, short_candidates))
            return _eval_dfs(df_candidates, attr_candidates)
        data = candidates.pop(target_length)
        # Add residuals as attrs
        attr_candidates.update(_residual_arrays(candidates, short_candidates))
    elif "max" in selection_mode:
        target_length = max(lengths)
        min_len = target_length
        data = candidates.pop(target_length)
        # Add residuals as attrs
        attr_candidates.update(_residual_arrays(candidates, short_candidates))
    elif "all" in selection_mode:
        # Combine all candidates that are long enough
        data = {k: v for d in candidates.values() for k, v in d.items()}
        # All short candidates become attrs
        attr_candidates.update(_residual_arrays(short_candidates))
    else:
        raise ValueError(f"Unknown selection_mode: {selection_mode}")

    # --- Handle columns: retain order ---
    # update columns to only include those in data
    columns = [str(c) for c in orig_keys if c in data]
    # make sure any keys in data missing from columns are added
    columns = columns + list(set(data.keys()) - set(columns))

    out_df = pd.DataFrame(data, columns=columns)
    out_df = out_df.loc[((out_df != 0) & ~out_df.isna()).any(axis=1)]

    if out_df.empty or _is_short(len(out_df)):
        return _eval_dfs(df_candidates, attr_candidates)

    num_cols = out_df.select_dtypes(include="number").columns
    out_df[num_cols] = out_df[num_cols].fillna(0)

    res = {"combined_data": out_df}
    res.update(df_candidates)
    return _eval_dfs(res, attr_candidates)


def _cleanup_attrs(attrs, df_id, key=None):
    """
    Cleans up the attributes for a DataFrame.

    Parameters:
    attrs (dict): The attributes to clean up.
    key (str): The parent key where the data is located.
    df (pd.DataFrame): The DataFrame to attach attributes to.

    Returns:
    dict: The cleaned-up attributes.
    """
    # Evaluate attributes safely
    cleaned_attrs = {k: safe_eval(v) for k, v in attrs.items()}

    # Add a UUID to the DataFrame attributes
    cleaned_attrs["py_id"] = id(df_id) if isinstance(df_id, pd.DataFrame) else df_id

    if key is None:
        key = cleaned_attrs.get("py_id", "")

    cleaned_attrs["name"] = str(attrs.get("name", key))
    return cleaned_attrs


def cleanup_dict(
    input_dict: dict,
    flatten_dict: bool = True,
    flatten_arr: bool = True,
    dropna: bool = True,
    **kwargs,
) -> dict:
    """
    Recursively cleans up a dictionary by flattening nested structures, removing NaNs, and dropping duplicates.
    The flatten, dropna, and drop_duplicates steps are optional.

    Parameters:
        input_dict (dict): The dictionary to clean up.
        parent_key (str): Prefix for nested keys.
        parent_dict (dict | None): Dictionary to update (used for recursion).
        flatten (bool): Whether to flatten array-like values.
        dropna (bool): Whether to drop NaN values.
        drop_duplicates (bool): Whether to drop duplicate values.

    Returns:
        dict: The cleaned-up dictionary.
    """
    cleaned_dict = dict(kwargs.pop("parent_dict", {}))
    ignore_ind = kwargs.get("ignore_index", True)
    drop_dict_duplicates = kwargs.get("drop_dict_duplicates", False)
    parent_key = kwargs.pop("parent_key", "")

    ops = ["reshape"]
    if dropna:
        ops.append("drop_na")
    if kwargs.get("drop_duplicates", True):
        ops.append("drop_dupl")

    for key, value in input_dict.items():
        new_key = str(parent_key) + "_" + str(key) if key in cleaned_dict else str(key)
        # Scalar or string values
        if (isinstance(value, str) or not hasattr(value, "__iter__")) and (value or not dropna):
            cleaned_dict[new_key] = value.item() if isinstance(value, np.generic) else value
        # Array-like values
        elif isinstance(value, (list, np.ndarray, tuple, set, pd.Series, pd.DataFrame)):

            if isinstance(value, set):
                value = list(value)

            if flatten_arr:
                array = evaluate_1D_array(value, operations=ops, ignore_index=ignore_ind)
            else:
                array = evaluate_nD_array(value, operations=ops, ignore_index=ignore_ind)

            if array.empty:
                continue
            if isinstance(array, pd.Series) and len(array) == 1:
                value = array.iloc[0]
                cleaned_dict[new_key] = value.item() if isinstance(value, np.generic) else value
            else:
                cleaned_dict[new_key] = array.to_numpy(copy=True).tolist()
        # Nested dicts
        elif isinstance(value, dict):
            res_dict = cleanup_dict(
                value,
                parent_key=new_key,
                parent_dict=cleaned_dict,
                flatten_arr=flatten_arr,
                dropna=dropna,
                **kwargs,
            )
            if not res_dict and dropna:
                continue
            if (
                drop_dict_duplicates
                and isinstance(res_dict, dict)
                and len(res_dict) > 0
                and all(v == list(res_dict.values())[0] for v in res_dict.values())
            ):
                cleaned_dict[new_key] = list(res_dict.values())[0]

            elif flatten_dict:
                cleaned_dict.update(res_dict)
            else:
                cleaned_dict[new_key] = res_dict
    return cleaned_dict


# ARCHIVE
