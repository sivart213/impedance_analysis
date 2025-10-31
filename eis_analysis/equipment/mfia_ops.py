# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""

import re
from typing import Any
from datetime import datetime as dt
from collections.abc import Callable

import pandas as pd

from ..dict_ops import (
    dict_level_ops,
    flip_dict_levels,
    recursive_concat,
    push_non_dict_items,
    parse_dict_of_datasets,
    rename_from_internal_df,
)
from ..string_ops import safe_eval, find_common_str
from ..data_treatment import (
    ensure_unique,
    modify_sub_dfs,
    sanitize_types,
    drop_common_index_key,
    convert_unix_time_array,
    convert_unix_time_value,
)
from ..system_utilities import get_file_stats
from ..utils.decorators import handle_dicts


def parse_mfia_files(pth):
    """
    Parses the given file path to extract specific components.
    Args:
        pth (Path): The file path to parse.
    Returns:
        list: A list containing extracted components from the file path.
    """
    str0, diffs = find_common_str(
        *[pth.stem, pth.parent.stem],
        sep="_",
        retry=False,
    )

    if len(str0) <= 5:
        if pth.suffix.startswith(".h"):
            str0 = pth.stem
        else:
            str0 = pth.stem.replace("dev6037_imps_", "")
            str0 = re.sub(r"0+", "0", str0)
            str0 = pth.parent.stem + "_" + str0

    diff1, diff2 = [
        int(match.group(1)) if (match := re.search(r"(\d+)$", d)) else 0 for d in diffs
    ]

    # Define the regex pattern to match the session format
    pattern = re.compile(r"[\\/]session_(\d{8})_(\d{6})_\d{2}[\\/]")

    # Search for the pattern in the path
    match = pattern.search(str(pth))
    if match:
        d_str, t_str = match.groups()
        sdate = stime = dt.combine(
            dt.strptime(d_str, "%Y%m%d").date(), dt.strptime(t_str, "%H%M%S").time()
        )
        # sdate = dt.strptime(sdate_str, '%Y%m%d')
        # stime = dt.strptime(stime_str, '%H%M%S')
    else:
        # Get the creation date of the file
        # ctime = os.path.getctime(pth)
        stats = get_file_stats(pth)
        ctime = stats.get("st_birthtime")
        if ctime is None:
            ctime = stats.get("st_ctime")
        sdate = stime = ctime

    return [str0, sdate, stime, diff2, diff1, pth]


def parse_labone_hdf5(pth):
    """
    Parses the given file path to extract specific components.
    Args:
        pth (Path): The file path to parse.
    Returns:
        list: A list containing extracted components from the file path.
    """
    # str0, [diff1, diff2] = common_substring(
    #     [pth.stem, pth.parent.stem], sep="_"
    # )
    str0, [diff1, diff2] = find_common_str(
        *[pth.stem, pth.parent.stem],
        sep="_",
        retry=False,
    )

    _, sdate, stime, _ = re.split(r"[_\s-]", pth.parent.parent.stem)

    try:
        return [str0, int(sdate), int(stime), int(diff2), int(diff1), pth]
    except ValueError:
        return [str0, sdate, stime, diff2, diff1, pth]


def convert_mfia_data(
    arg: tuple[dict, dict] | dict,
    attrs: dict | None = None,
    rename: bool = True,
    simplify: bool = True,
    sanitize: bool = True,
    flip: bool = True,
    flatten: bool | int = False,
    transpose_check: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Converts MFIA data into a structured format, applying various transformations.

    This function processes MFIA data, which can be provided as a tuple or other data structure.
    It applies a series of transformations such as renaming, simplifying, sanitizing, flipping,
    and flattening the data based on the provided parameters.

    Parameters:
    arg (tuple or other): The MFIA data to convert. If a tuple, it should contain the data and attributes.
    columns (dict or list, optional): Specifies which keys to include as columns in the resulting DataFrame.
    attrs (dict, optional): Additional attributes to attach to the resulting DataFrame.
    rename (bool, optional): If True, renames columns based on predefined rules. Default is True.
    simplify (bool, optional): If True, simplifies the data structure. Default is True.
    sanitize (bool, optional): If True, sanitizes the data by removing or correcting invalid entries. Default is True.
    flip (bool, optional): If True, flips the data orientation. Default is True.
    flatten (bool, optional): If True, flattens nested structures into a single level. Default is False.

    Returns:
    pd.DataFrame or dict: The resulting DataFrame or dictionary after applying the transformations.
    """
    # collection of functions for data conversion
    # Sanitize imports
    if isinstance(arg, tuple):
        if len(arg) == 2:
            attrs = arg[1]
            arg = arg[0]
        else:
            return

    # breakpoint()
    if attrs is None:
        attrs = {}
    attrs = push_non_dict_items(attrs)

    if sanitize:
        arg = sanitize_types(arg)
        attrs = sanitize_types(attrs)

    # Initial conversion
    res: dict = parse_dict_of_datasets(arg, attrs, kwargs.get("min_len", 3))  # type: ignore

    if transpose_check:
        res = handle_transposed_data(
            res, column_name="fieldname", min_len=kwargs.get("min_len", 3)
        )

    # Rename numeric keys. Must come after results are dicts
    if rename:
        res = rename_from_internal_df(res)

    # Flip if desired
    if flip:
        res = flip_dict_levels(res)

    if flatten:
        res = dict_level_ops(res, recursive_concat, level=int(flatten))
        # hopefully cleaning the multi-index will work like merge
        if simplify:
            res = drop_common_index_key(res, ["imps", "demods"])

    # breakpoint()
    modify_sub_dfs(
        res,
        check_mfia_data_attrs,
        convert_mfia_time,
    )

    return res


def check_mfia_data_attrs(data):
    """
    Ensures the time-related attributes in the MFIA data are properly prepared.

    This function checks for the presence of expected time-related keys in the `attrs` dictionary.
    If a key is missing, it looks for variations with underscores or relevant `st_<name>` attributes.
    If none are found, it assigns default values. It also handles type conversion for these attributes.

    Parameters:
    - data (dataframe): The attributes dictionary to check and update.

    Returns:
    - dataframe: The updated attributes dictionary.
    """

    attrs = data.attrs
    # Define the expected keys and their default values
    expected_keys = {
        "systemtime": 0,
        "createdtimestamp": None,
        "changedtimestamp": None,
        "timebase": 1 / (6e7),
    }
    available_timestamps = []
    flat_keys = []
    for k, v in attrs.items():
        if isinstance(v, dt):
            available_timestamps.append(k)
        # use re to sub all _ and \s with ""
        flat_keys.append(re.sub(r"[_\s]", "", k))

    # sort available timestamps which are dt objects so that oldest is first
    available_timestamps = sorted(available_timestamps)

    def attrs_is_arraylike(value: list | tuple | Any, modifier: int | Callable = 0):
        try:
            if isinstance(value, (list, tuple)):
                if callable(modifier):
                    return modifier([safe_eval(v) for v in value])
                elif isinstance(modifier, (int, float)) and int(modifier) < len(value):
                    return safe_eval(value[int(modifier)])
                else:
                    return value
            else:
                return value
        except (TypeError, ValueError, AttributeError):
            return value

    n_ts = 0
    # Check for each expected key
    for key, default_value in expected_keys.items():
        if key == "timebase":
            attrs[key] = attrs_is_arraylike(attrs.get(key, default_value), modifier=0)
            continue
        if key not in attrs:
            if key in flat_keys:
                # If the key is found in the flattened keys, assign its value
                new_value = attrs.pop(list(attrs.keys())[flat_keys.index(key)])
            # Look for relevant `st_<name>` attributes
            elif available_timestamps:
                if n_ts < len(available_timestamps):
                    new_value = attrs.get(available_timestamps[n_ts])
                    n_ts += 1
                else:
                    new_value = attrs.get(available_timestamps[-1])
            # Assign the default value if no alternative is found
            else:
                new_value = default_value
        else:
            new_value = attrs[key]
        if key == "systemtime" and isinstance(new_value, str):
            try:
                new_value = min(int(n) for n in re.split(r"\D+", new_value))
            except ValueError:
                val = re.search(r"[0-9]+", new_value)
                new_value = int(val[0]) if val is not None else 0
        attrs[key] = attrs_is_arraylike(new_value, modifier=min)

        if isinstance(attrs[key], dt) and attrs[key].year < 2000:
            # If the value is a datetime object and the year is less than 2000, convert it to a timestamp
            attrs[key] = attrs[key].timestamp()

    data.attrs = attrs
    return data


def convert_mfia_time(data):
    """
    Converts the time-related attributes in MFIA data to a standardized format.

    This function processes the time attributes in the provided MFIA data, such as timebase,
    created timestamp, and system time. It normalizes these times based on a base time and
    an optional starting time.

    Parameters:
    - data (pd.DataFrame): The MFIA data containing time-related attributes.

    Returns:
    - pd.DataFrame: The updated DataFrame with converted time values.
    """

    def time_eq(arr, base=1, t_0=None):
        if t_0 is None:
            if isinstance(arr, (int, float)):
                t_0 = 0
            else:
                t_0 = min(arr)
        return (arr - t_0) * base

    t_sys = data.attrs.get("systemtime", 0)
    t_start = data.attrs.get("createdtimestamp", None)
    t_end = data.attrs.get("changedtimestamp", None)
    t_base = data.attrs.get("timebase", 1 / (6e7))

    if not isinstance(t_sys, (dt)):
        t_sys = convert_unix_time_value(t_sys)

    data.attrs["createdtime"] = t_sys

    if isinstance(t_start, dt):
        data.attrs["mintime"] = t_start
    else:
        data.attrs["mintime"] = (
            convert_unix_time_value(time_eq(t_start, t_base, 0), t_sys)
            if t_start is not None
            else data.attrs["createdtime"]
        )

    if isinstance(t_end, dt):
        data.attrs["maxtime"] = t_end
    else:

        data.attrs["maxtime"] = (
            convert_unix_time_value(time_eq(t_end, t_base, t_start), t_sys)
            if t_end is not None
            else data.attrs["createdtime"]
        )

    for c in data.columns:
        if (isinstance(c, str) and "time" in c.lower()) or (
            isinstance(c, tuple) and any("time" in tc.lower() for tc in c)
        ):
            if isinstance(data[c], dt):
                # If the column is already a datetime object, skip conversion
                continue
            start = t_start
            if isinstance(t_start, dt) or (
                t_start and t_end and (min(data[c]) > t_end or not 0.1 < max(data[c]) / t_end < 10)
            ):
                start = None
            data[c] = convert_unix_time_array(
                time_eq(data[c], t_base, start).to_numpy(copy=True),
                t_sys,
            )
            try:
                data.attrs["mintime"] = min(data.attrs["mintime"], *data[c])
                data.attrs["maxtime"] = max(data.attrs["maxtime"], *data[c])
            except TypeError:
                continue
    return data


@handle_dicts
def handle_transposed_data(
    df: pd.DataFrame,
    column_name: str | None = None,
    min_len: int = 3,
) -> pd.DataFrame:
    """
    Transposes a DataFrame and uses one column as the new column names.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - column_name (str, optional): The column name to use as the new headers.

    Returns:
    - pd.DataFrame: The modified DataFrame with transposed data and updated headers.
    """
    if column_name is None or column_name not in df.columns:
        # Automatically detect the last column with all string values
        for col in reversed(df.columns):
            if df[col].apply(lambda x: isinstance(x, str)).all():
                column_name = col
                break
        else:
            # If no such column is found, return the original DataFrame
            return df

    col_index = int(df.columns.get_loc(column_name))  # type: ignore
    # Ensure the column_name exists in the DataFrame
    if column_name not in df.columns or len(df.columns) - col_index < min_len:
        return df

    # Evaluate attrs for columns to the left of column_name
    prior_columns = df.iloc[:, :col_index]
    skip_columns = []  # Track columns with len(unique_values) == 1
    for col in prior_columns.columns:
        unique_values = prior_columns[col].unique()
        if len(unique_values) == 1:
            # If all values are the same, store the single value and skip this column
            skip_columns.append(col)
            try:
                df.attrs[col] = unique_values[0].item()
            except (TypeError, AttributeError):
                df.attrs[col] = unique_values[0]
        else:
            # Otherwise, store the list of values
            df.attrs[col] = prior_columns[col].tolist()

    # Prepare a DataFrame with prior_columns + primary_column, excluding skip_columns
    unique_check_df = pd.concat(
        [prior_columns.drop(columns=skip_columns), df[[column_name]]], axis=1
    )

    # Use ensure_unique to ensure uniqueness in the primary column
    df[column_name] = ensure_unique(
        unique_check_df,
        primary_column=column_name,
        behavior="check_other_columns",
        sep="_",
    )

    # Transpose the DataFrame starting from the column_name
    t_df = df.iloc[:, col_index:].set_index(column_name).T

    # Reset the index to make it a clean DataFrame
    t_df = t_df.reset_index(drop=True)

    # Drop rows where all values are NaN or 0
    t_df = t_df[~((t_df.isna() | (t_df == 0)).all(axis=1))]

    if len(t_df) < min_len:
        return df

    return t_df


# ARCHIVE
