# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""

import re
import os
from copy import deepcopy
from datetime import datetime as dt

import pandas as pd


from ..data_treatment import (
    sanitize_types,
    convert_from_unix_time,
    modify_sub_dfs,
    insert_inverse_col,
    hz_label,
    drop_common_index_key,
)
# from ..data_treatment.ckt_analysis import (
#     hz_label,
# )
from ..dict_ops import (
    dict_level_ops,
    rename_from_subset,
    rename_from_internal_df,
    flip_dict_levels,
    push_non_dict_items,
    recursive_concat,
    merge_unique_sub_dicts,
    dict_to_df,
)
# from ..string_ops import common_substring
from ..string_ops import find_common_str, safe_eval


def parse_mfia_files(pth):
    """
    Parses the given file path to extract specific components.
    Args:
        pth (Path): The file path to parse.
    Returns:
        list: A list containing extracted components from the file path.
    """
    str0, diffs = find_common_str(
        *[pth.stem, pth.parent.stem], sep="_", retry=False,
    )

    if len(str0) <= 5:
        str0 = pth.stem

    diff1, diff2 = [
        int(match.group(1)) if (match := re.search(r'(\d+)$', d)) else 0
        for d in diffs
    ]

    # Define the regex pattern to match the session format
    pattern = re.compile(r"[\\/]session_(\d{8})_(\d{6})_\d{2}[\\/]")

    # Search for the pattern in the path
    match = pattern.search(str(pth))
    if match:
        d_str, t_str = match.groups()
        sdate = stime = dt.combine(dt.strptime(d_str, '%Y%m%d').date(), dt.strptime(t_str, '%H%M%S').time())
        # sdate = dt.strptime(sdate_str, '%Y%m%d')
        # stime = dt.strptime(stime_str, '%H%M%S')
    else:
        # Get the creation date of the file
        ctime = os.path.getctime(pth)
        sdate = stime = dt.fromtimestamp(ctime)

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
        *[pth.stem, pth.parent.stem], sep="_", retry=False,
    )

    _, sdate, stime, _ = re.split(r"[_\s-]", pth.parent.parent.stem)

    try:
        return [str0, int(sdate), int(stime), int(diff2), int(diff1), pth]
    except ValueError:
        return [str0, sdate, stime, diff2, diff1, pth]



def convert_mfia_data(
    arg,
    columns=None,
    attrs=None,
    rename=True,
    simplify=True,
    sanitize=True,
    flip=True,
    flatten=False,
):
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

    # # simplify dict
    # if simplify:
    #     arg = merge_unique_sub_dicts(arg, ["000", "imps", "demods"])
    #     attrs = merge_unique_sub_dicts(attrs, ["000", "imps", "demods"])

    # # Sanitize Data
    # if sanitize:
    #     arg = sanitize_types(arg)
    #     attrs = sanitize_types(attrs)

    # # Initial conversion
    # res = dict_to_df(arg, columns, attrs)

    # # Rename numeric keys. Must come after results are dicts
    # if rename:
    #     res = rename_from_subset(res)

    # # Flip if desired
    # if flip:
    #     res = flip_dict_levels(res)

    # if flatten:
    #     res = dict_level_ops(res, recursive_concat, level=int(flatten))
    # """
    # --- New target order ---
    # 1. sanitize_types
    # 2. convert to df
    # 3a. rename dict
    # 3b. convert time
    # 4. flip or merge dict?
    # 5. flatten dict
    # 5a. simplify multiindex if not merge in 4
    # 6. modify sub dfs
    # """

    # # Sanitize Data
    # if sanitize:
    #     arg = sanitize_types(arg)
    #     attrs = sanitize_types(attrs)

    # # Initial conversion
    # res0 = dict_to_df(arg, columns, attrs)

    # # Rename numeric keys. Must come after results are dicts
    # if rename:
    #     res0 = rename_from_internal_df(res0) # Should work like rename_from_subset without requiring merge previously

    # if simplify:
    #     res0 = merge_unique_sub_dicts(res0, ["000", "imps", "demods"])
    #     res0 = push_non_dict_items(res0) # may not be necessary

    # # Flip if desired
    # if flip:
    #     res0 = flip_dict_levels(res0)

    # if flatten:
    #     res0 = dict_level_ops(res0, recursive_concat, level=int(flatten))


    """
    --- New target order ---
    1. sanitize_types
    2. convert to df
    3a. rename dict
    3b. convert time
    4. flip or merge dict?
    5. flatten dict
    5a. simplify multiindex if not merge in 4
    6. modify sub dfs
    """

    # # Sanitize Data
    # if sanitize:
    #     arg = sanitize_types(arg)
    #     attrs = sanitize_types(attrs)

    # Initial conversion
    res = dict_to_df(arg, columns, attrs)

    # Rename numeric keys. Must come after results are dicts
    if rename:
        res = rename_from_internal_df(res) # Should work like rename_from_subset without requiring merge previously

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
        convert_mfia_time,
        # (insert_inverse_col, ("imps", "imagz")),
        # (
        #     hz_label,
        #     dict(
        #         kind="exp",
        #         space="",
        #         postfix="",
        #         targ_col=("imps", "frequency"),
        #         test_col=("imps", "imagz"),
        #         new_col=("imps", "flabel"),
        #     ),
        # ),
    )

    return res


# def convert_mfia_time(data):
#     """
#     Converts the time-related attributes in MFIA data to a standardized format.

#     This function processes the time attributes in the provided MFIA data, such as timebase,
#     created timestamp, and system time. It normalizes these times based on a base time and
#     an optional starting time.

#     Parameters:
#     data (pd.DataFrame): The MFIA data containing time-related attributes.

#     Returns:
#     np.ndarray: An array of converted time values.
#     """

#     def time_eq(arr, base=1, t_0=None):
#         if t_0 is None:
#             if isinstance(arr, (int, float)):
#                 t_0 = 0
#             else:
#                 t_0 = min(arr)
#         return (arr - t_0) * base

#     t_base = data.attrs["timebase"] if "timebase" in data.attrs.keys() else 1
#     t_start = (
#         data.attrs["createdtimestamp"]
#         if "createdtimestamp" in data.attrs.keys()
#         else None
#     )
#     t_sys = data.attrs["systemtime"] if "systemtime" in data.attrs.keys() else 0
#     if isinstance(t_sys, str):
#         try:
#             t_sys = min(int(n) for n in re.split(r"\D+", t_sys))
#         except ValueError:
#             val = re.search(r"[0-9]+", "sdfasdf")
#             t_sys = int(val[0]) if val is not None else 0

#     for c in data.columns:
#         if (isinstance(c, str) and "time" in c.lower()) or (
#             isinstance(c, tuple) and any("time" in tc.lower() for tc in c)
#         ):
#             data[c] = convert_from_unix_time(
#                 time_eq(data[c], t_base, t_start).to_numpy(),
#                 t_sys,
#             )
#     data.attrs["createdtime"] = convert_from_unix_time(t_sys)

#     return data

def convert_mfia_time(data):
    """
    Converts the time-related attributes in MFIA data to a standardized format.

    This function processes the time attributes in the provided MFIA data, such as timebase,
    created timestamp, and system time. It normalizes these times based on a base time and
    an optional starting time.

    Parameters:
    data (pd.DataFrame): The MFIA data containing time-related attributes.

    Returns:
    np.ndarray: An array of converted time values.
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
    t_base = data.attrs.get("timebase", 1/(6e7))
    
    if isinstance(t_sys, str):
        try:
            t_sys = min(int(n) for n in re.split(r"\D+", t_sys))
        except ValueError:
            val = re.search(r"[0-9]+", "sdfasdf")
            t_sys = int(val[0]) if val is not None else 0
    elif isinstance(t_sys, (list, tuple)):
        t_sys = min([safe_eval(v) for v in t_sys])
    if isinstance(t_start, (list, tuple)):
        t_start = min([safe_eval(v) for v in t_start])
    if isinstance(t_end, (list, tuple)):
        t_end = min([safe_eval(v) for v in t_end])
    if isinstance(t_base, (list, tuple)):
        t_base = t_base[0]

    data.attrs["createdtime"] = convert_from_unix_time(t_sys)
    
    data.attrs["mintime"] = (
        convert_from_unix_time(time_eq(t_start, t_base, 0), t_sys)
        if t_start is not None
        else data.attrs["createdtime"]
    )

    data.attrs["maxtime"] = (
        convert_from_unix_time(time_eq(t_end, t_base, t_start), t_sys)
        if t_end is not None
        else data.attrs["createdtime"]
    )


    for c in data.columns:
        if (isinstance(c, str) and "time" in c.lower()) or (
            isinstance(c, tuple) and any("time" in tc.lower() for tc in c)
        ):
            start = t_start
            if (
                t_start
                and t_end
                and (
                    min(data[c]) > t_end or not 0.1 < max(data[c]) / t_end < 10
                )
            ):
                start = None
            data[c] = convert_from_unix_time(
                time_eq(data[c], t_base, start).to_numpy(),
                t_sys,
            )
            try:
                data.attrs["mintime"] = min(data.attrs["mintime"], *data[c])
                data.attrs["maxtime"] = max(data.attrs["maxtime"], *data[c])
            except TypeError as e:
                continue
    return data


def convert_mfia_df_for_fit(raw_data):
    """
    Converts raw MFIA data into a DataFrame suitable for fitting.

    This function processes raw MFIA data, ensuring it contains the necessary columns
    ('frequency', 'realz', 'imagz'). It then creates a new DataFrame with columns renamed
    to 'freq', 'real', and 'imag', and sorts the data by frequency. If the imaginary part
    of any column is zero, it converts that column to its real part.

    Parameters:
    raw_data (pd.DataFrame or dict): The raw MFIA data to convert. Must be a DataFrame containing
                                     'frequency', 'realz', and 'imagz' columns.

    Returns:
    pd.DataFrame or None: The converted DataFrame if the necessary columns are present,
                          otherwise None.
    """
    if isinstance(raw_data, pd.DataFrame):
        native_cols = ["frequency", "realz", "imagz"]
        if isinstance(raw_data.columns, pd.MultiIndex):
            if raw_data.columns.nlevels == 2 and "imps" in raw_data.columns.get_level_values(0):
                return convert_mfia_df_for_fit(raw_data["imps"])

            # Identify the level containing native_cols
            level_with_native_cols = None
            for level in range(raw_data.columns.nlevels):
                if all(col in raw_data.columns.get_level_values(level) for col in native_cols):
                    level_with_native_cols = level
                    break

            if level_with_native_cols is None:
                return None

            # Create a list of length nlevels filled with slice(None)
            selectors = [slice(None)] * raw_data.columns.nlevels
            # Replace the appropriate level with native_cols
            selectors[level_with_native_cols] = native_cols

            # Select the desired columns
            raw_data = raw_data.loc[:, tuple(selectors)]
            # Remove levels by setting df.columns to native_cols
            raw_data.columns = native_cols

        if not all(nc in raw_data.columns for nc in native_cols):
            return None
        res = pd.DataFrame()
        try:
            res = pd.DataFrame(
                {
                    "freq": raw_data["frequency"].to_numpy(),
                    "real": raw_data["realz"].to_numpy(),
                    "imag": raw_data["imagz"].to_numpy(),
                }
            ).sort_values("freq", ignore_index=True)

            for col in res.iloc[:, 1:].columns:
                if res[col].to_numpy().imag.sum() == 0:
                    res[col] = res[col].to_numpy().real
        except KeyError:
            return None
        return res
    elif isinstance(
        raw_data, dict
    ):  # and all([isinstance(d, pd.DataFrame) for d in data.values()])
        res = {}
        for k, d in raw_data.items():
            val = convert_mfia_df_for_fit(d)
            if val is not None:
                res[k] = val
        if len(res) == 0:
            return None
        return res
    elif isinstance(raw_data, (list, tuple)):
        res = []
        for d in raw_data:
            val = convert_mfia_df_for_fit(d)
            if val is not None:
                res.append(val)
        if len(res) == 0:
            return None
        return res

    return raw_data


# def hz_label(
#     data,
#     test_arr=None,
#     prec=2,
#     kind="eng",
#     space=" ",
#     postfix="Hz",
#     label_rc=True,
#     targ_col="frequency",
#     test_col="imag",
#     new_col="flabel",
# ):
#     """
#     Generates frequency labels for MFIA data.

#     This function creates a new column in the provided DataFrame or processes a numpy array
#     to generate frequency labels based on the specified parameters. It uses the target column
#     for frequency values and the test column for additional calculations.

#     Parameters:
#     data (pd.DataFrame or np.ndarray): The data to process. If a DataFrame, it should contain
#                                        the target and test columns.
#     test_arr (np.ndarray, optional): An array for additional calculations. If not provided,
#                                      it is computed using a moving average of the test column.
#     prec (int, optional): The precision for the frequency labels. Default is 2.
#     kind (str, optional): The format kind for the labels ('eng' for engineering notation). Default is "eng".
#     space (str, optional): The space between the number and the postfix. Default is " ".
#     postfix (str, optional): The postfix for the frequency labels. Default is "Hz".
#     label_rc (bool, optional): If True, labels are generated in reverse order. Default is True.
#     targ_col (str, optional): The name of the target column for frequency values in the DataFrame. Default is "frequency".
#     test_col (str, optional): The name of the test column for additional calculations in the DataFrame. Default is "imag".
#     new_col (str, optional): The name of the new column to store the generated labels in the DataFrame. Default is "flabel".

#     Returns:
#     pd.DataFrame or np.ndarray: The DataFrame with the new column of frequency labels, or the processed numpy array.
#     """
#     if isinstance(data, pd.DataFrame):
#         targ_col = str_in_list(targ_col, data.columns)
#         test_col = str_in_list(test_col, data.columns)
#         if targ_col not in data.columns or test_col not in data.columns:
#             return data
#         data[new_col] = hz_label(
#             data[targ_col].to_numpy(),
#             test_arr=moving_average(-1 * data[test_col].to_numpy(), 5, True),
#             prec=prec,
#             kind=kind,
#             space=space,
#             postfix=postfix,
#             label_rc=label_rc,
#         )
#         return data
#     # if isinstance(data, pd.Series)
#     base = [float(10 ** (np.floor(np.log10(a)))) if a > 0 else 0 for a in data]
#     base_diff = np.diff(base)

#     res = [np.nan] * len(data)

#     for n, value in enumerate(data):
#         if value == 0:
#             continue
#         elif n == 0 or base_diff[n - 1] != 0:
#             res[n] = str(eng_not(base[n], 0, kind, space)) + postfix
#             if (
#                 label_rc
#                 and isinstance(test_arr, (list, np.ndarray))
#                 and test_arr[n] == max(abs(np.array(test_arr)))
#             ):
#                 res[n] = res[n] + " (RC)"
#         elif (
#             label_rc
#             and isinstance(test_arr, (list, np.ndarray))
#             and test_arr[n] == max(abs(np.array(test_arr)))
#         ):
#             try:
#                 if len(kind) > 2 and "exp" in kind.lower():
#                     res[n] = "RC (f=" + eng_not(data[n], prec, "eng", " ") + "Hz)"
#                 else:
#                     res[n] = "RC (f=" + eng_not(data[n], prec, kind, space) + "Hz)"
#             except TypeError:
#                 return res
#     return res
