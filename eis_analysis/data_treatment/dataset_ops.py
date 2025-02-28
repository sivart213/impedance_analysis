# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""

import re
from difflib import get_close_matches
from collections import Counter

import numpy as np
import pandas as pd


from ..utils.decorators import handle_dicts, handle_subdicts
from ..string_ops import eng_not  # str_in_list


def most_frequent(arg):
    """
    Finds the most frequent element in an array.

    This function takes an array-like input and returns the most frequently occurring
    element. If there are multiple elements with the same highest frequency, it returns
    the first one encountered.

    Parameters:
    arg (array-like): The input array to analyze.

    Returns:
    int: The most frequent element in the array.
    """
    unique, counts = np.unique(arg, return_counts=True)
    index = np.argmax(counts)
    return int(unique[index])


# def get_valid_keys(columns, target, return_unmatched=False):
#     """
#     Returns the correct column names for the DataFrame based on the target.
#     Args:
#         columns (pd.Index or pd.MultiIndex): The DataFrame columns.
#         target (str, list of str, or list of tuple of str): The target column names to match.
#     Returns:
#         list: A list of matched column names.
#     """
#     if isinstance(target, str):
#         target = [target]

#     # Convert columns to list if not already a list
#     if isinstance(columns, (pd.Index, pd.MultiIndex)):
#         columns = columns.tolist()

#     if isinstance(columns[0], (tuple, list)):
#         column_lists = [[".".join(map(str, col)) for col in columns]]
#         for level in range(len(columns[0])):
#             column_lists.append([str(col[level]) for col in columns])
#     else:
#         column_lists = [columns]

#     matched_columns = []
#     unmatched_columns = []
#     for t in target:
#         if isinstance(t, tuple):
#             t = ".".join(t)

#         if t.startswith("(") and t.endswith(")"):
#             t = re.sub(r'\s*,\s*', '.', t[1:-1])

#         match_found = False
#         cutoff = 0.8
#         while cutoff >= 0.1 and not match_found:
#             for column_list in column_lists:
#                 if matches := get_close_matches(t, column_list, cutoff=cutoff):
#                     match_index = column_list.index(matches[0])
#                     matched_columns.append(columns[match_index])
#                     match_found = True
#                     break

#             cutoff -= 0.1
#         if not match_found:
#             print(f"No match found for {t}")
#             unmatched_columns.append(t)

#     if return_unmatched:
#         return matched_columns, unmatched_columns
#     return matched_columns


def get_valid_keys(
    columns, target, return_unmatched=False, prevent_duplicates=True, slice_multi=False
):
    """
    Returns the correct column names for the DataFrame based on the target.
    Args:
        columns (pd.Index or pd.MultiIndex): The DataFrame columns.
        target (str, list of str, or list of tuple of str): The target column names to match.
    Returns:
        list: A list of matched column names.
    """
    if isinstance(target, str):
        target = [target]

    # Convert columns to list if not already a list
    if isinstance(columns, (pd.Index, pd.MultiIndex)):
        columns = columns.tolist()

    is_multi = False
    if isinstance(columns[0], (tuple, list)):
        is_multi = True
        column_lists = [[".".join(map(str, col)) for col in columns]]
        for level in range(len(columns[0])):
            column_lists.append([str(col[level]) for col in columns])
        # column_lists.append([str(col) for col in columns])
    else:
        column_lists = [columns]

    matched_columns = []
    unmatched_columns = []
    for t in target:
        if isinstance(t, tuple):
            t = ".".join(t)

        if t.startswith("(") and t.endswith(")"):
            t = re.sub(r"\s*,\s*", ".", t[1:-1])

        match_found = False
        cutoff = 0.8
        cutoff_min = 0.1
        while cutoff >= cutoff_min and not match_found:
            for n, column_list in enumerate(column_lists):
                if matches := get_close_matches(t, column_list, cutoff=cutoff):
                    match_index = column_list.index(matches[0])
                    res = columns[match_index]
                    if is_multi and slice_multi and n >= 1:
                        # if n >= 1 then it is a multi-index column and I want to "slice" it safely if slice_multi is True
                        for col in columns[match_index:]:
                            if col[n - 1] == res[n - 1] and col not in matched_columns:
                                matched_columns.append(col)
                            match_found = True
                            break

                    elif (is_multi and slice_multi) or prevent_duplicates:
                        for match in matches:
                            if (
                                m := columns[column_list.index(match)]
                            ) not in matched_columns:
                                matched_columns.append(m)
                                match_found = True
                                break
                        else:
                            cutoff_min = max(0.1, cutoff / 2)
                    else:
                        matched_columns.append(res)
                        match_found = True
                    break

            cutoff -= 0.1
        if not match_found:
            print(f"No match found for {t}")
            unmatched_columns.append(t)

    if return_unmatched:
        return matched_columns, unmatched_columns
    return matched_columns


def insert_inverse_col(df, name):
    """
    Inserts a new column in the DataFrame with the inverse of the specified column.

    This function searches for a column in the DataFrame that matches the given name
    (or a close match if the exact name is not found). It then creates a new column
    with the inverse values of the specified column and inserts it immediately after
    the original column.

    Parameters:
    df (pd.DataFrame): The DataFrame to modify.
    name (str): The name of the column to invert.

    Returns:
    pd.DataFrame: The modified DataFrame with the new inverse column inserted.
    """
    # name = str_in_list(name, df.columns)
    # if name in df.columns:
    names = get_close_matches(name, list(df.columns))
    if names:
        name = names[0]
        if isinstance(name, tuple):
            new_name = tuple(
                [
                    name[n] if n < len(name) - 1 else "inv_" + name[n]
                    for n in range(len(name))
                ]
            )
        else:
            new_name = "inv_" + name
        df.insert(df.columns.get_loc(name) + 1, new_name, -1 * df[name])
    return df


def modify_sub_dfs(data, *functions):
    """
    Applies a series of functions to a DataFrame or nested DataFrames within a dictionary.

    This function takes a DataFrame or a dictionary containing DataFrames and applies
    a series of functions to each DataFrame. Each function can be provided as a callable
    or as a tuple containing a callable and its arguments. The function modifies the
    DataFrame in place and returns the modified DataFrame.

    Parameters:
    data (pd.DataFrame or dict): The DataFrame or dictionary of DataFrames to modify.
    *functions (callable or tuple): A series of functions or tuples of functions and their arguments
                                to apply to the DataFrame(s).

    Returns:
    pd.DataFrame or dict: The modified DataFrame or dictionary of DataFrames.
    """
    if isinstance(data, pd.DataFrame):
        for f in functions:
            if isinstance(f, (tuple, list)):
                if len(f) == 1:
                    res = f[0](data)
                elif len(f) == 2:
                    res = (
                        f[0](data, **f[1])
                        if isinstance(f[1], dict)
                        else f[0](data, f[1])
                    )
                else:
                    res = f[0](data, *f[1:])
            else:
                res = f(data)
            if res is not None:
                data = res
        return data
    elif isinstance(data, dict):
        return {k: modify_sub_dfs(d, *functions) for k, d in data.items()}
    elif isinstance(data, (list, tuple)):
        return [modify_sub_dfs(d, *functions) for d in data]

    return data


# def remove_duplicate_datasets(data_dict, min_rows=1, verbose=False):
#     """
#     Removes duplicate datasets from a dictionary of datasets and filters based on minimum number of rows.

#     Parameters:
#     - data_dict (dict): Dictionary containing datasets.
#       Format: {filename: {sheet_name: pd.DataFrame}}
#     - min_rows (int): Minimum number of rows required to keep a dataset. Default is 1.
#     - verbose (bool): If True, prints the file and sheet name of rejected datasets. Default is False.

#     Returns:
#     - dict: Filtered dictionary with duplicates removed.
#     """
#     unique_datasets = {}
#     seen_createdtimes = {}

#     for filename, sheets in data_dict.items():
#         for sheet_name, df in sheets.items():
#             if len(df) <= min_rows:
#                 if verbose:
#                     print(
#                         f"Rejected dataset due to insufficient rows: {filename} - {sheet_name} (rows: {len(df)})"
#                     )
#                 continue

#             createdtime = df.attrs.get("createdtime")
#             if createdtime:
#                 if createdtime in seen_createdtimes:
#                     existing_filename, existing_sheet_name = seen_createdtimes[
#                         createdtime
#                     ]
#                     existing_df = unique_datasets[existing_filename][
#                         existing_sheet_name
#                     ]

#                     # Compare completeness (number of non-null values)
#                     if df.notnull().sum().sum() > existing_df.notnull().sum().sum():
#                         # Replace with the more complete dataset
#                         unique_datasets[existing_filename].pop(existing_sheet_name)
#                         if not unique_datasets[existing_filename]:
#                             unique_datasets.pop(existing_filename)
#                         unique_datasets.setdefault(filename, {})[sheet_name] = df
#                         seen_createdtimes[createdtime] = (filename, sheet_name)
#                         if verbose:
#                             print(
#                                 f"Replaced dataset: {existing_filename} - {existing_sheet_name} with {filename} - {sheet_name} due to more completeness"
#                             )
#                     elif (
#                         df.notnull().sum().sum() == existing_df.notnull().sum().sum()
#                         and "loaded" not in sheet_name
#                         and df.attrs.get("createdtimestamp", 0)
#                         != df.attrs.get("changedtimestamp", 1)
#                     ):
#                         # If completeness is the same, keep the dataset with the longer name
#                         if (
#                             existing_sheet_name.startswith(sheet_name)
#                             or len(filename) > len(existing_filename)
#                             or (len(filename) == len(existing_filename)
#                                 and len(sheet_name) > len(existing_sheet_name))
#                         ):
#                             unique_datasets[existing_filename].pop(existing_sheet_name)
#                             if not unique_datasets[existing_filename]:
#                                 unique_datasets.pop(existing_filename)
#                             unique_datasets.setdefault(filename, {})[sheet_name] = df
#                             seen_createdtimes[createdtime] = (
#                                 filename,
#                                 sheet_name,
#                             )
#                             if verbose:
#                                 print(
#                                     f"Replaced dataset: {existing_filename} - {existing_sheet_name} with {filename} - {sheet_name} due to longer name"
#                                 )
#                     else:
#                         if verbose:
#                             print(
#                                 f"Rejected dataset due to duplicate createdtime: {filename} - {sheet_name} (createdtime: {createdtime})"
#                             )
#                 else:
#                     unique_datasets.setdefault(filename, {})[sheet_name] = df
#                     seen_createdtimes[createdtime] = (filename, sheet_name)
#             else:
#                 # If no createdtime attribute, keep the dataset
#                 unique_datasets.setdefault(filename, {})[sheet_name] = df

#     return unique_datasets


# def find_duplicate_datasets(data_dict):
#     """
#     Finds duplicate datasets and returns a dictionary with the necessary information
#     to select all duplicate sets.

#     Parameters:
#     - data_dict (dict): Dictionary containing datasets.
#       Format: {filename: {sheet_name: pd.DataFrame}}

#     Returns:
#     - dict: Dictionary with createdtime as keys and lists of tuples (filename, sheet_name) as values.
#     """
#     duplicates = {}
#     seen_createdtimes = {}

#     for filename, sheets in data_dict.items():
#         for sheet_name, df in sheets.items():
#             createdtime = df.attrs.get("createdtime")
#             if createdtime:
#                 if createdtime in seen_createdtimes:
#                     if createdtime not in duplicates:
#                         duplicates[createdtime] = [seen_createdtimes[createdtime]]
#                     duplicates[createdtime].append((filename, sheet_name))
#                 else:
#                     seen_createdtimes[createdtime] = (filename, sheet_name)

#     return duplicates


def remove_duplicate_datasets(data_dict, min_rows=1, duplicates=None):
    """
    Removes duplicate datasets from a dictionary of datasets and filters based on minimum number of rows.

    Parameters:
    - data_dict (dict): Dictionary containing datasets.
      Format: {filename: {sheet_name: pd.DataFrame}}
    - min_rows (int): Minimum number of rows required to keep a dataset. Default is 1.
    - verbose (bool): If True, prints the file and sheet name of rejected datasets. Default is False.

    Returns:
    - dict: Filtered dictionary with duplicates removed.
    """
    if duplicates is None:
        duplicates = find_duplicate_datasets(data_dict)

    for key, val in data_dict.items():
        if isinstance(val, dict):
            sub_dict = remove_duplicate_datasets(val)
            duplicates.update(
                {k: duplicates.get(k, []) + v for k, v in sub_dict.items()}
            )
        elif isinstance(val, pd.DataFrame):
            if createdtime := val.attrs.get("createdtime"):
                duplicates[createdtime] = duplicates.get(createdtime, []) + [key]

    # return duplicates

    unique_datasets = {}
    seen_createdtimes = {}

    for filename, sheets in data_dict.items():
        for sheet_name, df in sheets.items():
            if len(df) <= min_rows:
                continue

            createdtime = df.attrs.get("createdtime")
            if createdtime:
                if createdtime in seen_createdtimes:
                    existing_filename, existing_sheet_name = seen_createdtimes[
                        createdtime
                    ]
                    existing_df = unique_datasets[existing_filename][
                        existing_sheet_name
                    ]

                    # Compare completeness (number of non-null values)
                    if df.notnull().sum().sum() > existing_df.notnull().sum().sum():
                        # Replace with the more complete dataset
                        unique_datasets[existing_filename].pop(existing_sheet_name)
                        if not unique_datasets[existing_filename]:
                            unique_datasets.pop(existing_filename)
                        unique_datasets.setdefault(filename, {})[sheet_name] = df
                        seen_createdtimes[createdtime] = (filename, sheet_name)
                    elif (
                        df.notnull().sum().sum() == existing_df.notnull().sum().sum()
                        and "loaded" not in sheet_name
                        and df.attrs.get("createdtimestamp", 0)
                        != df.attrs.get("changedtimestamp", 1)
                    ):
                        # If completeness is the same, keep the dataset with the longer name
                        if (
                            existing_sheet_name.startswith(sheet_name)
                            or len(filename) > len(existing_filename)
                            or (
                                len(filename) == len(existing_filename)
                                and len(sheet_name) > len(existing_sheet_name)
                            )
                        ):
                            unique_datasets[existing_filename].pop(existing_sheet_name)
                            if not unique_datasets[existing_filename]:
                                unique_datasets.pop(existing_filename)
                            unique_datasets.setdefault(filename, {})[sheet_name] = df
                            seen_createdtimes[createdtime] = (
                                filename,
                                sheet_name,
                            )
                else:
                    unique_datasets.setdefault(filename, {})[sheet_name] = df
                    seen_createdtimes[createdtime] = (filename, sheet_name)
            else:
                # If no createdtime attribute, keep the dataset
                unique_datasets.setdefault(filename, {})[sheet_name] = df

    return unique_datasets


def find_duplicate_datasets(data_dict):
    """
    Finds duplicate datasets and returns a dictionary with the necessary information
    to select all duplicate sets.

    Parameters:
    - data_dict (dict): Dictionary containing datasets.
      Format: {filename: {sheet_name: pd.DataFrame}}

    Returns:
    - dict: Dictionary with createdtime as keys and lists of tuples (filename, sheet_name) as values.
    """
    duplicates = {}

    def check_duplicates(old, new):
        if old is None:
            return new
        if (
            "loaded" not in new.get("name", "")
            and new.get("time_delta", 0) > 1
            and (
                new.get("name", "") >= old.get("name", "")
                or "loaded" in old.get("name", "")
            )
        ):
            new["duplicates"] = old["duplicates"] + new["duplicates"]
            new["py_ids"] = old["py_ids"] + new["py_ids"]
            return new
        old["duplicates"] = old["duplicates"] + new["duplicates"]
        old["py_ids"] = old["py_ids"] + new["py_ids"]
        return old

    for key, val in data_dict.items():
        if isinstance(val, dict):
            sub_dict = find_duplicate_datasets(val)
            for k, v in sub_dict.items():
                duplicates[k] = check_duplicates(duplicates.get(k), v)
        elif isinstance(val, pd.DataFrame):
            if createdtime := val.attrs.get("createdtime"):
                try:
                    time_delta = int(val.attrs.get("changedtimestamp", 1)
                    - val.attrs.get("createdtimestamp", 1))
                except TypeError:
                    time_delta = 0
                duplicates[createdtime] = check_duplicates(
                    duplicates.get(createdtime),
                    {
                        "data": val,
                        "name": key,
                        "py_id": val.attrs.get("py_id", id(val)),
                        "duplicates": [key],
                        "py_ids": [val.attrs.get("py_id", id(val))],
                        "time_delta": time_delta,
                    },
                )
    return duplicates

@handle_dicts
def simplify_multi_index(df, keep_keys=None, allow_merge=False, sep="."):
    """Perform simplification of multiindex columns in a DataFrame."""
    if not isinstance(df, pd.DataFrame):
        return df
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(drop=True)
    
    if isinstance(df.columns, pd.MultiIndex):
        df = simplify_columns(df)
    df = drop_common_index_key(df, keep_keys, allow_merge)
    df = flatten_multiindex_columns(df, sep)
    return df

@handle_dicts
def flatten_multiindex_columns(df, sep="."):
    """
    Flatten MultiIndex columns by merging them with a specified separator.
    """
    if not isinstance(df, pd.DataFrame):
        return df
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    # Flatten the MultiIndex columns
    df.columns = [sep.join(map(str, col)).strip() for col in df.columns.values]

    return df

# @handle_dicts
# def drop_common_index_key(df,  keep_keys=None, allow_merge=False):
#     """Drop levels with all unique names or single unique values."""
#     if not isinstance(df, pd.DataFrame):
#         return df
#     if not isinstance(df.columns, pd.MultiIndex):
#         return df
#     keep_keys = keep_keys or []
#     drop_levels = []
#     for level in range(df.columns.nlevels):
#         level_vals = df.columns.get_level_values(level)
#         if level_vals.is_unique and not keep_keys:
#             df.columns = level_vals
#             break
#         elif df.columns.get_level_values(level).nunique() == 1 and level_vals[0] not in keep_keys:
#             drop_levels.append(level)
#         elif allow_merge and level > 0 and all(val.isnumeric() for val in level_vals):
#             merged_level = [
#                 f"{prev}_{curr}" for prev, curr in zip(df.columns.get_level_values(level - 1), level_vals)
#             ]
#             df.columns = df.columns.set_levels(merged_level, level=level - 1)
#             drop_levels.append(level)

#     if isinstance(df.columns, pd.MultiIndex) and drop_levels != []:
#         drop_levels.sort(reverse=True)
#         for level in drop_levels:
#             df.columns = df.columns.droplevel(level)
#     return df

@handle_dicts
def drop_common_index_key(df, keep_keys=None, allow_merge=False):
    """Drop levels with all unique names or single unique values."""
    if not isinstance(df, pd.DataFrame):
        return df
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    keep_keys = keep_keys or []
    new_columns = []

    # Iterate through each level
    for level in range(df.columns.nlevels):
        level_vals = df.columns.get_level_values(level)
        if level_vals.is_unique and not keep_keys:
            new_columns = level_vals.to_list()
            break
        elif df.columns.get_level_values(level).nunique() == 1 and level_vals[0] not in keep_keys:
            continue
        elif allow_merge and level > 0 and all(val.isnumeric() for val in level_vals):
            merged_level = [
                f"{prev}_{curr}" for prev, curr in zip(df.columns.get_level_values(level - 1), level_vals)
            ]
            if new_columns:
                new_columns.pop()
            new_columns.append(merged_level)

        else:
            new_columns.append(level_vals.to_list())

    if len(new_columns) == len(df.columns):
        df.columns = new_columns
    else:
        df.columns = pd.MultiIndex.from_arrays(new_columns)

    return df

@handle_dicts
def simplify_columns(df):
    """
    Simplify columns by finding duplicated data and renaming columns in the multi-index.

    Parameters:
    df (pd.DataFrame): The DataFrame to simplify.

    Returns:
    pd.DataFrame: The simplified DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        return df

    df = df.copy(deep=True)

    # Identify duplicated columns
    unchecked_columns = list(df.columns)
    duplicated_columns = {}
    for col in df.columns:
        duplicated = []
        if col in unchecked_columns:
            unchecked_columns.remove(col)
        for other_col in unchecked_columns:
            if df[col].equals(df[other_col]):
                duplicated.append(other_col)
        if duplicated:
            for dup in duplicated:
                unchecked_columns.remove(dup)
            duplicated.insert(0, col)
            duplicated_columns[col] = duplicated

    for col, duplicates in duplicated_columns.items():
        if isinstance(df.columns, pd.MultiIndex):
            col_df = pd.DataFrame(columns=df.columns)
            dup_df =  pd.DataFrame(columns=df[duplicates].columns)
            new_col = []
            # iterate the levels
            for _ in range(len(col)):
                occurrences = Counter(list(col_df.columns.get_level_values(0)))
                level_val = ""
                for dup in dup_df.columns.get_level_values(0):
                    if not level_val:
                        level_val = dup
                    elif occurrences[dup] > occurrences[level_val]:
                        level_val = dup

                new_col.append(level_val)
                col_df = col_df[level_val]
                dup_df = dup_df[level_val]
            new_col = tuple(new_col)
        else:
            new_col = col

        if new_col not in df.columns:
            df.rename(columns={col: new_col}, inplace=True)
        if new_col in duplicates:
            duplicates.remove(new_col)
        df.drop(columns=duplicates, inplace=True)

    return df

# @handle_dicts
# def simplify_columns2(df, preferred=""):
#     """
#     Simplify columns by finding duplicated data and renaming columns in the multi-index.

#     Parameters:
#     df (pd.DataFrame): The DataFrame to simplify.

#     Returns:
#     pd.DataFrame: The simplified DataFrame.
#     """
#     if not isinstance(df, pd.DataFrame):
#         return df
#     if isinstance(preferred, str):
#         preferred = [preferred]
#     df = df.copy(deep=True)

#     # Identify duplicated columns
#     unchecked_columns = list(df.columns)
#     duplicated_columns = {}
#     for col in df.columns:
#         duplicated = []
#         if col in unchecked_columns:
#             unchecked_columns.remove(col)
#         for other_col in unchecked_columns:
#             if df[col].equals(df[other_col]):
#                 duplicated.append(other_col)
#         if duplicated:
#             for dup in duplicated:
#                 unchecked_columns.remove(dup)
#             duplicated.insert(0, col)
#         duplicated_columns[col] = duplicated
    
#     new_df = pd.DataFrame()
#     for col, duplicates in duplicated_columns.items():
#         if isinstance(df.columns, pd.MultiIndex):
#             col_df = pd.DataFrame(columns=df.columns)
#             dup_df =  pd.DataFrame(columns=df[duplicates].columns)
#             new_col = []
#             # iterate the levels
#             for _ in range(len(col)):
#                 occurrences = Counter(list(col_df.columns.get_level_values(0)))
#                 level_val = ""
#                 for dup in dup_df.columns.get_level_values(0):
#                     if not level_val:
#                         level_val = dup
#                     elif occurrences[dup] > occurrences[level_val]:
#                         level_val = dup

#                 new_col.append(level_val)
#                 col_df = col_df[level_val]
#                 dup_df = dup_df[level_val]
#             new_col = tuple(new_col)
#         else:
#             new_col = col

#         if new_col not in df.columns:
#             df.rename(columns={col: new_col}, inplace=True)
#         if new_col in duplicates:
#             duplicates.remove(new_col)
#         df.drop(columns=duplicates, inplace=True)

#     return df

                        # duplicated_columns[col].append(col)
    # duplicated_columns is a dictionary with the 1st duplicated columns as keys and the list of duplicates as values

    # Flatten the list of all strings and their occurrences
    # if isinstance(df.columns, pd.MultiIndex):
    #     flat_list = [item for sublist in df.columns for item in sublist]
    #     occurrences = Counter(flat_list)
    #     # occurrences = [Counter(sublist) for sublist in df.columns]
    #     # occurrences = [Counter(df.columns.get_level_values(i)) for i in range(df.columns.nlevels)]
    # else:
    #     # flat_list = list(df.columns)
    #     occurrences = Counter(list(df.columns))
    # occurrences = Counter(flat_list)
    # Iterate through duplicated columns
            # new_col = []
            # for level in range(len(col)):
            #     level_vals = [dup[level] for dup in duplicates]
            #     most_common_val = max(level_vals, key=lambda x: (occurrences[level][x], -duplicates.index(dup)))
            #     new_col.append(most_common_val)
            # new_col = []
            # for level in range(len(col)):
            #     level_val = duplicates[0][level]
            #     for dup in duplicates:
            #         # if occurrences[dup[level]] > occurrences[level_val]:
            #         if occurrences[level][dup[level]] > occurrences[level][level_val]:
            #             level_val = dup[level]
            #     new_col.append(level_val)
    # for col, vals in df.items():
    #     new_col = col
    #     if col in duplicated_columns:
    #         duplicates = duplicated_columns[col]
    #         new_col = []
    #         for i in range(len(col)):
    #             level_val = duplicates[0][i]
    #             for dup in duplicates:
    #                 if occurrences[dup[i]] > occurrences[level_val]:
    #                     level_val = dup[i]
    #             new_col.append(level_val)
    #         new_col = tuple(new_col)
            

# def simplify_columns(df):
#     """Simplify columns by finding main level and handling duplicates."""
#     unique_counts = [
#         df.columns.get_level_values(level).nunique()
#         for level in range(df.columns.nlevels)
#     ]
#     main_level = unique_counts.index(max(unique_counts))

#     common_keys = []
#     for level in range(df.columns.nlevels):
#         if level != main_level:
#             common_keys.append(
#                 df.columns.get_level_values(level).value_counts().idxmax()
#             )
#         else:
#             common_keys.append(None)
#     duplicates = df.columns.get_level_values(main_level).value_counts()
#     duplicates = [k for k, v in duplicates.items() if v > 1]

#     try:
#         new_df = pd.DataFrame()
#         for key in df.columns.get_level_values(main_level):
#             common_keys.pop(main_level)
#             common_keys.insert(main_level, key)
#             if key not in duplicates:
#                 new_df[key] = df.xs(key, level=main_level, axis=1)
#             else:
#                 new_df = handle_duplicates(df, new_df, key, main_level, common_keys)
#     except KeyError as e:
#         raise KeyError(
#             "Error while simplifying multi-index columns. Please check the DataFrame."
#         ) from e
#     return new_df

# def handle_duplicates(df, new_df, key, main_level, common_keys):
#     """Handle duplicates by comparing their data and simplifying columns."""
#     selectors = [slice(None)] * df.columns.nlevels
#     selectors[main_level] = key
#     group = df.loc[:, tuple(selectors)]
#     main_col = group[tuple(common_keys)]
#     for col in group.columns:
#         if not group[col].equals(main_col):
#             new_df[".".join(col)] = group[col]
#         elif key not in new_df.columns:
#             new_df[key] = main_col
#     return new_df

@handle_dicts
def simplify_multi_index2(df):
    """Perform simplification of multiindex columns in a DataFrame."""
    if not isinstance(df, pd.DataFrame):
        return df
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(drop=True)

    # Check for levels with all unique names and drop them
    # source of drop_unique_or_single_levels
    drop_levels = []
    for level in range(df.columns.nlevels):
        if df.columns.get_level_values(level).is_unique:
            df.columns = df.columns.get_level_values(level)
            break
        elif df.columns.get_level_values(level).nunique() == 1:
            drop_levels.append(level)

    if isinstance(df.columns, pd.MultiIndex) and drop_levels != []:
        for level in drop_levels:
            df.columns = df.columns.droplevel(level)

    # end prev, source for simplify_columns
    if isinstance(df.columns, pd.MultiIndex):
        # Find the level with the most unique column keys
        unique_counts = [
            df.columns.get_level_values(level).nunique()
            for level in range(df.columns.nlevels)
        ]
        main_level = unique_counts.index(max(unique_counts))

        # Get the most common column key for each of the other levels
        # source for get_common_keys
        common_keys = []
        for level in range(df.columns.nlevels):
            if level != main_level:
                common_keys.append(
                    df.columns.get_level_values(level).value_counts().idxmax()
                )
            else:
                common_keys.append(None)

        # Find duplicates of the main level and compare their data
        duplicates = df.columns.get_level_values(main_level).value_counts()
        duplicates = [k for k, v in duplicates.items() if v > 1]

        try:
            new_df = pd.DataFrame()
            for key in df.columns.get_level_values(main_level):
                common_keys.pop(main_level)
                common_keys.insert(main_level, key)
                if key not in duplicates:
                    new_df[key] = df.xs(key, level=main_level, axis=1)
                else:
                    selectors = [slice(None)] * df.columns.nlevels
                    selectors[main_level] = key
                    group = df.loc[:, tuple(selectors)]
                    main_col = group[*common_keys]
                    for col in group.columns:
                        if not group[col].equals(main_col):
                            new_df[".".join(col)] = group[col]
                        elif key not in new_df.columns:
                            new_df[key] = main_col
        except KeyError as e:
            breakpoint()
            raise KeyError(
                "Error while simplifying multi-index columns. Please check the DataFrame."
            ) from e
        df = new_df
    return df


@handle_subdicts
def impedance_concat(raw_data):
    """Perform Concat on DataFrames in a dictionary."""
    data = {
        k: v for k, v in raw_data.items() if isinstance(v, (pd.DataFrame, pd.Series))
    }
    comb = pd.concat(data.values(), sort=False, keys=data.keys())
    try:
        if isinstance(comb.columns, pd.MultiIndex):
            freq_col = get_close_matches("freq", comb.columns.get_level_values(1))
            if not freq_col:
                return comb
            return comb.sort_values(
                [
                    (
                        "imps",
                        freq_col[0],
                        # str_in_list("freq", comb.columns.get_level_values(1)),
                    )
                ]
            )
        else:
            freq_col = get_close_matches("freq", comb.columns)
            if not freq_col:
                return comb
            return comb.sort_values(
                freq_col,
                ignore_index=True,
                # str_in_list("freq", comb.columns), ignore_index=True
            )
    except KeyError:
        return comb


class TypeList(list):
    """Class to create a list with 'type' information."""

    def __init__(self, values):
        super().__init__(values)

    def of_type(self, *item_type):
        """Return a list of items of the given type."""
        if len(item_type) == 0:
            return self
        if len(item_type) == 1:
            item_type = item_type[0]
        if isinstance(item_type, str):
            return [
                item for item in self if item_type.lower() in str(type(item)).lower()
            ]
        elif isinstance(item_type, type) or (
            isinstance(item_type, tuple) and all(isinstance(i, type) for i in item_type)
        ):
            return [item for item in self if isinstance(item, item_type)]
        return []


def moving_average(arr, w=2, logscale=False):
    """
    Computes the moving average of an array.

    This function calculates the moving average of the input array `arr` with a specified
    window size `w`. If `logscale` is True, the logarithm (base 10) of the array values is
    used for the calculation. The function handles edge cases by adjusting the window size
    and returns the result as a list of floats.

    Parameters:
    arr (list or np.ndarray): The input array to compute the moving average for.
    w (int, optional): The window size for the moving average. Default is 2.
    logscale (bool, optional): If True, computes the moving average on the logarithm (base 10)
                               of the array values. Default is False.

    Returns:
    list of float: The moving average of the input array.
    """
    if logscale:
        arr = np.log10([a if a > 0 else 1e-30 for a in arr])
    res = list(np.convolve(arr, np.ones(w), "valid") / w)
    w -= 1
    while w >= 1:
        if w % 2:
            res.append((np.convolve(arr, np.ones(w), "valid") / w)[-1])
        else:
            res.insert(0, (np.convolve(arr, np.ones(w), "valid") / w)[0])
        w -= 1
    if logscale:
        return [float(10**f) for f in res]
    return [float(f) for f in res]


def hz_label(
    data,
    test_arr=None,
    prec=2,
    kind="eng",
    space=" ",
    postfix="Hz",
    label_rc=True,
    targ_col="frequency",
    test_col="imag",
    new_col="flabel",
):
    """
    Generates frequency labels for MFIA data.

    This function creates a new column in the provided DataFrame or processes a numpy array
    to generate frequency labels based on the specified parameters. It uses the target column
    for frequency values and the test column for additional calculations.

    Parameters:
    data (pd.DataFrame or np.ndarray): The data to process. If a DataFrame, it should contain
                                       the target and test columns.
    test_arr (np.ndarray, optional): An array for additional calculations. If not provided,
                                     it is computed using a moving average of the test column.
    prec (int, optional): The precision for the frequency labels. Default is 2.
    kind (str, optional): The format kind for the labels ('eng' for engineering notation). Default is "eng".
    space (str, optional): The space between the number and the postfix. Default is " ".
    postfix (str, optional): The postfix for the frequency labels. Default is "Hz".
    label_rc (bool, optional): If True, labels are generated in reverse order. Default is True.
    targ_col (str, optional): The name of the target column for frequency values in the DataFrame. Default is "frequency".
    test_col (str, optional): The name of the test column for additional calculations in the DataFrame. Default is "imag".
    new_col (str, optional): The name of the new column to store the generated labels in the DataFrame. Default is "flabel".

    Returns:
    pd.DataFrame or np.ndarray: The DataFrame with the new column of frequency labels, or the processed numpy array.
    """
    if isinstance(data, pd.DataFrame):
        # targ_col = str_in_list(targ_col, data.columns)
        # test_col = str_in_list(test_col, data.columns)
        targ_col = get_close_matches(targ_col, data.columns)
        test_col = get_close_matches(test_col, data.columns)

        if targ_col and test_col:
            targ_col = targ_col[0]
            test_col = test_col[0]
            # if targ_col not in data.columns or test_col not in data.columns:
            #     return data
            data[new_col] = hz_label(
                data[targ_col].to_numpy(),
                test_arr=moving_average(-1 * data[test_col].to_numpy(), 5, True),
                prec=prec,
                kind=kind,
                space=space,
                postfix=postfix,
                label_rc=label_rc,
            )
        return data
    # if isinstance(data, pd.Series)
    base = [float(10 ** (np.floor(np.log10(a)))) if a > 0 else 0 for a in data]
    base_diff = np.diff(base)

    res = [np.nan] * len(data)

    for n, value in enumerate(data):
        if value == 0:
            continue
        elif n == 0 or base_diff[n - 1] != 0:
            res[n] = str(eng_not(base[n], 0, kind, space)) + postfix
            if (
                label_rc
                and isinstance(test_arr, (list, np.ndarray))
                and test_arr[n] == max(abs(np.array(test_arr)))
            ):
                res[n] = res[n] + " (RC)"
        elif (
            label_rc
            and isinstance(test_arr, (list, np.ndarray))
            and test_arr[n] == max(abs(np.array(test_arr)))
        ):
            try:
                if len(kind) > 2 and "exp" in kind.lower():
                    res[n] = "RC (f=" + eng_not(data[n], prec, "eng", " ") + "Hz)"
                else:
                    res[n] = "RC (f=" + eng_not(data[n], prec, kind, space) + "Hz)"
            except TypeError:
                return res
    return res


def apply_extend(start, stop, count, extend_by=0, extend_to=None, logscale=True):
    if logscale:
        start = np.log10(start)
        stop = np.log10(stop)
        extend_to = (
            np.log10(extend_to)
            if isinstance(extend_to, (int, float)) and extend_to > 0
            else None
        )

    # Calculate delta
    delta = np.diff(np.linspace(start, stop, count)).mean()

    # Apply extend_by logic
    if extend_by < 0:
        start += delta * extend_by
    elif extend_by > 0:
        stop += delta * extend_by

    # Apply extend_to logic
    if extend_to is not None and isinstance(extend_to, (int, float)):
        if extend_to < start:
            start = start + delta * (1 + (extend_to - start) // delta)
        elif extend_to > stop:
            stop = stop + delta * ((extend_to - stop) // delta)

    # Update count based on new start and stop
    count = int(np.ceil((stop - start) / delta)) + 1

    return start, stop, count


def shiftspace(start, stop, num=50, shift=0, logscale=True, as_exp=False):
    if logscale:
        start = np.log10(start)
        stop = np.log10(stop)

    delta = np.diff(np.linspace(start, stop, num)).mean()

    new_start = start - delta * shift
    new_stop = stop + delta * shift

    if logscale and not as_exp:
        return float(10**new_start), float(10**new_stop), int(num + 2 * shift)

    return new_start, new_stop, int(num + 2 * shift)


def range_maker(
    start,
    stop,
    points_per_decade=24,
    shift=0,
    is_exp=False,
    fmt="mfia",
    extend_by=0,
    extend_to=None,
):
    if not is_exp:
        start = np.log10(start)
        stop = np.log10(stop)
        extend_to = (
            np.log10(extend_to)
            if isinstance(extend_to, (int, float)) and extend_to > 0
            else None
        )
    count = int(1 + points_per_decade * abs(start - stop))
    start, stop, count = shiftspace(start, stop, count, shift, False, True)

    start, stop, count = apply_extend(
        start, stop, count, extend_by, extend_to, logscale=False
    )

    if fmt.lower() in ["numpy", "np"]:
        return start, stop, count
    return {"start": float(10**start), "stop": float(10**stop), "samplecount": count}


# def extendspace(start, stop, num=50, ext=0, logscale=True, as_exp=False):
#     if logscale:
#         start = np.log10(start)
#         stop = np.log10(stop)

#     delta = np.diff(np.linspace(start, stop, num)).mean()

#     new_start = start - delta * ext
#     new_stop = stop + delta * ext

#     if logscale and not as_exp:
#         return 10**new_start, 10**new_stop, int(num + 2 * ext)

#     return new_start, new_stop, int(num + 2 * ext)


# def range_maker(start, stop, points_per_decade=24, ext=0, is_exp=False):
#     if not is_exp:
#         start = np.log10(start)
#         stop = np.log10(stop)
#     count = int(1 + points_per_decade * abs(start - stop))
#     start, stop, count = extendspace(start, stop, count, ext, False, True)
#     return {"start": 10**start, "stop": 10**stop, "samplecount": count}
