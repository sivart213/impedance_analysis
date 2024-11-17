# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""

import numpy as np
import pandas as pd

from ..utils.decorators import handle_dicts, handle_subdicts
from ..string_ops import str_in_list, eng_not

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
    name = str_in_list(name, df.columns)
    if name in df.columns:
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


def remove_duplicate_datasets(data_dict, min_rows=1, verbose=False):
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
    unique_datasets = {}
    seen_createdtimes = {}

    for filename, sheets in data_dict.items():
        for sheet_name, df in sheets.items():
            if len(df) <= min_rows:
                if verbose:
                    print(
                        f"Rejected dataset due to insufficient rows: {filename} - {sheet_name} (rows: {len(df)})"
                    )
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
                        if verbose:
                            print(
                                f"Replaced dataset: {existing_filename} - {existing_sheet_name} with {filename} - {sheet_name} due to more completeness"
                            )
                    elif df.notnull().sum().sum() == existing_df.notnull().sum().sum():
                        # If completeness is the same, keep the dataset with the longer name
                        if len(filename) > len(existing_filename):
                            unique_datasets[existing_filename].pop(existing_sheet_name)
                            if not unique_datasets[existing_filename]:
                                unique_datasets.pop(existing_filename)
                            unique_datasets.setdefault(filename, {})[sheet_name] = df
                            seen_createdtimes[createdtime] = (
                                filename,
                                sheet_name,
                            )
                            if verbose:
                                print(
                                    f"Replaced dataset: {existing_filename} - {existing_sheet_name} with {filename} - {sheet_name} due to longer name"
                                )
                        elif len(filename) == len(existing_filename) and len(
                            sheet_name
                        ) > len(existing_sheet_name):
                            unique_datasets[existing_filename].pop(existing_sheet_name)
                            if not unique_datasets[existing_filename]:
                                unique_datasets.pop(existing_filename)
                            unique_datasets.setdefault(filename, {})[sheet_name] = df
                            seen_createdtimes[createdtime] = (
                                filename,
                                sheet_name,
                            )
                            if verbose:
                                print(
                                    f"Replaced dataset: {existing_filename} - {existing_sheet_name} with {filename} - {sheet_name} due to longer sheet name"
                                )
                    else:
                        if verbose:
                            print(
                                f"Rejected dataset due to duplicate createdtime: {filename} - {sheet_name} (createdtime: {createdtime})"
                            )
                else:
                    unique_datasets.setdefault(filename, {})[sheet_name] = df
                    seen_createdtimes[createdtime] = (filename, sheet_name)
            else:
                # If no createdtime attribute, keep the dataset
                unique_datasets.setdefault(filename, {})[sheet_name] = df

    return unique_datasets

@handle_dicts
def simplify_multi_index(df):
    """Perform simplification of multiindex columns in a DataFrame."""
    if not isinstance(df, pd.DataFrame):
        return df
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(drop=True)

    # Check for levels with all unique names and drop them
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

    if isinstance(df.columns, pd.MultiIndex):
        # Find the level with the most unique column keys
        unique_counts = [
            df.columns.get_level_values(level).nunique()
            for level in range(df.columns.nlevels)
        ]
        main_level = unique_counts.index(max(unique_counts))

        # Get the most common column key for each of the other levels
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
            return comb.sort_values(
                [
                    (
                        "imps",
                        str_in_list("freq", comb.columns.get_level_values(1)),
                    )
                ]
            )
        else:
            # return comb
            return comb.sort_values(
                str_in_list("freq", comb.columns), ignore_index=True
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
        targ_col = str_in_list(targ_col, data.columns)
        test_col = str_in_list(test_col, data.columns)
        if targ_col not in data.columns or test_col not in data.columns:
            return data
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

