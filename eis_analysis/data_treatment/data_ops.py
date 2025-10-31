# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""

import re
from typing import Any, Literal, TypeAlias, overload
from itertools import combinations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

try:
    from .value_ops import convert_val
    from ..string_ops import eng_not
except ImportError:
    from eis_analysis.string_ops import eng_not
    from eis_analysis.data_treatment.value_ops import convert_val

IsFalse: TypeAlias = Literal[False]
IsTrue: TypeAlias = Literal[True]


def ensure_unique(
    data: np.ndarray | pd.Series | pd.DataFrame | list[str] | tuple[str, ...],
    primary_column: str | int | None = None,
    behavior: str | None = None,
    sep: str = "_",
    prefix: bool = True,
) -> pd.Series:
    """
    Ensures that all values in an array-like sequence or a DataFrame column are unique by
    modifying duplicates based on the specified behavior.

    Parameters:
    - data (array-like or pd.DataFrame): The input data to validate.
    - primary_column (str or int, optional): The primary column to validate if a DataFrame is
                                            passed. Defaults to the first column.
    - behavior (str or None): Behavior for handling non-uniqueness. Options:
                      - None: Defaults to "add_numbers_all" or "check_other_columns" for DF's else.
                      - "raise": Raise a ValueError.
                      - "add_numbers": Append numbers to duplicates.
                      - "add_numbers_all": Append numbers to all values if duplicates exist.
                      - "check_other_columns": Use other columns to create unique IDs (DF only).
    - sep (str): Separator to use when appending numbers or combining columns. Default is "_".

    Returns:
    - array-like or pd.Series: The modified data with unique values.

    Raises:
    - ValueError: If duplicates are found and behavior is "raise".
    """

    def _make_res(c_data, s, mod):
        if prefix:
            res = mod + s + c_data
        else:
            res = c_data + s + mod
        return res

    if isinstance(data, pd.DataFrame):
        if primary_column is None:
            primary_column = data.columns[0]
        if primary_column not in data.columns:
            raise ValueError(f"Primary column '{primary_column}' not found in DataFrame.")
        column_data = data[primary_column].astype(str)
        if behavior is None:
            behavior = "check_other_columns"
    else:
        column_data = pd.Series(data, dtype=str)

    if column_data.is_unique:
        return column_data

    if behavior is None:
        behavior = "add_numbers_all"

    if behavior == "raise":
        raise ValueError("Duplicate values found in the data.")

    elif behavior == "add_numbers":
        counts = column_data.groupby(column_data).cumcount()
        modifier = counts.astype(str).where(counts > 0, "")
        return _make_res(column_data, [sep if m else "" for m in modifier], modifier)

    elif behavior == "add_numbers_all":
        modifier = column_data.groupby(column_data).cumcount().astype(str)
        return _make_res(column_data, sep, modifier)

    elif behavior == "check_other_columns" and isinstance(data, pd.DataFrame):
        other_columns = [col for col in data.columns if col != primary_column]

        # Sort other columns by the average string length of their values
        other_columns.sort(key=lambda col: data[col].astype(str).str.len().mean())

        modifier = None
        for r in range(1, len(other_columns) + 1):
            for combo in combinations(other_columns, r):
                modifier = data[list(combo)].astype(str).agg(sep.join, axis=1)
                if (modifier + column_data).is_unique:
                    break
                else:
                    modifier = None
            if modifier is not None:
                break

        if modifier is None:
            return ensure_unique(
                column_data,
                behavior="add_numbers_all",
                sep=sep,
                prefix=prefix,
            )
        else:
            return _make_res(column_data, sep, modifier)

    else:
        raise ValueError(f"Invalid behavior '{behavior}' specified.")
    # Set column_data based on prefix


class TypeList(list):
    """Class to create a list with 'type' information."""

    def __init__(self, values):
        super().__init__(values)

    def of_type(self, *item_type: str | type):
        """Return a list of items of the given type."""
        if len(item_type) == 0:
            return self
        str_types = [t for t in item_type if isinstance(t, str)]
        type_types = tuple([t for t in item_type if isinstance(t, type)])
        return [
            item
            for item in self
            if isinstance(item, type_types)
            or any(st.lower() in str(type(item)).lower() for st in str_types)
        ]

        # if len(item_type) == 1:
        #     item_type = item_type[0]
        # if isinstance(item_type, str):
        #     return [item for item in self if item_type.lower() in str(type(item)).lower()]
        # elif isinstance(item_type, type) or (
        #     isinstance(item_type, tuple) and all(isinstance(i, type) for i in item_type)
        # ):
        #     return [item for item in self if isinstance(item, cast(tuple[type, ...], item_type))]
        # return []


def moving_average(
    array: np.ndarray | pd.Series | list[float | int] | tuple[float | int, ...],
    w: int = 2,
    logscale: bool = False,
) -> list[float]:
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
    # array = np.asarray(arr)
    if logscale:
        array = np.log10([a if a > 0 else 1e-30 for a in array])
    res = list(np.convolve(array, np.ones(w), "valid") / w)
    w -= 1
    while w >= 1:
        if w % 2:
            res.append((np.convolve(array, np.ones(w), "valid") / w)[-1])
        else:
            res.insert(0, (np.convolve(array, np.ones(w), "valid") / w)[0])
        w -= 1
    if logscale:
        return [float(10**f) for f in res]
    return [float(f) for f in res]


def generate_labels(
    data: np.ndarray | pd.Series | list[Any] | tuple[Any, ...],
    test_values: np.ndarray | pd.Series | list[Any] | tuple[Any, ...] | float | int | None = None,
    mode: str = "both",  # Options: "decade", "points", "both"
    prec: int = 2,
    kind: str = "eng",
    space: str = " ",
    postfix: str = "",
    prefix: str = "",
    test_label: str = "",
) -> list[str | float]:
    """
    Generates labels for an array-like sequence based on specified modes.

    This function creates labels for an array-like sequence. Labels can be generated
    for each decade, specific points (based on `test_values`), or both. The labels
    can include a prefix and postfix, and the format can be customized.

    Parameters:
    - data (array-like): The input data to label.
    - test_values (float, tuple, or list, optional): Specific value(s) to label. If not provided,
                                                    decade labels are generated (depends on mode).
    - mode (str, optional): The labeling mode. Options are:
                            - "decade": Label each decade.
                            - "points": Label specific points based on `test_values`.
                            - "both": Label both decades and specific points. Default is "both".
    - prec (int, optional): The precision for the labels. Default is 2.
    - kind (str, optional): The format for the labels (Default: 'eng' for engineering notation).
    - space (str, optional): The space between the number and the postfix. Default is " ".
    - postfix (str, optional): The postfix for the labels. Default is "".
    - prefix (str, optional): The prefix for the labels. Default is "".

    Returns:
    - list: A list of labels corresponding to the input data.
    """
    if isinstance(data, str) or not hasattr(data, "__iter__"):
        raise ValueError("Input data must be array-like.")

    if test_values is None:
        test_values = []
    elif isinstance(test_values, (float, int)):
        test_values = [test_values]
    elif isinstance(test_values, tuple):
        test_values = list(test_values)
    elif isinstance(test_values, str) or not hasattr(test_values, "__iter__"):
        raise ValueError("test_values must be a float, tuple, or list.")

    # Create an array of the decades
    base = [float(10 ** (np.floor(np.log10(a)))) if a > 0 else 0 for a in data]
    base_diff = np.diff(base)  # Array to indicate where the base changes

    res: list[str | float] = [np.nan] * len(data)

    for n, value in enumerate(data):
        str_val = value
        if value == 0:
            continue
        # Label decades
        elif mode in ["decade", "both"] and (n == 0 or base_diff[n - 1] != 0):
            str_val = eng_not(base[n], 0, kind, space)
            str_val = f"{str_val} {postfix}"
            if mode == "both":
                str_val = str_val + f" ({test_label})"
            res[n] = str_val

        # Label specific points
        elif mode in ["points", "both"] and value in test_values:
            try:
                str_val = f"{test_label} ({prefix}{eng_not(value, prec, kind, space)}{postfix})"
            except TypeError:
                continue

        if isinstance(str_val, str):
            str_val = re.sub(r"\s+", " ", str_val)  # Replace double spaces
            str_val = re.sub(r"\(+", "(", str_val)  # Replace double opening parentheses
            str_val = re.sub(r"\)+", ")", str_val)  # Replace double closing parentheses
            str_val = str_val.strip()  # Strip leading/trailing spaces

        res[n] = str_val

    return res


def apply_extend(
    start: float,
    stop: float,
    count: int,
    extend_by: int = 0,
    extend_to: float | None = None,
    logscale: bool = True,
) -> tuple[float, float, int]:
    """
    Adjusts the range of values by extending the start and stop points.

    This function modifies the range defined by `start` and `stop` by extending it
    based on the `extend_by` or `extend_to` parameters. It also recalculates the
    number of points (`count`) in the range. Logarithmic scaling can be applied
    if specified.

    Parameters:
    - start (float): The starting value of the range.
    - stop (float): The ending value of the range.
    - count (int): The number of points in the range.
    - extend_by (int, optional): The number of steps to extend the range by. Default is 0.
    - extend_to (float, optional): A specific value to extend the range to. Default is None.
    - logscale (bool, optional): If True, applies logarithmic scaling. Default is True.

    Returns:
    - tuple: A tuple containing the adjusted `start`, `stop`, and `count` values.
    """
    if logscale:
        start = np.log10(start)
        stop = np.log10(stop)
        extend_to = (
            np.log10(extend_to) if isinstance(extend_to, (int, float)) and extend_to > 0 else None
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


def shift_space(
    start: float,
    stop: float,
    num: int = 50,
    shift: int = 0,
    logscale: bool = True,
    as_exp: bool = True,
) -> tuple[float, float, int]:
    """
    Shifts the range of values by adjusting the start and stop points without modifying the length.

    This function modifies the range defined by `start` and `stop` by shifting it
    up or down based on the `shift` parameter. Logarithmic scaling can be
    applied if specified.

    Parameters:
    - start (float): The starting value of the range.
    - stop (float): The ending value of the range.
    - num (int, optional): The number of points in the range. Default is 50.
    - shift (int, optional): The number of steps to shift the range by. Default is 0.
    - logscale (bool, optional): If True, applies logarithmic scaling. Default is True.
    - as_exp (bool, optional): If True, returns the values in exponential form. Default is False.

    Returns:
    - tuple: A tuple containing the adjusted `start`, `stop`, and `num` values.
    """
    if logscale:
        start = np.log10(start)
        stop = np.log10(stop)

    delta = np.diff(np.linspace(start, stop, num)).mean()

    new_start = start + delta * shift
    new_stop = stop + delta * shift

    if logscale and not as_exp:
        return float(10**new_start), float(10**new_stop), num

    return new_start, new_stop, num


def range_maker(
    start: float | int,
    stop: float | int,
    points_per_decade: int = 24,
    shift: int = 0,
    is_exp: bool = False,
    fmt: str = "mfia",
    extend_by: int = 0,
    extend_to: float | None = None,
) -> dict[str, float | int] | tuple[float, float, int]:
    """
    Generates a range of values with optional extensions and shifts.

    This function creates a range of values between `start` and `stop` with a specified
    number of points per decade. The range can be extended or shifted, and logarithmic
    scaling can be applied. The output format can be customized.

    Parameters:
    - start (float): The starting value of the range.
    - stop (float): The ending value of the range.
    - points_per_decade (int, optional): The number of points per decade. Default is 24.
    - shift (int, optional): The number of steps to shift the range by. Default is 0.
    - is_exp (bool, optional): If True, treats the input as exponential values. Default is False.
    - fmt (str, optional): The output format ("mfia" or "numpy"). Default is "mfia".
    - extend_by (int, optional): The number of steps to extend the range by. Default is 0.
    - extend_to (float, optional): A specific value to extend the range to. Default is None.

    Returns:
    - dict or tuple: A dictionary with `start`, `stop`, and `samplecount` keys if `fmt="mfia"`,
                     or a tuple of (`start`, `stop`, `count`) if `fmt="numpy"`.
    """
    if not is_exp:
        start = np.log10(start)
        stop = np.log10(stop)
        extend_to = (
            np.log10(extend_to) if isinstance(extend_to, (int, float)) and extend_to > 0 else None
        )

    count = int(1 + points_per_decade * abs(start - stop))
    start, stop, count = shift_space(start, stop, count, shift, False, True)

    start, stop, count = apply_extend(start, stop, count, extend_by, extend_to, logscale=False)

    if fmt.lower() in ["numpy", "np"]:
        return start, stop, count
    return {"start": float(10**start), "stop": float(10**stop), "samplecount": count}


def convert_array(
    arr: ArrayLike,
    i_unit: str | Any | None = None,
    f_unit: str | Any | None = None,
    exponent: int = 1,
) -> ArrayLike:
    """
    Converts values from one unit to another by sanitizing the input and using sympy.convert_to()
    """
    if isinstance(arr, (list, tuple, np.ndarray, pd.Series)):
        res = [convert_val(v, i_unit, f_unit, exponent) for v in arr]  # type: ignore[assignment]
        if isinstance(arr, np.ndarray):
            return np.array(res)
        if isinstance(arr, pd.Series):
            return pd.Series(res)
        else:
            return res
    else:
        return arr


@overload
def clean_key_list(
    keys: list[tuple[Any, ...]],
    fill: Any,
    drop_mode: Literal["common", "minimize", None],
    append_name: bool,
    append_ints: bool,
    flatten: IsTrue,
    sep: str = "/",
    ensure_items_unique: bool = ...,
) -> list[str]: ...
@overload
def clean_key_list(
    keys: list[tuple[Any, ...]],
    fill: Any = ...,
    drop_mode: Literal["common", "minimize", None] = None,
    append_name: bool = False,
    append_ints: bool = False,
    *,
    flatten: IsTrue,
    sep: str = "/",
    ensure_items_unique: bool = ...,
) -> list[str]: ...
@overload
def clean_key_list(
    keys: list[tuple[Any, ...]],
    fill: Any = ...,
    drop_mode: Literal["common", "minimize", None] = None,
    append_name: bool = False,
    append_ints: bool = False,
    flatten: IsFalse = False,
    sep: str = "/",
    ensure_items_unique: bool = ...,
) -> list[tuple[Any, ...]]: ...


def clean_key_list(
    keys: list[tuple[Any, ...]],
    fill: Any = "",
    drop_mode: Literal["common", "minimize", None] = None,
    append_name: bool = False,
    append_ints: bool = False,
    flatten: bool = False,
    sep: str = "/",
    ensure_items_unique: bool = True,
) -> list[tuple[Any, ...]] | list[str]:
    """
    Cleans and normalizes a list of tuple values.
    Typically used to clean keys for dictionaries or DataFrame columns.

    Parameters:
        keys (List[Tuple]): List of tuple keys (from a flattened dict).
        fill (Any): Placeholder for missing levels (default: "").
        ensure_items_unique (bool): If True, ensures that the keys are unique.
        flatten (bool): If True, returns a flattened list of strings.
        drop_mode (str|None): Controls which tuple indices (positions) to drop:
            - "common": Drop only tuple indices where all tuples have the same value at that position.
            - "minimize": Drop as many tuple indices as possible while preserving uniqueness.
            - None: Keep all tuple indices.
        append_name (bool): If True, appends the dropped value to the next column.
        append_ints (bool): If True, appends dropped values as integers.
        sep (str): Separator to use when flattening the keys.

    Returns:
        List[Tuple]: Cleaned and squared-up list of tuple keys.
    """
    if not keys:
        return []
    # Squares up keys to ensure they are tuples of the same length
    df = pd.DataFrame([k if isinstance(k, (tuple, list)) else (k,) for k in keys]).fillna(fill)

    # Prepare the DataFrame for processing
    all_cols: set[int] = set(df.columns.astype(int))
    num_cols: set[int] = set(df.select_dtypes(include="number").columns.astype(int))

    # Ensure all value sets are unique
    if ensure_items_unique:
        df = df.drop_duplicates(ignore_index=True)

    # Section evaluates for unneeded value sets
    if drop_mode is not None and len(df) > 1:
        # Step 1: Always drop value sets where nunique == 1
        drop_cols = {int(i) for i in df.columns if df[i].nunique(dropna=False) == 1}
        keep: set[int] = all_cols - drop_cols

        # Step 1 check: Only possible if df was not unique to start
        if len(keep) == 0:
            keep = all_cols.copy()
            drop_cols = set()
        # Step 2: Drop value sets which do not contribute to uniqueness
        elif drop_mode == "minimize":
            # Try dropping columns from right to left (sorted for reproducibility)
            drop_cols_base = drop_cols.copy()
            for i in sorted(keep, reverse=True):
                if not df[sorted(keep - {i})].duplicated().any():
                    keep.remove(i)
                    drop_cols.add(i)

            for num_col in sorted(keep & num_cols):  # iterate over numeric keep columns
                for non_num_col in sorted(
                    drop_cols - num_cols
                ):  # iterate over non-numeric drop columns
                    test_keep = sorted(keep - {num_col}) + [non_num_col]
                    if not df[test_keep].duplicated().any():
                        keep.remove(num_col)
                        drop_cols.add(num_col)
                        keep.add(non_num_col)
                        drop_cols.remove(non_num_col)
                        break
            # Step 2 check: Only possible if df was not unique to start
            if len(keep) == 0:
                keep = all_cols - drop_cols_base
                drop_cols = drop_cols_base

        # Step 3: Clean up drop and add to names if directed
        if len(drop_cols) > 0:
            append_cols = []
            if append_name:
                df = df.astype(str)
                append_cols = sorted(drop_cols)
            elif append_ints:
                df = df.astype(str)
                append_cols = sorted(drop_cols & num_cols)
            if append_cols:
                keep_array = np.array(sorted(keep), dtype=int)
                df[append_cols] = "(" + df[append_cols] + ")"
                for col in append_cols:
                    if col > keep_array.max():
                        target = keep_array.max()
                    else:
                        target = keep_array[keep_array > col].min()
                    df[target] = df[target] + df[col]

        # Drop the columns
        df = df.drop(columns=sorted(drop_cols))

    # Convert to flat string if `flatten`
    if flatten:
        return list(map(sep.join, df.astype(str).itertuples(index=False, name=None)))

    return list(df.itertuples(index=False, name=None))


def evaluate_1D_array(
    val: list | tuple | np.ndarray | pd.Series | pd.DataFrame,
    operations: str | tuple | list = "shape",
    null_values: list | tuple | None = None,
    ignore_index: bool = True,
    as_type: str | type = "any",  # <-- new argument
) -> Any:
    """
    Reduce and clean a 1D array-like object (list, tuple, np.ndarray, pd.Series) using a sequence of operations.
    Supported operations: "shape" (ensure Series), "drop_null", "drop_na", "drop_dupl".
    """

    if not isinstance(null_values, (tuple, list)):
        null_values = []

    if isinstance(operations, str):
        operations = [operations]
    elif isinstance(operations, tuple):
        operations = list(operations)

    # Convert to DataFrame if not already
    arr = PD_Ops.to_pandas(val, pd.Series)

    # Apply operations in order
    for op in operations:
        if "shape" in op.lower():
            arr = PD_Ops.check_shape(arr)
        # request drop activity
        elif "drop" in op.lower():
            # `drop` without null values will always be a drop na
            # to get drop null, exclude na and pass null values
            if "dupl" in op.lower():
                arr = arr.drop_duplicates()
            elif "na" in op.lower() or not null_values:
                arr = PD_Ops.drop_na(arr)
            elif null_values:
                arr = PD_Ops.drop_vals(arr, null_values)
    arr = PD_Ops.check_shape(arr)
    if ignore_index:
        arr = arr.reset_index(drop=True)

    arr = PD_Ops.convert_type(arr, as_type)

    return arr


def evaluate_nD_array(
    val: list | tuple | np.ndarray | pd.DataFrame | pd.Series,
    operations: str | tuple | list = "shape",
    null_values: list | tuple | None = None,
    drop_on: Literal[0, 1, "both"] = 0,
    ignore_index: bool = True,
    as_type: str | type = "any",  # <-- new argument
) -> Any:  # pd.DataFrame | pd.Series:
    """
    Reduce a DataFrame or Series based on shape.
    """
    if not isinstance(null_values, (tuple, list)):
        null_values = []
    if isinstance(operations, str):
        operations = [operations]

    # Convert to DataFrame if not already
    df = PD_Ops.to_pandas(val, pd.DataFrame)

    # Apply operations in order
    for op in operations:
        if "shape" in op.lower():
            df = PD_Ops.check_shape(df)
        # request drop activity
        elif "drop" in op.lower():
            # `drop` without null values will always be a drop na
            # to get drop null, exclude na and pass null values
            if "dupl" in op.lower():
                df = df.drop_duplicates()
            elif "na" in op.lower() or not null_values:
                df = PD_Ops.drop_na(df, drop_on)
            elif null_values:
                df = PD_Ops.drop_vals(df, null_values, drop_on)
    df = PD_Ops.check_shape(df)
    if ignore_index:
        df = df.reset_index(drop=True)

    df = PD_Ops.convert_type(df, as_type)

    return df


class PD_Ops:
    """
    Class for performing operations on Pandas DataFrames and Series.
    This class provides methods to drop null values, check shapes, and convert types.
    """

    type_map = {
        "series": pd.Series,
        "dataframe": pd.DataFrame,
        "array": np.ndarray,
        "list": list,
        "tuple": tuple,
        "dict": dict,
    }

    @classmethod
    def str_to_type(cls, val, default=pd.DataFrame) -> type:
        """Convert a string representation of a type to the actual type."""
        if isinstance(val, str):
            return cls.type_map.get(val.lower(), default)
        return default

    @overload
    @classmethod
    def convert_type(
        cls, arr: pd.Series | pd.DataFrame, ret_type: Literal["any"]
    ) -> pd.Series | pd.DataFrame: ...
    @overload
    @classmethod
    def convert_type(
        cls, arr: pd.Series | pd.DataFrame, ret_type: Literal["series"] | type[pd.Series]
    ) -> pd.Series: ...
    @overload
    @classmethod
    def convert_type(
        cls, arr: pd.Series | pd.DataFrame, ret_type: Literal["dataframe"] | type[pd.DataFrame]
    ) -> pd.DataFrame: ...
    @overload
    @classmethod
    def convert_type(
        cls, arr: pd.Series | pd.DataFrame, ret_type: Literal["array"] | type[np.ndarray]
    ) -> np.ndarray: ...
    @overload
    @classmethod
    def convert_type(
        cls, arr: pd.Series | pd.DataFrame, ret_type: Literal["list"] | type[list]
    ) -> list: ...
    @overload
    @classmethod
    def convert_type(
        cls, arr: pd.Series | pd.DataFrame, ret_type: Literal["tuple"] | type[tuple]
    ) -> tuple: ...
    @overload
    @classmethod
    def convert_type(
        cls, arr: pd.Series | pd.DataFrame, ret_type: Literal["dict"] | type[dict]
    ) -> dict: ...
    @overload
    @classmethod
    def convert_type(cls, arr: pd.Series | pd.DataFrame, ret_type: str | type) -> Any: ...

    @classmethod
    def convert_type(cls, arr: pd.Series | pd.DataFrame, ret_type: str | type) -> Any:
        """
        Convert arr to the requested type.
        as_type: "any" (no forced conversion), "dataframe", "series"
        """
        if isinstance(ret_type, str):
            if ret_type.lower() == "any":
                return arr
            ret_type = cls.str_to_type(ret_type, pd.DataFrame)

        if isinstance(arr, ret_type):
            return arr

        if ret_type is pd.DataFrame:
            return pd.DataFrame(arr)
        elif ret_type is pd.Series:
            return pd.Series(arr.to_numpy(copy=True).flatten())
        elif ret_type is dict:
            return arr.to_dict()
        elif ret_type is np.ndarray:
            return arr.to_numpy(copy=True)
        elif ret_type is list:
            return arr.to_numpy(copy=True).tolist()
        elif ret_type is tuple:
            return tuple(arr.to_numpy(copy=True).tolist())
        return arr

    @overload
    @classmethod
    def to_pandas(
        cls, val: Any, target_type: Literal["series"] | type[pd.Series]
    ) -> pd.Series: ...
    @overload
    @classmethod
    def to_pandas(
        cls, val: Any, target_type: Literal["dataframe"] | type[pd.DataFrame]
    ) -> pd.DataFrame: ...

    @classmethod
    def to_pandas(cls, val: Any, target_type: str | type = pd.Series) -> pd.Series | pd.DataFrame:
        """
        Convert any array-like input to a pandas Series or DataFrame.

        Parameters:
            val: The input value (array-like, Series, DataFrame, etc.)
            target_type: pd.Series or pd.DataFrame (default: pd.Series)

        Returns:
            pd.Series or pd.DataFrame
        """
        if isinstance(target_type, str):
            target_type = cls.type_map.get(target_type.lower(), "")
        if target_type not in (pd.Series, pd.DataFrame):
            raise ValueError("target_type must be pd.Series or pd.DataFrame")

        if isinstance(val, target_type):
            return val.copy()

        try:
            df = pd.DataFrame(val)
        except Exception as exc:
            print(f"Input is not a DataFrame: {exc}")
            try:
                df = pd.DataFrame(np.asarray(val))
            except Exception as exc:
                raise ValueError(f"Cannot convert input to DataFrame: {exc}. ") from exc
        if isinstance(df, target_type):
            return df
        return cls.convert_type(df, target_type)

    @staticmethod
    def _is_null(null_values):
        def wrapper(x):
            # Handles both scalar and array-like
            if pd.isna(x):
                return True
            return x in null_values

        return wrapper

    @classmethod
    def drop_vals(cls, arr, null_values, drop_on: Any = 0):
        null_values = cls._is_null(null_values)
        mask = ~arr.map(null_values)
        if isinstance(arr, pd.Series):
            return arr[mask]
        if drop_on in ["both", 0]:
            arr = arr.loc[mask.any(axis=1)]
        if drop_on in ["both", 1]:
            arr = arr.loc[:, mask.any(axis=0)]
        return arr

    @staticmethod
    def drop_na(arr, drop_on: Any = 0):
        if drop_on in ["both", 0]:
            arr = arr.dropna(how="all")
        if isinstance(arr, pd.DataFrame) and drop_on in ["both", 1]:
            arr = arr.dropna(axis=1, how="all")
        return arr

    @staticmethod
    def check_shape(arr):
        arr = arr.squeeze()
        if not isinstance(arr, (pd.Series, pd.DataFrame)):
            return pd.Series([arr])
        return arr


# keys = [
#     ("a", "A", "x", 1, "B"),
#     ("a", "A", "x", 2, "B"),
#     ("b", "A", "y", 1, "B"),
#     ("b", "A", "y", 2, "B"),
# ]
# test = clean_key_list(keys, "/", True, True, "minimize", True, True)
