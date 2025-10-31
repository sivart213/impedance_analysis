# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
import sys
from typing import Any, TypeVar, cast
from collections.abc import Callable

import numpy as np
import pandas as pd

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


# %% Decorators
def handle_collection(func: F) -> F:
    """
    A decorator to handle collections (tuple, list, set, np.ndarray).
    Recursively applies the decorated function to each element.
    """

    def wrapper(arg, *args, **kwargs) -> Any:
        if isinstance(arg, (tuple, list, set)):
            converted_vals = [
                (
                    wrapper(t, *args, **kwargs)
                    if not (isinstance(t, float) and np.isnan(t))
                    else float("nan")
                )
                for t in arg
            ]
            return type(arg)(converted_vals)
        elif isinstance(arg, np.ndarray):
            converted_vals = [
                (
                    wrapper(t, *args, **kwargs)
                    if not (isinstance(t, float) and np.isnan(t))
                    else float("nan")
                )
                for t in arg
            ]
            return np.array(converted_vals)
        else:
            return func(arg, *args, **kwargs)

    return cast(F, wrapper)


def handle_pandas(func: F) -> F:
    """
    A decorator to handle pandas Series and DataFrames.
    Recursively applies the decorated function to each element.
    """

    def wrapper(arg, *args, **kwargs) -> Any:
        if "pandas" in sys.modules:
            if isinstance(arg, pd.Series):
                converted_vals = [
                    (
                        wrapper(t, *args, **kwargs)
                        if not (isinstance(t, float) and np.isnan(t))
                        else float("nan")
                    )
                    for t in arg
                ]
                return pd.Series(converted_vals, index=arg.index)
            elif isinstance(arg, pd.DataFrame):
                df: pd.DataFrame = arg.copy()
                return df.applymap(
                    lambda t: (
                        wrapper(t, *args, **kwargs)
                        if not (isinstance(t, float) and np.isnan(t))
                        else float("nan")
                    )
                )  # type: ignore
            else:
                return func(arg, *args, **kwargs)
        else:
            return func(arg, *args, **kwargs)

    return cast(F, wrapper)


def handle_dicts(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    A decorator to handle dictionaries.
    Recursively applies the decorated function to each value.
    Assumes that the target data structure is of non-dict types.
    """

    def wrapper(arg, *args, **kwargs) -> Any:
        if isinstance(arg, dict):
            return {
                k: (
                    wrapper(t, *args, **kwargs)
                    if not (isinstance(t, float) and np.isnan(t))
                    else float("nan")
                )
                for k, t in arg.items()
            }
        else:
            return func(arg, *args, **kwargs)

    return cast(Callable[..., Any], wrapper)


# def handle_dicts(*funcs: Any) -> Any:
#     """
#     A decorator to handle dictionaries.
#     Recursively applies the decorated function(s) to each value.
#     Assumes that the target data structure is of non-dict types.

#     Notes:
#         - args passed with data will be ignored if args are passed to the function
#         - kwargs passed with data will be overwritten by shared kwargs passed to the function
#     """

#     def wrapper(arg, *args, **kwargs) -> Any:
#         if isinstance(arg, dict):
#             return {
#                 k: (
#                     wrapper(t, *args, **kwargs)
#                     if not (isinstance(t, float) and np.isnan(t))
#                     else float("nan")
#                 )
#                 for k, t in arg.items()
#             }
#         else:
#             res = None
#             for f in funcs:
#                 # As long as the type of the result is the same as the input, the loop will
#                 # ensure that the arg is updated with the result, otherwise update relies on
#                 # the functions modifying the arg directly
#                 if callable(f):
#                     res = f(arg, *args, **kwargs)
#                 elif isinstance(f[-1], dict):
#                     res = f[0](arg, *(f[1:-1] or args), **{**kwargs, **f[-1]})
#                 else:
#                     res = f[0](arg, *(f[1:] or args), **kwargs)
#                 arg = res if isinstance(res, type(arg)) else arg
#             return arg

#     return wrapper


# from eis_analysis.functions import sig_figs_ceil
def handle_subdicts(func: F) -> F:
    """
    A decorator to handle dictionaries.
    Recursively applies the decorated function to each value.
    Assumes that the target data structure is a dictionary of non-dict types.
    """

    def wrapper(arg, *args, **kwargs) -> Any:
        if isinstance(arg, dict) and all(isinstance(a, dict) for a in arg.values()):
            res = {k: wrapper(t, *args, **kwargs) for k, t in arg.items()}
            return res
        else:
            return func(arg, *args, **kwargs)

    return cast(F, wrapper)


def recursive(func: F) -> F:
    """
    A main decorator to apply a function recursively to elements within various data structures.
    """

    def wrapper(arg, *args, **kwargs):
        if isinstance(arg, (tuple, list, set, np.ndarray)):
            return handle_collection(wrapper)(arg, *args, **kwargs)
        elif isinstance(arg, dict):
            return handle_dicts(wrapper)(arg, *args, **kwargs)  # type: ignore
        elif "pandas" in sys.modules and isinstance(arg, (pd.Series, pd.DataFrame)):
            return handle_pandas(wrapper)(arg, *args, **kwargs)
        else:
            return func(arg, *args, **kwargs)

    return cast(F, wrapper)


def sanitized_input(
    func=None,
    /,
    *,
    accept_types=None,
    accept_none=False,
    conv_none=False,
    conv_nan=False,
    null_value=0,
):
    """
    A decorator to handle single values (int, float, str, etc..).
    """
    if accept_types is None:
        accept_types = [int, float, np.integer, np.number, complex, str, bool, bytes]
    else:
        if np.integer in accept_types and int not in accept_types:
            accept_types.append(int)
        if np.number in accept_types and float not in accept_types:
            accept_types.append(float)

    if accept_none and not isinstance(null_value, tuple(accept_types)):
        null_value = None
    else:
        for t in accept_types:
            try:
                null_value = t(null_value)
                break
            except (TypeError, ValueError):
                continue

    def is_valid_float(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def decorator(func):
        def wrapper(arg, *args, **kwargs):
            # if not isinstance(arg, str) and hasattr(arg, "__iter__"):
            #     return func(arg, *args, **kwargs)
            if arg is None and (accept_none or conv_none):
                arg = null_value if conv_none else None
                return func(arg, *args, **kwargs)
            if isinstance(arg, tuple(accept_types)):
                # the following sanitizes the value which is known to be of an acceptable type
                if isinstance(arg, (int, np.integer)):
                    arg = int(arg)
                elif isinstance(arg, (float, np.number)):
                    arg = float(arg) if not (conv_nan and np.isnan(arg)) else null_value
                return func(arg, *args, **kwargs)

            # last is specific to string
            if isinstance(arg, bytes):
                arg = arg.decode("ascii", errors="ignore")
                if arg not in str(arg):
                    encodes = [
                        "utf_8",
                        "utf_16",
                        "utf_32",
                        "utf_7",
                        "cp037",
                        "cp437",
                        "utf_8_sig",
                    ]
                    for enc in encodes:
                        try:
                            arg = arg.decode(enc)  # type: ignore
                            break
                        except UnicodeDecodeError:
                            continue
                if isinstance(arg, tuple(accept_types)):
                    return func(arg, *args, **kwargs)
            # both attempt and force conversion use the following checks but if force is False, it will raise an error
            if any(t in accept_types for t in [int, float, np.integer, np.number]):
                # these are numeric types which can be converted
                if isinstance(arg, str):
                    if arg.isnumeric():
                        arg = int(arg)
                    elif is_valid_float(arg):
                        arg = float(arg)
                elif isinstance(arg, bool):
                    arg = int(arg)
                elif isinstance(arg, complex):
                    arg = abs(arg)
                # backup in case preferred conversion (int or float) is not acceptable
                if not isinstance(arg, tuple(accept_types)) and isinstance(
                    arg, (int, float, np.integer, np.number)
                ):
                    arg = float(arg) if isinstance(arg, (int, np.integer)) else int(arg)

            return func(arg, *args, **kwargs)

        return wrapper

    if func is None:
        return decorator

    return decorator(func)


def sanitized_after_recursion(
    func=None,
    /,
    *,
    accept_types=None,
    accept_none=False,
    conv_none=False,
    conv_nan=False,
    null_value=0,
):
    """
    A main decorator to apply a function recursively to elements within various data structures.
    Combines handle_collection, handle_dicts, handle_pandas, and sanitized_value.
    """

    def decorator(func):
        def wrapper(arg, *args, **kwargs):
            # Primary wrapper: Handles recursive traversal of data structures
            if isinstance(arg, (tuple, list, set, np.ndarray)):
                return handle_collection(wrapper)(arg, *args, **kwargs)
            elif isinstance(arg, dict):
                return handle_dicts(wrapper)(arg, *args, **kwargs)  # type: ignore
            elif "pandas" in sys.modules and isinstance(arg, (pd.Series, pd.DataFrame)):
                return handle_pandas(wrapper)(arg, *args, **kwargs)
            else:
                # Secondary wrapper: Applies the sanitized_value decorator to individual values
                @sanitized_input(
                    accept_types=accept_types,
                    accept_none=accept_none,
                    conv_none=conv_none,
                    conv_nan=conv_nan,
                    null_value=null_value,
                )
                def inner(sub_arg):
                    return func(sub_arg, *args, **kwargs)

                return inner(arg)

        return wrapper

    if func is None:
        return decorator

    return decorator(func)


def raise_error_on_invalid(
    func=None,
    /,
    *,
    accept_types=None,
):
    """
    A decorator to handle single values (int, float, str, etc..).
    """
    if accept_types is None:
        accept_types = [int, float, np.integer, np.number, complex, str, bool, bytes]

    def decorator(func):
        def wrapper(arg, *args, **kwargs):
            if isinstance(arg, tuple(accept_types)):
                return func(arg, *args, **kwargs)
            else:
                raise TypeError(
                    f"Argument of type {type(arg)} is not acceptable. Acceptable types: {accept_types}"
                )

        return wrapper

    if func is None:
        return decorator

    return decorator(func)


def sanitized_after_recursion_w_error(
    func=None,
    /,
    *,
    accept_types=None,
    accept_none=False,
    conv_none=False,
    conv_nan=False,
    null_value=0,
):
    """
    A main decorator to apply a function recursively to elements within various data structures with a final type check.
    Combines handle_collection, handle_dicts, handle_pandas, sanitized_input, and raise_error_on_invalid.
    """

    def decorator(func):
        """
        A main decorator to apply a function recursively to elements within various data structures.
        """

        def wrapper(arg, *args, **kwargs):
            # Primary wrapper: Handles recursive traversal of data structures
            if isinstance(arg, (tuple, list, set, np.ndarray)):
                return handle_collection(wrapper)(arg, *args, **kwargs)
            elif isinstance(arg, dict):
                return handle_dicts(wrapper)(arg, *args, **kwargs)  # type: ignore
            elif "pandas" in sys.modules and isinstance(arg, (pd.Series, pd.DataFrame)):
                return handle_pandas(wrapper)(arg, *args, **kwargs)
            else:
                # Secondary wrapper: Applies the sanitized_value decorator to individual values
                @raise_error_on_invalid(accept_types=accept_types)
                @sanitized_input(
                    accept_types=accept_types,
                    accept_none=accept_none,
                    conv_none=conv_none,
                    conv_nan=conv_nan,
                    null_value=null_value,
                )
                def inner(sub_arg):
                    return func(sub_arg, *args, **kwargs)

                return inner(arg)

        return wrapper

    if func is None:
        return decorator

    return decorator(func)


# %% Data Treatment Functions
def sanitize_types(data, decode_bytes=True):
    """
    Recursively sanitizes the types of elements in the input data.

    This function processes the input data, which can be a dictionary, list, tuple,
    numpy array, or basic data type, and converts its elements to standard Python types.
    Optionally, it can decode byte strings using various encodings.

    Parameters:
    data (any): The input data to sanitize. Can be a dictionary, list, tuple, numpy array,
                or basic data type (int, float, bool, complex, str, bytes).
    decode_bytes (bool, optional): If True, attempts to decode byte strings using various
                                   encodings. Default is True.

    Returns:
    any: The sanitized data with elements converted to standard Python types.
    """
    res = data

    if isinstance(data, dict):
        res = {k: sanitize_types(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple, np.ndarray)):
        res = [sanitize_types(d) for d in data]
        if len(res) == 1:
            res = res[0]

    # if a type that can be sanitized
    if isinstance(data, (int, np.integer)):
        res = int(data)
    elif isinstance(data, (float, np.number)):
        res = float(data)
    elif isinstance(data, bool):
        res = bool(data)
    elif isinstance(data, complex):
        res = complex(data)
    elif isinstance(data, str):
        res = str(data)
    elif isinstance(data, bytes):
        res = bytes(data)
        if decode_bytes:
            encodes = [
                "utf_8",
                "utf_16",
                "utf_32",
                "utf_7",
                "cp037",
                "cp437",
                "utf_8_sig",
            ]
            res = data.decode("ascii")
            n = 0
            while n < len(encodes) and res not in str(data):
                try:
                    res = data.decode(encodes[n])
                    n += 1
                except UnicodeDecodeError:
                    n += 1

    return res


# Archive

# def handle_dicts(func):
#     """
#     A decorator to handle dictionaries.
#     Recursively applies the decorated function to each value.
#     Assumes that the target data structure is of non-dict types.
#     """

#     def wrapper(arg, *args, **kwargs):
#         if isinstance(arg, dict):
#             return {
#                 k: (
#                     wrapper(t, *args, **kwargs)
#                     if not (isinstance(t, float) and np.isnan(t))
#                     else float("nan")
#                 )
#                 for k, t in arg.items()
#             }
#         else:
#             return func(arg, *args, **kwargs)

#     return wrapper
