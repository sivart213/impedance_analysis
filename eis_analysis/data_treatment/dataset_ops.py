# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""

import re
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast, overload
from difflib import get_close_matches
from itertools import chain
from collections.abc import Callable, Iterator

import pandas as pd

if TYPE_CHECKING:
    from ..utils._typings import (
        PandasKey,
        PandasMap,
        PandasKeys,
        PandasLike,
        IPandasKeys,
        MIPandasKeys,
    )

try:
    from ..utils.decorators import handle_dicts, handle_subdicts
except ImportError:
    from eis_analysis.utils.decorators import handle_dicts, handle_subdicts


IsFalse: TypeAlias = Literal[False]
IsTrue: TypeAlias = Literal[True]


class CachedColumnSelector:
    """
    Class to manage and validate DataFrame columns using a cache.
    """

    def __init__(self, initial_columns: list | tuple | set | None = None):
        self._cache = []
        # Initialize the cache with default column names
        self.cache = initial_columns

    @property
    def cache(self) -> list[list[PandasKey]]:
        """
        Return the cache list.
        """
        return self._cache

    @cache.setter
    def cache(self, value: Any):
        if isinstance(value, (tuple, list, set)):
            self._cache.append(list(value))

    @overload
    def get_valid_columns(
        self,
        df: pd.DataFrame,
        reducing_keys: ...,
        get_keys: IsTrue,
        cutoff: float = 0.6,
    ) -> list: ...

    @overload
    def get_valid_columns(
        self,
        df: pd.DataFrame,
        reducing_keys: str | list[str] | None = None,
        *,
        get_keys: IsTrue,
        cutoff: float = 0.6,
    ) -> list[str]: ...

    @overload
    def get_valid_columns(
        self,
        df: pd.DataFrame,
        reducing_keys: str | list[str] | tuple[str, ...] | None = None,
        get_keys: IsFalse = False,
        cutoff: float = 0.6,
    ) -> pd.DataFrame: ...

    def get_valid_columns(
        self,
        df: pd.DataFrame,
        reducing_keys: str | list[str] | tuple[str, ...] | None = None,
        get_keys: bool = False,
        cutoff: float = 0.6,
    ) -> pd.DataFrame | list:
        """
        Check if the DataFrame contains a valid set of columns.
        If not, call `get_valid_keys` to find a valid set and update the cache.

        Args:
            df (pd.DataFrame): The DataFrame to validate.
            reducing_keys (tuple/list, optional): Keys to iteratively reduce the DataFrame columns.
            get_keys (bool, optional): If True, return the valid keys instead of updating the DF.

        Returns:
            pd.DataFrame or list: Updated DataFrame or valid keys, depending on `get_keys`.
        """

        def recombine_keys(keys, common_keys: list[str]) -> PandasKeys:
            """
            Recombine keys with the preceding common_keys.
            """
            if not common_keys:
                return keys
            return [
                tuple(common_keys) + (key if isinstance(key, tuple) else (key,)) for key in keys
            ]

        key_len = len(self._cache[0])
        used_r_keys = []

        columns = set(df.columns.tolist())
        for col_set in self._cache:
            if set(col_set).issubset(columns):
                if get_keys:
                    return col_set
                df = df[col_set]
                df.columns = self._cache[0]
                return df

        # Handle MultiIndex columns with reducing_keys
        if reducing_keys is not None and isinstance(df.columns, pd.MultiIndex):
            if isinstance(reducing_keys, str):
                reducing_keys = [reducing_keys]
            used_r_keys: list[str] = []
            for key in reducing_keys:
                if key in df.columns and len(df[key].columns) >= key_len:
                    df = df[key]  # type: ignore
                    used_r_keys.append(key)
                    if not isinstance(df.columns, pd.MultiIndex):
                        break
            columns = set(df.columns.tolist())

        # If no valid set is found, call `get_valid_keys`

        n_cache = 0
        valid_keys = []
        while n_cache < len(self._cache) and not valid_keys:
            valid_keys = get_valid_keys(df, self._cache[n_cache], all_or_none=True, cutoff=cutoff)
            valid_keys = (
                [] if not valid_keys or len(valid_keys) > len(set(valid_keys)) else valid_keys
            )
            n_cache += 1

        if valid_keys:
            self.cache = recombine_keys(valid_keys, used_r_keys)
            if get_keys:
                return self._cache[-1]
            df = df[valid_keys]
            df.columns = self._cache[0]
            return df
        elif used_r_keys or cutoff >= 0.2:
            return self.get_valid_columns(df, None, get_keys=get_keys, cutoff=cutoff - 0.1)

        # Return None if no valid columns are found
        if get_keys:
            return []
        return df[[]]

    def get_valid_keys_from_list(
        self,
        columns: list,
        cutoff: float = 0.4,
    ) -> list:
        """
        Validate a list of column names and return the valid keys.

        Args:
            columns (list): List of column names (strings or tuples) to validate.
            cutoff (float, optional): Fuzzy matching cutoff for `get_valid_keys`.

        Returns:
            list: Valid keys from the list of column names.
        """
        # Check against cached column sets
        columns_set = set(columns)
        for col_set in self.cache:
            if set(col_set).issubset(columns_set):
                return col_set  # No modification logic for lists

        # If no valid set is found, call `get_valid_keys`

        n_cache = 0
        valid_keys = []
        while n_cache < len(self._cache) and not valid_keys:
            valid_keys = get_valid_keys(
                columns,
                self._cache[n_cache],
                all_or_none=True,
                cutoff=cutoff,
            )
            valid_keys = (
                [] if not valid_keys or len(valid_keys) > len(set(valid_keys)) else valid_keys
            )
            n_cache += 1

        # valid_keys = get_valid_keys(columns, self._cache[0], all_or_none=True, cutoff=cutoff)
        if valid_keys:
            self.cache.append(valid_keys)
            return valid_keys

        # Return None if no valid keys are found
        return []

    def reset_cache(self, initial_columns: list[PandasKey] | tuple[PandasKey] | None = None):
        """
        Reset the cache to the initial columns.

        Args:
            initial_columns (tuple/list, optional): Initial columns to set in the cache.
        """

        if not self._cache:
            pass
        elif initial_columns is not None:
            self._cache = []
            self.cache = initial_columns
        else:
            self._cache = [self._cache[0]]


@overload
def get_valid_keys(
    source: Any,
    target: str | list[str] | tuple[str, ...],
    as_list: IsTrue = True,
    slice_multi: bool = ...,
    all_or_none: bool = ...,
    cutoff: float = ...,
) -> list[PandasKey]: ...
@overload
def get_valid_keys(
    source: Any,
    target: str | list[str] | tuple[str, ...],
    as_list: IsFalse,
    slice_multi: bool = ...,
    all_or_none: bool = ...,
    cutoff: float = ...,
) -> PandasMap: ...
def get_valid_keys(
    source: Any,
    target: str | list[str] | tuple[str, ...],
    as_list: bool = True,
    slice_multi: bool = False,
    all_or_none: bool = False,
    cutoff: float = 0.6,
) -> list[PandasKey] | PandasMap:
    """
    [DEPRECATED] Returns the correct column names for the DataFrame based on the target.

    This function is retained for compatibility with legacy code and may be removed in the future.
    For new code, use the KeyMatcher class directly.

    Args:
        source (pd.DataFrame, pd.Index or pd.MultiIndex): The DataFrame or DataFrame columns.
        target (str, list of str, or list of tuple of str): The target column names to match.
        slice_multi (bool): If True, uses the treated target to slice.
        as_list (bool): If True, returns a list of matched columns.
        all_or_none (bool): If True, returns [] if not all targets are matched.
        cutoff (float): Fuzzy matching cutoff for get_close_matches.

    Returns:
        list: A list of matched column names.
    """
    return KeyMatcher(source, target, cutoff, slice_multi, as_list, all_or_none).result


class KeyMatcher:
    """
    Class to match target column names to DF columns, supporting both flat and MultiIndex columns.
    Handles exact and fuzzy matching, and prevents duplicates using a set of prior matches.
    """

    def __init__(
        self,
        source: Any | None = None,
        target: str | list[str] | tuple[str, ...] | None = None,
        cutoff: float | None = None,
        slice_multi: bool | None = None,
        as_list: bool | None = None,
        all_or_none: bool | None = None,
    ):
        """
        Initialize the KeyMatcher and immediately perform matching.

        Args:
            source (pd.DataFrame, pd.Index, pd.MultiIndex, list): The columns or DF to match.
            target (str, list, tuple): The target column names to match.
            cutoff (float): Fuzzy matching cutoff for get_close_matches.
            slice_multi (bool): If True, enables slicing for MultiIndex columns.
            as_list (bool): If True, returns a list of matched columns.
            all_or_none (bool): If True, returns [] if not all targets are matched.
        """
        self.priors = set()
        self.matched_mapping: PandasMap = {}
        self.cutoff = 0.6
        self._targets = set()
        self.info = {}
        self.process_info = dict(
            slice_multi=False,
            as_list=True,
            all_or_none=False,
        )
        self.keys = [[], []]
        self._result = {}
        # as_list=True,
        # all_or_none=False

        self._preprocess(source, target, cutoff, slice_multi, as_list, all_or_none)

    @property
    def result(self) -> list[PandasKey] | PandasMap:
        """Return result of matching."""
        if not self._result:
            self.get_valid_keys()
        return cast(list[PandasKey] | PandasMap, self._result)

    @property
    def result_list(self) -> list[PandasKey]:
        """Return result of matching."""
        if not self.matched_mapping:
            self.get_valid_keys()
        return self._postprocess(self.process_info["all_or_none"])

    @property
    def result_dict(self) -> PandasMap:
        """Return result of matching."""
        if not self.matched_mapping:
            self.get_valid_keys()
        return self.matched_mapping

    def _parse_target(self, target: str | list[str] | tuple[str, ...] | None):
        """
        Parse the target input to ensure it's a list of strings or tuples.
        """
        if target is None:
            return
        if isinstance(target, str):
            target = [target]

        if isinstance(target, (tuple, list)):
            try:
                self._targets = set(target)
            except TypeError:
                self._targets = set(str(t) for t in target)
        else:
            raise ValueError("Invalid target type. Must be str, list of str/tuples.")

        self.info.update({"target": target})

    def _parse_source(self, source: Any | None):
        """
        Parse the source input
        """
        if source is None:
            return
        # Always convert columns to a list
        if isinstance(source, pd.DataFrame):
            columns = source.columns.to_list()
        elif isinstance(source, pd.Index):
            columns = source.to_list()
        elif isinstance(source, (list, tuple)):
            columns = list(source)
        else:
            raise ValueError(
                "Invalid columns type. Must be pd.DataFrame, pd.Index, or list/tuple."
            )

        is_multi = isinstance(columns[0], tuple) and not any(
            isinstance(col, (str, int, float)) for col in columns
        )

        self.info.update(
            {
                "columns": columns,
                "c_len": len(columns),
                "levels": len(columns[0]) if is_multi else 1,
                "is_multi": is_multi,
            }
        )

    def _preprocess(
        self,
        source: Any | None,
        target: str | list[str] | tuple[str, ...] | None,
        cutoff: float | None,
        slice_multi: bool | None,
        as_list: bool | None,
        all_or_none: bool | None,
    ):
        """
        Initialize the KeyMatcher and immediately perform matching.
        """
        if source is not None:
            self._parse_source(source)

        # Normalize targets
        if target is not None:
            self._parse_target(target)

        if isinstance(cutoff, (float, int)):
            self.cutoff = cutoff

        if isinstance(slice_multi, bool):
            self.process_info["slice_multi"] = slice_multi

        if isinstance(as_list, bool):
            self.process_info["as_list"] = as_list

        if isinstance(all_or_none, bool):
            self.process_info["all_or_none"] = all_or_none

        if len(self.info) == 5 and not self.keys[0]:
            # Prepare keys for matching
            if self.info["is_multi"]:
                # keys[0]: full multiindex as joined string;
                # keys[1]: all single-level keys as strings
                self.keys = [
                    [".".join(map(str, col)) for col in self.info["columns"]],
                    [str(col) for col in chain.from_iterable(self.info["columns"])],
                ]
                self.keys[0].extend([k.lower() for k in self.keys[0]])
                self.keys[1].extend([k.lower() for k in self.keys[1]])

            else:
                self.keys = [[str(col) for col in self.info["columns"]], []]
                self.keys[0].extend([col.lower() for col in self.info["columns"]])

                self.process_info["slice_multi"] = False

    def _postprocess(self, all_or_none: bool) -> list[PandasKey]:
        """
        Post-process the result to return as list, handle slice_multi and all_or_none.
        """
        # Use the original target order if provided, else sorted keys
        matched_columns: list[PandasKey] = [
            self.matched_mapping[x] for x in self.info["target"] if x in self.matched_mapping
        ]
        if self.process_info["slice_multi"]:
            matched_columns = list(
                dict.fromkeys(
                    chain.from_iterable(
                        [value] if not isinstance(value, list) else value
                        for value in matched_columns
                    )
                )
            )
        if all_or_none and len(matched_columns) != len(self.info["target"]):
            return []

        return matched_columns

    def _get_mat_value(self, mat: int) -> PandasKey:
        """
        Get the column value for a flat index.
        """
        return self.info["columns"][mat % self.info["c_len"]]

    def _get_multi_mat_value(self, mat: int) -> list[PandasKey]:
        """
        Get all columns matching a specific level value in a MultiIndex.
        """
        match_index = mat % self.info["c_len"]
        level = match_index % self.info["levels"]
        col_key = self.info["columns"][match_index // self.info["levels"]][level]
        return [col for col in self.info["columns"] if col[level] == col_key]

    def _is_in_filter(
        self,
        target: str,
        keys: list[str],
        value_getter: Callable,
        cutoff: float,
    ) -> tuple[list, Callable, float]:
        """
        Helper for short target logic in _find_best_match.
        Returns updated keys, value_getter, and cutoff.
        """

        def _value_getter(vg, oi):
            def wrapped(idx):
                return vg(oi[idx])

            return wrapped

        filtered = [
            (i, key) for i, key in enumerate(keys) if target in key or target.lower() in key
        ]
        if filtered:
            orig_indices, filtered_keys = zip(*filtered)
            f_keys = list(filtered_keys)
            min_key_lens = (len(target), min(len(k) for k in keys))
            max_possible = 2 * min(min_key_lens) / sum(min_key_lens)
            f_cutoff = (1 - (1 - max_possible) * (1 - cutoff) / 2) * min(max_possible, cutoff)
            return f_keys, _value_getter(value_getter, orig_indices), f_cutoff

        return keys, value_getter, cutoff

    def _find_best_match(
        self,
        target: str,
        keys: list[str],
        value_getter: Callable,
    ) -> Any:
        """
        Helper to find the best match for a target in keys using value_getter.
        Returns the first value not in self.priors, or the first match if all are taken.

        Args:
            target (str): The target string to match.
            keys (list): List of candidate keys (strings).
            value_getter (callable): Function to get the column value from a key index.

        Returns:
            The best-matched column value, or None if no match is found.
        """

        cutoff = self.cutoff
        val_func = value_getter
        if len(target) <= 3:
            keys, val_func, cutoff = self._is_in_filter(target, keys, value_getter, cutoff)

        matches = self.get_close_matches_caseless(target, keys, n=5, cutoff=cutoff)

        if matches:
            val = next(
                (v for mat in matches if (v := val_func(keys.index(mat))) not in self.priors),
                val_func(keys.index(matches[0])),
            )
            self.priors.add(val)
            return val
        return None

    def match_exact(self) -> PandasMap:
        """
        Match targets to exact column names.

        Returns:
            dict: Mapping of target to matched column.
        """
        # Perform exact matching once in __init__ for both flat and multiindex
        exact_matches = self._targets.intersection(self.info["columns"])
        self.matched_mapping = dict(zip(exact_matches, exact_matches))
        self._targets.difference_update(exact_matches)

        return self.matched_mapping

    def match_flat(self) -> PandasMap:
        """
        Match targets to flat (single-level) columns using fuzzy matching.

        Returns:
            dict: Mapping of target to matched column.
        """
        self.match_exact()
        for target in sorted(self._targets, key=lambda x: len(str(x)), reverse=True):
            val = self._find_best_match(str(target), self.keys[0], self._get_mat_value)
            if val is not None:
                self.matched_mapping[target] = val

        return self.matched_mapping

    def match_multi(self) -> PandasMap:
        """
        Match targets to MultiIndex columns using fuzzy matching.
        Handles both full multiindex keys and single-level keys.

        Returns:
            dict: Mapping of target to matched column(s).
        """
        self.match_exact()
        slice_multi_res = {}
        for target in sorted(self._targets, key=lambda x: len(str(x)), reverse=True):
            # Branch 1: tuple/list or stringified tuple (full MultiIndex key)
            modify = False
            if isinstance(target, (tuple, list)) or (
                (modify := str(target)[0] in "([" and "," in str(target))
            ):
                mod_target = tuple(re.split(r"\s*,\s*", str(target)[1:-1])) if modify else target
                val_v1 = self._find_best_match(
                    ".".join(map(str, mod_target)), self.keys[0], self._get_mat_value
                )
                if val_v1 is not None:
                    self.matched_mapping[target] = val_v1
            # Branch 2: single-level key (matches all columns at that level)
            else:
                val_v2 = self._find_best_match(
                    str(target), self.keys[1], self._get_multi_mat_value
                )
                if val_v2 is not None:
                    if len(val_v2) == 1:
                        self.matched_mapping[target] = val_v2[0]
                    else:
                        slice_multi_res[target] = val_v2

        if slice_multi_res:
            if self.process_info["slice_multi"]:
                self.matched_mapping.update(slice_multi_res)
            else:
                self.matched_mapping.update(
                    {
                        key: next(
                            (v for v in value if v not in self.matched_mapping.values()), value[0]
                        )
                        for key, value in slice_multi_res.items()
                    }
                )
        return self.matched_mapping

    # @overload
    # def get_valid_keys(
    #     source: Any | None ,
    #     target: list | None,
    #     as_list: IsTrue = True,
    #     slice_multi: bool | None = ...,
    #     all_or_none: bool | None = ...,
    #     cutoff: float | None = ...,
    # ) -> list[PandasKey]: ...
    # @overload
    # def get_valid_keys(
    #     source: Any | None ,
    #     target: list | None,
    #     as_list: IsFalse,
    #     slice_multi: bool | None = ...,
    #     all_or_none: bool | None = ...,
    #     cutoff: float | None = ...,
    # ) -> PandasMap: ...

    def get_valid_keys(
        self,
        source: Any | None = None,
        target: list | None = None,
        as_list: bool | None = None,
        slice_multi: bool | None = None,
        all_or_none: bool | None = None,
        cutoff: float | None = None,
    ) -> list[PandasKey] | PandasMap:
        """
        Initialize the  and immediately perform matching.

        Args:
            source (pd.DataFrame, pd.Index, pd.MultiIndex, list): The columns or DF to match.
            target (str, list, tuple): The target column names to match.
            cutoff (float): Fuzzy matching cutoff for get_close_matches.
            slice_multi (bool): If True, enables slicing for MultiIndex columns.
            as_list (bool): If True, returns a list of matched columns.
            all_or_none (bool): If True, returns [] if not all targets are matched.
        """
        self._preprocess(source, target, cutoff, slice_multi, as_list, all_or_none)

        if len(self.info) != 5:
            raise ValueError(
                "KeyMatcher not fully initialized. "
                "Please provide a source/target list at initialization or via get_valid_keys()."
            )

        if self.info["is_multi"]:
            self.match_multi()
        else:
            self.match_flat()

        if self.process_info["as_list"]:
            self._result = self._postprocess(self.process_info["all_or_none"])
        else:
            self._result = self.matched_mapping

        return self._result

    @staticmethod
    def get_close_matches_caseless(
        target: str,
        possibilities: list,
        n: int = 5,
        cutoff: float = 0.6,
    ) -> list[str]:
        """
        Return close matches to 'target' in 'possibilities'.
        Ignores case of target if first pass fails.

        Args:
            target (str): The string to match.
            possibilities (list): List of candidate strings.
            n (int): Maximum number of close matches to return.
            cutoff (float): Minimum similarity ratio.

        Returns:
            list: List of close matches.
        """
        results = get_close_matches(target, possibilities, n, cutoff)
        return results or get_close_matches(target.lower(), possibilities, n, cutoff)


# used by convert_mfia_data and is a primary function


@handle_dicts
def modify_sub_dfs(
    data: pd.DataFrame,
    *functions: Callable[..., Any] | tuple[Any, ...],
    args: tuple[Any, ...] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Applies a series of functions to a DataFrame or nested DataFrames within a dictionary.

    This function takes a DataFrame or a dictionary containing DataFrames and applies
    a series of functions to each DataFrame. Each function can be provided as a callable
    or as a tuple containing a callable and its arguments. The function modifies the
    DataFrame in place and returns the modified DataFrame.

    Parameters:
    data pd.DataFrame: The DataFrame to modify.
    *functions (callable or tuple): A series of functions/tuples of functions and their arguments
                                to apply to the DataFrame(s).

    Returns:
    pd.DataFrame: The modified DataFrame .
    """
    if args is None:
        args = tuple([])
    res = None
    for f in functions:
        if callable(f):
            res = f(data, *args, **kwargs)
        elif isinstance(f[-1], dict):
            res = f[0](data, *(f[1:-1] or args), **{**kwargs, **f[-1]})
        else:
            res = f[0](data, *(f[1:] or args), **kwargs)
        data = res if isinstance(res, type(data)) else data
    return data


@overload
def sanitize_df_col(args: MIPandasKeys, as_index: IsTrue) -> pd.MultiIndex: ...
@overload
def sanitize_df_col(args: IPandasKeys | PandasMap, as_index: IsTrue) -> pd.Index: ...
@overload
def sanitize_df_col(args: Any, as_index: IsFalse = False) -> pd.DataFrame: ...


def sanitize_df_col(
    args: PandasLike | IPandasKeys | MIPandasKeys | PandasMap,
    as_index: bool = False,
) -> pd.DataFrame | pd.MultiIndex | pd.Index:
    """
    Converts various input types (e.g., MultiIndex, Index, list, tuple) into a DataFrame
    or optionally back into a MultiIndex/Index.

    Parameters:
    args: Input to convert (e.g., pd.MultiIndex, pd.Index, list, tuple).
    as_index (bool): If True, converts the result back into a MultiIndex or Index.

    Returns:
    pd.DataFrame or pd.MultiIndex/pd.Index: Converted DataFrame or Index.
    """
    if isinstance(args, pd.DataFrame):
        args = args.columns
    if isinstance(args, (tuple, list)):
        # Check if it's a list of tuples (MultiIndex-like)
        if all(isinstance(item, tuple) for item in args):
            result = pd.DataFrame(args)
        # Check if it's a list of strings (flat index-like)
        elif all(isinstance(item, str) for item in args):
            result = pd.DataFrame({0: args})
        else:
            raise ValueError(
                "Invalid list/tuple format. Expected a list of tuples (MultiIndex-like)"
                " or a list of strings (flat index-like)."
            )
    elif isinstance(args, dict):
        result = pd.DataFrame.from_dict(args, orient="index").reset_index(drop=True)
    elif isinstance(args, pd.Series):
        result = args.to_frame()
    elif isinstance(args, (pd.Index, pd.MultiIndex)):
        result = args.to_frame(index=False)
    else:
        try:
            if isinstance(args, str):
                raise ValueError("strings are not supported")
            args = list(args)
            return sanitize_df_col(args, as_index=as_index)
        except Exception as exc:
            # Handle cases where args is not iterable
            raise ValueError("Invalid arguments. Expected an interable.") from exc

    # Ensure output is disconnected from the original object
    result = result.copy(deep=True)

    # Convert back to MultiIndex or Index if requested
    if as_index:
        if result.shape[1] > 1:
            return pd.MultiIndex.from_frame(result)
        return pd.Index(result[0])

    return result


# Not actively used
def sort_column_levels(
    df: pd.DataFrame,
    level_index: int | list | None = None,
    return_index: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, list[int]]:
    """
    Sorts the levels of a DataFrame's columns based on the number of unique values in each level
    or a specified level index.

    Parameters:
    df (pd.DataFrame): The DataFrame to sort.
    level_index (int or list, optional): Specific level(s) to sort by.
        Default: sorts by uniqueness.
    return_index (bool): If True, returns the sorted index instead of the sorted DataFrame.

    Returns:
    pd.DataFrame or pd.Index: The sorted DataFrame or index.
    """
    # Convert columns to a DataFrame for consistent handling
    columns_df = sanitize_df_col(df)

    # Handle single-level columns
    if len(columns_df.columns) == 1:
        if return_index:
            return df, [0]
        return df

    sorted_columns = []

    # Sort the columns by the number of unique values in each level
    if level_index is None:
        unique_counts = columns_df.nunique()
        sorted_columns = unique_counts.sort_values().index
        sorted_df = pd.DataFrame(columns_df[sorted_columns])
    else:
        level_index = [level_index] if not isinstance(level_index, list) else level_index
        try:
            if any(idx not in columns_df.columns for idx in level_index):
                sorted_df = pd.DataFrame(columns_df.iloc[:, level_index])
            else:
                sorted_df = pd.DataFrame(columns_df[level_index])
        except Exception as exc:
            raise ValueError(
                f"Invalid level index in sort_column_levels: {level_index}. Error: {exc}"
            ) from exc

    if isinstance(df, pd.DataFrame):
        cache = [col if col in df.columns.names else None for col in sorted_df.columns]
        df = df.copy(deep=True)
        df.columns = pd.MultiIndex.from_frame(sorted_df, names=cache)
    if return_index:
        return df, list(sorted_columns)
    return df


def get_most_common_keys(
    columns: PandasLike | PandasKeys | PandasMap,
    context: PandasLike | PandasKeys | PandasMap | None = None,
) -> Iterator[Any]:
    """
    Iteratively selects the most common key at each level of a MultiIndex,
    narrowing down the options for subsequent levels. Uses Pandas indexing
    and sorting methods for improved performance.

    Parameters:
    columns (pd.MultiIndex): The MultiIndex columns to analyze.
    context (pd.MultiIndex, optional): A subset of columns to limit the options for the most common
                                        key. Defaults to None, in which case it uses the
                                        original columns.

    Yields:
    tuple: The most common key for the current level.
    """
    # Convert MultiIndex to a DataFrame for easier manipulation
    columns_df = sanitize_df_col(columns)
    context_df = sanitize_df_col(context) if context is not None else columns_df.copy(deep=True)

    if all(columns_df.columns != context_df.columns):
        raise ValueError("Columns and context must have the number of levels")

    for col in columns_df.columns:
        # Count occurrences of each value in the current col of the original columns
        level_counts = columns_df[col].value_counts()

        # Filter the context to the current col
        context_values = context_df[col].unique()

        # Find the most common key within the context
        most_common_key = level_counts.loc[context_values].idxmax()

        # Yield the most common key for this col
        yield most_common_key

        # Narrow down the columns and context for the next col
        columns_df = columns_df[columns_df[col] == most_common_key].drop(columns=[col])
        context_df = context_df[context_df[col] == most_common_key].drop(columns=[col])


@overload
def evaluate_df_index_key(
    col: str,
    clean_map: dict[str, str],
    sep: str | list | None = None,
) -> str: ...
@overload
def evaluate_df_index_key(
    col: tuple[str, ...],
    clean_map: dict[tuple[str, ...], tuple[str, ...]],
    sep: str | list | None = None,
) -> tuple[str, ...]: ...


def evaluate_df_index_key(
    col: str | tuple[str, ...],
    clean_map: dict[Any, Any],
    sep: str | list | None = None,
) -> str | tuple[str, ...]:
    """
    Helper function to evaluate and complete a tuple based on the base columns and common keys.

    Parameters:
    col (tuple): The tuple to evaluate.
    clean_map (dict): Mapping of clean column names.
    sep (str or list, optional): Separator(s) to use for splitting the string.

    Returns:
    tuple: The completed tuple.
    """
    if isinstance(col, str):
        return col
    # Reverse both col and completed_col for simplified reverse iteration
    eval_col = list(col)[::-1]
    base_col_df = sanitize_df_col(clean_map)
    completed_col: list[str] = list(get_most_common_keys(clean_map))[::-1]

    base_col_df = base_col_df[base_col_df.columns[::-1]].T.reset_index(drop=True).T

    # Track available indices in completed_col
    available_indices = list(range(len(completed_col)))

    if sep and isinstance(sep, str):
        pattern = re.sub(r"\\+", r"\\", re.escape(sep))
        sep = [pattern]
    elif isinstance(sep, (list, tuple)):
        sep = [re.sub(r"\\+", r"\\", re.escape(s)) for s in sep]
        pattern = (
            "("
            + "|".join(f"{s}(?!.*[{''.join(sep[:i])}])" if i > 0 else s for i, s in enumerate(sep))
            + ")"
        )
    else:
        sep = [r"\.", r"\-", "/", "_"]
        # sep = r"(\.|-(?!.*\.)|\/(?!.*[-.])|_(?!.*[\/\-.]))"
        pattern = r"(\.|\-(?!.*[\.])|/(?!.*[\.\-])|_(?!.*[\.\-/]))"

    highest_sep_priority_used = len(sep) - 1
    # ignored_split = []
    # Use a while loop instead of recursion
    idx = 0
    while idx < len(eval_col):
        col_str = eval_col[idx]

        # Check if `.`, `-`, `/`, or `_` is in the string
        # Then if either of the split parts are in the base_col_df
        match = re.search(pattern, col_str)
        if match:
            split_parts = [s for s in re.split(re.escape(match.group(1)), col_str) if s]
            #
            if len(split_parts) > 1:
                sep_priority = sep.index(match.group(1))
                if len(split_parts) - 1 <= len(completed_col) - len(eval_col):
                    if any(part in base_col_df.values for part in split_parts):
                        # Update the tuple with split parts
                        eval_col = eval_col[:idx] + split_parts[::-1] + eval_col[idx + 1 :]
                        # Reset idx to re-evaluate the current position
                        col_str = eval_col[idx]
                    # else:
                    #     ignored_split.append((sep_priority, col_str))
                elif sep_priority < highest_sep_priority_used:
                    if any(part in base_col_df.values for part in split_parts):
                        # Update the tuple with split parts
                        i = col.index(col_str)
                        new_col = list(col)[:i] + split_parts + list(col)[i + 1 :]
                        return evaluate_df_index_key(tuple(new_col), clean_map, sep=sep)
                    # else:
                    #     ignored_split.append((sep_priority, col_str))

        # Determine the index (level_idx) where col_str should be placed
        # Note: If col_str is not in base_col_df.values, level_idx will default to 0
        if (level_idx := base_col_df.isin([col_str]).any().idxmax()) in available_indices:
            # Place col_str in the determined index
            completed_col[level_idx] = col_str
            # Remove the used index from available_indices
            available_indices.remove(level_idx)
        else:
            # Use the next smallest available index if level_idx is not valid
            next_idx = available_indices.pop(0)
            completed_col[next_idx] = col_str

        # Increment idx to move to the next part of col
        idx += 1

    # Reverse completed_col back to its original order before returning
    return tuple(completed_col[::-1])


@handle_dicts
def rename_columns_with_mapping(
    df: pd.DataFrame,
    mapping: PandasMap,
    **_,
) -> pd.DataFrame:
    """
    Renames columns in a DataFrame based on a provided mapping dictionary.

    Parameters:
    df (pd.DataFrame): The DataFrame whose columns need to be renamed.
    mapping (dict): A dictionary mapping old column names to new column names.
                    Keys are old names, values are new names.

    Returns:
    pd.DataFrame: The DataFrame with renamed columns.
    """
    if not isinstance(df, pd.DataFrame) or not isinstance(mapping, dict):
        return df

    # Convert columns to a DataFrame for consistent handling
    columns_df = sanitize_df_col(df.columns)

    # Apply the mapping to rename columns
    # dict[new_name: tuple[str],
    # list[tuple[<old matching name>: str | tuple[str], new_name: tuple[str]))]
    new_columns = {}
    for col in columns_df.itertuples(index=False, name=None):
        new_name = mapping.get(
            col, mapping.get(str(col), mapping.get(col[0], col) if len(col) == 1 else col)
        )
        new_name = new_name if isinstance(new_name, tuple) else (new_name,)
        new_columns.setdefault(len(new_name), []).append((col, new_name))

    # dict[<old matching name>: str | tuple[str], new_name: tuple[str] ]
    clean_map = {col[0]: col[1] for col in new_columns[max(new_columns.keys())]}

    # insert sorting
    # Evaluate the other new_column groups excluding the max and 0 in descending order
    for i in range(len(new_columns) - 1, 0, -1):
        active_columns = sorted(new_columns[i], key=lambda x: len(str(x[1])), reverse=True)

        # Evaluate each tuple (aka column name)
        for original_col, col in active_columns:
            completed_col = evaluate_df_index_key(col, clean_map)
            # base_columns.append(completed_col)
            clean_map[original_col] = completed_col

    # restore original order
    # Update the MultiIndex columns
    df.columns = pd.MultiIndex.from_tuples(
        [clean_map[col if isinstance(col, tuple) else (col,)] for col in df.columns]
    )
    return df


@handle_dicts
def dataframe_manager(
    df: pd.DataFrame,
    simplify: bool = True,
    columns: list | tuple | dict | None = None,
    mapping: dict | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Perform simplification of multiindex columns in a DataFrame."""
    if not isinstance(df, pd.DataFrame):
        return df

    df = df.copy(deep=True)
    if isinstance(columns, (tuple, list)):
        valid_columns = [c for c in columns if c in df.columns]
        if valid_columns:
            df = df[valid_columns]

        if isinstance(mapping, dict) and any(col in mapping for col in valid_columns):
            df = rename_columns_with_mapping(df, mapping)
    elif isinstance(columns, dict):
        valid_columns = [c for c in columns if c in df.columns]
        if valid_columns:
            df = df[valid_columns]
        df = rename_columns_with_mapping(df, columns)
    elif isinstance(mapping, dict):
        df = rename_columns_with_mapping(df, mapping)

    if simplify:
        df = simplify_multi_index(df, **kwargs)

    df = df.dropna(
        how="all",
        subset=df.columns[df.nunique(dropna=False) >= 2],
        ignore_index=(
            True
            if isinstance(df.index, pd.RangeIndex) or pd.api.types.is_integer_dtype(df.index)
            else False
        ),
    )

    return df


# used in parse_files.gui_workers
@handle_dicts
def simplify_multi_index(
    df: pd.DataFrame,
    keep_keys: list | None = None,
    allow_merge: bool = False,
    sep: str = ".",
    **_,
) -> pd.DataFrame:
    """Perform simplification of multiindex columns in a DataFrame."""
    if not isinstance(df, pd.DataFrame):
        return df
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    # Flatten MultiIndex index
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(drop=True)

    # Simplify MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df = simplify_columns(df)

    # Make sure all levels are unique
    df = drop_common_index_key(df, keep_keys, allow_merge)
    df = flatten_multiindex_columns(df, sep)
    return df


# only used by simplify_multi_index
@handle_dicts
def flatten_multiindex_columns(
    df: pd.DataFrame,
    sep: str = ".",
) -> pd.DataFrame:
    """
    Flatten MultiIndex columns by merging them with a specified separator.
    """
    if not isinstance(df, pd.DataFrame):
        return df

    # Convert columns to a DataFrame for consistent handling
    columns_df = sanitize_df_col(df.columns)

    # Flatten the columns
    df.columns = [
        sep.join(map(str, col)).strip() for col in columns_df.itertuples(index=False, name=None)
    ]

    return df


# used by simplify_multi_index and convert_mfia_data of mfia_ops (an important function)
@handle_dicts
def drop_common_index_key(
    df: pd.DataFrame,
    keep_keys: list | None = None,
    allow_merge: bool = False,
) -> pd.DataFrame:
    """
    Drop levels with all unique names or single unique values in MultiIndex columns.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    keep_keys (list, optional): Keys to retain even if they are unique or single-valued.
    allow_merge (bool, optional): Whether to merge numeric levels.

    Returns:
    pd.DataFrame: The DataFrame with simplified MultiIndex columns.
    """
    if not isinstance(df, pd.DataFrame):
        return df
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    keep_keys = keep_keys or []

    # Convert columns to a DataFrame for consistent handling
    columns_df = sanitize_df_col(df.columns)

    # Case 1: Handle unique levels upfront
    unique_mask = columns_df.nunique() == len(columns_df)
    if unique_mask.any():
        unique_level = unique_mask.idxmax()  # Get the first unique level
        unique_values = set(columns_df[unique_level])

        # Check if keep_keys is empty or all strings in keep_keys are in the unique level
        if not keep_keys or all(key in unique_values for key in keep_keys):
            df.columns = columns_df[unique_level]
            return df

    # Initialize a list to track levels to drop
    drop_levels = []
    numeric_level = None
    # Handle remaining cases in a for loop
    # for level in range(columns_df.shape[1]):
    for level in columns_df.columns:
        level_vals = columns_df[level]

        # Case 2: If the level has a single unique value and it's not in keep_keys, drop this level
        if level_vals.nunique() == 1 and level_vals.iloc[0] not in keep_keys:
            drop_levels.append(level)

        # Case 3: If allow_merge is True and the level is numeric, merge it with the previous level
        elif allow_merge and level_vals.str.isnumeric().all():
            numeric_level = (
                level_vals if numeric_level is None else numeric_level + "_" + level_vals
            )
            drop_levels.append(level)

        # Case 4: Otherwise, keep this level (do nothing)
        # No action needed for levels that don't meet the above criteria

    # Drop the identified levels
    columns_df = columns_df.drop(columns=drop_levels)

    # Check if all levels were dropped
    if columns_df.empty:
        return df  # Return the unmodified DataFrame

    # if numeric_level, merge with the first column:
    if numeric_level is not None:
        columns_df.iloc[:, 0] = columns_df.iloc[:, 0] + "_" + numeric_level

    # Convert back to MultiIndex or flat index
    if columns_df.shape[1] == 1:
        df.columns = pd.Index(columns_df.iloc[:, 0])
    else:
        df.columns = pd.MultiIndex.from_frame(columns_df)

    return df


# only used by simplify_multi_index
@handle_dicts
def simplify_columns(df: pd.DataFrame) -> pd.DataFrame:
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

    columns_df = sanitize_df_col(df.columns)

    duplicated_columns = {}
    while not columns_df.empty:
        col = tuple(columns_df.iloc[0])  # Get the first column as a tuple
        duplicated = []

        # Compare col with the remaining columns
        for other_col in columns_df.iloc[1:].itertuples(index=False, name=None):
            if df[col].equals(df[other_col]):
                duplicated.append(other_col)

        if duplicated:
            duplicated.insert(0, col)
            duplicated_columns[col] = duplicated

            # Remove all columns in `duplicated` from `columns_df`
            columns_df = columns_df[~columns_df.apply(tuple, axis=1).isin(duplicated)]
        else:
            columns_df = columns_df.iloc[1:]

    for col, duplicates in duplicated_columns.items():
        # Use get_most_common_key to determine the new column name
        new_col = tuple(get_most_common_keys(df.columns, context=duplicates))

        # Rename and drop duplicates
        if new_col not in df.columns:
            df.rename(columns={col: new_col}, inplace=True)
        if new_col in duplicates:
            duplicates.remove(new_col)
        df.drop(columns=duplicates, inplace=True)

    return df


@handle_subdicts
def impedance_concat(raw_data: dict) -> pd.DataFrame:
    """
    Perform Concat on DataFrames in a dictionary.

    Parameters:
    raw_data (dict): Dictionary containing DataFrames to concatenate.

    Returns:
    pd.DataFrame: Concatenated DataFrame.
    """
    data = {k: v for k, v in raw_data.items() if isinstance(v, (pd.DataFrame, pd.Series))}
    comb = pd.concat(data.values(), sort=False, keys=data.keys())
    try:
        if isinstance(comb.columns, pd.MultiIndex):
            freq_col = get_close_matches("freq", comb.columns.get_level_values(1))
            if not freq_col:
                return comb
            return comb.sort_values(("imps", freq_col[0]))
        else:
            freq_col = get_close_matches("freq", comb.columns)
            if not freq_col:
                return comb
            return comb.sort_values(freq_col, ignore_index=True)
    except KeyError:
        return comb


# ARCHIVE
