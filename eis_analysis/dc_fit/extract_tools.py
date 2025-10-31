# -*- coding: utf-8 -*-
import re
import time
import warnings
from typing import Any, overload
from pathlib import Path
from collections import defaultdict
from collections.abc import Callable, Hashable

import numpy as np
import pandas as pd
from numpy.exceptions import RankWarning

try:
    import winsound
except ImportError:
    winsound = None

import matplotlib.pyplot as plt

from eis_analysis.system_utilities import save, load_file  # noqa: F401

np.seterr(invalid="raise")

DEFAULT_DIR = Path(
    r"D:\Online\ASU Dropbox\Jacob Clenney\Work Docs\Data\Analysis\IS\EVA\DC Analysis\rnd 2\converted"
)


warnings.filterwarnings("ignore", category=RankWarning)


# %% Functions
BASE_KEYS = ("sample_name", "condition", "temp", "sodium", "run")
DEFAULT_KEYS = BASE_KEYS + ("segment",)
DEFAULT_DTYPES = {
    "sample_name": ["cln2", "100", "200", "300", "301"],
    "condition": ["pre", "dh", "dry"],
    "temp": float,
    "sodium": int,
    "run": str,
    "segment": int,
    "peaks": int,
}


def form_std_df_index(
    df: pd.DataFrame,
    names: tuple[str, ...] = DEFAULT_KEYS,
    dtypes: dict[str, type | list] | None = None,
    **sort_kwargs,
) -> pd.DataFrame:
    """
    Format the DataFrame index to standard MultiIndex with correct names and ordering.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose index should be set to MultiIndex with standard names and ordering.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized MultiIndex.
    """
    if dtypes is None:
        dtypes = {}
    dtypes = {**DEFAULT_DTYPES, **dtypes}
    # if "sodium" in names:
    #     sort_kwargs.setdefault("level", "sodium")
    name_list = list(names)

    if df.index.nlevels != len(names) and not (
        df.index.nlevels == 1 and isinstance(df.index[0], tuple) and len(df.index[0]) == len(names)
    ):
        names_missing = [name for name in names if name not in df.columns]
        if not names_missing:
            # If all names are in the columns, set index directly and assume the index is not important
            if not isinstance(df.index, pd.RangeIndex):
                df = df.reset_index(drop=False)
            df = df.set_index(name_list)
        else:
            names_index = []
            for n in df.index.names:
                if n is None and names_missing:
                    # set name from next available unused name
                    names_index.append(names_missing.pop(0))
                elif n in names_missing:
                    # keep name and remove from missing
                    names_index.append(n)
                    names_missing.remove(str(n))
                else:
                    # keep existing name
                    names_index.append(n)
            if names_missing:
                raise ValueError(
                    f"Not all names were found in the index or dataframe: {names_missing}. "
                )
            # update the names
            df.index.names = names_index

            # Move index to columns to simplify next step
            df = df.reset_index(drop=False)
            # Set the index to the names, retains names order and ensures only names are in index
            df = df.set_index(name_list)

    df_mi = pd.MultiIndex.from_tuples(
        list(df.index),
        names=name_list,
    )
    for key, dtype in dtypes.items():
        if key in name_list:
            ind = name_list.index(key)
            if isinstance(dtype, list):
                df_mi = df_mi.set_levels(
                    df_mi.levels[ind].astype(pd.CategoricalDtype(dtype, ordered=True)),
                    level=ind,
                )
            else:
                df_mi = df_mi.set_levels(df_mi.levels[ind].astype(dtype), level=ind)

    df.index = df_mi
    sort_kwargs.setdefault("level", name_list)
    df = df.sort_index(**sort_kwargs)
    return df


def create_std_df(
    source_idxs: list | None = None,
    source_cols: list | None = None,
    source_df: pd.DataFrame | None = None,
    col_filter: Callable = lambda x: True,
) -> pd.DataFrame:
    """
    Create or update the fit_df DataFrame for fit results.

    Parameters
    ----------
    source_idxs : list
        List of MultiIndex tuples for DataFrame index.
    source_cols : list
        List of columns for the DataFrame.
    source_df : pd.DataFrame, optional
        DataFrame of mean values for initial guesses.

    Returns
    -------
    pd.DataFrame
        DataFrame with fit values for each main_param.
    """
    if not source_idxs and source_df is not None:
        source_idxs = list(source_df.index)
    elif not source_idxs:
        raise ValueError("source_idxs cannot be empty or None.")

    if not source_cols and source_df is not None:
        source_cols = [col for col in source_df.columns if col_filter(col)]
    elif not source_cols:
        source_cols = ["Voltage", "Current", "Resistance", "Temperature"]
    else:
        source_cols = [col for col in source_cols if col_filter(col)]

    return form_std_df_index(
        pd.DataFrame(
            index=source_idxs,
            columns=source_cols,
            dtype=float,
        )
    )


def _digit_suffix(input_set: set[str]) -> tuple[set[str], int]:
    """Returns a set of strings from input_set that end with a digit."""
    digit_set = {s for s in input_set if s and s[-1].isdigit()}
    max_digit = max([int(re.search(r"\d+$", s).group()) for s in digit_set], default=0)  # type: ignore
    return digit_set, max_digit


def _parse_at_suffix(input_str: str, max_digit: int) -> str:
    """Parses a string containing '@' and returns a modified string based on the suffix."""
    if "@" not in input_str:
        return input_str

    base, mode = input_str.split("@", 1)
    if "min" in mode.lower() or max_digit == 0:
        digit = 0
    elif "mid" in mode.lower():
        digit = max_digit // 2
    else:  # default to max
        digit = max_digit

    return f"{base.strip()}{digit}"


def bias_prior_fits(
    prior_fits: dict[str, pd.DataFrame] | None,
    *group_keys: str,
    bias: float = 0.0,
    agg_method: str = "mean",
    method_map: dict[str, Any] | tuple[str, ...] | list[str] = (),
) -> dict[str, pd.DataFrame] | None:
    """
    Return a new prior_fits dict with values biased toward grouped averages.

    Parameters
    ----------
    prior_fits : dict[str, pd.DataFrame] or None
        Dictionary of DataFrames (as used in fit_preparation). If None, returns None.
    bias : float
        Value from 0 to 1. 0 = original values, 1 = grouped average, 0.5 = average of both.
        Thus the smaller the value, the more the original values are retained.
    group_keys : tuple of str
        Index level(s) to group by (must be present in the DataFrame MultiIndex).

    Returns
    -------
    dict[str, pd.DataFrame] or None
        New dict with values biased toward grouped averages.
    """
    if prior_fits is None:
        return None
    bias = np.clip(bias, 0.0, 1.0).item()

    gr_key_list = list(group_keys) if group_keys else ["sample_name", "condition", "temp"]

    if isinstance(method_map, (list, tuple)):
        method_map = {col: None for col in method_map}

    result = {}
    for key, df in prior_fits.items():
        # Do not modify "Equations" or non-DataFrame entries
        if not isinstance(df, pd.DataFrame) or key == "Equations":
            result[key] = df
            continue
        df = form_std_df_index(df)
        # Only operate if all gr_key_list are in the index
        if not all(k in df.index.names for k in gr_key_list):
            result[key] = df.copy(deep=True)
            continue

        # Create a copy of the DataFrame for the result
        geo_df = df.copy(deep=True)
        # Get relavent columns (those that end with a digit)
        main_cols, max_p = _digit_suffix(set(df.columns))

        # Process dictionary mapping columns to aggregation methods
        column_transforms = defaultdict(list)
        for col, agg in method_map.items():
            col = _parse_at_suffix(col, max_p)
            if col in main_cols:
                column_transforms[agg].append(col)
                main_cols.remove(col)

        column_transforms[agg_method] += list(main_cols)

        for agg, cols in column_transforms.items():
            if agg is not None:
                # Compute grouped aggregate for this column
                targ_df = geo_df[cols]
                grouped_df = targ_df.abs().groupby(level=gr_key_list, observed=True).transform(agg)
                # Apply the transformation with bias
                geo_df[cols] = (targ_df.abs() ** (1 - bias) * grouped_df**bias) * np.sign(targ_df)

        result[key] = geo_df

    return result


def insert_polarity_column(
    data: pd.DataFrame,
    reference: str | tuple | list | pd.Series | np.ndarray = "Voltage",
    polarity_name: str = "Polarity",
    location: int | str | tuple = 0,
    after: bool = False,
) -> pd.DataFrame:
    """
    Insert a polarity column at a specified location, using a reference column or array.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to modify.
    reference : str or array-like, optional
        Name of the column or array/series to use for polarity (default: "Voltage").
    polarity_name : str, optional
        Name of the polarity column to insert (default: "Polarity").
    location : int or str, optional
        Location to insert the polarity column (default: 0).
        If str, uses the column name to determine index.
    after : bool, optional
        If True, inserts after the specified location (default: False).

    Returns
    -------
    pd.DataFrame
        DataFrame with the polarity column inserted.
    """
    df = data.copy()
    df.attrs |= data.attrs.copy()  # Copy attributes to the new DataFrame
    # Determine polarity values
    try:
        if isinstance(reference, (str, tuple)) and reference in df.columns:
            polarity = np.sign(df[reference])
        elif hasattr(reference, "__len__") and len(reference) == len(df):
            polarity = np.sign(reference)
        else:
            return df
    except TypeError:
        return df

    # Determine insert location
    if not isinstance(location, int):
        location = df.columns.tolist().index(location) if location in df.columns else 0  # type: ignore
    if after:
        location += 1

    # Insert or update polarity column
    if polarity_name in df.columns:
        df[polarity_name] = polarity
    else:
        df.insert(location, polarity_name, polarity)
    return df


def aggregate_primary_columns(
    fit_df: pd.DataFrame,
    index_levels: list[Hashable] | None = None,
    primary_cols: list[str] | tuple[str, ...] = (
        "Voltage",
        "Current",
        "Resistance",
        "Temperature",
    ),
    split_by_polarity: bool = True,
    polarity_col: str = "Polarity",
) -> pd.DataFrame:
    """
    Aggregate primary columns by index levels, computing mean and std for each.
    Optionally splits by polarity.

    Parameters
    ----------
    fit_df : pd.DataFrame
        DataFrame to aggregate.
    index_levels : list of str, optional
        Index levels to group by. If None, uses all index levels except 'segment'.
    primary_cols : list of str, optional
        Columns to aggregate (default: ["Voltage", "Current", "Resistance", "Temperature"]).
    split_by_polarity : bool, optional
        If True, split aggregation by polarity column.
    polarity_col : str, optional
        Name of the polarity column (default: "Polarity").

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with mean and std columns for each primary column.
    """
    df = fit_df.copy()
    df.attrs |= fit_df.attrs.copy()  # Copy attributes to the new DataFrame
    # Ensure polarity column exists if splitting
    if split_by_polarity and polarity_col not in df.columns:
        df = insert_polarity_column(df)
        if polarity_col not in df.columns:
            return df

    # Invert index_levels: group by all index levels except those in index_levels
    if index_levels is None:
        # Default: exclude 'segment'
        exclude_levels = {"run", "segment"}  # TODO: Add "sodium" ?
    else:
        exclude_levels = set(index_levels)

    group_levels = [lvl for lvl in df.index.names if lvl not in exclude_levels]
    cols_to_agg = [col for col in primary_cols if col in df.columns]
    if not cols_to_agg:
        cols_to_agg = df.columns.tolist()
    if split_by_polarity:
        group_levels = group_levels + [polarity_col]
        if polarity_col not in cols_to_agg:
            cols_to_agg.append(polarity_col)

    # Group and aggregate
    result = df[cols_to_agg].groupby(group_levels, observed=True).agg(["mean", "std"])

    # Flatten MultiIndex columns
    result.columns = [
        f"{col}" if stat == "mean" else f"{col}_{stat}" for col, stat in result.columns
    ]
    # If splitting by polarity, pivot polarity index into columns
    if split_by_polarity:
        # Polarity is the last index level
        pol_level = result.index.names[-1]
        # Pivot so that polarity becomes columns: e.g., Current_mean_neg, Current_mean_pos, etc.
        result = pd.DataFrame(result.unstack(pol_level))
        # Flatten the new MultiIndex columns
        cols = [f"{'neg' if int(pol) < 0 else 'pos'} {col}" for col, pol in result.columns]
        result.columns = cols

        reordered = []
        for i in range(0, len(cols), 4):
            group = cols[i : i + 4]
            # Swap the middle two to get: mean-pos, std-pos, mean-neg, std-neg
            group[1], group[2] = group[2], group[1]
            # If "neg" is first, swap first two with last two
            if "neg" in group[0]:
                group = group[2:] + group[:2]
            reordered.extend(group)
        result = result[reordered]
    result = result.fillna(result.mean())
    return result


def reduce_df_index(
    df0: pd.DataFrame,
    index: int | str = 0,
    rename: Callable = lambda x: x,
    drop_level: bool = False,
) -> dict:
    """
    Split a MultiIndex DataFrame into a dictionary of DataFrames keyed by the first index level.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a MultiIndex.
    index : int or str, optional
        If int, specifies the first-level index to use (default: 0).
        If str, specifies the name of the first-level index to use.

    Returns
    -------
    dict
        Dictionary mapping first-level index values to DataFrames with the remaining index levels.
    """
    df = df0.copy()
    df.attrs |= df0.attrs.copy()  # Copy attributes to the new DataFrame

    if not isinstance(df.index, pd.MultiIndex) and index not in df.columns:
        raise ValueError("Input DataFrame must have a MultiIndex.")

    if index in df.columns:
        df = df.set_index(index, append=True)
        gr_level = index
    elif isinstance(index, int):
        if index < 0 or index >= df.index.nlevels:
            index = 0
        gr_level = df.index.names[index]
    elif index not in df.index.names:
        gr_level = df.index.names[0]
    else:
        gr_level = index

    # gr_level = df.index.names[0]
    result = {}
    for key, sub_df in df.groupby(level=gr_level, observed=True):
        # Drop the first level from the index for each sub-DataFrame
        if drop_level:
            sub_df = sub_df.droplevel(gr_level)
        result[rename(key)] = sub_df
    return result


def polarize_params(
    grouped_data: dict,
    data_sets: set[str] | list[str] | tuple[str, ...] = (),
    simplify: bool = True,
) -> dict:
    """
    Apply Arrhenius fits to specified columns in grouped DataFrames.

    Parameters
    ----------
    grouped_data : dict
        Dictionary of DataFrames with MultiIndex including a temperature level.
    columns : list[str], optional
        Columns to fit. Default is ["pos Current", "neg Current", "pos Resistance", "neg Resistance"].

    Returns
    -------
    dict
        Dictionary with the same structure as input, with fit results stored in DataFrame attrs.
    """
    if not data_sets:
        data_sets = {"Current", "Resistance", "Voltage", "Temperature"}

    revised_data = defaultdict(dict)
    for top_key, vals in grouped_data.items():
        if "vals" not in top_key:
            if not simplify:
                revised_data[top_key] = vals
            continue
        pol_ref = vals["Voltage"]["b0"].copy()
        for key, df0 in vals.items():
            if key not in data_sets or len(df0) != len(pol_ref):
                if not simplify or key != "Equations":
                    revised_data[top_key][key] = df0
                continue
            revised_data[top_key][key] = insert_polarity_column(
                df0,
                reference=pol_ref,
                location="timestamp" if "timestamp" in df0.columns else "Error",
                after=True,
            )
            revised_data[top_key] |= reduce_df_index(
                revised_data[top_key][key],
                "Polarity",
                lambda x: f"{key} neg" if x < 0 else f"{key} pos",
                True,
            )
    if simplify:
        simplified_data = {}
        for key, datasets in revised_data.items():
            n_key = key.replace("vals", "").strip()
            simplified_data |= {f"{n_key} {k}": v for k, v in datasets.items()}
        return simplified_data

    return dict(revised_data)


def polarize_points(points: dict) -> dict:
    """
    Insert polarity column and reduce index for fit keys.

    Parameters
    ----------
    points : dict
        Dictionary of points DataFrames.

    Returns
    -------
    pol_points : dict
        Dictionary of points with polarity column inserted and reduced index for fit keys.
    """
    pol_points = {}
    for key in list(points):
        pol_points[key] = insert_polarity_column(points[key])
        if re.match(r"^fit\s\d{1,2}$", key.lower().strip()):
            pol_points |= reduce_df_index(
                pol_points[f"{key}"],
                "Polarity",
                lambda x: f"{key} neg" if x < 0 else f"{key} pos",
                True,
            )

    return pol_points


def group_points(
    points: dict,
    match_str: str = r"^fit\s\d{1,2}$",
    aggregate: bool = True,
    polarize: bool = True,
    gr_levels: list | tuple = (0, 1, (0, 1)),
    include_unchanged: bool = True,
) -> dict:
    """
    Group points by primary columns and polarity for fit keys.

    Parameters
    ----------
    points : dict
        Dictionary of points DataFrames.

    Returns
    -------
    grp_points : dict
        Dictionary of grouped points.
    """
    grp_points = {}
    for key in list(points):
        if re.match(match_str, key.lower().strip()):
            g_key = key
            unchanged = False
            if aggregate:
                g_key = f"{key} grouped"
                grp_points[g_key] = aggregate_primary_columns(points[key], split_by_polarity=True)
            elif polarize and "Polarity" not in points[key].columns:
                grp_points[g_key] = insert_polarity_column(points[key])
            else:
                grp_points[g_key] = points[key].copy()
                grp_points[g_key].attrs |= points[
                    key
                ].attrs.copy()  # Copy attributes to the new DataFrame
                unchanged = True

            for g_lvl in gr_levels:
                if isinstance(g_lvl, (str, int)):
                    grp_points |= reduce_df_index(grp_points[g_key], g_lvl, lambda k: f"{key} {k}")
                elif isinstance(g_lvl, (tuple, list)):
                    level_dfs = reduce_df_index(
                        grp_points[g_key], g_lvl[0], lambda k: f"{key} {k}"
                    )
                    for i in range(1, len(g_lvl)):
                        next_dfs = {}
                        for l_key, l_df in level_dfs.items():
                            # Process the next level
                            next_dfs |= reduce_df_index(l_df, g_lvl[i], lambda k: f"{l_key} {k}")
                        # Update grp_points and level_dfs for the next iteration
                        level_dfs = next_dfs
                    grp_points |= level_dfs
                    # for lvl in points[key].index.get_level_values(g_lvl[0]).unique():
                    #     l_key = f"{key} {lvl}"
                    #     if l_key in grp_points:
                    #         grp_points |= reduce_df_index(
                    #             grp_points[l_key], g_lvl[1], lambda k: f"{l_key} {k}"
                    #         )
            if unchanged and not include_unchanged:
                del grp_points[g_key]

    return grp_points


def get_job(
    job_key: str,
    job_source: dict | None = None,
    job_profiles: dict | None = None,
    job_names: list[str] | tuple[str, ...] = (),
    **kwargs,
) -> dict:
    """
    Returns a copy of the job dict with prior_fit inserted if prior_fit_num is not None.

    Parameters
    ----------
    job_key : str
        The key for the job in all_jobs_dict.
    job_source : dict, optional
        The source dictionary containing job definitions.
    job_profiles : dict, optional
        Dictionary of job profiles to merge into the job dict.
    job_names : list or tuple of str, optional
        list of prefixes for job profiles/kwargs needed to update the job dict.
        Typically correlates to different functions.

    Returns
    -------
    dict
        The job dict, with prior_fit if specified.
    """
    job = job_source.get(job_key, {}).copy() if job_source else {}
    job = job | kwargs
    job_profiles = job_profiles or {}

    for name in job_names:
        profile, prof_kwargs = f"{name}_profile", f"{name}_kwargs"
        p_list = job.get(profile, [])
        if isinstance(p_list, str):
            p_list = [p_list]
        merged = {}
        for p_key in p_list:
            prof = job_profiles.get(p_key, {})
            for k, v in prof.items():
                if isinstance(merged.get(k), dict) and isinstance(v, dict):
                    merged[k] |= v
                else:
                    merged[k] = v
        # merge the profile's kwargs into the profile and save the job's profile
        job[profile] = merged | job.pop(prof_kwargs, {})

    return job


def any_all_func(how: Callable, ref: tuple = (), default: bool = True) -> Callable:
    """
    Convert a string "any" or "all" to the corresponding built-in function.

    Parameters
    ----------
    any_all : str or Callable
        If str, must be "any" or "all". If Callable, returned as is.

    Returns
    -------
    Callable
        The corresponding built-in function.
    """

    def func(targ) -> bool:
        if not ref:
            return default
        # if isinstance(targ, (list, tuple, np.ndarray, pd.Series)):
        #     return how(key == val for key in ref for val in targ)
        return how(key in targ for key in ref)

    return func


@overload
def partial_selection(
    raw_data: dict, *keys: str | float, any_all: str | Callable = ...
) -> dict: ...
@overload
def partial_selection(
    raw_data: pd.DataFrame, *keys: str | float, any_all: str | Callable = ...
) -> pd.DataFrame: ...


def partial_selection(
    raw_data: dict | pd.DataFrame, *keys: str | float, any_all: str | Callable = "any"
) -> dict | pd.DataFrame:
    """
    Return a new dict containing only the specified keys from raw_data.
    Returns a subset of the DataFrame with fuzzy matching where the index matches any of the provided keys.
    For a flat index, matches if the key is a substring of the index value.
    For a MultiIndex, matches if the key is a substring of any value in the index tuple.

    Parameters
    ----------
    raw_data : dict | pd.DataFrame
        The original dictionary or DataFrame to filter.
    *keys : str
        Keys to select.

    Returns
    -------
    dict
        Filtered dictionary with only the specified keys.
    """
    if isinstance(any_all, str):
        if any_all not in ["any", "all"]:
            any_all = "all"
        func = eval(any_all)
    else:
        func = any_all

    parse_func = func
    if func in [any, all]:
        parse_func = any_all_func(func, keys, isinstance(raw_data, dict))

    if isinstance(raw_data, dict):
        return {k: v for k, v in raw_data.items() if parse_func(k)}
    else:
        if raw_data.index.nlevels == 1:
            # std dataframe with single index
            mask = raw_data.index.to_series().astype(str).apply(parse_func)
        else:
            # dataframe with multiindex
            mask = pd.Series(
                [parse_func(idx_tuple) for idx_tuple in raw_data.index],
                index=raw_data.index,
            )
        return raw_data[mask]


def flatten_multiindex(
    *dfs: pd.DataFrame, sep: str = "_", copy: bool = False
) -> tuple[pd.DataFrame, ...]:
    """
    Flatten a MultiIndex on a DataFrame into a single string index joined by underscores.

    Parameters
    ----------
    sep : str, optional
        Separator to join MultiIndex levels, by default "_"
    copy : bool, optional
        If True, return a copy of the DataFrame(s) with flattened index, by default False
    *dfs : pd.DataFrame
        DataFrames to flatten.

    Returns
    -------
    tuple[pd.DataFrame, ...]
        Tuple of DataFrames with flattened index.
    """
    res = []
    for df in dfs:
        if copy:
            df = df.copy(deep=True)
        if isinstance(df.index, pd.MultiIndex):
            df.index = pd.Index([sep.join(map(str, idx)) for idx in df.index.to_flat_index()])
        res.append(df)
    return tuple(res)


def trimmed_midpoint(
    *series: tuple | list | np.ndarray | pd.Series, q: float = 0.25, logspace=True
) -> float:
    """
    Calculates a trimmed mean midpoint in log space from one or more pd.Series.

    Parameters
    ----------
    *series : pd.Series or array-like
        Variable number of input sequences representing initial guesses (e.g. tau0, tau1...).
    q : float in [0, 0.5]
        Quantile for symmetric trimming in log-space. q=0.25 keeps the IQR.

    Returns
    -------
    float
        Robust midpoint in original data space.
    """
    q = np.clip(q, 0.0, 0.5)

    means = []
    for s in series:
        s = pd.to_numeric(abs(np.asarray(s)), errors="coerce")
        s = np.log10(s[np.isfinite(s) & (s > 0)]) if logspace else s[np.isfinite(s)]
        if s.size:
            means.append(np.mean(np.quantile(s, [q, 1 - q])))

    if not means:
        return float(np.nanmedian([np.nanmedian(s) for s in series]))

    if logspace:
        return float(10 ** (np.mean(means)))
    return float(np.mean(means))


def _fix_dfs(data: dict, mi_cols: bool = False, is_std: bool = True) -> dict:
    """
    Fix DataFrames in the provided data dictionary to ensure they have a standard index.

    Parameters
    ----------
    data : dict
        Dictionary containing DataFrames to be fixed.
    """
    for key, df in data.items():
        if is_std:
            df = form_std_df_index(df)
        else:
            names = [n for n in DEFAULT_KEYS if n in df.columns]
            if names:
                df = df.set_index(names)

        if mi_cols and all(isinstance(col, str) and "." in col for col in df.columns):
            # Split columns and create MultiIndex
            split_cols = [tuple(col.split(".")) for col in df.columns]
            df.columns = pd.MultiIndex.from_tuples(split_cols)
        data[key] = df

    return data


def load_dc_data(data_dir: str | Path, *files: str) -> dict:
    """
    Load all DC fitting data from disk, optionally limiting to specified files.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing data files.
    *files : str
        Optional list of file keys to load (e.g., "points", "curves", "params", "grouped", "perm_peaks", "perms", etc.).
        For keys like "curves", all files starting with "curves" will be loaded.

    Returns
    -------
    data : dict
        Dictionary containing all loaded data keyed by file type and subkey.
    """

    def base_entry(string: str) -> bool:
        check = not string.startswith("fit") or re.match(r"^fit\s\d{1,2}$", string.lower().strip())
        return bool(check)

    def fix_tuplekey(key: str) -> tuple:
        t_key: list[Any] = key.split(" ")
        t_key[2] = float(t_key[2][:2])
        for nkey in range(3, len(t_key)):
            if t_key[nkey].isnumeric():
                t_key[nkey] = int(t_key[nkey])
        return tuple(t_key)

    def fix_dtypes(
        df0: pd.DataFrame,
        defined_types: dict[str, type] | None = None,
    ) -> pd.DataFrame:
        """Convert all columns in a DataFrame to float except those specified in defined_types."""
        df = df0.copy()
        df.attrs |= df0.attrs.copy()  # Copy attributes to the new DataFrame

        if defined_types is None:
            defined_types = {}
        for col in df.columns:
            if col in defined_types:
                df[col] = df[col].astype(defined_types[col])
            else:
                df[col] = df[col].astype(float)
        return df

    data_path = Path(data_dir)
    data = {}

    # If no files specified, load all recognized types from extract_responses.py
    file_keys = (
        set(files) if files else {"points", "grouped", "params", "perm_peaks", "curves", "perms"}
    )
    glob_str = "*[!0-9].xls*"
    if "fits" in file_keys:
        glob_str = "*.xls*"

    # Load points
    # form_std_df_index
    if "points" in file_keys:
        points_file = data_path / "points.xlsx"
        if points_file.exists():
            print(f"{time.ctime()}: Loading points")
            points_data = load_file(points_file)
            data["points"] = _fix_dfs({k: d for k, d in points_data[0].items() if base_entry(k)})
    # Load grouped
    if "grouped" in file_keys:
        grouped_file = data_path / "grouped.xlsx"
        if grouped_file.exists():
            print(f"{time.ctime()}: Loading grouped points")
            grouped_data = load_file(grouped_file)
            data["grouped"] = _fix_dfs(grouped_data[0], is_std=False)

    # Load params
    if "params" in file_keys:
        params_file = data_path / "params.xlsx"
        if params_file.exists():
            print(f"{time.ctime()}: Loading parameters")
            params_data = load_file(params_file)
            params = defaultdict(dict)
            for flat_key, flat_df in params_data[0].items():
                parts = flat_key.split()
                if len(parts) < 3 or parts[0] != "fit":
                    continue
                fit_n = parts[1]
                # Handle equations case
                if parts[2].lower() == "equations":
                    # Set all columns before "Equation" as index
                    eq_idx = flat_df.columns.get_loc("Equation")
                    if eq_idx > 0:
                        flat_df = flat_df.set_index(flat_df.columns[:eq_idx].tolist())
                    params[f"fit {fit_n} conditions"].update({"Equations": flat_df})
                    params[f"fit {fit_n} vals"].update({"Equations": flat_df})
                else:
                    params[f"fit {fit_n} {parts[3]}"].update(
                        _fix_dfs({parts[2].title(): flat_df}, mi_cols=(parts[3] == "conditions"))
                    )

            data["params"] = dict(params)

    # Load perm_peaks
    if "perm_peaks" in file_keys:
        perm_peaks_file = data_path / "perm_peaks.xlsx"
        if perm_peaks_file.exists():
            print(f"{time.ctime()}: Loading permittivity peaks")
            perm_peaks_data = load_file(perm_peaks_file)
            data["perm_peaks"] = _fix_dfs(perm_peaks_data[0])

    # Load curves (all files starting with curves)
    if "curves" in file_keys:
        print(f"{time.ctime()}: Loading curves")
        curves_data = load_file(data_path, glob="curves" + glob_str, load_to_dict=True)
        if isinstance(curves_data, tuple):
            # If a tuple is returned, it means we have a single file with multiple sheets
            data["curves"] = {fix_tuplekey(k): v for k, v in curves_data[0].items()}
        else:
            curves = defaultdict(dict)
            for key, entry in curves_data.items():
                key_name = key.replace("curves_", "").replace("fit_", "fit ")
                curves[key_name] = {fix_tuplekey(k): v for k, v in entry[0].items()}
            data["curves"] = dict(curves)

    # Load perms (all files starting with perms)
    if "perms" in file_keys:
        print(f"{time.ctime()}: Loading permittivity curves")
        perms_data = load_file(data_path, glob="perms" + glob_str, load_to_dict=True)
        if isinstance(perms_data, tuple):
            # If a tuple is returned, it means we have a single file with multiple sheets
            data["perms"] = {fix_tuplekey(k): v for k, v in perms_data[0].items()}
        else:
            curves = defaultdict(dict)
            for key, entry in perms_data.items():
                key_name = key.replace("perms_", "").replace("fit_", "fit ")
                curves[key_name] = {
                    fix_tuplekey(k): fix_dtypes(v, {"permittivity": complex})
                    for k, v in entry[0].items()
                }
            data["perms"] = dict(curves)

    return data


def _flatten_params(params: dict) -> dict:
    """
    Flatten nested fit parameters for saving.

    Parameters
    ----------
    params : dict
        Dictionary of fit parameters.

    Returns
    -------
    params_flt : dict
        Dictionary of flattened parameters.
    """
    params_flt = {}
    for fit_key, subdict in params.items():
        if not fit_key.startswith("fit "):
            continue
        parts = fit_key.split()
        if len(parts) < 3:
            continue
        for param, value in subdict.items():
            if param == "Equations":
                params_flt[f"fit {parts[1]} Equations"] = value
            else:
                params_flt[f"fit {parts[1]} {param} {parts[2]}"] = value
    return params_flt


def save_results(
    data_path: str | Path,
    *targets: str,
    save_mode: str = "w",
    attrs: bool = False,
    verbose: bool = True,
    **datasets: dict,
) -> None:
    """
    Save post-fitting results to disk if provided.
    If points or params are passed, they are converted to the appropriate save dicts.

    Parameters
    ----------
    data_path : str
        Path to the data directory.
    points : dict, optional
        Raw points dictionary to be processed and saved.
    params : dict, optional
        Raw params dictionary to be processed and saved.
    pol_points : dict, optional
        Points with polarity.
    grp_points : dict, optional
        Grouped points.
    params_flt : dict, optional
        Flattened parameters.
    perm_peaks : dict, optional
        Permittivity peak summaries.
    perm_curves : dict, optional
        Permittivity curves.
    curves : dict, optional
        Curve DataFrames.
    save_mode : str, optional
        File save mode ("w" for write, "a" for append).
    """
    output_dir = Path(data_path)

    skipped = {}
    if verbose:
        print(f"{time.ctime()}: Saving data...")
    try:
        saved = []
        for name, value in datasets.items():
            # if verbose:
            #     # print(f"{time.ctime()}: Saving {name}...")
            #     print(f"\r Currently saving {name}...", end="")
            if name == "params":
                value = _flatten_params(value)
            elif name == "points":
                value = polarize_points(value)
            priors = f"Saved [{', '.join(saved)}] | " if saved else ""

            if isinstance(next(iter(value.values())), pd.DataFrame):
                if verbose:
                    print(
                        f"\r{priors}{time.ctime().split()[3]}: Saving {name}...",
                        end="",
                        flush=True,
                    )
                try:
                    save(value, output_dir, name, file_type="xls", attrs=attrs, mode=save_mode)
                    saved.append(name)
                except PermissionError:
                    skipped[name] = datasets[name]
            else:
                for key in value:
                    if not targets or key in targets:
                        if verbose:
                            print(
                                f"\r{priors}{time.ctime().split()[3]}: Saving {name}_{key}...",
                                end="",
                                flush=True,
                            )
                            # print(f"    Currently saving {name}_{key}...")
                        try:
                            save(
                                value[key],
                                output_dir,
                                f"{name}_{key}",
                                file_type="xls",
                                attrs=attrs,
                                mode=save_mode,
                            )
                            saved.append(f"{name}_{key}")
                        except PermissionError:
                            skipped.setdefault(name, {})[key] = datasets[name][key]
    except KeyboardInterrupt:
        if verbose:
            print()
        raise KeyboardInterrupt

    if verbose:
        priors = f"Saved [{', '.join(saved)}]" if saved else ""
        print(priors)

    if winsound is not None:
        winsound.MessageBeep(winsound.MB_ICONHAND)
    else:
        print("\a")

    if skipped:
        redo = input(
            "Some files were skipped due to permission errors. "
            "Do you want to retry saving them? (y/n): "
        )
        if redo.lower() == "y":
            save_results(
                data_path,
                *targets,
                save_mode=save_mode,
                attrs=attrs,
                verbose=verbose,
                **skipped,
            )


def plot_fit_curves(
    fit_curves: dict,
    targets: set[tuple[str, tuple]] | None = None,
    groupby: list[int | str] | int | str | None = None,
    columns: str | list[str] | None = None,
    plot_fits: bool = True,
    **kwargs,
) -> None:
    """
    Plot data and fit curves for the specified targets.

    Parameters
    ----------
    fit_curves : dict[tuple, pd.DataFrame]
        Dictionary of fitted curves for each segment and main_param.
    targets : set of (str, tuple), optional
        Set of (column, info_index) pairs specifying which curves to plot.
        If None, constructed from fit_curves and columns.
    groupby : list[int] | int | None, optional
        Indices of info_index to group by. If None, all curves are plotted on a single plot.
        If int or list of ints, a separate plot is made for each group.
    plot_fits : bool, optional
        Whether to plot fit curves (default: True).
    x_data : str, optional
        Column to use for x-axis (default: "time").
    legend : bool, optional
        Whether to show legend (default: False).
    columns : str or list of str, optional
        Columns to plot. If None, defaults to ["Voltage", "Current", "Resistance", "Temperature"].

    Notes
    -----
    - Dashed lines are used for fits, matching the color of the data.
    - Grid is enabled on all plots.
    - Grouping is always by column first, then by groupby indices as subgroups.
    """
    if not fit_curves:
        print("No fit curves available to plot.")
        return

    def get_common_and_unique_indices(gr_ts: list[tuple]) -> tuple[list, list]:
        c_keys = []
        u_idxs = []
        if isinstance(gr_ts[0], tuple):
            columns = list(map(set, zip(*gr_ts)))
            for i, col in enumerate(columns):
                if len(col) == 1:
                    c_keys.append(col.pop())
                else:
                    u_idxs.append(i)
        return c_keys, u_idxs

    default_columns = ["Voltage", "Current", "Resistance", "Temperature"]
    group_str = list(DEFAULT_KEYS)

    if isinstance(columns, str):
        columns = [columns]
    if isinstance(columns, list) and "default" in columns:
        columns.remove("default")
        columns += default_columns

    # If targets is None, construct from fit_curves and columns_to_plot
    if targets is None:
        # Handle columns argument
        targets = set()
        for col in columns or default_columns:
            targets = targets | set((col, idx) for idx in fit_curves)

    if groupby is None:
        groupby = []
    if isinstance(groupby, (int, str)):
        groupby = [groupby]

    valid_groups: set[int] | list[int] = set()
    for gr in groupby:
        if isinstance(gr, str) and gr in group_str:
            valid_groups.add(group_str.index(gr))
        elif isinstance(gr, int) and 0 <= gr < 5:
            valid_groups.add(gr)

    valid_groups = sorted(valid_groups)

    groups = {}
    for col, idx in targets:
        if idx in fit_curves and col in fit_curves[idx] and (columns is None or col in columns):
            group_key = (col,) + tuple(idx[i] for i in valid_groups)
            groups.setdefault(group_key, set()).add(idx)

    x_data = kwargs.get("x_data", "time")
    legend = kwargs.get("legend", False)
    y_pad = kwargs.get("y_pad", 0.1)
    log_y = kwargs.get("log_y", False)

    for group, group_targets in groups.items():
        gr_targets = sorted(group_targets)
        if not gr_targets:
            continue  # Skip empty groups

        y_min = 0
        y_max = 0
        col = group[0]
        apply_log = False
        if log_y and ("curr" in col.lower() or "res" in col.lower()):
            apply_log = True

        common_keys, unique_idxs = get_common_and_unique_indices(gr_targets)

        fig, ax = plt.subplots()
        for idx in gr_targets:
            curve_df = fit_curves.get(idx)
            # Already checked for existence above, but double-check for safety
            if (
                curve_df is None
                or f"{col}" not in curve_df
                or not isinstance(curve_df, pd.DataFrame)
            ):
                continue
            label_idx = tuple(idx[i] for i in unique_idxs) if unique_idxs else idx
            # Plot data

            data = abs(curve_df[f"{col}"]) if apply_log else curve_df[f"{col}"]
            y_min = min(y_min, data.min()) if y_min else data.min()
            y_max = max(y_max, data.max()) if y_max else data.max()

            line = ax.plot(
                curve_df[x_data],
                data,
                label=f"{col} data {label_idx}",
                alpha=0.8,
            )[0]
            # Plot fit with dashed line, matching color
            if plot_fits and f"{col}_fit" in curve_df:
                fit_data = abs(curve_df[f"{col}_fit"]) if apply_log else curve_df[f"{col}_fit"]
                ax.plot(
                    curve_df[x_data],
                    fit_data,
                    linestyle="--",
                    color=line.get_color(),
                    label=f"{col} fit {label_idx}",
                    alpha=0.8,
                )

        y_max = y_max or 1.0

        if apply_log:
            ax.set_yscale("log")
            if y_min == 0:
                y_min = 1e-16
            lims = (
                10 ** (np.floor(np.log10(y_min) * (1 - y_pad))),
                10 ** (np.ceil(np.log10(y_max) * (1 + y_pad))),
            )
        else:
            lims = (y_min * (1 - y_pad), y_max * (1 + y_pad))

        if min(lims) != max(lims):
            ax.set_ylim(min(lims), max(lims))

        ax.set_xlabel(str(x_data).title())
        ax.set_ylabel(str(col).title())
        ax.grid(True)
        if legend:
            ax.legend()
        # Show group in title, including both col and groupby values if present

        if len(group) == 1:
            if common_keys:
                ax.set_title(f"Measurement: {common_keys}")
            else:
                ax.set_title(f"Column: {col}")
        else:
            common_keys = [k for k in common_keys if k not in group[1:]]
            if common_keys:
                ax.set_title(f"Group: {group[1:]}, Measurement: {common_keys}")
            else:
                ax.set_title(f"Group: {group[1:]}")
        plt.tight_layout()
        plt.show()


def plot_selected_curves(
    data: dict,
    curve_selectors: list,
    targets: set[tuple[str, tuple]] | None = None,
    groupby: list[int | str] | int | str | None = None,
    columns: str | list[str] | None = None,
    plot_fits: bool = True,
    **kwargs,
) -> None:
    """
    Selects curves using partial_selection with provided argument sets and plots them.

    Parameters
    ----------
    data : dict
        Dictionary containing curve data.
    curve_selectors : list
        List of argument tuples for partial_selection. Each tuple:
        - First value is the key in `data` (e.g., "cleaned").
        - Remaining values are positional args for partial_selection.
        - If the last value is in [any, all, "any", "all"], it is used as any_all.
    plot_args : dict
        Dictionary of keyword arguments for plot_fit_curves.

    Returns
    -------
    None
    """
    target_curves = {}
    for args in curve_selectors:
        if not args:
            continue
        if args[0] in data:
            data_dict = data[args[0]].copy()
            sel_args = list(args[1:])
        else:
            data_dict = data.copy()
            sel_args = list(args)
        any_all = "any"
        if sel_args and sel_args[-1] in [any, all, "any", "all"]:
            any_all = sel_args.pop()
        target_curves |= partial_selection(data_dict, *sel_args, any_all=any_all)

    plot_fit_curves(
        target_curves,
        targets=targets,
        groupby=groupby,
        columns=columns,
        plot_fits=plot_fits,
        **kwargs,
    )


if __name__ == "__main__":
    # Example usage: load data and run permittivity analysis
    data_dir = r"D:\Online\ASU Dropbox\Jacob Clenney\Work Docs\Data\Analysis\IS\EVA\DC Analysis\rnd 2\converted"

    # test_dict = {
    #     "sample1_high": np.array([1, 1, 1]),
    #     "sample2_high": np.array([2, 2, 2]),
    #     "sample1_low": np.array([3, 3, 3]),
    #     "sample3_medium": np.array([4, 4, 4]),
    # }

    # # Test with "any" (default)
    # result_any = partial_selection(test_dict, "sample1", "high")
    # print("Any match:", list(result_any.keys()) == ["sample1_high", "sample2_high", "sample1_low"])
    # # Expected: ['sample1_high']

    # # Test with "all"
    # result_all = partial_selection(test_dict, "sample1", "high", any_all="all")
    # print("All match:", list(result_all.keys()) == ["sample1_high"])
    # # Expected: ['sample1_high']

    # # Test with no matches for "all"
    # result_no_match = partial_selection(test_dict, "sample1", "medium", any_all="all")
    # print("No match:", list(result_no_match.keys()) == [])
    # # Expected: []

    # # Test with single key
    # result_single = partial_selection(test_dict, "sample3")
    # print("Single key:", list(result_single.keys()) == ["sample3_medium"])

    # # DataFrame with single index
    # df_single = pd.DataFrame(
    #     {"value": [10, 20, 30, 40, 50]},
    #     index=["sample1_A", "sample2_B", "sample1_C", "sample3_D", "sample2_C"],
    # )

    # # Test with "any" (default)
    # result_df_any = partial_selection(df_single, "sample1")
    # print("Any match - single index:", result_df_any.index.tolist() == ["sample1_A", "sample1_C"])

    # # Expected: DataFrame with rows for "sample1_A", "sample1_C"

    # # Test with "all"
    # result_df_all = partial_selection(df_single, "sample1", "C", any_all="all")
    # print("All match - single index:", result_df_all.index.tolist() == ["sample1_C"])
    # # Expected: DataFrame with row for "sample1_C"

    # # Test with no matches
    # result_df_no_match = partial_selection(df_single, "sample4")
    # print("No match - single index:", result_df_no_match.index.tolist() == [])
    # # Expected: Empty DataFrame

    # # DataFrame with MultiIndex
    # arrays = [
    #     ["sample1", "sample1", "sample2", "sample2", "sample3"],
    #     ["high", "low", "high", "low", "medium"],
    # ]
    # tuples = list(zip(*arrays))
    # index = pd.MultiIndex.from_tuples(tuples, names=["sample", "level"])
    # df_multi = pd.DataFrame({"value": [10, 20, 30, 40, 50]}, index=index)

    # # Test with "any" (default)
    # result_multi_any = partial_selection(df_multi, "sample1")
    # print(
    #     "Any match - multi index:",
    #     result_multi_any.index.tolist() == [("sample1", "high"), ("sample1", "low")],
    # )
    # # Expected: DataFrame with first two rows (sample1/high, sample1/low)

    # # Test with "all"
    # result_multi_all = partial_selection(df_multi, "sample1", "high", any_all="all")
    # print("All match - multi index:", result_multi_all.index.tolist() == [("sample1", "high")])
    # # Expected: DataFrame with first row only (sample1/high)

    # # Test with value across different index levels
    # result_multi_cross = partial_selection(df_multi, "high")
    # print(
    #     "Cross-level match:",
    #     result_multi_cross.index.tolist() == [("sample1", "high"), ("sample2", "high")],
    # )
    # # Expected: DataFrame with rows containing "high"

    # # Define a custom lambda function using regex
    # # This will match entries that start with "sample" followed by a digit
    # def sample_digit_matcher(x):
    #     return bool(re.match(r"^(sample\d|\('sample\d)", str(x)))
    #     # return bool(
    #     #     re.match(r"sample\d", str(x))
    #     #     if isinstance(x, str)
    #     #     else any(re.match(r"sample\d", str(val)) for val in x)
    #     # )

    # # Test with dictionary
    # result_custom_dict = partial_selection(test_dict, any_all=sample_digit_matcher)
    # print(
    #     "Regex match - dictionary:",
    #     list(result_custom_dict.keys())
    #     == ["sample1_high", "sample2_high", "sample1_low", "sample3_medium"],
    # )
    # # Expected: All keys (all start with "sample" followed by a digit)

    # # Test with single index DataFrame
    # result_custom_df = partial_selection(df_single, any_all=sample_digit_matcher)
    # print(
    #     "Regex match - single index DataFrame:",
    #     result_custom_df.index.tolist()
    #     == ["sample1_A", "sample2_B", "sample1_C", "sample3_D", "sample2_C"],
    # )
    # # Expected: All rows (all indices start with "sample" followed by a digit)

    # # Test with MultiIndex DataFrame
    # result_custom_multi = partial_selection(df_multi, any_all=sample_digit_matcher)
    # # print("Regex match - multi index DataFrame:", result_custom_multi.index.tolist() == [])
    # print(
    #     "Regex match - multi index DataFrame:",
    #     result_custom_multi.index.tolist()
    #     == [
    #         ("sample1", "high"),
    #         ("sample1", "low"),
    #         ("sample2", "high"),
    #         ("sample2", "low"),
    #         ("sample3", "medium"),
    #     ],
    # )
    # # Expected: All rows (all contain a value starting with "sample" followed by a digit)

# %%

# ARCHIVE:

# def sort_exp_params(
#     *values: list[float] | np.ndarray,
# ) -> tuple[list[float], ...]:
#     """
#     Sorts exponential fit parameters so that (a, tau) pairs are ordered by tau ascending.
#     All additional arrays (i.e. stds or names) in *values are sorted in the same order as values[0].

#     Parameters
#     ----------
#     *values : list[float] or np.ndarray
#         Flat list/array of parameters: [a0, tau0, a1, tau1, ..., (b0)]

#     Returns
#     -------
#     tuple[list[float], ...]
#         Sorted arrays, each with (a, tau) pairs ordered by tau ascending, offset (if present) last.
#     """
#     vals = [list(arr) for arr in values]

#     if not vals or len(vals[0]) <= 3:
#         return tuple(vals)

#     offsets = [[arr[-1]] if len(arr) % 2 else [] for arr in vals]

#     pairs = [[(arr[2 * i], arr[2 * i + 1]) for arr in vals] for i in range(len(vals[0]) // 2)]

#     sorted_vals = [
#         [val[j][i] for j in range(len(val))]
#         for val in sorted(pairs, key=lambda x: x[0][1])
#         for i in range(2)
#     ]

#     return tuple(list(v) + o for v, o in zip(zip(*sorted_vals), offsets))


# def sort_exp_params(
#     params: list[float] | np.ndarray,
#     stds: list[float] | np.ndarray,
# ) -> tuple[list[float], list[float]]:
#     """
#     Sorts exponential fit parameters so that (a, tau) pairs are ordered by tau ascending.
#     If stds are provided, sorts them in the same order as their corresponding params.

#     The input should be a flat list or array: [a0, tau0, a1, tau1, ..., b0?]
#     If there is an odd number of parameters, the last is assumed to be the offset (b0) and is left in place.

#     Parameters
#     ----------
#     params : list[float] or np.ndarray
#         Flat list/array of parameters: [a0, tau0, a1, tau1, ..., (b0)]
#     stds : list[float] or np.ndarray, optional
#         Flat list/array of std devs, same length/order as params.

#     Returns
#     -------
#     sorted_params : list[float]
#         Sorted parameter list with (a, tau) pairs ordered by tau ascending, offset (if present) last.
#     sorted_stds : list[float] or None
#         Sorted std devs in the same order as sorted_params, or None if stds was not provided.
#     """
#     params = list(params)
#     stds = list(stds)
#     if len(params) <= 3:
#         return params, stds

#     offset = params[-1] if len(params) % 2 else None
#     offset_std = stds[-1] if len(stds) % 2 else None

#     vals = [
#         (params[2 * i], params[2 * i + 1], stds[2 * i], stds[2 * i + 1])
#         for i in range(len(params) // 2)
#     ]
#     sorted_vals = [
#         (v, val[i + 2]) for val in sorted(vals, key=lambda x: x[1]) for i, v in enumerate(val[:2])
#     ]
#     sorted_params, sorted_stds = map(list, zip(*sorted_vals))
#     if offset is not None:
#         sorted_params.append(offset)
#         sorted_stds.append(offset_std)
#     return sorted_params, sorted_stds
# def remove_low_freq_baseline(
#     arr: np.ndarray, fs: float, cutoff: float = 0.01, order: int = 2
# ) -> np.ndarray:
#     """
#     Remove low-frequency baseline using a Butterworth high-pass filter.

#     Parameters
#     ----------
#     arr : np.ndarray
#         Input signal.
#     fs : float
#         Sampling frequency (Hz).
#     cutoff : float
#         Cutoff frequency for high-pass filter (Hz).
#     order : int
#         Filter order.

#     Returns
#     -------
#     np.ndarray
#         Filtered signal.
#     """
#     b, a = butter(order, cutoff / (0.5 * fs), btype="high")  # type: ignore
#     return filtfilt(b, a, arr)


# def transform_to_permittivity(
#     df: pd.DataFrame,
#     current_col: str = "Current",
#     voltage_col: str = "Voltage",
#     time_col: str = "time",
#     *,
#     dims_cm: tuple[float, float] = (25.0, 0.045),
#     decay_mod: dict | None = None,
#     window_type: str | tuple | None = None,
#     remove_mean: int | float | Callable | None = 0.075,
#     padding: bool = True,
#     logspace_freq: bool = True,
#     equal_length: bool = True,
#     filter_cutoff: float = 0.0,
#     filter_order: int = 2,
#     dt_function: Callable[[np.ndarray], float] | None = None,
#     pre_spline: str = "cubic",
#     post_spline: str = "cubic",
#     **kwargs: Any,
# ) -> pd.DataFrame:
#     """
#     Transform current/voltage decay data to time-domain and frequency-domain permittivity.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         DataFrame with columns for time, current, and voltage.
#     current_col : str, optional
#         Name of the current column (default: "Current").
#     voltage_col : str, optional
#         Name of the voltage column (default: "Voltage").
#     time_col : str, optional
#         Name of the time column (default: "time").
#     dims_cm : tuple of float, optional
#         Sample geometry as (area_cm2, thickness_cm) (default: (25.0, 0.045)).
#     decay_mod : dict or None, optional
#         If True, apply gradient and smoothing to decay signal (default: True).
#     window_type : str, tuple, or None, optional
#         Window type for get_window (default: ("tukey", 0.5)).
#     remove_mean : int, float, Callable, or None, optional
#         Whether to subtract the steady-state mean before FFT (default: 0.075).
#     padding : bool, optional
#         Whether to zero-pad to next power of two for FFT (default: True).
#     logspace_freq : bool, optional
#         Whether to use logspace for frequency axis (default: True).
#     equal_length : bool, optional
#         Whether to spline FFT result to output frequency axis (default: True).
#     filter_cutoff : float, optional
#         Cutoff frequency for high-pass filter (Hz, default: 0.01).
#     filter_order : int, optional
#         Filter order for high-pass filter (default: 2).
#     dt_function : Callable or None, optional
#         Function to compute dt from np.diff(time_array). Should accept the diff array and return a float.
#         If None, defaults to np.median.
#     pre_spline : str, optional
#         Spline type for interpolation ("cubic" for CubicSpline, "pchip" for PchipInterpolator).
#     **kwargs : dict
#         Additional keyword arguments.

#     Returns
#     -------
#     pd.DataFrame
#         DataFrame with columns:
#         - time, voltage, current, decay_raw, decay, ss decay
#         - frequency, permittivity, real perm, imag perm, mag perm, phase perm, loss tangent
#     """
#     kwargs.setdefault("extrapolate", True)
#     kwargs.setdefault("bc_type", "natural")
#     # --- Calculate decay implementation ---
#     if dims_cm[1] >= 50:
#         dims_cm = (dims_cm[0], float(dims_cm[1]) * 1e-4)

#     # Use smallest dt for uniform grid
#     t = df[time_col].to_numpy(copy=True)
#     diffs = np.diff(t)
#     if dt_function is None:
#         dt = float(np.median(diffs))
#     else:
#         dt = dt_function(diffs)
#     n_points = int(np.round((t[-1] - t[0]) / dt)) + 1

#     # Use CubicSpline for interpolation and optional extrapolation
#     if pre_spline == "pchip":
#         interp_current = PchipInterpolator(
#             t, df[current_col].to_numpy(copy=True), extrapolate=kwargs["extrapolate"]
#         )
#         interp_voltage = PchipInterpolator(
#             t, df[voltage_col].to_numpy(copy=True), extrapolate=kwargs["extrapolate"]
#         )
#     else:
#         interp_current = CubicSpline(t, df[current_col].to_numpy(copy=True), **kwargs)
#         interp_voltage = CubicSpline(t, df[voltage_col].to_numpy(copy=True), **kwargs)

#     t_uniform = np.linspace(t[0], t[-1], n_points)
#     current_uniform = interp_current(t_uniform)
#     voltage_uniform = interp_voltage(t_uniform)

#     decay_uniform = (current_uniform / float(dims_cm[0])) / (
#         VACUUM_PERMITTIVITY * (voltage_uniform / float(dims_cm[1]))
#     )

#     if filter_cutoff > 0:
#         fs = 1.0 / dt  # Sampling frequency in Hz
#         decay_uniform = remove_low_freq_baseline(
#             decay_uniform, fs, cutoff=filter_cutoff, order=filter_order
#         )

#     n_fft = n_points
#     decay_uniform_p = decay_uniform.copy()
#     if decay_mod is not None:
#         if decay_mod.get("mode", "interp") == "gradient":
#             decay_uniform_p = np.gradient(
#                 decay_uniform, t_uniform, edge_order=decay_mod.get("edge_order", 1)
#             )
#         else:
#             decay_mod["delta"] = dt
#             decay_mod.setdefault("deriv", 1)
#             decay_mod.setdefault("polyorder", 2)
#             decay_mod.setdefault("window_length", 3)
#             win_len = (
#                 int(abs(decay_mod["window_length"]) * n_points)
#                 if abs(decay_mod["window_length"]) < 1
#                 else int(abs(decay_mod["window_length"]))
#             )
#             decay_mod["window_length"] = max(decay_mod["polyorder"] + 1, win_len)
#             decay_mod["window_length"] = min(decay_mod["window_length"], n_points - 1)
#             decay_uniform_p = savgol_filter(decay_uniform, **decay_mod)

#     # --- get_window implementation ---
#     # Use get_window for flexible windowing; Tukey is default, alpha can be adjusted
#     if window_type is not None:
#         window = get_window(window_type, n_points)
#     else:
#         window = np.ones(n_points)
#     # ---------------------------------

#     # --- New suggestion: windowing always applied before FFT ---
#     # This helps suppress edge spikes and improves amplitude reliability
#     final_mean = 0.0
#     if callable(remove_mean):
#         final_mean = remove_mean(decay_uniform)
#         decay_uniform_p = decay_uniform - final_mean
#     elif isinstance(remove_mean, int) and remove_mean < 0:
#         final_mean = np.nanmean(decay_uniform[remove_mean:])
#         decay_uniform_p = decay_uniform - final_mean
#     elif isinstance(remove_mean, (int, float)):
#         perc = remove_mean if 0 < remove_mean < 1 else abs(remove_mean) / 100.0
#         final_mean = np.nanmean(decay_uniform[-max(1, int(n_points * perc)) :])
#         decay_uniform_p = decay_uniform - final_mean
#     decay_uniform_p_windowed = decay_uniform_p * window
#     # ----------------------------------------------------------

#     if remove_mean and padding:
#         n_fft = 2 ** int(np.ceil(np.log2(n_points)))
#         perm_fft = fft(np.pad(decay_uniform_p_windowed, (0, n_fft - n_points), "constant"))
#     else:
#         perm_fft = fft(decay_uniform_p_windowed)

#     freq = fftfreq(n_fft, d=dt)
#     pos_mask = freq >= 0
#     freq = freq[pos_mask]
#     permittivity_compl = perm_fft[pos_mask]  # type: ignore
#     permittivity_real = np.real(permittivity_compl)
#     permittivity_imag = np.imag(permittivity_compl)
#     permittivity_mag = np.abs(permittivity_compl)
#     permittivity_phase = np.angle(permittivity_compl, deg=True)

#     if equal_length:
#         # Spline the FFT result to a logspace or linspace frequency axis
#         freq_min = np.max([np.min(freq[freq > 0]), 1e-10])  # type: ignore
#         freq_max = np.max(freq)
#         if logspace_freq:
#             freq_out = np.logspace(np.log10(freq_min), np.log10(freq_max), n_points)
#         else:
#             freq_out = np.linspace(freq_min, freq_max, n_points)

#         # Only interpolate where FFT is defined (freq > 0)
#         # Use selected spline for post-processing interpolation
#         if post_spline == "pchip":
#             # s_fd = PchipInterpolator(freq, permittivity_compl, extrapolate=kwargs["extrapolate"])
#             s_real = PchipInterpolator(freq, permittivity_real, extrapolate=kwargs["extrapolate"])(
#                 freq_out
#             )
#             s_imag = PchipInterpolator(freq, permittivity_imag, extrapolate=kwargs["extrapolate"])(
#                 freq_out
#             )
#             s_mag = PchipInterpolator(freq, permittivity_mag, extrapolate=kwargs["extrapolate"])(
#                 freq_out
#             )
#             s_phase = PchipInterpolator(
#                 freq, permittivity_phase, extrapolate=kwargs["extrapolate"]
#             )(freq_out)
#             s_fd = s_real + 1j * s_imag
#         else:
#             s_fd = CubicSpline(freq, permittivity_compl, **kwargs)(freq_out)
#             s_real = CubicSpline(freq, permittivity_real, **kwargs)(freq_out)
#             s_imag = CubicSpline(freq, permittivity_imag, **kwargs)(freq_out)
#             s_mag = CubicSpline(freq, permittivity_mag, **kwargs)(freq_out)
#             s_phase = CubicSpline(freq, permittivity_phase, **kwargs)(freq_out)

#         try:
#             loss = s_imag / s_real
#         except (FloatingPointError, ValueError):
#             imag = s_imag
#             real = s_real
#             loss = imag / np.where(real != 0, real, 1e-32)

#         result = pd.DataFrame(
#             {
#                 "time": t_uniform,
#                 "voltage": voltage_uniform,
#                 "current": current_uniform,
#                 "decay_raw": decay_uniform,
#                 "decay": decay_uniform_p,
#                 "ss decay": final_mean,
#                 "frequency": freq_out,
#                 "permittivity": s_fd,
#                 "real perm": s_real,
#                 "imag perm": s_imag,
#                 "mag perm": s_mag,
#                 "phase perm": s_phase,
#                 "loss tangent": loss,
#             }
#         )
#     else:
#         delta_fd = [np.nan] * (n_points - len(freq)) if n_points > len(freq) else []
#         result = pd.DataFrame(
#             {
#                 "time": t_uniform,
#                 "voltage": voltage_uniform,
#                 "current": current_uniform,
#                 "decay_raw": decay_uniform,
#                 "decay": decay_uniform_p,
#                 "ss decay": final_mean,
#                 "frequency": np.concatenate([freq, delta_fd]),
#                 "permittivity": np.concatenate([permittivity_compl, delta_fd]),
#                 "real perm": np.concatenate([permittivity_real, delta_fd]),
#                 "imag perm": np.concatenate([permittivity_imag, delta_fd]),
#                 "mag perm": np.concatenate([permittivity_mag, delta_fd]),
#                 "phase perm": np.concatenate([permittivity_phase, delta_fd]),
#                 "loss tangent": np.concatenate([permittivity_imag / permittivity_real, delta_fd]),
#             }
#         )

#     result.attrs = {
#         "ss decay": final_mean,
#         "DC permittivity": permittivity_real[0],
#         "DC imag": permittivity_real[0],
#         "inf permittivity": result["real perm"].dropna().iloc[-1],
#         "inf imag": result["imag perm"].dropna().iloc[-1],
#     }

#     if np.any(result["real perm"] < 0):
#         percent_negative = 100.0 * np.sum(result["real perm"] < 0) / len(result)
#         warnings.warn(
#             f"Negative values found in real permittivity: {percent_negative:.2f}% < 0. "
#             "This may indicate a malformed FFT conversion.",
#             RuntimeWarning,
#             stacklevel=2,
#         )

#     return result


# def is_log_spaced(arr: np.ndarray, rtol: float = 1e-2) -> bool:
#     """
#     Check if an array is approximately logarithmically spaced.

#     Parameters
#     ----------
#     arr : np.ndarray
#         Input array (must be positive and 1D).
#     rtol : float, optional
#         Relative tolerance for uniformity of log spacing.

#     Returns
#     -------
#     bool
#         True if array is log-spaced, False otherwise.
#     """
#     arr = np.asarray(arr)
#     if np.any(arr <= 0):
#         return False
#     log_diffs = np.diff(np.log10(arr))
#     return np.allclose(log_diffs, log_diffs[0], rtol=rtol)


# def find_wide_peaks(
#     freq: np.ndarray,
#     arr: np.ndarray,
#     min_width: float = 5,
#     min_prominence: float = 0.05,
#     background_subtract: bool = True,
#     min_peaks: int = 2,
#     weighting_func: Callable | None = None,
# ) -> pd.DataFrame:
#     """
#     Find wide, prominent peaks in the input array, optionally using background subtraction.

#     Parameters
#     ----------
#     freq : np.ndarray
#         Frequency array.
#     arr : np.ndarray
#         Input array (e.g., 'mag perm', 'imag perm', etc.).
#     min_width : float, optional
#         Minimum width (in number of points) for a peak to be considered valid.
#     min_prominence : float, optional
#         Minimum prominence for a peak to be considered valid.
#     background_subtract : bool, optional
#         If True, also search for peaks after background subtraction or if no peaks found.
#     min_peaks : int, optional
#         Minimum number of peaks to find before reducing min_prominence recursively.
#     weighting_func : callable(width, prominences) | None, optional
#         If provided, a function that takes two arguments: the normalized width and prominences arrays,
#         and returns a new weight array. If None, the default weighting is used.

#     Returns
#     -------
#     pd.DataFrame
#         DataFrame with columns: 'peak_freq', 'peak_arr', 'width', 'prominence', 'weight', 'bkg_subtracted'.
#     """

#     def calc_weights(wid, prom, ar):
#         """
#         Calculate weights based on widths and prominences.
#         If weighting_func is provided, use it to calculate weights.
#         """
#         if not len(wid):
#             return wid, prom, np.array([])
#         n_widths = wid / len(ar)
#         n_prominences = prom / (np.ptp(ar) if np.ptp(ar) > 0 else 1e-32)
#         if weighting_func is not None:
#             # weighting_func(width, prominences)
#             return n_widths, n_prominences, weighting_func(n_widths, n_prominences)
#         return n_widths, n_prominences, np.sqrt(n_widths * n_prominences)

#     peaks, props = find_peaks(arr, width=min_width, prominence=min_prominence)
#     widths, prominences, weights = calc_weights(props["widths"], props["prominences"], arr)
#     result = pd.DataFrame(
#         {
#             "peak_freq": freq[peaks],
#             "peak_arr": arr[peaks],
#             "width": widths,
#             "prominence": prominences,
#             "weight": weights,
#             "bkg_subtracted": 0.0,  # False
#         },
#         dtype=float,
#     )

#     if (background_subtract or result.empty) and len(arr) > 2:
#         x = np.log10(freq) if is_log_spaced(freq) else freq
#         coeffs = np.polyfit(x, arr, 1)
#         baseline = np.polyval(coeffs, x)
#         peaks, props = find_peaks(arr - baseline, width=min_width, prominence=min_prominence)
#         widths, prominences, weights = calc_weights(props["widths"], props["prominences"], arr)
#         result_bkg = pd.DataFrame(
#             {
#                 "peak_freq": freq[peaks],
#                 "peak_arr": arr[peaks],
#                 "width": widths,
#                 "prominence": prominences,
#                 "weight": weights,
#                 "bkg_subtracted": 1.0,  # True
#             },
#             dtype=float,
#         )

#         result = pd.concat([result, result_bkg], ignore_index=True)
#         result = result.drop_duplicates(subset="peak_freq", keep="first")

#     # Recursively reduce min_prominence if not enough peaks found
#     if len(result) < min_peaks and min_prominence > 1e-6:
#         return find_wide_peaks(
#             freq,
#             arr,
#             min_width=min_width,
#             min_prominence=min_prominence / 2,
#             background_subtract=background_subtract,
#             min_peaks=min_peaks,
#         )

#     # If still no peaks found, use the max of arr as a fallback
#     if result.empty:
#         widths, prominences, weights = calc_weights(
#             np.array([1.0]), np.array([max(np.min(np.abs(np.diff(arr))), 1e-32)]), arr
#         )
#         result = pd.DataFrame(
#             {
#                 "peak_freq": freq[np.argmax(arr)],
#                 "peak_arr": np.max(arr),
#                 "width": widths,
#                 "prominence": prominences,
#                 "weight": weights,
#                 "bkg_subtracted": -1.0,
#             },
#             dtype=float,
#         )

#     return result


# def _fit_peak_trend(
#     all_peaks_df: pd.DataFrame,
#     step_size: int = 10,
#     slope_min: float = -np.inf,
#     slope_max: float = np.inf,
#     mask_mode: str | tuple[str, ...] = "quantile",
#     mode_use: str = "backup",
#     select_method: str = "median, abs_max",
# ) -> tuple[float, float]:
#     """
#     Fit a global trend (slope) of log10(peak_freq) vs temp, masking a fraction of edge peaks by value.
#     Iteratively increases edge_frac until the std of the residuals stabilizes.
#     Limits slope to [slope_min, slope_max] and prefers expected_slope if provided.

#     Parameters
#     ----------
#     all_peaks_df : pd.DataFrame
#         DataFrame with MultiIndex including 'temp' and a column for 'peak_freq'.
#     step_size : int, optional
#         Step size for increasing edge_frac (default: 10).
#     slope_min : float, optional
#         Minimum allowed slope value.
#     slope_max : float, optional
#         Maximum allowed slope value.
#     mask_mode : str or tuple of str, optional
#         Masking mode(s) to apply. Can be a single string or a tuple of strings.
#     mode_use : str, optional
#         Determines how mask_mode(s) are applied when multiple are provided:
#             - "parallel": All mask_modes are applied at once, so the mask becomes more restrictive
#               with each mode (mask is the intersection of all).
#             - "sequential": Each mask_mode is tried independently, updating a list of valid fits.
#             - "backup": Only the first mask_mode is used, however recursion is used to
#               try the next mask_mode if not enough valid fits are found. Default case.
#     select_method : str, optional
#         String specifying how to select the "best" slope from valid fits. Multiple methods can be combined
#         using spaces, commas, or both (e.g., "median, abs_max min_std"). Methods evaluate valid slopes
#         or the standard error of the estimate (SEE) of the fit for a valid slope.
#         Valid methods: [median, mean, min, max, abs_median, abs_mean, abs_min, abs_max, median_see,
#             mean_see, min_see, max_see].
#             Note: 'med' is a valid alias for 'median' and 'std' is a valid alias for 'see'.

#     Returns
#     -------
#     tuple[float, float]
#         Slope and intercept of the global fit (log10(peak_freq) vs temp).
#     """
#     df = all_peaks_df[all_peaks_df["peak_freq"].notna()].copy()

#     if len(df) < 3:
#         raise ValueError("Not enough points for global peak trend fit.")

#     if slope_min > slope_max:
#         slope_min, slope_max = slope_max, slope_min
#     elif slope_min == slope_max:
#         raise ValueError("slope_min and slope_max must be different values.")

#     peak_df = pd.DataFrame()
#     peak_df["temps"] = df.index.get_level_values("temp").astype(float).to_numpy(copy=True)
#     peak_df["peaks"] = np.log10(df["peak_freq"].to_numpy(copy=True))
#     peak_df["weights"] = df["weight"].to_numpy(copy=True)
#     peak_df = peak_df.sort_values("temps").reset_index(drop=True)

#     coeffs = list(np.polyfit(peak_df["temps"], peak_df["peaks"], 1, w=peak_df["weights"]))
#     see_history = [np.std(peak_df["peaks"] - np.polyval(coeffs, peak_df["temps"]))]
#     coeffs_history = [coeffs]
#     valid_see = []
#     valid_slopes = []
#     if slope_min <= coeffs[0] <= slope_max:
#         valid_slopes.append(coeffs[0])
#         valid_see.append(see_history[-1])

#     n_temps = peak_df["temps"].nunique()
#     if len(peak_df) <= n_temps:
#         return coeffs[0], coeffs[1]

#     step_size = int(np.clip(step_size, 1, 16))

#     if isinstance(mask_mode, str):
#         mask_mode = (mask_mode,)

#     remaining_masks = ()
#     if mode_use == "parallel":
#         # mode_groups: tuple[tuple[str, ...]]; Setup to iter in the inner loop
#         mode_groups = (mask_mode,)
#     elif mode_use == "sequential":
#         # mode_groups: tuple[tuple[str], ...]; Setup to iter in the outer loop
#         mode_groups = tuple((m,) for m in mask_mode)
#     else:  # backup/default
#         # mode_groups: tuple[tuple[str]]; Setup to recurse the function as needed
#         remaining_masks = mask_mode[1:]
#         mode_groups = ((mask_mode[0],),)

#     min_valid = 3
#     for m_group in mode_groups:
#         # Only one iteration unless mode_use is "sequential"
#         n_temps = peak_df["temps"].nunique()
#         for frac in range(step_size, 50 // step_size * step_size, step_size):
#             mask = np.ones(len(peak_df), dtype=bool)
#             for m_mode in m_group:
#                 # Only one iteration unless mode_use is "parallel"
#                 if m_mode == "quantile":
#                     lower_thresh = peak_df["peaks"].quantile(frac / 100)
#                     upper_thresh = peak_df["peaks"].quantile((100 - frac) / 100)
#                     mask *= (peak_df["peaks"] >= lower_thresh) & (peak_df["peaks"] <= upper_thresh)
#                 elif m_mode == "weight":
#                     lower_thresh = peak_df["weights"].quantile(frac / 100)
#                     mask *= peak_df["weights"] >= lower_thresh
#                 elif "temp" in m_mode:
#                     n_temps -= 1
#                     if n_temps < 2:
#                         break
#                     is_rev = True if "low" in m_mode else False
#                     temp_list = sorted(peak_df["temps"].unique(), reverse=is_rev)
#                     mask *= peak_df["temps"].isin(temp_list[: -int(frac / step_size)])

#             if n_temps < 2:
#                 break

#             temps_masked = peak_df.loc[mask, "temps"]
#             if temps_masked.nunique() < n_temps:
#                 if step_size > 1:
#                     return _fit_peak_trend(
#                         all_peaks_df,
#                         step_size=step_size // 2,
#                         slope_min=slope_min,
#                         slope_max=slope_max,
#                         mask_mode=mask_mode,
#                         mode_use=mode_use,
#                         select_method=select_method,
#                     )
#                 break

#             coeffs = list(
#                 np.polyfit(
#                     temps_masked, peak_df.loc[mask, "peaks"], 1, w=peak_df.loc[mask, "weights"]
#                 )
#             )
#             see_history.append(
#                 np.std(peak_df.loc[mask, "peaks"] - np.polyval(coeffs, temps_masked))
#                 / np.sqrt(len(temps_masked))
#             )

#             coeffs_history.append(coeffs)
#             if slope_min <= coeffs[0] <= slope_max:
#                 valid_slopes.append(coeffs[0])
#                 valid_see.append(see_history[-1])

#             if len(valid_slopes) >= min_valid and all(
#                 abs(n - np.median(valid_see[-3:])) < 1e-3 for n in valid_see[-3:]
#             ):
#                 min_valid += 3
#                 break

#     if remaining_masks and len(valid_slopes) < 3:
#         # Only called as a backup and when mode_use is not "sequential" or "parallel"
#         return _fit_peak_trend(
#             all_peaks_df,
#             step_size=step_size,
#             slope_min=slope_min,
#             slope_max=slope_max,
#             mask_mode=remaining_masks,
#             mode_use=mode_use,
#             select_method=select_method,
#         )
#     elif valid_slopes:
#         best_slopes = []
#         methods = re.split(r"[,\s]+", select_method)
#         if "med" in methods or "median" in methods:
#             best_slopes.append(np.median(valid_slopes))
#         if "mean" in methods:
#             best_slopes.append(np.mean(valid_slopes))
#         if "max" in methods:
#             best_slopes.append(valid_slopes[np.argmax(valid_slopes)])
#         if "min" in methods:
#             best_slopes.append(valid_slopes[np.argmin(valid_slopes)])
#         if "abs_med" in methods or "abs_median" in methods:
#             best_slopes.append(np.median(np.abs(valid_slopes)))
#         if "abs_mean" in methods:
#             best_slopes.append(np.mean(np.abs(valid_slopes)))
#         if "abs_max" in methods:
#             best_slopes.append(valid_slopes[np.argmax(np.abs(valid_slopes))])
#         if "abs_min" in methods:
#             best_slopes.append(valid_slopes[np.argmin(np.abs(valid_slopes))])
#         if "min_std" in methods or "min_see" in methods:
#             best_slopes.append(valid_slopes[np.argmin(valid_see)])
#         if "max_std" in methods or "max_see" in methods:
#             best_slopes.append(valid_slopes[np.argmax(valid_see)])
#         if "mean_std" in methods or "mean_see" in methods:
#             weights = 1 / (np.array(valid_see) + 1e-32)
#             weights /= weights.sum()
#             best_slopes.append(np.sum(np.array(valid_slopes) * weights))
#         if (
#             "median_std" in methods
#             or "med_std" in methods
#             or "median_see" in methods
#             or "med_see" in methods
#         ):
#             best_slopes.append(
#                 valid_slopes[np.argmin(np.abs(np.array(valid_see) - np.median(valid_see)))]
#             )

#         best_slope = np.mean(best_slopes) if best_slopes else np.median(valid_slopes)
#         idx = np.argmin([abs(s[0] - best_slope) for s in coeffs_history])
#         coeffs_history.append(coeffs_history[idx])
#     else:
#         # If no valid slopes found, return the best slope and intercept from the last coeffs
#         if slope_min <= 0 <= slope_max:
#             slope = 0.0
#         else:
#             slope = slope_min if abs(slope_min) < abs(slope_max) else slope_max
#         intercept = np.average(
#             peak_df["peaks"] - slope * peak_df["temps"], weights=peak_df["weights"]
#         )
#         coeffs_history.append([slope, intercept])

#     return coeffs_history[-1][0], coeffs_history[-1][1]


# def collect_peak_summary_df(
#     perm_dict: dict,
#     column: str = "imag perm",
#     min_width: float = 0.05,
#     min_prominence: float = 0.05,
#     min_peaks: int = 2,
#     weighting_func: Callable | None = None,
#     fit_step: int = 10,
#     slope_min: float = -np.inf,
#     slope_max: float = np.inf,
#     fit_mask_mode: str | tuple[str, ...] = "quantile",
#     fit_mode_use: str = "sequential",
#     fit_select_method: str = "median, abs_max",
#     weight_diff: bool = False,
#     normalize_weight_mode: str | tuple[str, ...] | None = None,
# ) -> tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     Summarize the main peak for each entry in a permittivity dictionary,
#     using all peaks and temperature trends to select the best peak for each entry.
#     If group slope is too far from global, revert to global slope.

#     Parameters
#     ----------
#     perm_dict : dict
#         Dictionary of DataFrames from transform_to_permittivity, keys are MultiIndex tuples.
#     column : str, optional
#         Column to evaluate for peak finding (default: "imag perm").
#     min_width : float, optional
#         Minimum width (in number of points) for a peak to be considered valid.
#     min_prominence : float, optional
#         Minimum prominence for a peak to be considered valid.
#     min_peaks : int, optional
#         Minimum number of peaks to find before reducing min_prominence recursively.
#     slope_min : float, optional
#         Minimum allowed slope value.
#     slope_max : float, optional
#         Maximum allowed slope value.
#     global_slope_tol : float, optional
#         Relative tolerance for group slope vs global slope (default: 0.2).

#     Returns
#     -------
#     pd.DataFrame
#         DataFrame with MultiIndex (from keys + 'peaks') and columns:
#         'peak_freq', 'peak_mag', 'width', 'prominence', 'bkg_subtracted'
#     """
#     peaks_df = pd.DataFrame(
#         index=list(perm_dict.keys()),
#         columns=["peak_freq", "peak_arr", "width", "prominence", "weight", "bkg_subtracted"],
#         dtype=float,
#     )
#     peaks_df = form_std_df_index(peaks_df)

#     all_peaks_dict = {}
#     for key, df in perm_dict.items():
#         freq = df["frequency"].to_numpy(copy=True)
#         arr = df[column].to_numpy(copy=True)
#         if np.sum(arr < 0) > 0.5 * len(arr):
#             arr = -1 * arr
#         min_width_pts = max(1, int(np.round(len(arr) * min_width)))
#         all_peaks_dict[key] = find_wide_peaks(
#             freq,
#             arr,
#             min_width=min_width_pts,
#             min_prominence=min_prominence,
#             min_peaks=min_peaks,
#             weighting_func=weighting_func,
#         )
#         if len(all_peaks_dict[key]) < min_peaks and min_width_pts > 3:
#             all_peaks_dict[key] = find_wide_peaks(
#                 freq,
#                 arr,
#                 min_width=3,
#                 min_prominence=min_prominence,
#                 min_peaks=min_peaks,
#                 weighting_func=weighting_func,
#             )
#         if all_peaks_dict[key].empty:
#             all_peaks_dict[key].loc[0] = np.nan

#     names = ["sample_name", "condition", "temp", "run", "segment", "peaks"]

#     all_peaks_df = form_std_df_index(pd.concat(all_peaks_dict, names=names), names=tuple(names))

#     all_peaks_df = _normalize_peak_weights(all_peaks_df, normalize_weight_mode)

#     # all_peaks_df["weight"] = all_peaks_df["weight"].where(all_peaks_df["weight"] > 0, 1e-32)

#     # Compute global slope and intercept for all peaks
#     global_coeffs = _fit_peak_trend(
#         all_peaks_df,
#         step_size=fit_step,
#         slope_min=slope_min,
#         slope_max=slope_max,
#         mask_mode=fit_mask_mode,
#         mode_use=fit_mode_use,
#         select_method=fit_select_method,
#     )

#     # Add "m", "b", and "diffs" columns to all_peaks_df and peaks_df
#     for df in (all_peaks_df, peaks_df):
#         df["global_coeffs"] = 0.0
#         for col in ["pred_fit", "m", "b", "fit", "diffs", "peak_quantile"]:
#             df[col] = np.nan

#     grouped = all_peaks_df.groupby(["sample_name", "condition"], observed=True)
#     for key, group in grouped:
#         # Fit the group to find the slope and intercept using the helper
#         fit_df = group.dropna(subset=["peak_freq"]).copy()

#         if fit_df.empty:
#             continue

#         fit_df["peak_quantile"] = fit_df["peak_freq"].apply(np.log10).rank(pct=True)

#         if len(fit_df) < 3:
#             coeffs = global_coeffs
#             fit_df["global_coeffs"] = 1.0
#         else:
#             # Use the helper to get group slope and intercept
#             coeffs = _fit_peak_trend(
#                 fit_df,
#                 step_size=fit_step,
#                 slope_min=slope_min,
#                 slope_max=slope_max,
#                 mask_mode=fit_mask_mode,
#                 mode_use=fit_mode_use,
#                 select_method=fit_select_method,
#             )

#         peaks_df.attrs[key] = coeffs
#         fit_df["m"] = coeffs[0]
#         fit_df["b"] = coeffs[1]

#         fit_df["fit"] = coeffs[0] * fit_df.index.get_level_values("temp").astype(float) + coeffs[1]
#         fit_df["pred_fit"] = 10 ** fit_df["fit"]
#         fit_df["diffs"] = abs(np.log10(fit_df["peak_freq"]) - fit_df["fit"])

#         all_peaks_df.update(fit_df)

#         for _, group in fit_df.groupby(level=list(fit_df.index.names)[:-1], observed=True):
#             best_idx = group["diffs"].idxmin()
#             if weight_diff:
#                 best_idx = (group["diffs"] / group["weight"]).idxmin()
#             peaks_df.loc[best_idx[:-1]] = fit_df.loc[best_idx]  # type: ignore

#     # Add selection columns
#     for col in [
#         "peak_temp_quantile",
#         "peak_local_quantile",
#     ]:
#         all_peaks_df[col] = np.nan

#     for col in [
#         "best_local_diff",
#         "best_local_prominence",
#         "best_local_width",
#         "best_local_weight",
#         "best_diff",
#         "best_prominence",
#         "best_width",
#         "best_weight",
#     ]:
#         all_peaks_df[col] = 0.0

#     # "bests" for each segment
#     grouped = all_peaks_df.groupby(all_peaks_df.index.names[:-1], observed=True)
#     for _, group in grouped:
#         if not group["peak_freq"].isna().all():
#             all_peaks_df.loc[group.index, "peak_local_quantile"] = (
#                 group["peak_freq"].apply(np.log10).rank(pct=True)
#             )
#         # By minimum diff
#         if not group["diffs"].isna().all():
#             all_peaks_df.loc[group["diffs"].idxmin(), "best_local_diff"] = 1.0
#         # By maximum prominence
#         if not group["prominence"].isna().all():
#             all_peaks_df.loc[group["prominence"].idxmax(), "best_local_prominence"] = 1.0
#         # By maximum width
#         if not group["width"].isna().all():
#             all_peaks_df.loc[group["width"].idxmax(), "best_local_width"] = 1.0

#         if not group["weight"].isna().all():
#             all_peaks_df.loc[group["weight"].idxmax(), "best_local_weight"] = 1.0

#     # "Bests" for each set
#     grouped = all_peaks_df.groupby(["sample_name", "condition", "temp"], observed=True)
#     for _, group in grouped:
#         if not group["peak_freq"].isna().all():
#             all_peaks_df.loc[group.index, "peak_temp_quantile"] = (
#                 group["peak_freq"].apply(np.log10).rank(pct=True)
#             )
#         # By minimum diff
#         if not group["diffs"].isna().all():
#             all_peaks_df.loc[group["diffs"].idxmin(), "best_diff"] = 1.0
#         # By maximum prominence
#         if not group["prominence"].isna().all():
#             all_peaks_df.loc[group["prominence"].idxmax(), "best_prominence"] = 1.0
#         # By maximum width
#         if not group["width"].isna().all():
#             all_peaks_df.loc[group["width"].idxmax(), "best_width"] = 1.0

#         if not group["weight"].isna().all():
#             all_peaks_df.loc[group["weight"].idxmax(), "best_weight"] = 1.0

#     return peaks_df, all_peaks_df


# def _normalize_peak_weights(
#     peaks_df: pd.DataFrame,
#     mode: str | tuple[str, ...] | None = None,
#     min_weight: float = 1e-32,
# ) -> pd.DataFrame:
#     """
#     Normalize the 'weight' column in peaks_df according to the specified mode.

#     Parameters
#     ----------
#     peaks_df : pd.DataFrame
#         DataFrame with a 'weight' column to normalize.
#     mode : str or None, optional
#         Normalization mode:
#         - None or "None": do not normalize.
#         - "dataset": normalize weights within each DataFrame returned by find_wide_peaks.
#         - "all": normalize weights across all entries in peaks_df.
#         - "temp": normalize weights for each (sample_name, condition, temp) group.
#         - "condition": normalize weights for each (sample_name, condition) group.
#     min_weight : float, optional
#         Minimum weight value after normalization (default: 1e-32).

#     Returns
#     -------
#     pd.DataFrame
#         DataFrame with normalized weights.
#     """
#     df = peaks_df.copy()

#     if df.empty:
#         return df

#     df["weight"] = df["weight"].where(df["weight"] > 0, 1e-32)

#     if isinstance(mode, tuple):
#         if len(mode) == 1:
#             mode = mode[0]
#         elif len(mode) > 1:
#             df["weight"] = 0.0
#             for md in mode:
#                 df["weight"] += _normalize_peak_weights(peaks_df, md, min_weight=min_weight)[
#                     "weight"
#                 ]
#             df["weight"] /= len(mode)
#             return df

#     if mode is None or str(mode).lower() == "none":
#         return df

#     def norm_func(x):
#         x = x.astype(float)
#         x_min = np.nanmin(x)
#         x_max = np.nanmax(x)
#         if x_max == x_min:
#             return np.full_like(x, min_weight)
#         normed = (x - x_min) / (x_max - x_min)
#         normed = np.clip(normed, min_weight, 1.0)
#         return normed

#     if mode == "all":
#         df["weight"] = norm_func(df["weight"].values)
#     elif mode == "dataset":
#         group_levels = df.index.names[:-1]
#         df["weight"] = df.groupby(group_levels, observed=True)["weight"].transform(norm_func)
#     elif mode == "temp":
#         group_levels = ["sample_name", "condition", "temp"]
#         df["weight"] = df.groupby(group_levels, observed=True)["weight"].transform(norm_func)
#     elif mode == "condition":
#         group_levels = ["sample_name", "condition"]
#         df["weight"] = df.groupby(group_levels, observed=True)["weight"].transform(norm_func)
#     elif mode == "sample":
#         group_levels = ["sample_name"]
#         df["weight"] = df.groupby(group_levels, observed=True)["weight"].transform(norm_func)

#     return df

# for idx in fit_df.index.droplevel("peaks").unique():
#     # Pick the best peak for each row of peaks_df based on residuals of possible peaks
#     sub = fit_df.xs(idx, level=(0, 1, 2, 3, 4), drop_level=False)
#     if sub.empty:
#         continue
#     fit_val = line_df.loc[idx, "fit_line"]
#     diffs = abs(np.log10(sub["peak_freq"]) - fit_val)
#     best_idx = diffs.idxmin()  # type: ignore
#     peaks_df.loc[best_idx[:-1]] = sub.loc[best_idx]


# def collect_peak_summary_df(
#     perm_dict: dict,
#     column: str = "imag perm",
#     min_width: float = 0.05,
#     min_prominence: float = 0.05,
# ) -> pd.DataFrame:
#     """
#     Summarize the main peak for each entry in a permittivity dictionary,
#     iteratively using group means and temperature trends to select the best peak for each entry.

#     Parameters
#     ----------
#     perm_dict : dict
#         Dictionary of DataFrames from transform_to_permittivity, keys are MultiIndex tuples.
#     column : str, optional
#         Column to evaluate for peak finding (default: "imag perm").
#     min_width : float, optional
#         Minimum width (in number of points) for a peak to be considered valid.
#     min_prominence : float, optional
#         Minimum prominence for a peak to be considered valid.

#     Returns
#     -------
#     pd.DataFrame
#         DataFrame with MultiIndex (from keys) and columns:
#         'peak_freq', 'peak_mag', 'width', 'prominence', 'bkg_subtracted'
#     """
#     group_levels = ["sample_name", "condition", "temp"]

#     # Create the output DataFrame with MultiIndex
#     peaks_df = pd.DataFrame(
#         index=list(perm_dict.keys()),
#         columns=["peak_freq", "peak_mag", "width", "prominence", "bkg_subtracted"],
#     )
#     peaks_df = form_std_df_index(peaks_df)

#     # Build a dict of all peaks for each entry, including bkg_subtracted column
#     all_peaks_dict = {}
#     for key, df in perm_dict.items():
#         freq = df["frequency"].to_numpy(copy=True)
#         arr = np.abs(df[column].to_numpy(copy=True))
#         min_width_pts = max(1, int(np.round(len(arr) * min_width)))
#         all_peaks_dict[key] = find_wide_peaks(
#             freq,
#             arr,
#             min_width=min_width_pts,
#             min_prominence=min_prominence,
#         )
#         if all_peaks_dict[key].empty:
#             all_peaks_dict[key] = find_wide_peaks(
#                 freq,
#                 arr,
#                 min_width=min(3, min_width_pts),
#                 min_prominence=min_prominence,
#             )

#     # First, select the most prominent peak for each entry
#     for key in peaks_df.index:
#         peaks = all_peaks_dict[key]
#         if not peaks.empty:
#             idx = peaks["prominence"].idxmax()
#             peak = peaks.loc[idx]
#             peaks_df.loc[key, "peak_freq"] = peak["peak_freq"]
#             peaks_df.loc[key, "peak_mag"] = peak["peak_arr"]
#             peaks_df.loc[key, "width"] = peak["width"]
#             peaks_df.loc[key, "prominence"] = peak["prominence"]
#             peaks_df.loc[key, "bkg_subtracted"] = peak["bkg_subtracted"]
#         else:
#             peaks_df.loc[
#                 key, ["peak_freq", "peak_mag", "width", "prominence", "bkg_subtracted"]
#             ] = np.nan

#     peaks_df = peaks_df.astype(
#         {
#             "peak_freq": float,
#             "peak_mag": float,
#             "width": float,
#             "prominence": float,
#             "bkg_subtracted": bool,
#         }
#     )

#     # Now iterate: for each group, use the group mean to select the closest peak for each entry in the group
#     group_means = peaks_df.groupby(group_levels, observed=True)["peak_freq"].mean()
#     for group_key, group in peaks_df.groupby(group_levels, observed=True):
#         group_mean = group_means.loc[group_key]
#         for key in group.index:
#             peaks = all_peaks_dict[key]
#             if not peaks.empty and not np.isnan(group_mean):
#                 # Select the peak closest to the group mean
#                 idx = (np.log10(peaks["peak_freq"]) - np.log10(group_mean)).abs().idxmin()
#                 peak = peaks.loc[idx]
#                 peaks_df.loc[key, "peak_freq"] = peak["peak_freq"]
#                 peaks_df.loc[key, "peak_mag"] = peak["peak_arr"]
#                 peaks_df.loc[key, "width"] = peak["width"]
#                 peaks_df.loc[key, "prominence"] = peak["prominence"]
#                 peaks_df.loc[key, "bkg_subtracted"] = (
#                     peak["bkg_subtracted"]
#                     if "bkg_subtracted" in peak
#                     else peak.get("found_by_bkg_sub", False)
#                 )

#     # Secondary refinement: check overall trend of peak_freq vs temp for each (sample_name, condition)
#     grouped = peaks_df.groupby(["sample_name", "condition"], observed=True)
#     for (sample_name, condition), group in grouped:
#         # Get temps and corresponding peak_freqs, drop NaNs
#         temps = [pd.to_numeric(idx[-1]) for idx in group.index]
#         peak_freqs = group.index.get_level_values("temp").astype(float).tolist()
#         valid = ~np.isnan(peak_freqs)
#         temps = np.array(temps)[valid]
#         peak_freqs = np.array(peak_freqs)[valid]
#         if len(temps) > 2:
#             # Fit a line to log10(peak_freq) vs temp
#             try:
#                 coeffs = np.polyfit(temps, np.log10(peak_freqs), 1)
#                 trend = coeffs[0]
#             except Exception:
#                 trend = None
#             # If trend is negative or near zero, try to find better peaks for outliers
#             if trend is not None and trend < 0.05:
#                 for idx, temp in zip(group.index, temps):
#                     peaks = all_peaks_dict[idx]
#                     if not peaks.empty:
#                         # Find peaks with freq above the group median and similar prominence
#                         median_freq = np.median(peak_freqs)
#                         candidate = peaks[
#                             (peaks["peak_freq"] > median_freq)
#                             & (
#                                 np.abs(peaks["prominence"] - peaks_df.loc[idx, "prominence"])
#                                 < 0.2 * peaks_df.loc[idx, "prominence"]
#                             )
#                         ]
#                         if not candidate.empty:
#                             idx_new = (
#                                 (candidate["peak_freq"] - peaks_df.loc[idx, "peak_freq"])
#                                 .abs()
#                                 .idxmin()
#                             )
#                             peak = candidate.loc[idx_new]
#                             peaks_df.loc[idx, "peak_freq"] = peak["peak_freq"]
#                             peaks_df.loc[idx, "peak_mag"] = peak["peak_arr"]
#                             peaks_df.loc[idx, "width"] = peak["width"]
#                             peaks_df.loc[idx, "prominence"] = peak["prominence"]
#                             peaks_df.loc[idx, "bkg_subtracted"] = (
#                                 peak["bkg_subtracted"]
#                                 if "bkg_subtracted" in peak
#                                 else peak.get("found_by_bkg_sub", False)
#                             )

#     return peaks_df


# def collect_peak_summary_df(
#     perm_dict: dict,
#     column: str = "mag perm",
#     min_width: float = 5,
#     min_prominence: float = 0.05,
#     ignore_low_freq: float = 0.0,
# ) -> pd.DataFrame:
#     """
#     Summarize the main peak for each entry in a permittivity dictionary,
#     refining selection using group means and temperature trends.

#     Parameters
#     ----------
#     perm_dict : dict
#         Dictionary of DataFrames from transform_to_permittivity, keys are MultiIndex tuples.
#     column : str, optional
#         Column to evaluate for peak finding (default: "mag perm").
#     min_width : float, optional
#         Minimum width (in number of points) for a peak to be considered valid.
#     min_prominence : float, optional
#         Minimum prominence for a peak to be considered valid.
#     ignore_low_freq : float, optional
#         Ignore peaks below this frequency (Hz).

#     Returns
#     -------
#     pd.DataFrame
#         DataFrame with MultiIndex (from keys) and columns:
#         'peak_freq', 'peak_mag', 'width', 'prominence'
#     """
#     group_levels = ("sample_name", "condition", "temp")

#     # Build a dict of all peaks for each entry
#     all_peaks_dict = {}
#     for key, df in perm_dict.items():
#         freq = df["frequency"].values
#         mag = np.abs(df[column].values)
#         peaks_df = find_wide_peaks(
#             freq,
#             mag,
#             min_width=min_width,
#             min_prominence=min_prominence,
#             ignore_low_freq=ignore_low_freq,
#         )
#         all_peaks_dict[key] = peaks_df if not peaks_df.empty else None

#     # Create the output DataFrame with MultiIndex
#     peaks_df = pd.DataFrame(
#         index=list(perm_dict.keys()), columns=["peak_freq", "peak_mag", "width", "prominence"]
#     )
#     peaks_df = form_std_df_index(peaks_df)

#     # First, select the most prominent peak for each entry
#     for key in peaks_df.index:
#         peaks = all_peaks_dict.get(key)
#         if peaks is not None:
#             idx = peaks["prominence"].idxmax()
#             peak = peaks.loc[idx]
#             peaks_df.loc[key, "peak_freq"] = peak["peak_freq"]
#             peaks_df.loc[key, "peak_mag"] = peak["peak_mag"]
#             peaks_df.loc[key, "width"] = peak["width"]
#             peaks_df.loc[key, "prominence"] = peak["prominence"]
#         else:
#             peaks_df.loc[key, ["peak_freq", "peak_mag", "width", "prominence"]] = np.nan

#     peaks_df = peaks_df.astype(float)

#     # Iterative refinement: use group mean and temperature trend
#     # 1. For each group (sample_name, condition), get all rows and their temps/peaks
#     grouped = peaks_df.groupby(["sample_name", "condition"], observed=True)
#     for (sample_name, condition), group in grouped:
#         # Get group mean (excluding NaN)
#         group_mean = group["peak_freq"].mean(skipna=True)


#         # For each row in group, select the peak closest to group mean,
#         # but also check if peak_freq increases with temp (monotonicity)
#         for key in group.index:
#             peaks = all_peaks_dict.get(key)
#             if peaks is not None and not np.isnan(group_mean):
#                 # Find peak closest to group mean
#                 idx_mean = (np.log10(peaks["peak_freq"]) - np.log10(group_mean)).abs().idxmin()
#                 # Optionally, check for monotonicity with temp
#                 # Get all temp/peak pairs in group
#                 temp_peak_pairs = [
#                     (pd.to_numeric(idx[-1]), peaks_df.loc[idx, "peak_freq"])
#                     for idx in group.index
#                     if not np.isnan(peaks_df.loc[idx, "peak_freq"])
#                 ]
#                 temp_peak_pairs = sorted(temp_peak_pairs, key=lambda x: x[0])
#                 # Check if peak_freq is generally increasing with temp
#                 is_monotonic = all(
#                     y2 >= y1 for (_, y1), (_, y2) in zip(temp_peak_pairs, temp_peak_pairs[1:])
#                 )
#                 # If not monotonic, still use group mean as main selector
#                 peak = peaks.loc[idx_mean]
#                 peaks_df.loc[key, "peak_freq"] = peak["peak_freq"]
#                 peaks_df.loc[key, "peak_mag"] = peak["peak_mag"]
#                 peaks_df.loc[key, "width"] = peak["width"]
#                 peaks_df.loc[key, "prominence"] = peak["prominence"]

#     return peaks_df

# def collect_peak_summary_df(
#     perm_dict: dict,
#     column: str = "mag perm",
#     min_width: float = 5,
#     min_prominence: float = 0.05,
#     ignore_low_freq: float = 0.0,
#     group_levels: tuple[str, ...] = ("sample_name", "condition", "temp"),
# ) -> pd.DataFrame:
#     """
#     Summarize the main peak for each entry in a permittivity dictionary,
#     using group means to guide peak selection.

#     Parameters
#     ----------
#     perm_dict : dict
#         Dictionary of DataFrames from transform_to_permittivity, keys are MultiIndex tuples.
#     column : str, optional
#         Column to evaluate for peak finding (default: "mag perm").
#     min_width : float, optional
#         Minimum width (in number of points) for a peak to be considered valid.
#     min_prominence : float, optional
#         Minimum prominence for a peak to be considered valid.
#     ignore_low_freq : float, optional
#         Ignore peaks below this frequency (Hz).
#     group_levels : tuple of str, optional
#         Index levels to group by for mean calculation.

#     Returns
#     -------
#     pd.DataFrame
#         DataFrame with MultiIndex (from keys) and columns:
#         'peak_freq', 'peak_mag', 'width', 'prominence'
#     """
#     # First pass: collect all peaks for all entries

#     peaks_df = pd.DataFrame(index=list(perm_dict.keys()), columns=["peak_freq", "peak_mag", "width", "prominence"])
#     peaks_df = form_std_df_index(peaks_df)

#     # Find peaks for each entry and fill the DataFrame
#     for key in peaks_df.index:
#         df = perm_dict[key]
#         freq = df["frequency"].values
#         mag = np.abs(df[column].values)
#         peaks = find_wide_peaks(
#             freq, mag,
#             min_width=min_width,
#             min_prominence=min_prominence,
#             ignore_low_freq=ignore_low_freq,
#         )
#         if not peaks.empty:
#             idx = peaks["prominence"].idxmax()
#             peak = peaks.loc[idx]
#             peaks_df.loc[key, "peak_freq"] = peak["peak_freq"]
#             peaks_df.loc[key, "peak_mag"] = peak["peak_mag"]
#             peaks_df.loc[key, "width"] = peak["width"]
#             peaks_df.loc[key, "prominence"] = peak["prominence"]
#         else:
#             peaks_df.loc[key, ["peak_freq", "peak_mag", "width", "prominence"]] = np.nan

#     # Convert columns to float
#     peaks_df = peaks_df.astype(float)

#     # Calculate group means for peak_freq
#     group_means = peaks_df.groupby(list(group_levels), observed=True)["peak_freq"].mean()


# def bias_prior_fits(
#     prior_fits: dict[str, pd.DataFrame] | None,
#     *group_keys: str,
#     bias: float = 0.0,
#     agg_method: str = "mean",
#     ignore: tuple[str, ...] | list[str] = (),
# ) -> dict[str, pd.DataFrame] | None:
#     """
#     Return a new prior_fits dict with values biased toward grouped averages.

#     Parameters
#     ----------
#     prior_fits : dict[str, pd.DataFrame] or None
#         Dictionary of DataFrames (as used in fit_preparation). If None, returns None.
#     bias : float
#         Value from 0 to 1. 0 = original values, 1 = grouped average, 0.5 = average of both.
#     group_keys : tuple of str
#         Index level(s) to group by (must be present in the DataFrame MultiIndex).

#     Returns
#     -------
#     dict[str, pd.DataFrame] or None
#         New dict with values biased toward grouped averages.
#     """
#     if prior_fits is None:
#         return None
#     bias = np.clip(bias, 0.0, 1.0).item()

#     gr_key_list = list(group_keys) if group_keys else ["sample_name", "condition", "temp"]

#     result = {}
#     for key, df in prior_fits.items():
#         # Do not modify "Equations" or non-DataFrame entries
#         if not isinstance(df, pd.DataFrame) or key == "Equations":
#             result[key] = df
#             continue
#         df = form_std_df_index(df)
#         # Only operate if all gr_key_list are in the index
#         if not all(k in df.index.names for k in gr_key_list):
#             result[key] = df.copy(deep=True)
#             continue

#         # Compute grouped mean for all columns
#         df0 = df.select_dtypes(float)
#         df1 = df.select_dtypes(exclude=float)
#         grouped_df = df0.abs().groupby(level=gr_key_list, observed=True).transform(agg_method)  # type: ignore

#         new_df = (1 - bias) * df0 + bias * grouped_df * np.sign(df0)
#         if 0 < bias < 1:
#             geo_df = (df0.abs() ** (1 - bias) * grouped_df**bias) * np.sign(df0)
#             df_abs = df0.abs().replace(0, 1e-32)
#             mask = df0.columns[np.log10(df_abs.max() / df_abs.min()) >= 3]
#             new_df[mask] = geo_df[mask]

#         for col in df1.columns:
#             new_df[col] = df1[col]

#         # Preserve index/columns
#         new_df.index = df.index
#         new_df.columns = df.columns
#         for col in ignore:
#             if col in df.columns:
#                 new_df[col] = df[col].copy()

#         result[key] = new_df

#     return result

# def fit_arrhenius_to_grouped_data(
#     grouped_data: dict,
#     columns: list[str] | None = None,
# ) -> dict:
#     """
#     Apply Arrhenius fits to specified columns in grouped DataFrames.

#     Parameters
#     ----------
#     grouped_data : dict
#         Dictionary of DataFrames with MultiIndex including a temperature level.
#     columns : list[str], optional
#         Columns to fit. Default is ["pos Current", "neg Current", "pos Resistance", "neg Resistance"].

#     Returns
#     -------
#     dict
#         Dictionary with the same structure as input, with fit results stored in DataFrame attrs.
#     """
#     if columns is None:
#         columns = ["pos Current", "neg Current", "pos Resistance", "neg Resistance"]

#     for df in grouped_data.values():
#         # Get unique temperatures and sort them
#         temps = pd.Series(df.index.get_level_values("temp"), dtype=float)

#         # Apply fits to each specified column
#         for col in columns:
#             res = perform_arrhenius_fit(temps, df[col])
#             df.attrs |= {f"{col}_{k}": v for k, v in res.items()}

#     return grouped_data


# def save_results(
#     data_path: str | Path,
#     *targets: str,
#     points: dict | None = None,
#     params: dict | None = None,
#     pol_points: dict | None = None,
#     grp_points: dict | None = None,
#     trend_params: dict | None = None,
#     params_flt: dict | None = None,
#     grp_params: dict | None = None,
#     perm_peaks: dict | None = None,
#     perm_curves: dict | None = None,
#     curves: dict | None = None,
#     save_mode: str = "w",
#     attrs: bool = False,
#     **kwargs: Any,
# ) -> None:
#     """
#     Save post-fitting results to disk if provided.
#     If points or params are passed, they are converted to the appropriate save dicts.

#     Parameters
#     ----------
#     data_path : str
#         Path to the data directory.
#     points : dict, optional
#         Raw points dictionary to be processed and saved.
#     params : dict, optional
#         Raw params dictionary to be processed and saved.
#     pol_points : dict, optional
#         Points with polarity.
#     grp_points : dict, optional
#         Grouped points.
#     params_flt : dict, optional
#         Flattened parameters.
#     perm_peaks : dict, optional
#         Permittivity peak summaries.
#     perm_curves : dict, optional
#         Permittivity curves.
#     curves : dict, optional
#         Curve DataFrames.
#     save_mode : str, optional
#         File save mode ("w" for write, "a" for append).
#     """
#     output_dir = Path(data_path)

#     # If raw points are provided, process them
#     if points is not None:
#         pol_points = pol_points or polarize_points(points)
#         grp_points = grp_points or group_points(points)
#     if params is not None:
#         params_flt = _flatten_params(params)

#     if pol_points is not None:
#         print(f"{time.ctime()}: Saving points with polarity...")
#         save(pol_points, output_dir, "points", file_type="xls", attrs=attrs, mode=save_mode)
#     if grp_points is not None:
#         print(f"{time.ctime()}: Saving grouped points...")
#         save(grp_points, output_dir, "grouped", file_type="xls", attrs=attrs, mode=save_mode)
#     if params_flt is not None:
#         print(f"{time.ctime()}: Saving flattened parameters...")
#         save(params_flt, output_dir, "params", file_type="xls", attrs=attrs, mode=save_mode)
#     if grp_params is not None:
#         print(f"{time.ctime()}: Saving grouped parameters...")
#         save(grp_params, output_dir, "grp_params", file_type="xls", attrs=attrs, mode=save_mode)
#     if trend_params is not None:
#         print(f"{time.ctime()}: Saving point trend params...")
#         save(
#             trend_params, output_dir, "trend_params", file_type="xls", attrs=attrs, mode=save_mode
#         )
#     if perm_peaks is not None:
#         print(f"{time.ctime()}: Saving permittivity peaks...")
#         save(perm_peaks, output_dir, "perm_peaks", file_type="xls", attrs=attrs, mode=save_mode)
#     if perm_curves is not None:
#         print(f"{time.ctime()}: Saving permittivity curves...")
#         for key in perm_curves:
#             if not targets or key in targets:
#                 print(f"    Now saving perms_{key}...")
#                 save(
#                     perm_curves[key],
#                     output_dir,
#                     f"perms_{key}",
#                     file_type="xls",
#                     attrs=attrs,
#                     mode=save_mode,
#                 )
#     if curves is not None:
#         print(f"{time.ctime()}: Saving curves...")
#         for key in curves:
#             if not targets or key in targets:
#                 print(f"    Now saving curves_{key}...")
#                 save(
#                     curves[key],
#                     output_dir,
#                     f"curves_{key}",
#                     file_type="xls",
#                     attrs=attrs,
#                     mode=save_mode,
#                 )

#     if winsound is not None:
#         winsound.MessageBeep(winsound.MB_ICONHAND)
