import re
import time
from typing import Any
from collections import defaultdict
from collections.abc import Callable

import numpy as np
import pandas as pd

from eis_analysis.dc_fit.extract_tools import (
    DEFAULT_DIR,
    group_points,
    load_dc_data,
    save_results,  # noqa: F401
    polarize_params,
    polarize_points,
    form_std_df_index,
    partial_selection,  # noqa: F401
)
from eis_analysis.dc_fit.fit_functions import (
    data_group_trend_eval,
    perform_arrhenius_fit,
    nested_data_group_trend_eval,
)

np.seterr(invalid="raise")


def fit_arrhenius_for_points(
    grouped_data: dict,
    columns: set[str] | list[str] | tuple[str, ...] = (),
    infer: bool = False,
    skip: Callable = lambda x: False,
    multi_pol_as_pos: bool = True,
    fit_resid: bool = False,
    infer_func: Callable = lambda col: col[-1].isdigit(),
) -> tuple[dict[str, pd.DataFrame], dict]:
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
    if not columns:
        columns = {"pos Current", "neg Current", "pos Resistance", "neg Resistance"}

    revised_data = {}
    fit_results = defaultdict(dict)
    for key, df0 in grouped_data.items():
        if skip(df0):
            revised_data[key] = df0
            continue
        # Get unique temperatures and sort them
        fit_cols = [s for s in df0.columns if infer_func(s)] if infer else columns
        df = df0[[s for s in df0.columns if "_fit" not in s]].copy()

        df.attrs = df0.attrs.copy()
        temps = pd.Series(df.index.get_level_values("temp"), dtype=float).to_numpy()

        # Apply fits to each specified column
        for col in fit_cols:
            if col not in df.columns:
                continue
            if multi_pol_as_pos and any(df[col] < 0) and any(df[col] > 0):
                df[col] = np.abs(df[col])
            e_weights = df.get("Error", None)
            if e_weights is not None:
                e_weights = e_weights.max() - e_weights
                e_weights = np.clip(e_weights / e_weights.max(), 1e-32, 1)

            if col == "b0" and "curr" in key.lower():
                res = perform_arrhenius_fit(
                    temps, df[col].to_numpy(), weights=e_weights, Ea=0.5914
                )
            else:
                res = perform_arrhenius_fit(temps, df[col].to_numpy(), weights=e_weights)

            res = {f"{col}_{k}": v for k, v in res.items()}
            df[f"{col}_fit"] = res.pop(f"{col}_fit")
            if fit_resid:
                df[f"{col}_fit_resid"] = df[col] - df[f"{col}_fit"]

            df.attrs |= res
            fit_results[key] |= res
        revised_data[key] = df
        # Store the fit results in the DataFrame attrs
    return revised_data, dict(fit_results)


def fit_arrhenius_for_params(
    grouped_data: dict,
    data_sets: set[str] | list[str] | tuple[str, ...] = (),
    multi_pol_as_pos: bool = False,
) -> tuple[dict, dict]:
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
        data_sets = {"Current", "Resistance"}

    revised_data = defaultdict(dict)
    fit_results = defaultdict(dict)
    for top_key, vals in grouped_data.items():
        if "vals" not in top_key:
            revised_data[top_key] = vals
            continue
        for key, df0 in vals.items():
            if key not in data_sets:
                revised_data[top_key][key] = df0
                continue
            fit_cols = [s for s in df0.columns if s[-1].isdigit()]
            df = df0[[s for s in df0.columns if "_fit" not in s]].copy()
            df.attrs = df0.attrs.copy()

            # Get unique temperatures and sort them
            temps = pd.Series(df.index.get_level_values("temp"), dtype=float).to_numpy()

            # Apply fits to each specified column
            for col in fit_cols:
                if col not in df.columns:
                    continue
                if multi_pol_as_pos and any(df[col] < 0) and any(df[col] > 0):
                    df[col] = np.abs(df[col])
                e_weights = df.get("Error", None)
                if e_weights is not None:
                    e_weights = e_weights.max() - e_weights
                    # e_weights = (e_weights - e_weights.min()) / (e_weights.max() - e_weights.min())
                    e_weights = e_weights / e_weights.max()
                    e_weights[e_weights <= 0] = 1e-32
                if key == "Current" and col == "b0":
                    res = perform_arrhenius_fit(temps, df[col], weights=e_weights, Ea=0.5914)
                else:
                    res = perform_arrhenius_fit(temps, df[col], weights=e_weights)
                res = {f"{col}_{k}": v for k, v in res.items()}
                df[f"{col}_fit"] = res.pop(f"{col}_fit")
                df.attrs |= res
                fit_results[f"{top_key} {key}"] |= res
            revised_data[top_key][key] = df
        # Store the fit results in the DataFrame attrs
    return dict(revised_data), dict(fit_results)


def fit_arrhenius_for_points2(
    grouped_data: dict,
    columns: set[str] | list[str] | tuple[str, ...] = (),
    infer: bool = False,
    skip: Callable = lambda x: False,
    sign_default: int = 1,
    fit_resid: bool = False,
    infer_func: Callable = lambda col: col[-1].isdigit(),
    **kwargs: Any,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """
    Apply Arrhenius fits to specified columns in grouped DataFrames.

    Parameters
    ----------
    grouped_data : dict
        Dictionary of DataFrames with MultiIndex including a temperature level.
    columns : list[str], optional
        Columns to fit. Default is ["pos Current", "neg Current", "pos Resistance", "neg Resistance"].
    infer : bool, default False
        Whether to infer columns to fit using infer_func
    skip : Callable, default lambda x: False
        Function to determine if a DataFrame should be skipped
    multi_pol_as_pos : bool, default True
        Whether to convert mixed-sign data to absolute values
    fit_resid : bool, default False
        Whether to calculate fit residuals
    infer_func : Callable, default lambda col: col[-1].isdigit()
        Function to infer which columns to fit if infer=True

    Returns
    -------
    tuple[dict, dict]
        Dictionary with the same structure as input with fit results stored in DataFrame attrs,
        and a dictionary of extracted fit results.
    """
    if not columns:
        columns = {"pos Current", "neg Current", "pos Resistance", "neg Resistance"}

    kwargs.setdefault("cond_kwargs", {"Ea": 0.5914})
    # Use the generalized data_group_trend_eval function
    revised_data, fit_results = data_group_trend_eval(
        data=grouped_data,
        x_data="temp",
        columns=() if infer else columns,
        skip=skip,
        col_selector=infer_func if infer else lambda col: True,
        pass_kwargs_eval=lambda key, col: col == "b0" and "curr" in key.lower(),
        fit_func="arrhenius",
        sign_default=sign_default,
        fit_resid=fit_resid,
        **kwargs,
    )

    return revised_data, fit_results


def fit_arrhenius_for_params2(
    grouped_data: dict,
    data_sets: set[str] | list[str] | tuple[str, ...] = (),
    sign_default: int = 0,
    **kwargs: Any,
) -> tuple[dict, dict]:
    """
    Apply Arrhenius fits to specified columns in grouped DataFrames.

    Parameters
    ----------
    grouped_data : dict
        Dictionary of DataFrames with MultiIndex including a temperature level.
    data_sets : set[str] | list[str] | tuple[str, ...], default ()
        Data sets to fit. Default is {"Current", "Resistance"}.
    multi_pol_as_pos : bool, default False
        Whether to convert mixed-sign data to absolute values

    Returns
    -------
    tuple[dict, dict]
        Dictionary with the same structure as input with fit results stored in DataFrame attrs,
        and a dictionary of extracted fit results.
    """
    if not data_sets:
        data_sets = {"Current", "Resistance"}

    kwargs.setdefault("cond_kwargs", {"Ea": 0.5914})
    # Use the generalized nested_data_group_trend_eval function
    revised_data, fit_results = nested_data_group_trend_eval(
        grouped_data=grouped_data,
        x_data="temp",
        data_sets=data_sets,
        skip=lambda x: "vals" not in x,
        col_selector=lambda col: col[-1].isdigit(),
        pass_kwargs_eval=lambda key, col: col == "b0" and "curr" in key.lower(),
        fit_func="arrhenius",
        sign_default=sign_default,
        **kwargs,
    )

    return revised_data, fit_results


def collect_point_stats(
    curves: dict,
) -> dict[str, pd.DataFrame]:
    """
    Extracts _error, _change, _updated attrs from each DataFrame in each curve_dict,
    returns a dict of DataFrames (one per curve_dict) and a summary DataFrame.

    Parameters
    ----------
    curves : dict
        Dict of dicts of DataFrames, e.g., curves[group][idx] = DataFrame

    Returns
    -------
    dict[str, pd.DataFrame]
        Dict of DataFrames: one per curve_dict, plus a summary DataFrame.
    """
    patterns = re.compile(r".*(_error|_change|_updated)$")
    result: dict[str, pd.DataFrame] = {}

    # Collect stats for each fit group
    sum_stats: dict[str, Any] = {"all": defaultdict(dict)}
    all_stats = {}
    u_stats = {}
    # names = set()
    for group, curve_dict in curves.items():
        if "fit" not in group:
            continue
        df_stats = defaultdict(dict)

        df = form_std_df_index(
            pd.DataFrame(
                {
                    i: {k: d.attrs[k] for k in d.attrs.keys() if patterns.match(k)}
                    for i, d in curve_dict.items()
                }
            ).T
        )
        for col in df.columns:
            # names.add(col)
            # name = [s.strip() for s in col.split("_")]
            # names.setdefault("base_name", set()).add(name[0])
            name = col.split("_")[0].strip()
            # names.add(name)
            if "updated" in col:
                # names.setdefault("updated", set()).add(col)
                df.attrs[col] = df[col].sum()
                # all_stats[col] = float(all_stats.get(col, 0) + df[col].sum())
                sum_stats["all"][name].setdefault("updated", 0.0)
                sum_stats["all"][name]["updated"] += df[col].sum()
            else:
                vals = df[col]
                df.attrs[f"{col}_median"] = vals.median()
                df.attrs[f"{col}_mean"] = vals.mean()
                df.attrs[f"{col}_std"] = vals.std()
                df.attrs[f"{col}_min"] = vals.min()
                df.attrs[f"{col}_max"] = vals.max()
                all_stats.setdefault(col, []).extend(vals.tolist())

                if any(df.get(f"{name}_updated", [False])):
                    vals = vals[df[f"{name}_updated"].astype(bool)]

                df.attrs[f"{col}_umedian"] = vals.median()
                df.attrs[f"{col}_umean"] = vals.mean()
                df.attrs[f"{col}_ustd"] = vals.std()
                df.attrs[f"{col}_umin"] = vals.min()
                df.attrs[f"{col}_umax"] = vals.max()
                u_stats.setdefault(col, []).extend(vals.tolist())

        for key, attr in df.attrs.items():
            parts = key.split("_")  # type: ignore[assignment]
            df_stats[parts[0].strip()]["_".join(parts[1:])] = attr

        result[group] = df
        sum_stats[group] = pd.DataFrame(df_stats).T

    for col in all_stats:
        if isinstance(all_stats[col], list):
            name = col.split("_")[0].strip()
            suffix = "_".join(col.split("_")[1:])
            for arr, mod in zip([all_stats[col], u_stats[col]], ["_", "_u"]):
                sum_stats["all"][name][f"{suffix}{mod}median"] = np.nanmedian(arr)
                sum_stats["all"][name][f"{suffix}{mod}mean"] = np.nanmean(arr)
                sum_stats["all"][name][f"{suffix}{mod}std"] = np.nanstd(arr)
                sum_stats["all"][name][f"{suffix}{mod}min"] = np.nanmin(arr)
                sum_stats["all"][name][f"{suffix}{mod}max"] = np.nanmax(arr)

    sum_stats["all"] = pd.DataFrame(sum_stats["all"]).T

    # fmt: off
    p_map = {"curr": 0, "resi": 1, "res": 1, "volt": 2, "temp": 3}

    re_cols = [
        "updated",
        "error_median", "error_umedian", "change_median", "change_umedian",
        "error_mean", "error_std", "error_umean", "error_ustd",
        "change_mean", "change_std", "change_umean", "change_ustd",
        "error_min", "error_max", "error_umin", "error_umax",
        "change_min", "change_max", "change_umin", "change_umax",
    ]
    summary = pd.concat(sum_stats, axis=0)[re_cols]

    # fmt: on
    si: pd.MultiIndex = summary.index  # type: ignore[assignment]
    summary.index = si.set_levels(
        si.levels[1].astype(
            pd.CategoricalDtype(
                sorted(
                    si.get_level_values(1).unique(),
                    key=lambda n: (p_map.get(n.lower()[:4], 999), len(n), n),
                ),
                ordered=True,
            )
        ),
        level=1,
    )
    summary.index.names = ["fit", "measurement"]
    result["summary"] = summary.sort_index()

    return result


def run_point_fitting(
    points: dict | None = None,
    params: dict | None = None,
    curves: dict | None = None,
    perm_peaks: dict | None = None,
    verbose: bool = True,
) -> dict[str, dict]:
    """
    Apply Arrhenius fits to different dictionaries of DataFrames.

    This function handles different data structures and applies the appropriate
    fitting function to each.

    Parameters
    ----------
    points : dict, optional
        Dictionary of DataFrames for point-based data, by default None
    grp_points : dict, optional
        Dictionary of DataFrames for grouped point data, by default None
    params : dict, optional
        Nested dictionary of DataFrames for parameter data, by default None
    perm_peaks : dict, optional
        Dictionary of DataFrames for permittivity peaks, by default None

    Returns
    -------
    dict[str, dict]
        Dictionary containing the fitting results for each input type
    """
    results = {}
    # Process points data
    grp_points = {}
    gr_params = {}
    pol_points = {}
    pol_params = {}

    results["trend_params"] = {}

    if points:
        if verbose:
            print(f"{time.ctime()}: Evaluating points...")
        grp_points = group_points(points)
        pol_points = polarize_points(points)

        data, fit_results = fit_arrhenius_for_points(pol_points, columns=["Current", "Resistance"])
        results["points"] = data
        results["trend_params"]["points"] = pd.DataFrame(fit_results).T

    if pol_points:
        if verbose:
            print(f"{time.ctime()}: Evaluating polarized points...")
        pol_points = group_points(pol_points, r"^fit\s\d{1,2}", aggregate=False)
        data, fit_results = fit_arrhenius_for_points(
            pol_points, columns=["Voltage", "Current", "Resistance"]
        )
        results["pol_points"] = data
        results["trend_params"]["pol_points"] = pd.DataFrame(fit_results).T

    # Process grp_points data (already formatted correctly for the function)
    if grp_points:
        if verbose:
            print(f"{time.ctime()}: Evaluating grouped points...")
        data, fit_results = fit_arrhenius_for_points(grp_points)
        results["grp_points"] = data
        results["trend_params"]["grp_points"] = pd.DataFrame(fit_results).T

    # Process params data (nested structure)
    if params:
        if verbose:
            print(f"{time.ctime()}: Evaluating parameters...")
        pol_params = polarize_params(params, simplify=True)
        gr_params = pol_params | group_points(pol_params, r"^fit\s\d{1,2}\s(cur|res)", False)
        gr_params |= group_points(pol_params, r"^fit\s\d{1,2}\s(cur|res)[^\s]*\s(neg|pos)$", False)
        data, fit_results = fit_arrhenius_for_params(params)
        results["params"] = data
        results["trend_params"]["params"] = pd.DataFrame(fit_results).T

    if pol_params:
        if verbose:
            print(f"{time.ctime()}: Evaluating polarized parameters...")
        data, fit_results = fit_arrhenius_for_points(pol_params, infer=True)
        results["pol_params"] = data
        results["trend_params"]["pol_params"] = pd.DataFrame(fit_results).T

    if gr_params:
        if verbose:
            print(f"{time.ctime()}: Evaluating grouped parameters...")
        data, fit_results = fit_arrhenius_for_points(gr_params, infer=True)
        nested = defaultdict(dict)
        for key, df in data.items():
            keys = key.split(" ")
            nested[" ".join(keys[:2])][" ".join(keys[2:])] = df
        results["grp_params"] = dict(nested)
        # results["grp_params"] = data
        results["trend_params"]["grp_params"] = pd.DataFrame(fit_results).T

    if curves:
        if verbose:
            print(f"{time.ctime()}: Evaluating curves...")
        results["trend_params"] |= collect_point_stats(curves)

    # # Process permittivity peaks if provided
    # if perm_peaks:
    #     data, fit_results = fit_arrhenius_for_points(perm_peaks)
    #     results["perm_peaks"] = data
    #     results["trend_params"]["perm_peaks"] = fit_results

    return results


# %%
if __name__ == "__main__":
    try:
        points = (
            globals()["points"]
            if "points" in globals()
            else load_dc_data(DEFAULT_DIR, "points")["points"]
        )
        curves = (
            globals()["curves"]
            if "curves" in globals()
            else load_dc_data(DEFAULT_DIR, "curves", "fits")["curves"]
        )
        params = (
            globals()["params"]
            if "params" in globals()
            else load_dc_data(DEFAULT_DIR, "params")["params"]
        )
        # perm_peaks = (
        #     globals()["perm_peaks"]
        #     if "perm_peaks" in globals()
        #     else load_dc_data(DEFAULT_DIR, "perm_peaks")["perm_peaks"]
        # )

    except Exception:
        points = load_dc_data(DEFAULT_DIR, "points")["points"]
        curves = load_dc_data(DEFAULT_DIR, "curves")["curves"]
        params = load_dc_data(DEFAULT_DIR, "params")["params"]
        # perm_peaks = load_dc_data(DEFAULT_DIR, "perm_peaks")["perm_peaks"]

    # %% Calculate permittivity
    results = run_point_fitting(
        points=points,
        params=params,
        curves=curves,
        # perm_peaks=perm_peaks,
    )

    # %% Save results
    save_results(
        DEFAULT_DIR,
        attrs=True,
        **results,  # type: ignore[call-arg]
    )

    # save_results(
    #     DEFAULT_DIR,
    #     trend_params=results,
    #     params_flt=results.get("params"),
    #     grp_points=results.get("grp_points"),
    #     points=results.get("points"),
    #     grp_params=results.get("gr_params"),
    #     perm_peaks=results.get("perm_peaks"),
    #     attrs=True,
    # )
