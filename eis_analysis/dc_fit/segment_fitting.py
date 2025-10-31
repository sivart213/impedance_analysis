import re
import time
import warnings
from typing import Any
from functools import partial
from itertools import chain, product
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning

from eis_analysis.data_treatment import Statistics
from eis_analysis.dc_fit.extract_tools import (
    create_std_df,
    trimmed_midpoint,
    form_std_df_index,
)
from eis_analysis.dc_fit.fit_functions import (
    exp_func,
    pow_func,
    n_exp_func,
    linear_func,
    double_exp_func,
    run_segment_fit,
    sort_exp_params,
    stretch_exp_func,
    initial_guess_exp_func,
    initial_guess_pow_func,
    initial_guess_poly_func,
    initial_guess_str_exp_func,
)

np.seterr(invalid="raise")


# %% Fitting Functions
fit_func_map = {
    "linear_func": (linear_func, None),
    "exp_func": (exp_func, 1),
    "double_exp_func": (double_exp_func, 2),
    "pow_func": (pow_func, None),
    "stretch_exp_func": (stretch_exp_func, 1),
}

default_func = {
    "Voltage": linear_func,
    "Temperature": linear_func,
    "Current": exp_func,
    "Resistance": exp_func,
}


def get_fit_func(val: str | Callable) -> tuple[Callable, int | None]:
    """Get fit function by name, or return linear_func as default."""
    if callable(val):
        func_name = getattr(val, "__name__", "")
        return val, get_fit_func(func_name)[1]
    if val in fit_func_map:
        return fit_func_map[val]
    if "exp" in val:
        # if val in fit_func_map:
        #     return (double_exp_func, 2) if "double" in val else (exp_func, 1)
        special_match = re.match(r"^(\d{1,2})_exp_func$", val)
        if special_match:
            special = int(special_match.group(1))
            return n_exp_func, special
    if val in default_func:
        return get_fit_func(default_func[val])
    return linear_func, None


def summarize_array(arr, edge_items=5):
    """
    Return a string summarizing the array as [n1, n2, ..., nN-1, nN].
    Shows up to edge_items from each end.
    """
    arr = np.asarray(arr)
    if arr.size <= 2 * edge_items:
        return np.array2string(arr, separator=", ")
    head = ", ".join(f"{x:.3g}" for x in arr[:edge_items])
    tail = ", ".join(f"{x:.3g}" for x in arr[-edge_items:])
    return f"[{head}, ..., {tail}]"


def prevent_param_overlaps(
    fit_params_df: pd.DataFrame,
    base_names: list[str] | tuple[str, ...],
    group_keys: list | tuple = (),
) -> pd.DataFrame:
    """
    Prevents overlaps in boundaries for related parameters (e.g., tau0 and tau1).

    Parameters
    ----------
    fit_params_df : pd.DataFrame
        DataFrame with MultiIndex columns ([bound_type, param_name]) containing parameter bounds.
    base_names : list[str] or tuple[str, ...]
        Base names of parameters to check for overlaps (e.g., ["tau", "a"]).
    group_keys : list[str] or tuple[str, ...], optional
        Index level(s) to group by. If empty, operates on the whole dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame with adjusted boundaries to prevent overlaps.
    """
    if not isinstance(fit_params_df, pd.DataFrame):
        raise ValueError("fit_params_df must be a DataFrame")

    # Make a copy to avoid modifying the original
    df = fit_params_df.copy()

    # Check if column structure is a MultiIndex
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("fit_params_df must have MultiIndex columns [bound_type, param_name]")

    # Group by the provided group keys if any
    group_keys = list(group_keys)

    if group_keys:
        if not isinstance(group_keys[0], (tuple, list)):
            group_keys = [group_keys] * len(base_names)
        if len(group_keys) < len(base_names):
            group_keys += [[]] * len(base_names)

    param_cols = df.columns.get_level_values(1).unique()
    # Process each base name
    for idx, base in enumerate(base_names):
        # Find parameters with this base name (those that start with base and end with a digit)
        params = [p for p in param_cols if p.startswith(base)]

        # Skip if fewer than 2 parameters with this base name
        if len(params) < 2:
            continue

        base_params = sorted(params, key=lambda p: df[("p0", p)].abs().median())
        is_log = (
            np.max(df.loc[:, ("p0", base_params)].abs())  # type: ignore
            / np.min(df.loc[:, ("p0", base_params)].abs())  # type: ignore
            > 10
        )

        g_key = group_keys[idx] if group_keys else []
        # Apply processing based on grouping
        if g_key and all(key in df.index.names for key in g_key):
            # Process each group separately
            for _, group in df.groupby(level=g_key, observed=True):
                df.update(_process_group(group, base_params, is_log))
        else:
            # Process the entire DataFrame as one group
            df.update(_process_group(df, base_params, is_log))

    return df


def _process_group(data, sorted_params: list[str], use_geo=True) -> pd.DataFrame:
    """
    Process a group of parameters to prevent overlapping bounds.
    Uses absolute values to simplify overlap logic.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with MultiIndex columns ([bound_type, param_name]) containing parameter bounds.
    sorted_params : list
        List of parameter names sorted in ascending order (by median value).
    use_geo : bool, optional
        Whether to use geometric (multiplicative) or arithmetic (additive) scaling.

    Returns
    -------
    pd.DataFrame
        DataFrame with adjusted boundaries to prevent overlaps.
    """
    abs_df = data.loc[:, ["lb", "p0", "ub"]].abs().copy(deep=True)
    abs_df = validate_parameter_bounds(
        abs_df,
        use_multiply=use_geo,
        param_names=sorted_params,
    )

    # Process each adjacent pair of parameters
    for i in range(len(sorted_params) - 1):
        min_param = sorted_params[i]
        max_param = sorted_params[i + 1]

        # Check if there's an overlap (simpler with absolute values)
        if abs_df[("ub", min_param)].max() > abs_df[("lb", max_param)].min():
            # Calculate midpoint for new bounds
            min_p0_max = abs_df[("p0", min_param)].max()
            max_p0_min = abs_df[("p0", max_param)].min()

            if use_geo:
                # For geometric scaling, calculate ratios
                min_lb_delta = abs_df[("lb", min_param)] / abs_df[("ub", min_param)]
                max_ub_delta = abs_df[("ub", max_param)] / abs_df[("lb", max_param)]
            else:
                # For arithmetic scaling, calculate differences
                min_lb_delta = abs_df[("ub", min_param)] - abs_df[("lb", min_param)]
                max_ub_delta = abs_df[("ub", max_param)] - abs_df[("lb", max_param)]

            if min_p0_max <= max_p0_min:
                # No overlap in p0 values - use geometric or arithmetic mean
                mid_point = trimmed_midpoint([min_p0_max, max_p0_min], q=0.25, logspace=use_geo)
            else:
                # Overlap in p0 values - use average of all values
                mid_point = trimmed_midpoint(
                    abs_df[("p0", min_param)], abs_df[("p0", max_param)], q=0.25, logspace=use_geo
                )

            # Apply new bounds to ALL rows (not just those with violations)
            # Set upper bound of min_param and lower bound of max_param to mid_point
            abs_df[("ub", min_param)] = mid_point
            abs_df[("lb", max_param)] = mid_point

            # Check for any remaining bound violations
            min_violations = abs_df[("lb", min_param)] > abs_df[("ub", min_param)]
            max_violations = abs_df[("lb", max_param)] > abs_df[("ub", max_param)]

            if use_geo:
                # Apply geometric adjustment to lower bound
                abs_df.loc[min_violations, ("lb", min_param)] = (
                    abs_df[("ub", min_param)][min_violations] * min_lb_delta[min_violations]
                )
                # Apply geometric adjustment to upper bound
                abs_df.loc[max_violations, ("ub", max_param)] = (
                    abs_df[("lb", max_param)][max_violations] * max_ub_delta[max_violations]
                )
            else:
                # Apply arithmetic adjustment to lower bound
                abs_df.loc[min_violations, ("lb", min_param)] = (
                    abs_df[("ub", min_param)][min_violations] - min_lb_delta[min_violations]
                )
                # Apply arithmetic adjustment to upper bound
                abs_df.loc[max_violations, ("ub", max_param)] = (
                    abs_df[("lb", max_param)][max_violations] + max_ub_delta[max_violations]
                )

    data[abs_df.columns] = abs_df * np.sign(data["p0"])

    return validate_parameter_bounds(
        data,
        use_multiply=use_geo,
        param_names=sorted_params,
    )


def validate_parameter_bounds(
    df: pd.DataFrame,
    shift_value: float = 1e-14,
    use_multiply: bool = True,
    param_names: list[str] | tuple[str, ...] = (),
) -> pd.DataFrame:
    """
    Validates and fixes parameter bounds to ensure lb <= p0 <= ub for each parameter.
    Simply swaps bounds if lb > ub, or shifts both if lb == ub.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with MultiIndex columns ([bound_type, param_name]) containing parameter bounds.
    shift_value : float, optional
        Value to use for shifting bounds when violations are detected.
        For multiplicative shifts, this is directly used as the factor.
        For additive shifts, this represents the fraction of |p0| to shift by.
    use_multiply : bool, optional
        If True, uses multiplication for shifting bounds; if False, uses addition.
    param_names : list[str] or tuple[str,...], optional
        Specific parameter names to check. If None, checks all parameters.

    Returns
    -------
    pd.DataFrame
        DataFrame with corrected bounds.
    """

    # Get all parameter names if not provided
    param_names = param_names or df.columns.get_level_values(1).unique().tolist()

    # Process each parameter
    for param in param_names:
        # Check for bound violations (lb > ub)
        # Handle the case where bounds are equal (unlikely but possible)
        equal_bounds = df[("lb", param)] == df[("ub", param)]
        violations = df[("lb", param)] > df[("ub", param)]

        if any(equal_bounds):
            # For equal bounds, shift both outward from p0
            if use_multiply:
                shift_amount = abs(df[("lb", param)][equal_bounds]) * shift_value
            else:
                shift_amount = shift_value
            df.loc[equal_bounds, ("lb", param)] = df[("lb", param)][equal_bounds] - shift_amount
            df.loc[equal_bounds, ("ub", param)] = df[("ub", param)][equal_bounds] + shift_amount

        if any(violations):
            # Simple approach: swap the bounds where they're inverted
            lb_values = df[("lb", param)][violations].copy()
            ub_values = df[("ub", param)][violations].copy()

            df.loc[violations, ("lb", param)] = ub_values
            df.loc[violations, ("ub", param)] = lb_values

        df.loc[:, ("p0", param)] = np.clip(
            df[("p0", param)],
            df[("lb", param)] + np.spacing(df[("lb", param)].abs()) * 100,
            df[("ub", param)] - np.spacing(df[("ub", param)].abs()) * 100,
        )

    return df


def collect_segment_stats(
    segment_dict: dict, columns: list[str] | tuple[str, ...] = ()
) -> pd.DataFrame:
    """
    Create a DataFrame with key statistics for each segment to optimize bounds calculation.

    This function efficiently gathers data needed for determining proper bounds for fitting
    parameters, particularly b0, without repeatedly accessing the original segment data.

    Parameters
    ----------
    segment_dict : dict
        Dictionary mapping segment identifiers to DataFrames of segment data.
    columns : list[str]
        List of column names to analyze.

    Returns
    -------
    pd.DataFrame
        DataFrame with segment indexes and statistics needed for fit parameter bounds.
    """
    # Create base DataFrame with proper structure
    if not columns:
        columns = list(next(iter(segment_dict.values())).columns)[1:]

    col_names = ["time"] + [c for col in columns for c in (f"{col}_0", f"{col}_f")]
    stats_df = create_std_df(list(segment_dict.keys()), col_names)

    # Collect statistics for each segment and column
    for idx, df in segment_dict.items():
        # Store maximum time for tau calculations
        stats_df.loc[idx, "time"] = df.iloc[-1, 0]
        q_len = len(df) // 4
        if q_len < 1:
            continue
        # For each column, store first and last values
        for col in columns:
            if col in df.columns and not df[col].isna().all():
                values = df[col].to_numpy(copy=True)
                min_max = [np.argmin(abs(values)), np.argmax(abs(values))]
                start = np.min(min_max)
                stop = np.max(min_max)
                # if start > q_len:  # Ensure b_idx is in the 1st quarter of values
                #     start = 0
                stats_df.loc[idx, f"{col}_0"] = values[start] if start <= q_len else values[0]
                stats_df.loc[idx, f"{col}_f"] = values[stop] if stop > 3 * q_len else values[-1]

    # stats_df = stats_df.dropna(axis=1, how="all")
    stats_df.attrs["max_time"] = max(float(stats_df["time"].max()), 2.0)

    return stats_df


def fit_preparation(
    segment_dict: dict,
    skipcols: list[str] | set[str] | None = None,
    fit_raw: bool = False,
    base_fit_map: dict | None = None,
    base_scale_map: dict | None = None,
    initial_guesses: dict | None = None,
    prior_fit: dict | None = None,
    stats_df: pd.DataFrame | None = None,
    **kwargs,
) -> tuple[dict, dict, dict]:
    """
    Prepare fit_map and fit_params_dict for segment fitting.

    Parameters
    ----------
    segment_dict : dict
        Dictionary of segment DataFrames.
    mean_df : pd.DataFrame, optional
        DataFrame of mean values for initial guesses.
    skipcols : list[str], optional
        Columns to skip.
    fit_raw : bool, optional
        Whether to include raw columns.
    base_fit_map : dict, optional
        Mapping of base column names to fit functions.

    base_scale_map : dict, optional
        Mapping of column name to scale type(s) for each parameter.

    Returns
    -------
    fit_map : dict
        Mapping of column name to fit function.
    fit_params_dict : dict
        Dict: {col: pd.DataFrame}, columns are MultiIndex (["lb", "p0", "ub", "scale"], param_names), index is segment idx.
        Also includes "Equations" key for equations DataFrame.
    """
    if not segment_dict:
        raise ValueError("segment_dict cannot be empty or None.")

    if base_fit_map is None:
        base_fit_map = {}
    if base_scale_map is None:
        base_scale_map = {}

    def get_value(map_dict, col, param, default, override=None):
        # Most specific to least specific
        if override is not None:
            default = override

        if not map_dict:
            return default
        for key in [
            (col, param),
            ("*", param),
            (col, "*"),
            ("*", "*"),
        ]:
            if key in map_dict:
                return map_dict[key]
        return default

    if skipcols is None:
        skipcols = set()
    elif not isinstance(skipcols, set):
        skipcols = set(skipcols)

    # --- Build maps ---
    example_cols = list(next(iter(segment_dict.values())).columns)[1:]  # Skip 'time' column

    fit_map = {}
    spec_map = {}
    scale_map = {}
    equation_names = {}  # Add this line

    for col in example_cols:
        if col in skipcols or ("raw" in col and not fit_raw):
            continue
        col_base = col.replace("raw_", "").split(" ")[0]

        fit_func = base_fit_map.get(col, base_fit_map.get(col_base, col_base))
        equation_names[col] = default_func.get(fit_func, fit_func)
        if callable(equation_names[col]):
            equation_names[col] = equation_names[col].__name__

        fit_map[col], spec_map[col] = get_fit_func(fit_func)
        scale_map[col] = base_scale_map.get(col, base_scale_map.get(col_base, "linear"))

    p0_map = kwargs.get("p0_map", {})
    lb_map = kwargs.get("lb_map", {})
    ub_map = kwargs.get("ub_map", {})

    lb_mod = partial(get_value, kwargs.get("lb_mod_map", {}), default=kwargs.get("mod", 10))
    ub_mod = partial(get_value, kwargs.get("ub_mod_map", {}), default=kwargs.get("mod", 10))

    # --- Build fit_params_dict ---
    fit_params_dict = {}
    fit_param_val_dict = {}
    equations = {}

    if stats_df is None:
        stats_df = collect_segment_stats(segment_dict, list(fit_map.keys()))
    # example_df = create_std_df(list(segment_dict.keys()), list(fit_map.keys()))
    # mi: pd.MultiIndex = stats_df.index  # type: ignore

    for col, fit_func in fit_map.items():
        # --- Parameter names and equation string ---
        ini_df = None
        prior_df = (
            prior_fit[col].copy(deep=True) if prior_fit is not None and col in prior_fit else None
        )
        # prior_df = prior_fit.get(col) if prior_fit is not None else None

        # t_max = pd.Series({idx: segment_dict[idx].iloc[-1, 0] for idx in mi}, index=mi)
        n_pairs = 1
        # r_vals = pd.Series(dtype=float)
        if fit_func is linear_func:
            param_names = ["a0", "b0"]
            eqn = "a0 * t + b0"
            g_func = initial_guess_poly_func

        elif fit_func in [exp_func, double_exp_func, n_exp_func]:
            n_pairs = spec_map.get(col, 1)
            param_names = [p for i in range(n_pairs) for p in (f"a{i}", f"tau{i}")] + ["b0"]
            eqn = " + ".join([f"a{i} * exp(-t / tau{i})" for i in range(n_pairs)]) + " + b0"
            g_func = partial(initial_guess_exp_func, n_exp=n_pairs)

        elif fit_func is pow_func:
            param_names = ["a0", "tau0", "b0"]
            eqn = "a0 * t ** (-tau0) + b0"
            g_func = initial_guess_pow_func

        elif fit_func is stretch_exp_func:
            param_names = ["a0", "tau0", "beta0", "b0"]
            eqn = "a0 * exp(-(t / tau0) ** beta0) + b0"
            g_func = initial_guess_str_exp_func

        else:
            param_names = [f"p{i}" for i in range(fit_func.__code__.co_argcount - 1)]
            eqn = " + ".join(param_names)
            g_func = partial(initial_guess_poly_func, deg=len(param_names) - 1)

        p0_df = pd.DataFrame(
            [
                g_func(
                    segment_dict[idx][["time", col]], b0=stats_df.loc[idx].get(f"{col}_f", np.nan)
                )
                for idx in stats_df.index
            ],
            index=stats_df.index,
            columns=param_names,
            dtype=float,
        )
        # sign_df = np.sign(p0_df)

        equations[col] = eqn

        tuples = [(b, p) for p in param_names for b in ["lb", "p0", "ub", "scale"]]
        columns = pd.MultiIndex.from_tuples(tuples, names=["bound", "param"])

        df = pd.DataFrame(index=stats_df.index, columns=columns, dtype=float)

        res_df = pd.DataFrame(
            index=stats_df.index,
            columns=[item for p in param_names for item in (p, f"{p}_std")] + ["Error"],
            dtype=float,
        )
        res_df["timestamp"] = pd.Series(pd.NaT, dtype="datetime64[ns]")

        scale_val = scale_map.get(col, "linear")

        # --- Define p0 values ---
        # 1. manually provided defaults, max priority
        for mapper, gr in zip([p0_map, lb_map, ub_map], ["p0", "lb", "ub"]):
            for keys in mapper:
                if keys[1] == "*":
                    df[gr] = get_value(mapper, col, keys[1], np.nan)
                    break
                elif keys[1] in param_names:
                    df[(gr, keys[1])] = get_value(mapper, col, keys[1], np.nan)

        # 2. values from history
        if initial_guesses is not None and col in initial_guesses:
            ini_df = initial_guesses[col].copy(deep=True)
            if "tau0" in param_names:
                dif = n_pairs - max([int(p[-1]) + 1 for p in ini_df.columns if p[-1].isdigit()])
                if dif > 0:
                    # more new pairs than in ini_df
                    ini_df = ini_df.rename(
                        columns=lambda c: re.sub(
                            r"(tau|a)(\d+)", lambda x: f"{x.group(1)}{int(x.group(2)) + dif}", c
                        )
                    )

                elif dif < 0:
                    # fewer new pairs than in ini_df
                    dif = abs(dif)

                    ini_df[f"a{dif}"] = ini_df[[f"a{i}" for i in range(dif + 1)]].mean(axis=1)
                    ini_df[f"tau{dif}"] = ini_df[[f"tau{i}" for i in range(dif + 1)]].mean(axis=1)
                    ini_df[f"a{dif}_std"] = ini_df[f"tau{dif}_std"] = 1

                    ini_df = ini_df.rename(
                        columns=lambda c: re.sub(
                            r"(tau|a)(\d+)", lambda x: f"{x.group(1)}{int(x.group(2)) - dif}", c
                        )
                    )

            df["p0"] = df["p0"].fillna(ini_df)

        # 3. p0_df (data-driven initial guesses)
        df["p0"] = df["p0"].fillna(p0_df)

        for par in param_names:
            p0_sign = np.sign(df[("p0", par)])
            is_wrong = ((np.sign(df[("lb", par)]) != p0_sign) & ~df[("lb", par)].isna()) | (
                (np.sign(df[("ub", par)]) != p0_sign) & ~df[("ub", par)].isna()
            )

            if any(is_wrong):
                lb_old = df[("lb", par)][is_wrong].copy()
                ub_old = df[("ub", par)][is_wrong].copy()
                df.loc[is_wrong, ("lb", par)] = abs(ub_old) * p0_sign[is_wrong]
                df.loc[is_wrong, ("ub", par)] = abs(lb_old) * p0_sign[is_wrong]

        # Make sure p0 is within bounds when bounds are not nan
        df["p0"] = np.clip(
            df["p0"],
            df["lb"].fillna(-np.inf) + 1e-16,  # Use -inf when lb is nan
            df["ub"].fillna(np.inf) - 1e-16,  # Use inf when ub is nan
        )

        if prior_df is not None and list(prior_df.columns) != list(res_df.columns):
            prior_df = None

        if prior_df is not None:
            res_df["Error"] = prior_df.get("Error", 0.0)
            res_df["timestamp"] = prior_df.get("timestamp", pd.NaT)

        elif ini_df is not None:
            res_df["Error"] = 0.0
            res_df["timestamp"] = ini_df.get("timestamp", pd.NaT)

        # --- Define bounds ---
        tau_lb_mod = -1
        tau_ub_mod = 1
        taus_idx = []
        if "tau0" in param_names:
            taus_idx = [f"tau{i}" for i in range(n_pairs)]
            tau_df: pd.DataFrame = df["p0"][taus_idx]  # type: ignore[assignment]

            t_min_exp = -np.log10(stats_df.attrs["max_time"])
            t_max_exp = np.log10(max(kwargs.get("tau_p0_max", 1), 1))

            t_min_exp = min(t_min_exp, np.log10(tau_df.min().min()))
            t_max_exp = max(t_max_exp, np.log10(tau_df.max().max()))

            tau_lb_mod = -(t_max_exp - t_min_exp) - abs(lb_mod(col, "tau", override=5))
            tau_ub_mod = tau_df.mean().max() * ub_mod(col, "tau")

            lb_tau = (tau_df * 10**tau_lb_mod).combine(
                tau_df * [lb_mod(col, n) for n in taus_idx], func=np.minimum
            )
            lb_tau = lb_tau.where(abs(lb_tau) >= 1e-32, np.sign(lb_tau) * 1e-32)

            ub_tau = (tau_df / tau_ub_mod).combine(
                tau_df * [ub_mod(col, n) for n in taus_idx], func=np.maximum
            )
            ub_limit = np.maximum(10, [ub_mod(col, n) for n in taus_idx])
            ub_tau = ub_tau.where(ub_tau <= ub_limit, tau_df + ub_limit)

            # df.loc[:, ("lb", taus_idx)] = df.loc[
            #     :, (["lb"], taus_idx)
            # ].fillna(lb_tau)

            df[[("lb", i) for i in taus_idx]] = df[[("lb", i) for i in taus_idx]].fillna(lb_tau)
            df[[("ub", i) for i in taus_idx]] = df[[("ub", i) for i in taus_idx]].fillna(ub_tau)

            if "b0" in param_names:
                # Define b0 bounds for tau system
                k = lb_mod(col, "b0", override=0.05)
                k = k if abs(k) < abs(1 / k) else 1 / k

                if f"{col}_0" in stats_df.columns:
                    a_sum = stats_df[f"{col}_0"] - stats_df[f"{col}_f"]
                    df[("p0", "b0")] = df[("p0", "b0")].where(
                        np.sign(stats_df[f"{col}_0"] - df[("p0", "b0")]) == np.sign(a_sum),
                        stats_df[f"{col}_f"],
                    )
                    min_bounds = df[("p0", "b0")] + a_sum * abs(k)
                    max_bounds = df[("p0", "b0")] - a_sum * abs(ub_mod(col, "b0", override=2) - k)
                else:
                    a_sum = df[[("p0", f"a{i}") for i in range(n_pairs)]].sum(axis=1)
                    min_bounds = df[("p0", "b0")] + a_sum * k
                    max_bounds = df[("p0", "b0")] - a_sum * (ub_mod(col, "b0", override=2) - k)

                # Close b0 bounds (within dataset)
                lb_min_mask = df[("lb", "b0")].isna() & (a_sum <= 0)  # increasing
                ub_min_mask = df[("ub", "b0")].isna() & (a_sum > 0)  # decreasing

                df[("lb", "b0")] = df[("lb", "b0")].where(~lb_min_mask, min_bounds)
                df[("ub", "b0")] = df[("ub", "b0")].where(~ub_min_mask, min_bounds)

                # Far b0 bounds (beyond dataset and/or towards SS)
                # m = ub_mod(col, "b0", override=2)
                ub_max_mask = df[("ub", "b0")].isna() & (a_sum <= 0)  # increasing
                lb_max_mask = df[("lb", "b0")].isna() & (a_sum > 0)  # decreasing

                df[("ub", "b0")] = df[("ub", "b0")].where(~ub_max_mask, max_bounds)
                df[("lb", "b0")] = df[("lb", "b0")].where(~lb_max_mask, max_bounds)

        # --- Vectorized bounds for beta ---
        if "beta0" in param_names:
            df[("lb", "beta0")] = df[("lb", "beta0")].fillna(1e-32)
            df[("ub", "beta0")] = df[("ub", "beta0")].fillna(1.0)

        # --- Vectorized bounds for all other parameters ---
        lb_consts, ub_consts = zip(*[(lb_mod(col, p), ub_mod(col, p)) for p in param_names])

        lb_other = (df["p0"] / lb_consts).combine(df["p0"] * lb_consts, func=np.minimum)
        ub_other = (df["p0"] / ub_consts).combine(df["p0"] * ub_consts, func=np.maximum)
        df["lb"] = df["lb"].fillna(lb_other)
        df["ub"] = df["ub"].fillna(ub_other)

        df.update(
            prevent_param_overlaps(
                df,
                kwargs.get("check_overlaps", []),
                kwargs.get("overlap_group_keys", []),
            )
        )

        df = validate_parameter_bounds(df, use_multiply=False, param_names=param_names)

        # --- Iterate over parameters ---
        for n, p_name in enumerate(param_names):
            # --- Set scale ---
            if isinstance(scale_val, (list, tuple)):
                df[("scale", p_name)] = scale_val[n].lower() if n < len(scale_val) else "linear"
            elif isinstance(scale_val, str):
                df[("scale", p_name)] = scale_val.lower()

            # --- Fill res_df from p0's, priors, and or ini_df ---
            res_df[p_name] = df[("p0", p_name)].astype(float)
            res_df[f"{p_name}_std"] = 1.0
            if prior_df is not None and p_name in prior_df:
                res_df[p_name] = prior_df[p_name]
                res_df[f"{p_name}_std"] = prior_df[f"{p_name}_std"]
            elif ini_df is not None and p_name in ini_df:
                res_df[p_name] = ini_df[p_name]
                # res_df[f"{p_name}_std"] = ini_df[f"{p_name}_std"]

            if df[("scale", p_name)].str.contains("const").any():
                df[("p0", p_name)] = res_df[p_name]
                df[("lb", p_name)] = res_df[p_name] - np.spacing(res_df[p_name].abs()) * 1e3
                df[("ub", p_name)] = res_df[p_name] + np.spacing(res_df[p_name].abs()) * 1e3
                df[("scale", p_name)] = "linear"

        fit_params_dict[col] = df
        fit_param_val_dict[col] = res_df

    fit_params_dict["Equations"] = pd.DataFrame.from_dict(
        {col: {"Equation": equations[col], "Eqn Name": equation_names[col]} for col in equations},
        orient="index",
        columns=["Equation", "Eqn Name"],
    )
    fit_param_val_dict["Equations"] = fit_params_dict["Equations"].copy()

    return fit_map, fit_params_dict, fit_param_val_dict


def fit_segments(
    segment_dict: dict,
    fit_map: dict,
    fit_params_dict: dict,
    *,
    fit_df: pd.DataFrame | None = None,
    fit_curves: dict[tuple, pd.DataFrame] | None = None,
    fit_params: dict | None = None,
    targets: set[tuple[str, tuple]] | None = None,  # (col, idx) pairs
    weight_inflect: int | float | None = None,
    retain_best: bool = True,
    error_func: str | Callable = "mean_abs_perc_err",
    status: dict | None = None,
    verbose: bool = True,
    **kwargs,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Fit all segments for each sample and update fit results.

    Terminology
    -----------
    info_index : tuple
        A tuple of identifying information for each segment, in the form
        (sample_name, condition, temp, run, segment). Used as the MultiIndex for all
        major DataFrames and dictionaries in this module.
    main_param
        Any primary measured or calculated quantity, such as Voltage, Current,
        Resistance, or Temperature.
    main_param_columns
        The full set of primary measured or calculated quantities (see main_param).
        This includes rms & abs forms of Voltage & Current (e.g. `<main_param> rms`)
        plus raw forms (e.g., `raw_<main_param>`).
    fit_param_columns
        For each fit parameter, columns alternate between the fit value and its standard
        deviation, e.g., "A", "A_std", "tau", "tau_std", ..., "Error".

    Parameters
    ----------
    segment_dict : dict[info_index, pd.DataFrame]
        Dictionary mapping each info_index to a DataFrame of segment data.
    fit_map : dict[str, Callable]
        Mapping of column name (main_param) to fit function.
    fit_params_dict : dict[str, pd.DataFrame]
        Dictionary mapping each main_param to a DataFrame of fit parameter bounds and
        initial guesses. Columns are a MultiIndex (["lb", "p0", "ub", "scale"], param_names).
        Index is info_index. Also includes "Equations" key for equations DataFrame.
    weight_inflect : float or None, optional
        Controls the inflection point and direction of the weighting profile.
        If None, no weighting is applied, default.
        If not None, weights are initialized from normalized time array (0 to 1), then:
        - If weight_inflect < 0: V-shaped weights (1→min→1) with inflection at |weight_inflect|
        - If weight_inflect > 0: Λ-shaped weights (0→max→0) with inflection at |weight_inflect|
        - If weight_inflect = 0 or -1: weights decrease monotonically from 1 to 0
    Values are clipped to the range [-1.0, 1.0].
    fit_df : pd.DataFrame, optional
        DataFrame to update with fit values for each main_param. If None, a new DataFrame
        is created.
    fit_curves : dict, optional
        Dictionary to update with fitted curves for each segment and main_param.
        If None, a new dictionary is created.
    fit_params : dict, optional
        Dictionary to update with fit parameters and statistics for each main_param.
        If None, a new dictionary is created.
    targets : set of (str, tuple), optional
        set of (column, info_index) pairs specifying which fits to perform.
        If None, all columns and indices are fit.
    **kwargs
        Additional keyword arguments passed to the fitting function.

    Returns
    -------
    fit_df : pd.DataFrame
        DataFrame with fit values for each main_param.
        Columns -> main_param_columns; Index -> info_index.
    fit_curves : dict[info_index, pd.DataFrame]
        Dictionary of fitted curves for each segment and main_param.
        Keys -> info_index;
        Values -> DataFrames with columns: "time", main_param, main_param_fit, etc.
        Index -> measurement time.
    fit_params : dict[str, pd.DataFrame]
        Dictionary of fit parameters and statistics for each fit main_param.
        Keys -> main_param (fitted + pre-existing (if passed));
        Values -> DataFrames with columns:
            For each parameter, columns alternate between the fit value and its standard
            deviation (e.g., "A", "A_std", "tau", "tau_std", ..., "Error").
        Index -> info_index.
        Also includes "Equations" key with a DataFrame of fit equations.

    Notes
    -----
    - The term "info_index" is used throughout to refer to the MultiIndex tuple
      (sample_name, condition, temp, run, segment).
    - For each fit parameter, columns alternate between the fit value and its standard
      deviation, followed by "Error".
    - All outputs are updated in place if provided; otherwise, new objects are created.
    """
    # Remove "Equations" from fit_params_dict for iteration
    fit_param_cols = [col for col in fit_params_dict if col != "Equations"]
    fit_param_idxs = list(segment_dict.keys())

    # if isinstance(error_func, str):
    e_func: Callable = (
        error_func
        if callable(error_func)
        else getattr(Statistics, error_func, Statistics.mean_abs_perc_err)
    )

    # Create fit_df if not provided
    if fit_df is None:
        fit_df = create_std_df(fit_param_idxs, fit_param_cols)

    if add_cols := [col for col in fit_param_cols if col not in fit_df.columns]:
        fit_df[add_cols] = np.nan

    if len(fit_df) < len(segment_dict):
        df = create_std_df(fit_param_idxs, fit_param_cols)
        if fit_df.empty:
            fit_df[df.columns[0]] = [np.nan] * len(df)
            fit_df.index = df.index
        else:
            for idx in set(df.index) - set(fit_df.index):
                fit_df.loc[idx] = np.nan
        fit_df.sort_index(inplace=True)

    if not isinstance(fit_curves, dict):
        fit_curves = {}

    if not isinstance(fit_params, dict):
        fit_params = {}

    fit_params.setdefault("Equations", fit_params_dict["Equations"].copy())

    maps = {k: {} for k in ["name", "lb", "p0", "ub", "scale"]}
    for col in list(dict.fromkeys(list(fit_params.keys()) + fit_param_cols)):
        # Iterate over fit_param_cols and fit_params_dict (unique) columns

        # Skip "Equations" column
        if col == "Equations":
            continue

        # Update Equations in fit_params
        fit_params["Equations"].loc[col] = fit_params_dict["Equations"].loc[col]

        # Initialize maps for this column
        # Names generated from conditions p0's
        maps["name"][col] = [
            item for p in fit_params_dict[col]["p0"].columns for item in (p, f"{p}_std")
        ] + ["Error", "timestamp"]

        # Create separate lists for lb, p0, ub, and scale
        for var in ["lb", "p0", "ub", "scale"]:
            maps[var].update({col: fit_params_dict[col][var].T.to_dict(orient="list")})

        # Manage existing fit_params DataFrame
        if col in fit_params:
            if not fit_params[col].index.equals(fit_df.index):
                fit_params[col] = fit_params[col].reindex(index=fit_df.index)
            if list(fit_params[col].columns) != maps["name"][col]:
                fit_params[col] = fit_params[col].reindex(columns=maps["name"][col])
                # new_cols = [
                #     c
                #     for c in fit_params[col].columns
                #     if c not in maps["name"][col] and (c.endswith("_std") or c[-1].isdigit())
                # ] + maps["name"][col]
                # fit_params[col] = fit_params[col].reindex(columns=new_cols)
        else:
            # Or create a new DataFrame for this column
            fit_params[col] = pd.DataFrame(
                index=fit_df.index,
                columns=maps["name"][col],
                dtype=float,
            )
            # if "timestamp" in fit_params_dict[col].columns:
            fit_params[col] = fit_params[col].astype({"timestamp": "datetime64[ns]"})

        fit_params[col] = form_std_df_index(fit_params[col])
        fit_params[col]["status"] = 0.0

    if targets is not None and targets:
        target_cols, target_idx = map(set, zip(*targets))
        fit_param_cols = [col for col in fit_param_cols if col in target_cols]
        fit_param_idxs = [idx for idx in fit_param_idxs if idx in target_idx]
        target_set = set(targets)
    else:
        target_set = set(product(fit_param_cols, fit_param_idxs))

    # Initialize status dict if provided
    if status is None and verbose:
        status = {}

    if status is not None:
        status.clear()
        status["min left"] = np.inf  # Add eta field
        status["percent"] = 0.0
        status["percent_updated"] = 0.0
        status["ave_change"] = 0.0
        status["ave_run_error"] = 0.0
        status |= {f"error_of_{col.lower()}": 0.0 for col in fit_param_cols}
        if retain_best:
            status["updated"] = 0
        status["completed"] = 0
        status["total"] = len(fit_param_cols) * len(fit_param_idxs)
        status["start_time"] = time.ctime()
        # status["total_change"] = 0.0
        status["current_idx"] = None

    # w_inflect is (in effect) the inv slope of the first part, which becomes the inflection point
    weight_inflect = (
        float(np.clip(weight_inflect, -1.0, 1.0)) if weight_inflect is not None else None
    )

    start_time = time.time()
    for idx in fit_param_idxs:
        df = segment_dict[idx]
        t_arr = df.iloc[:, 0].to_numpy(copy=True)

        curve_df = fit_curves.get(
            idx,
            pd.DataFrame({"time": t_arr}, index=df.index, dtype=float).rename_axis("meas_time"),
        )
        curve_df.attrs |= df.attrs.copy()

        fit_df.loc[idx, "timestamp"] = df.attrs.get("timestamp", pd.Timestamp.now())

        weights = None
        if weight_inflect is not None:
            # 'not None' indicates that weights should be applied
            # Normalized time => default weights (monotonically increasing)
            # Effectively weight_inflect == 1
            weights = (t_arr - t_arr.min()) / (t_arr.max() - t_arr.min())

            if abs(weight_inflect) == 1.0:
                # Avoid division by zero (would occur via weights.max())
                weights = 1 - weights
            elif weight_inflect != 0.0:
                # Avoid division by zero (would occur via weights.min())
                weights = 2 * weights - abs(weight_inflect)
                weights[weights == 0] = 1 / (t_arr.size - 1) * 1e-8
                weights = np.where(weights < 0, weights / weights.min(), weights / weights.max())

            if weight_inflect >= 0:
                weights = 1 - weights

            weights[weights == 0] = 1 / (t_arr.size - 1) * 1e-8

        for col in fit_param_cols:
            if "timestamp" in fit_params[col].columns:
                fit_params[col].loc[idx, "timestamp"] = fit_df.loc[idx, "timestamp"]

            curve_df.attrs.setdefault(f"{col}_error", 0.0)
            curve_df.attrs.setdefault(f"{col}_change", 0.0)
            curve_df.attrs.setdefault(f"{col}_updated", 0.0)

            fit_params[col].loc[idx, "status"] = 0.0

            if status is not None:
                status["current_idx"] = idx
                status["completed"] += 1
                status["percent"] = round(status["completed"] / status["total"] * 100, 2)
                status[f"error_of_{col.lower()}"] = float(fit_params[col]["Error"].mean())

                if status["percent"] > 1.5:
                    status["min left"] = round(
                        ((time.time() - start_time) / status["completed"])
                        * (status["total"] - status["completed"])
                        / 60,
                        2,
                    )

                if verbose:
                    print(
                        f"\rProgress: {status['percent']:.2f}% | "
                        f"Min left: ~{status['min left']:.2f} min   ",
                        end="",
                        flush=True,
                    )

            fit_func = fit_map[col]
            arr = df[col].to_numpy(copy=True)

            if not isinstance(arr, np.ndarray) or arr.size < 3 or np.all(np.isnan(arr)):
                fit_params[col].loc[idx, "status"] = -1.0
                continue
            if (col, idx) not in target_set:
                fit_params[col].loc[idx, "status"] = -1.0
                continue

            curve_df.attrs[f"{col}_change"] = 0.0
            curve_df.attrs[f"{col}_updated"] = 0.0
            fit_params[col].loc[idx, "status"] += 0.5

            # Extract p0 and bounds from fit_params_dict DataFrame
            try:
                p0 = maps["p0"][col][idx]
                kwargs["bounds"] = (maps["lb"][col][idx], maps["ub"][col][idx])
                kwargs["scale"] = maps["scale"][col][idx]
            except KeyError as exc:
                raise KeyError(
                    f"Missing p0/bounds for idx={idx}, col={col} in fit_params_dict."
                ) from exc

            p0_arr = fit_func(t_arr, *p0)
            p0_error = float(e_func(arr, p0_arr, weights))
            old_error = fit_params[col].loc[idx, "Error"]
            if np.isnan(old_error):
                fit_params[col].loc[idx, "Error"] = old_error = p0_error

            if curve_df.attrs[f"{col}_error"] == 0.0:
                curve_df.attrs[f"{col}_error"] = old_error

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", OptimizeWarning)
                try:
                    fit_result = run_segment_fit(
                        t_arr,
                        arr,
                        fit_func,
                        p0=p0,
                        # bounds=(lb, ub),
                        weights=weights,
                        **kwargs,
                    )
                except Exception as exc:
                    raise exc
                for warn in w:
                    if issubclass(warn.category, OptimizeWarning):
                        print(
                            f"OptimizeWarning for fit: {warn.message}\n"
                            f"  index: {idx}\n"
                            f"  column: {col}\n"
                            f"  fit_func: {fit_func.__name__}\n"
                            f"  p0: {p0}\n"
                            f"  arr: {summarize_array(arr)}\n"
                        )

            if fit_result is not None:
                if "tau0" in fit_params[col].columns and fit_func is not stretch_exp_func:
                    # Sort exponential parameters if applicable
                    fit_result["params"], fit_result["cov"] = sort_exp_params(
                        fit_result["params"], fit_result["cov"]
                    )

                res_arr = fit_func(t_arr, *fit_result["params"])
                res_error = float(e_func(arr, res_arr, weights))

                if p0_error < res_error:
                    res_arr = p0_arr
                    res_error = p0_error
                    fit_result["params"] = np.asarray(p0)
                    fit_result["cov"] = np.ones_like(fit_result["cov"])
                    fit_params[col].loc[idx, "status"] += 0.1

                # Build param row for this column/segment
                param_row: list[Any] = list(
                    chain.from_iterable(zip(fit_result["params"], fit_result["cov"]))
                )

                param_row.append(res_error)
                param_row.append(fit_df.loc[idx, "timestamp"])
                param_row.append(fit_params[col].loc[idx, "status"])

                # If old == p0 then it's likely the first fit => prevent calc to trigger setting in next block
                if old_error != p0_error and old_error != 0.0:
                    curve_df.attrs[f"{col}_change"] = np.clip(
                        (old_error - res_error) / old_error * 100, -100, 100
                    )

                if (
                    not retain_best
                    or curve_df.attrs[f"{col}_change"] >= 0
                    or np.isnan(fit_df.loc[idx, col])
                ):
                    fit_params[col].loc[idx] = pd.Series(
                        param_row, index=maps["name"][col] + ["status"]
                    )
                    curve_df.attrs[f"{col}_error"] = old_error = res_error
                    curve_df.attrs[f"{col}_updated"] = 1.0
                    fit_params[col].loc[idx, "status"] += 0.5

                    # Store fit value (e.g., final value for exp, mean for linear)
                    if fit_func is linear_func:
                        value = np.mean(res_arr)
                    elif fit_func is pow_func:
                        value = res_arr[-1]
                    else:
                        value = fit_result["params"][-1]

                    # Add data and fit columns to curve_df
                    curve_df[col] = arr
                    curve_df[f"{col}_fit"] = res_arr

                    fit_df.loc[idx, col] = value

                # elif not p0_error:
                #     p0_error = float(e_func(arr, fit_func(t_arr, *p0), weights))
                #     if p0_error < old_error:
                #         curve_df.attrs[f"{col}_error"] = old_error = param_row[-2] = p0_error

                if status is not None:
                    if retain_best:
                        status["updated"] += curve_df.attrs[f"{col}_updated"]

                    status["ave_change"] = float(
                        np.round(
                            status["ave_change"]
                            + (curve_df.attrs[f"{col}_change"] - status["ave_change"])
                            / status["completed"],
                            3,
                        )
                    )
                    status["ave_run_error"] = float(
                        status["ave_run_error"]
                        + (old_error - status["ave_run_error"]) / status["completed"]
                    )
                    status["percent_updated"] = float(
                        np.round(
                            status["percent_updated"]
                            + (100 * curve_df.attrs[f"{col}_updated"] - status["percent_updated"])
                            / status["completed"],
                            3,
                        )
                    )

        if len(curve_df) > 0:
            fit_curves[idx] = curve_df

    fit_df = fit_df.dropna(axis=1, how="all")

    for col in fit_params:
        if col == "Equations":
            continue
        fit_params[col].drop("status", axis=1, errors="ignore", inplace=True)

    return fit_df, fit_curves, fit_params


BASIC_STATS = {
    "mean": np.mean,
    "median": np.median,
    "std": np.std,
    "quantile": np.quantile,
    "min": np.min,
    "max": np.max,
    "sum": np.sum,
    "var": np.var,
}


def create_target_mask(
    df: pd.DataFrame,
    expr: str = "Error > 1e-10",
) -> pd.Series:
    """
    Generate a boolean mask for rows in `df` where the logical expression `expr` is True.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to evaluate.
    expr : str, optional
        Logical expression string to evaluate for each row. This should be '==' (used to select
        all rows) or a valid pandas expression referencing columns in `df`, e.g.:
            - "Error > 1e-10"
            - "a0 < 0.5 and Error < 0.01"
            - "tau0 >= 1 and tau1 <= 10"
        The expression is passed to `df.eval(expr)`. If expr is "==", all rows are selected.

    Returns
    -------
    pd.Series
        Boolean mask (index matches `df.index`), True where `expr` is satisfied.

    Notes
    -----
    - If `expr` is invalid or evaluation fails, returns a mask of all False.
    - If `expr` is "==", returns a mask of all True.
    - Use column names as they appear in `df` for the expression.
    """
    if expr == "==":
        expr = f"{df.columns[0]} == {df.columns[0]}"
    elif "@" in expr:
        # Extract function names from expressions like @function(args)
        func_names = re.findall(r"@(\w+)\(", expr)
        for func_name in func_names:
            if func_name not in BASIC_STATS and hasattr(np, func_name):
                # Add the numpy function to BASIC_STATS
                BASIC_STATS[func_name] = getattr(np, func_name)
    try:
        mask = df.eval(expr, local_dict=BASIC_STATS)
        if isinstance(mask, pd.Series):
            return mask.astype(bool)
        if not isinstance(mask, pd.DataFrame):
            return pd.Series(
                mask if isinstance(mask, np.ndarray) else [bool(mask)] * len(df),
                index=df.index,
                dtype=bool,
            )
    except Exception:
        return pd.Series([False] * len(df), index=df.index, dtype=bool)
    return pd.Series([True] * len(df), index=df.index, dtype=bool)


def create_target_set(
    fit_params: dict,
    column: str,
    expr: str = "Error > 1e-10",
) -> set[tuple[str, tuple]]:
    """
    Return a set of (column, index) pairs for rows in `fit_params[column]` where `expr` is True.

    Parameters
    ----------
    fit_params : dict
        Dictionary mapping column names to DataFrames (typically fit parameter tables).
    column : str
        Key in `fit_params` specifying which DataFrame to use.
    expr : str, optional
        Logical expression string evaluated in `create_target_mask` to select rows. This should be '=='
        (used to select all rows) or a valid pandas expression referencing columns in `df`, e.g.:
            - "Error > 1e-10"
            - "a0 < 0.5 and Error < 0.01"
            - "tau0 >= 1 and tau1 <= 10"
        The expression is passed to `df.eval(expr)`. If expr is "==", all rows are selected.

    Returns
    -------
    set of (str, tuple)
        Set of (column, index) pairs where the row in `fit_params[column]` satisfies `expr`.

    Notes
    -----
    - If `column` is not in `fit_params`, returns an empty set.
    - The `expr` string is passed to `pandas.DataFrame.eval`.
    - Use column names as they appear in the DataFrame for the expression.
    """
    df = fit_params.get(column)
    if df is None:  # or param not in df.columns:
        return set()
    mask = create_target_mask(df, expr)  # param, op, value)
    return set((column, idx) for idx in df.index[mask])


def targeted_df_update(
    target_df: pd.DataFrame,
    source_df: pd.DataFrame,
    reference: pd.DataFrame,
    expr: str = "Error > 1e-10",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Update target_df rows from source_df where param meets the condition, using a mask.

    Parameters
    ----------
    target_df : pd.DataFrame
        DataFrame to update (will not be modified in place).
    source_df : pd.DataFrame
        DataFrame to copy values from.
    param : str, optional
        Column to apply the condition to.
    op : str, optional
        Operator as a string: one of "<", "<=", ">", ">=", "==", "!=".
    value : float, optional
        Value to compare against.
    columns : list[str] or None, optional
        Columns to update. If None, all columns are updated.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame (copy).
    """
    mask = create_target_mask(reference, expr)  # , param, op, value)
    updated_df = target_df.copy(deep=True)
    cols_to_update = pd.Index(columns) if columns is not None else source_df.columns
    # Only update rows where mask is True and index exists in both DataFrames
    common_idx = updated_df.index.intersection(source_df.index[mask])
    updated_df.loc[common_idx, cols_to_update] = source_df.loc[common_idx, cols_to_update].copy()  # type: ignore
    return updated_df


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


# ARCHIVE:

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


# def fit_preparation(
#     segment_dict: dict,
#     mean_df: pd.DataFrame | None = None,
#     skipcols: list[str] | set[str] | None = None,
#     fit_raw: bool = False,
#     base_fit_map: dict | None = None,
#     base_scale_map: dict | None = None,
#     prior_fits: dict | None = None,
#     **kwargs,
# ) -> tuple[dict, dict]:
#     """
#     Prepare fit_map and fit_params_dict for segment fitting.

#     Parameters
#     ----------
#     segment_dict : dict
#         Dictionary of segment DataFrames.
#     mean_df : pd.DataFrame, optional
#         DataFrame of mean values for initial guesses.
#     skipcols : list[str], optional
#         Columns to skip.
#     fit_raw : bool, optional
#         Whether to include raw columns.
#     base_fit_map : dict, optional
#         Mapping of base column names to fit functions.

#     base_scale_map : dict, optional
#         Mapping of column name to scale type(s) for each parameter.

#     Returns
#     -------
#     fit_map : dict
#         Mapping of column name to fit function.
#     fit_params_dict : dict
#         Dict: {col: pd.DataFrame}, columns are MultiIndex (["lb", "p0", "ub", "scale"], param_names), index is segment idx.
#         Also includes "Equations" key for equations DataFrame.
#     """
#     if not segment_dict:
#         raise ValueError("segment_dict cannot be empty or None.")

#     if base_fit_map is None:
#         base_fit_map = {}
#     if base_scale_map is None:
#         base_scale_map = {}

#     def get_fit_func(val: str | Callable) -> tuple[Callable, int | None]:
#         """Get fit function by name, or return linear_func as default."""
#         if callable(val):
#             return val, None
#         special_match = re.match(r"^\d{1,2}_exp_func$", val)
#         if special_match:
#             special = int(special_match.group(0).split("_")[0])
#             return n_exp_func, special
#         # if isinstance(val, str):
#         if val in fit_func_map:
#             return fit_func_map[val], None
#         elif val in default_func:
#             return default_func[val], None
#         return linear_func, None

#     def get_value(map_dict, col, param, default, override=None):
#         # Most specific to least specific
#         if override is not None:
#             default = override

#         if not map_dict:
#             return default
#         for key in [
#             (col, param),
#             ("*", param),
#             (col, "*"),
#             ("*", "*"),
#         ]:
#             if key in map_dict:
#                 return map_dict[key]
#         return default

#     if skipcols is None:
#         skipcols = []
#     skipcols = set(skipcols)

#     # --- Build maps ---
#     examples = next(iter(segment_dict.values()))
#     example_cols = list(examples.columns)[1:]
#     example_idxs = list(examples.index)
#     fit_map = {}
#     spec_map = {}
#     scale_map = {}

#     for col in example_cols:
#         if col in skipcols or ("raw" in col and not fit_raw):
#             continue
#         col_base = col.replace("raw_", "").split(" ")[0]

#         fit_map[col], spec_map[col] = get_fit_func(
#             base_fit_map.get(col, base_fit_map.get(col_base, col_base))
#         )
#         scale_map[col] = base_scale_map.get(col, base_scale_map.get(col_base, "linear"))

#     get_p0 = partial(get_value, kwargs.get("p0_map", {}))
#     get_lb = partial(get_value, kwargs.get("lb_map", {}))
#     get_ub = partial(get_value, kwargs.get("ub_map", {}))
#     lb_mod = partial(get_value, kwargs.get("lb_mod_map", {}), default=kwargs.get("mod", 10))
#     ub_mod = partial(get_value, kwargs.get("ub_mod_map", {}), default=kwargs.get("mod", 10))

#     # --- Build fit_params_dict ---
#     fit_params_dict = {}
#     equations = {}

#     for col, fit_func in fit_map.items():


#         if fit_func is linear_func:
#             param_names = ["a0", "b0"]
#             eqn = "a0 * t + b0"
#         elif fit_func is exp_func:
#             param_names = ["a0", "tau0", "b0"]
#             eqn = "a0 * exp(-t / tau0) + b0"
#         elif fit_func is double_exp_func:
#             param_names = ["a0", "tau0", "a1", "tau1", "b0"]
#             eqn = "a0 * exp(-t / tau0) + a1 * exp(-t / tau1) + b0"
#         elif fit_func is n_exp_func:
#             n_exp_n = spec_map.get(col, 1)
#             param_names = [p for i in range(n_exp_n) for p in (f"a{i}", f"tau{i}")] + ["b0"]
#             eqn = " + ".join([f"a{i} * exp(-t / tau{i})" for i in range(n_exp_n)]) + " + b0"
#         elif fit_func is pow_func:
#             param_names = ["a0", "tau0"]
#             eqn = "a0 * t ** (-tau0)"
#         else:
#             # fallback
#             param_names = [f"p{i}" for i in range(fit_func.__code__.co_argcount - 1)]
#             eqn = " + ".join(param_names)

#         equations[col] = eqn

#         guesses = pd.DataFrame(index=example_idxs, columns=param_names)
#         if prior_fits is not None and col in prior_fits:
#             guesses = prior_fits[col]
#         elif mean_df is not None and col in mean_df.columns:
#             guesses["b0"] = mean_df.loc[:, [col]]

#         # Prepare scale info for this column
#         scale_val = scale_map.get(col, "linear")
#         if isinstance(scale_val, str):
#             scale_list = [scale_val] * len(param_names)
#         else:
#             scale_list = list(scale_val)
#             if len(scale_list) < len(param_names):
#                 scale_list += ["linear"] * len(param_names)
#             if len(scale_list) > len(param_names):
#                 scale_list = scale_list[: len(param_names)]

#         # MultiIndex columns: (bound/scale, param)
#         tuples = []
#         for pname in param_names:
#             for bound_type in ["lb", "p0", "ub", "scale"]:
#                 tuples.append((bound_type, pname))
#         columns = pd.MultiIndex.from_tuples(tuples, names=["bound", "param"])

#         rows = []  # lists of lists: holds the bounds, p0, and scale for each parameter
#         idxs = []
#         for idx, df in segment_dict.items():
#             t = df.iloc[:, 0].to_numpy(copy=True)
#             arr = df[col].to_numpy(copy=True)
#             priors = guesses.get(idx, pd.Series())

#             mean_guess = (
#                 float(mean_df.loc[idx, col])  # type: ignore
#                 if mean_df is not None and np.isfinite(mean_df.loc[idx, col])
#                 else None
#             )
#             if fit_func is exp_func:
#                 pri_b0 = priors.get("b0", arr[-1])
#                 # mean_guess = guesses.get.get("b0", arr[-1])
#                 mean_guess = mean_guess if mean_guess is not None else arr[-1]
#                 p0 = [(arr[0] - mean_guess), 1, mean_guess]  # Initial guess: a0, tau0, b0
#                 lb = [
#                     min(p0[0] / lb_mod(col, "a0"), p0[0] * lb_mod(col, "a0")),  # a0
#                     1e-5,  # tau0
#                     min(mean_guess / lb_mod(col, "b0"), mean_guess * lb_mod(col, "b0")),  # b0
#                 ]
#                 ub = [
#                     max(p0[0] / ub_mod(col, "a0"), p0[0] * ub_mod(col, "a0")),  # a0
#                     10.0,  # tau0
#                     max(mean_guess / ub_mod(col, "b0"), mean_guess * ub_mod(col, "b0")),  # b0
#                 ]
#             elif fit_func is double_exp_func:
#                 mean_guess = mean_guess if mean_guess is not None else arr[-1]
#                 # Initial guess: a0, tau0, a1, tau1, b0
#                 p0 = [(arr[0] - mean_guess), 1.5, (arr[0] - mean_guess) / 4, 6.5, mean_guess]
#                 lb = [
#                     min(p0[0] / lb_mod(col, "a0"), p0[0] * lb_mod(col, "a0")),  # a0
#                     1e-5,  # tau0
#                     min(p0[2] / lb_mod(col, "a1"), p0[2] * lb_mod(col, "a1")),  # a1
#                     1.0,  # tau1
#                     min(mean_guess / lb_mod(col, "b0"), mean_guess * lb_mod(col, "b0")),  # b0
#                 ]
#                 ub = [
#                     max(p0[0] / ub_mod(col, "a0"), p0[0] * ub_mod(col, "a0")),  # a0
#                     3.0,  # tau0
#                     max(p0[2] / ub_mod(col, "a1"), p0[2] * ub_mod(col, "a1")),  # a1
#                     10.0,  # tau1
#                     max(mean_guess / ub_mod(col, "b0"), mean_guess * ub_mod(col, "b0")),  # b0
#                 ]
#             elif fit_func is n_exp_func:
#                 mean_guess = mean_guess if mean_guess is not None else arr[-1]
#                 n_exp_n = spec_map.get(col, 1)
#                 t_min_exp = -np.log10(max(t.max(), 2))
#                 t_max_exp = np.log10(max(kwargs.get("tau_p0_max", 1), 1))

#                 amps = [(arr[0] - mean_guess) / (n + 1) for n in range(n_exp_n)]
#                 lb_amps = [
#                     min(a / lb_mod(col, f"a{i}"), a * lb_mod(col, f"a{i}"))
#                     for i, a in enumerate(amps)
#                 ]
#                 ub_amps = [
#                     max(a / ub_mod(col, f"a{i}"), a * ub_mod(col, f"a{i}"))
#                     for i, a in enumerate(amps)
#                 ]

#                 taus = np.logspace(t_min_exp, t_max_exp, n_exp_n)
#                 lb_taus = taus * 10 ** (
#                     -(t_max_exp - t_min_exp) - abs(lb_mod(col, "tau", override=5))
#                 )
#                 ub_taus = taus / np.max(taus) * ub_mod(col, "tau")

#                 p0 = [val for pair in zip(amps, taus) for val in pair] + [mean_guess]
#                 # Interleave amp/tau bounds
#                 lb = [val for pair in zip(lb_amps, lb_taus) for val in pair] + [
#                     min(mean_guess / lb_mod(col, "b0"), mean_guess * lb_mod(col, "b0"))
#                 ]
#                 ub = [val for pair in zip(ub_amps, ub_taus) for val in pair] + [
#                     max(mean_guess / ub_mod(col, "b0"), mean_guess * ub_mod(col, "b0"))
#                 ]

#             elif fit_func is pow_func:
#                 t_one = find_nearest(t, 1.0)
#                 arr_one = arr[t_one]
#                 mean_guess = mean_guess if mean_guess is not None else arr[-1]
#                 r_guess = np.log10(abs(mean_guess)) / np.log10(abs(arr_one) * t[-1])
#                 p0 = [arr_one, r_guess]  # Initial guess: a0, tau0
#                 lb = [min(p0[0] / lb_mod(col, "a0"), p0[0] * lb_mod(col, "a0")), 1e-20]  # a0, tau0
#                 ub = [max(p0[0] / ub_mod(col, "a0"), p0[0] * ub_mod(col, "a0")), 10.0]  # a0, tau0
#             else:  # Defaults to linear_func
#                 mean_guess = mean_guess if mean_guess is not None else arr.mean()
#                 p0 = [0.0, mean_guess]  # Initial guess: a0, b0
#                 lb = [
#                     -1e32,
#                     min(mean_guess / lb_mod(col, "a0"), mean_guess * lb_mod(col, "b0")),
#                 ]  # a0, b0
#                 ub = [
#                     1e32,
#                     max(mean_guess / ub_mod(col, "a0"), mean_guess * ub_mod(col, "b0")),
#                 ]  # a0, b0

#             # Compose row as [lb, p0, ub, scale] for each param, grouped by param
#             row = []
#             for i, p_name in enumerate(param_names):
#                 row.extend(
#                     [
#                         get_lb(col, p_name, lb[i]),
#                         get_p0(col, p_name, p0[i]),
#                         get_ub(col, p_name, ub[i]),
#                         scale_list[i],
#                     ]
#                 )

#             rows.append(row)
#             idxs.append(idx)
#         # fit_params_dict[col] = pd.DataFrame(rows, index=idxs, columns=columns)
#         multi_index = pd.MultiIndex.from_tuples(
#             idxs, names=["sample_name", "condition", "temp", "run", "segment"]
#         )
#         fit_params_dict[col] = pd.DataFrame(rows, index=multi_index, columns=columns)

#     fit_params_dict["Equations"] = pd.DataFrame.from_dict(
#         equations, orient="index", columns=["Equation"]
#     )
#     return fit_map, fit_params_dict

# def plot_fit_curves(
#     fit_curves: dict,
#     targets: set[tuple[str, tuple]],
#     groupby: list[int] | int | None = None,
#     legend: bool = False,
# ):
#     """
#     Plot data and fit curves for the specified targets.

#     Parameters
#     ----------
#     fit_curves : dict[tuple, pd.DataFrame]
#         Dictionary of fitted curves for each segment and main_param.
#     targets : set of (str, tuple)
#         Set of (column, info_index) pairs specifying which curves to plot.
#     groupby : list[int] | int | None, optional
#         Indices of info_index to group by. If None, all curves are plotted on a single plot.
#         If int or list of ints, a separate plot is made for each group.

#     Notes
#     -----
#     - Dashed lines are used for fits, matching the color of the data.
#     - Grid is enabled on all plots.
#     """
#     if not fit_curves:
#         print("No fit curves available to plot.")
#         return
#     # Organize targets by group
#     if groupby is None:
#         groups = {None: list(targets)}
#     else:
#         if isinstance(groupby, int):
#             groupby = [groupby]
#         groups = {}
#         for col, idx in targets:
#             group_key = tuple(idx[i] for i in groupby)
#             groups.setdefault(group_key, []).append((col, idx))

#     for group, group_targets in groups.items():
#         fig, ax = plt.subplots()
#         for col, idx in group_targets:
#             curve_df = fit_curves.get(idx)
#             if curve_df is None or f"{col}" not in curve_df:
#                 continue
#             # Plot data
#             line = ax.plot(
#                 curve_df["time"],
#                 curve_df[f"{col}"],
#                 label=f"{col} data {idx}",
#                 alpha=0.8,
#             )[0]
#             # Plot fit with dashed line, matching color
#             if f"{col}_fit" in curve_df:
#                 ax.plot(
#                     curve_df["time"],
#                     curve_df[f"{col}_fit"],
#                     linestyle="--",
#                     color=line.get_color(),
#                     label=f"{col} fit {idx}",
#                     alpha=0.8,
#                 )
#         ax.set_xlabel("Time")
#         ax.set_ylabel("Value")
#         ax.grid(True)
#         if legend:
#             ax.legend()
#         if group is not None:
#             ax.set_title(f"Group: {group}")
#         plt.tight_layout()
#         plt.show()


# def clean_segment_data(
#     standardized_dict: dict[tuple, pd.DataFrame],
#     min_length: int = 3,
#     shift_start: int = 0,
#     shift_end: int = 0,
#     stable_voltage: bool = False,
#     check_exp_start: bool = False,
#     check_exp_end: bool = False,
#     thresholds: float | list[float] = 0.25,
#     alpha: float = 0.05,
#     allowed_deviations: int = 2,
#     reset_time_zero: bool = True,
#     smooth: bool = False,
#     window_length: int | float = 0.05,
#     polyorder: int = 2,
#     delta: float | int | None = None,
#     columns_to_smooth: list[str] | None = None,
#     **kwargs,
# ) -> dict[tuple, pd.DataFrame]:
#     """
#     Clean segment data by trimming to stable voltage and/or exponential current regions,
#     with optional smoothing.

#     Parameters
#     ----------
#     standardized_dict : dict[tuple, pd.DataFrame]
#         Dictionary mapping segment identifiers to DataFrames of segment data.
#     min_length : int, optional
#         Minimum number of rows to retain in each segment after trimming (default: 3).
#     shift_start : int, optional
#         Number of additional rows to trim from the start after stability checks (default: 0).
#     shift_end : int, optional
#         Number of additional rows to trim from the end after stability checks (default: 0).
#     stable_voltage : bool, optional
#         If True, trims segment to region where voltage is within threshold of its median (default: False).
#     check_exp_start : bool, optional
#         If True, trims segment start using robust exponential region detection (default: False).
#     check_exp_end : bool, optional
#         If True, trims segment end where normalized current difference exceeds threshold (default: False).
#     alpha : float, optional
#         Significance level for Grubbs' test in outlier detection (default: 0.05).
#     allowed_deviations : int, optional
#         Number of consecutive non-outlier iterations to stop outlier removal (default: 2).
#     thresholds : float or list[float], optional
#         Threshold(s) for voltage and current checks. If a single float, used for both.
#         If a list, thresholds[0] is for voltage, thresholds[1] for current (default: 0.25).
#     reset_time_zero : bool, optional
#         If True, resets the "time" column to start at zero after trimming (default: True).
#     smooth : bool, optional
#         If True, applies Savitzky-Golay smoothing to specified columns (default: False).
#     window_length : int or float, optional
#         Window length for Savitzky-Golay filter. If float in (0, 1), interpreted as fraction of data length (default: 5%).
#     polyorder : int, optional
#         Polynomial order for Savitzky-Golay filter (default: 2).
#     delta : float or int or None, optional
#         Spacing of samples for Savitzky-Golay filter. If None, estimated from "time" column (default: None).
#     columns_to_smooth : list[str] or None, optional
#         List of columns to smooth. If None, all numeric columns except "time" are smoothed (default: None).
#     **kwargs
#         Additional keyword arguments passed to Savitzky-Golay filter.

#     Returns
#     -------
#     cleaned_dict : dict[tuple, pd.DataFrame]
#         Dictionary mapping segment identifiers to cleaned DataFrames.

#     Notes
#     -----
#     - Trimming is performed if any of `stable_voltage`, `check_exp_end`, `shift_start`, or `shift_end` are set.
#     - All four trimming options can be used independently or in combination.
#     - Voltage is normalized by its median and checked for deviation.
#     - Current's rate of change at the endpoints are checked for deviations.
#     - Trimming ensures at least `min_length` rows remain after all operations.
#     - Smoothing is applied after trimming if enabled.
#     """

#     cleaned_dict = {}
#     min_length = max(abs(min_length), 1)
#     if isinstance(thresholds, (int, float)):
#         thresholds = [float(thresholds)] * 2
#     polyorder = max(0, int(polyorder))

#     cols_to_smooth = set()
#     if columns_to_smooth:
#         cols_to_smooth = set(columns_to_smooth)

#     for idx, df in standardized_dict.items():
#         df_clean = df.copy()

#         # --- Trim to stable voltage region ---
#         if (
#             stable_voltage or check_exp_end or check_exp_start or shift_start or shift_end
#         ) and len(df_clean) >= 2 + min_length:
#             voltage = abs(
#                 df_clean["Voltage"].to_numpy(copy=True) / np.nanmedian(df_clean["Voltage"].to_numpy(copy=True)) - 1
#             )

#             start = 0
#             end = len(voltage)

#             while start < end - min_length and stable_voltage and voltage[start] > thresholds[0]:
#                 start += 1

#             start = min(start + abs(shift_start), end - min_length)

#             if check_exp_start:
#                 while (
#                     end > start + min_length
#                     and stable_voltage
#                     and voltage[end - 1] > thresholds[0]
#                 ):
#                     end -= 1

#                 end = max(end - abs(shift_end), start + min_length)

#                 current_arr = df_clean["Current"].to_numpy(copy=True)[start:end]
#                 exp_start_idx, exp_stop_idx = clean_exponential_data(
#                     current_arr,
#                     alpha=alpha,
#                     allowed_deviations=allowed_deviations,
#                     eval_endpoint=check_exp_end,
#                 )
#                 # Adjust indices relative to original DataFrame
#                 if check_exp_end:
#                     end = start + exp_stop_idx
#                 start += exp_start_idx
#                 # If not check_exp_end, end remains as previously set

#             else:
#                 current_diff = (
#                     (df_clean["Current"] / float(np.nanmedian(df_clean["Current"])))
#                     .diff()
#                     .bfill()
#                     .to_numpy(copy=True)
#                 )

#                 while end > start + min_length and (
#                     (stable_voltage and voltage[end - 1] > thresholds[0])
#                     or (check_exp_end and current_diff[end - 1] > thresholds[1])
#                 ):
#                     end -= 1

#                 end = max(end - abs(shift_end), start + min_length)

#             df_clean = df_clean.iloc[start:end].copy()
#             if reset_time_zero and "time" in df_clean.columns and not df_clean.empty:
#                 df_clean["time"] = df_clean["time"] - df_clean["time"].iloc[0]

#         # --- Smoothing ---
#         if smooth:
#             # if columns_to_smooth is None:
#             #     columns_to_smooth = [col for col in df_clean.columns if col != "time"]
#             valid_cols = set(df_clean.columns) - {"time"}
#             if cols_to_smooth:
#                 valid_cols &= cols_to_smooth

#             if not valid_cols:
#                 continue  # No valid columns to smooth

#             if delta is None:
#                 delta = float(np.diff(df_clean["time"]).mean())

#             if isinstance(window_length, float):
#                 if 0 < window_length < 1:
#                     # Treat as percentage of data length, allow for minimum window length
#                     window_length = max(int(len(df_clean) * window_length), polyorder + 1)
#                 else:
#                     # Assume it's a valid absolute window length in float format (e.g., 5.0)
#                     window_length = int(window_length)

#             if window_length > len(df_clean):
#                 window_length = polyorder + 1

#             if window_length % 2 == 0:
#                 window_length = window_length + 1

#             if len(df_clean) > window_length > polyorder:
#                 for col in valid_cols:
#                     if (
#                         np.issubdtype(str(df_clean[col].dtype), np.number)
#                         and not df_clean[col].isna().any()
#                     ):
#                         try:
#                             df_clean[col] = savgol_filter(
#                                 df_clean[col].to_numpy(copy=True),
#                                 window_length,
#                                 polyorder,
#                                 delta=delta,
#                                 **kwargs,
#                             )
#                         except Exception:
#                             pass  # If smoothing fails, leave column as is

#         cleaned_dict[idx] = df_clean

#     return cleaned_dict


# def create_std_df(
#     source_idxs: list | None = None,
#     source_cols: list | None = None,
#     mean_df: pd.DataFrame | None = None,
# ) -> pd.DataFrame:
#     """
#     Create or update the fit_df DataFrame for fit results.

#     Parameters
#     ----------
#     source_idxs : list
#         List of MultiIndex tuples for DataFrame index.
#     source_cols : list
#         List of columns for the DataFrame.
#     mean_df : pd.DataFrame, optional
#         DataFrame of mean values for initial guesses.

#     Returns
#     -------
#     pd.DataFrame
#         DataFrame with fit values for each main_param.
#     """
#     if mean_df is not None:
#         return mean_df.copy(deep=True)

#     if not source_idxs:
#         raise ValueError("source_idxs cannot be empty or None.")

#     if not source_cols:
#         source_cols = ["Voltage", "Current", "Resistance", "Temperature"]
#     else:
#         source_cols = [col for col in source_cols if col != "Equations"]

#     df = pd.DataFrame(
#         index=source_idxs,
#         columns=source_cols,
#         dtype=float,
#     )
#     df_mi = pd.MultiIndex.from_tuples(
#         source_idxs,
#         names=["sample_name", "condition", "temp", "run", "segment"],
#     )
#     df.index = df_mi.set_levels(
#         df_mi.levels[1].astype(pd.CategoricalDtype(["pre", "dh", "dry"], ordered=True)),
#         level=1,
#     )
#     df = df.sort_index()
#     return df

# def fit_preparation(
#     segment_dict: dict,
#     # mean_df: pd.DataFrame | None = None,
#     skipcols: list[str] | set[str] | None = None,
#     fit_raw: bool = False,
#     base_fit_map: dict | None = None,
#     base_scale_map: dict | None = None,
#     prior_fits: dict | None = None,
#     **kwargs,
# ) -> tuple[dict, dict, dict]:
#     """
#     Prepare fit_map and fit_params_dict for segment fitting.

#     Parameters
#     ----------
#     segment_dict : dict
#         Dictionary of segment DataFrames.
#     mean_df : pd.DataFrame, optional
#         DataFrame of mean values for initial guesses.
#     skipcols : list[str], optional
#         Columns to skip.
#     fit_raw : bool, optional
#         Whether to include raw columns.
#     base_fit_map : dict, optional
#         Mapping of base column names to fit functions.

#     base_scale_map : dict, optional
#         Mapping of column name to scale type(s) for each parameter.

#     Returns
#     -------
#     fit_map : dict
#         Mapping of column name to fit function.
#     fit_params_dict : dict
#         Dict: {col: pd.DataFrame}, columns are MultiIndex (["lb", "p0", "ub", "scale"], param_names), index is segment idx.
#         Also includes "Equations" key for equations DataFrame.
#     """
#     if not segment_dict:
#         raise ValueError("segment_dict cannot be empty or None.")

#     if base_fit_map is None:
#         base_fit_map = {}
#     if base_scale_map is None:
#         base_scale_map = {}

#     def get_value(map_dict, col, param, default, override=None):
#         # Most specific to least specific
#         if override is not None:
#             default = override

#         if not map_dict:
#             return default
#         for key in [
#             (col, param),
#             ("*", param),
#             (col, "*"),
#             ("*", "*"),
#         ]:
#             if key in map_dict:
#                 return map_dict[key]
#         return default

#     if skipcols is None:
#         skipcols = []
#     skipcols = set(skipcols)

#     # --- Build maps ---
#     examples = next(iter(segment_dict.values()))
#     example_cols = list(examples.columns)[1:]
#     example_df = create_std_df(list(segment_dict.keys()), example_cols)

#     fit_map = {}
#     spec_map = {}
#     scale_map = {}
#     equation_names = {}  # Add this line

#     for col in example_cols:
#         if col in skipcols or ("raw" in col and not fit_raw):
#             continue
#         col_base = col.replace("raw_", "").split(" ")[0]

#         fit_func = base_fit_map.get(col, base_fit_map.get(col_base, col_base))
#         equation_names[col] = default_func.get(fit_func, fit_func)
#         if callable(equation_names[col]):
#             equation_names[col] = equation_names[col].__name__

#         fit_map[col], spec_map[col] = get_fit_func(fit_func)
#         scale_map[col] = base_scale_map.get(col, base_scale_map.get(col_base, "linear"))

#     p0_map = kwargs.get("p0_map", {})
#     lb_map = kwargs.get("lb_map", {})
#     ub_map = kwargs.get("ub_map", {})

#     lb_mod = partial(get_value, kwargs.get("lb_mod_map", {}), default=kwargs.get("mod", 10))
#     ub_mod = partial(get_value, kwargs.get("ub_mod_map", {}), default=kwargs.get("mod", 10))

#     # --- Build fit_params_dict ---
#     fit_params_dict = {}
#     fit_param_val_dict = {}
#     equations = {}

#     mi = example_df.index

#     for col, fit_func in fit_map.items():
#         # --- Parameter names and equation string ---
#         priors = None
#         t_max = pd.Series({idx: segment_dict[idx].iloc[-1, 0] for idx in mi}, index=mi)
#         n_pairs = 1
#         # r_vals = pd.Series(dtype=float)
#         if fit_func is linear_func:
#             param_names = ["a0", "b0"]
#             eqn = "a0 * t + b0"
#             g_func = initial_guess_poly_func

#             # means = pd.Series([segment_dict[idx][col].mean() for idx in mi], index=mi)
#             # bases = pd.Series(np.zeros(len(segment_dict), dtype=float), index=mi)

#         elif fit_func in [exp_func, double_exp_func, n_exp_func]:
#             n_pairs = spec_map.get(col, 1)
#             param_names = [p for i in range(n_pairs) for p in (f"a{i}", f"tau{i}")] + ["b0"]
#             eqn = " + ".join([f"a{i} * exp(-t / tau{i})" for i in range(n_pairs)]) + " + b0"
#             g_func = partial(initial_guess_exp_func, n_exp=n_pairs)

#             # means = pd.Series([segment_dict[idx][col].iloc[-1] for idx in mi], index=mi)
#             # bases = pd.Series(
#             #     [segment_dict[idx][col].iloc[0] - means[idx] for idx in mi], index=mi
#             # )
#         elif fit_func is pow_func:
#             param_names = ["a0", "tau0", "b0"]
#             eqn = "a0 * t ** (-tau0) + b0"
#             g_func = initial_guess_pow_func

#             # guesses = list(
#             #     zip(*[initial_guess_pow_func(segment_dict[idx][["time", col]]) for idx in mi])
#             # )

#             # means = pd.Series([segment_dict[idx][col].iloc[-1] for idx in mi], index=mi)
#             # bases = pd.Series(guesses[0], index=mi)
#             # r_vals = pd.Series(guesses[1], index=mi)
#         elif fit_func is stretch_exp_func:
#             param_names = ["a0", "tau0", "beta0", "b0"]
#             eqn = "a0 * exp(-(t / tau0) ** beta0) + b0"
#             g_func = initial_guess_str_exp_func

#             # means = pd.Series([segment_dict[idx][col].iloc[-1] for idx in mi], index=mi)
#             # bases = pd.Series(
#             #     [segment_dict[idx][col].iloc[0] - means[idx] for idx in mi], index=mi
#             # )

#         else:
#             param_names = [f"p{i}" for i in range(fit_func.__code__.co_argcount - 1)]
#             eqn = " + ".join(param_names)
#             g_func = partial(initial_guess_poly_func, deg=len(param_names) - 1)

#             # means = pd.Series([segment_dict[idx][col].iloc[-1] for idx in mi], index=mi)
#             # bases = pd.Series(
#             #     [segment_dict[idx][col].iloc[0] - means[idx] for idx in mi], index=mi
#             # )

#         p0_df = pd.DataFrame(
#             [g_func(segment_dict[idx][["time", col]]) for idx in mi],
#             index=mi,
#             columns=param_names,
#             dtype=float,
#         )

#         equations[col] = eqn

#         tuples = [(b, p) for p in param_names for b in ["lb", "p0", "ub", "scale"]]
#         columns = pd.MultiIndex.from_tuples(tuples, names=["bound", "param"])

#         fit_params_df = pd.DataFrame(index=mi, columns=columns, dtype=float)

#         res_df = pd.DataFrame(
#             index=mi,
#             columns=[item for p in param_names for item in (p, f"{p}_std")] + ["Error"],
#             dtype=float,
#         )

#         scale_val = scale_map.get(col, "linear")

#         # --- Default/init values ---
#         # manually provided defaults, max priority
#         for mapper, gr in zip([p0_map, lb_map, ub_map], ["p0", "lb", "ub"]):
#             for keys in mapper:
#                 if keys[1] == "*":
#                     fit_params_df[gr] = get_value(mapper, col, keys[0], np.nan)
#                     break
#                 elif keys[1] in param_names:
#                     fit_params_df[(gr, keys[1])] = get_value(mapper, col, keys[1], np.nan)

#         # values from history
#         if prior_fits is not None and col in prior_fits:
#             priors = prior_fits[col]
#             fit_params_df["p0"] = fit_params_df["p0"].fillna(priors)

#         # 3. p0_df (data-driven initial guesses)
#         fit_params_df["p0"] = fit_params_df["p0"].fillna(p0_df)

#         if priors is not None:
#             res_df["Error"] = priors.get("Error", np.inf)

#         # if r_vals.empty:
#         #     if prior_fits is not None and col in prior_fits:
#         #         priors = prior_fits[col]
#         #         fit_params_df["p0"] = fit_params_df["p0"].fillna(priors)
#         #     if (
#         #         mean_df is not None
#         #         and col in mean_df.columns
#         #         and fit_params_df[("p0", "b0")].hasnans
#         #     ):
#         #         fit_params_df[("p0", "b0")] = fit_params_df[("p0", "b0")].fillna(mean_df[col])
#         # else:
#         #     if (
#         #         prior_fits is not None
#         #         and col in prior_fits
#         #         and prior_fits["Equations"].loc[col, "Equation"] == eqn
#         #     ):
#         #         # ensure tau0 is not from an exp fit prior_fits["Equations"].loc[col] == eqn
#         #         priors = prior_fits[col]
#         #         fit_params_df["p0"] = fit_params_df["p0"].fillna(priors)
#         #     else:
#         #         fit_params_df[("p0", "tau0")] = fit_params_df[("p0", "tau0")].fillna(r_vals)

#         # tau_vals = 1 / np.arange(1, n_pairs + 1)[::-1]
#         tau_lb_mod = -1
#         tau_ub_mod = 1
#         taus_idx = []
#         if "tau0" in param_names:
#             taus_idx = [f"tau{i}" for i in range(n_pairs)]
#             tau_df: pd.DataFrame = fit_params_df["p0"][taus_idx]  # type: ignore[assignment]

#             t_min_exp = -np.log10(max(float(t_max.max()), 2))
#             t_max_exp = np.log10(max(kwargs.get("tau_p0_max", 1), 1))

#             t_min_exp = min(t_min_exp, np.log10(tau_df.min().min()))
#             t_max_exp = max(t_max_exp, np.log10(tau_df.max().max()))

#             tau_lb_mod = -(t_max_exp - t_min_exp) - abs(lb_mod(col, "tau", override=5))
#             tau_ub_mod = tau_df.mean().max() * ub_mod(col, "tau")

#             # lb_tau = np.minimum(
#             #     tau_df * 10**tau_lb_mod, tau_df * [lb_mod(col, n) for n in taus_idx]
#             # )
#             lb_tau = (tau_df * 10**tau_lb_mod).combine(
#                 tau_df * [lb_mod(col, n) for n in taus_idx], func=np.minimum
#             )
#             # lb_tau = np.where(abs(lb_tau) < 1e-32, np.sign(lb_tau) * 1e-32, lb_tau)
#             lb_tau = lb_tau.where(abs(lb_tau) >= 1e-32, np.sign(lb_tau) * 1e-32)
#             # ub_tau = np.maximum(tau_df / tau_ub_mod, tau_df * [ub_mod(col, n) for n in taus_idx])
#             ub_tau = (tau_df / tau_ub_mod).combine(tau_df * [ub_mod(col, n) for n in taus_idx], func=np.maximum)
#             ub_limit = np.maximum(10, [ub_mod(col, n) for n in taus_idx])
#             # ub_tau = np.where(ub_tau > ub_limit, tau_df + ub_limit, ub_tau)
#             ub_tau = ub_tau.where(ub_tau <= ub_limit, tau_df + ub_limit)
#             # fit_params_df["lb"][taus_idx] = lb_tau
#             # fit_params_df[[("lb", i) for i in taus_idx]] = lb_tau
#             fit_params_df[[("lb", i) for i in taus_idx]] = fit_params_df[[("lb", i) for i in taus_idx]].fillna(lb_tau)
#             # fit_params_df["ub"][taus_idx] = ub_tau
#             # fit_params_df[[("ub", i) for i in taus_idx]] = ub_tau
#             fit_params_df[[("ub", i) for i in taus_idx]] = fit_params_df[[("ub", i) for i in taus_idx]].fillna(ub_tau)

#             if "b0" in param_names:
#                 # b0_df = fit_params_df["p0"][["b0"]]
#                 # a_sum = fit_params_df["p0"][[f"a{i}" for i in range(n_pairs)]].sum(axis=1) # type: ignore[assignment]
#                 k = lb_mod(col, "b0", override=0.05)
#                 k  = k if abs(k) < abs(1/k) else 1/k
#                 a_sum = fit_params_df[[("p0", f"a{i}") for i in range(n_pairs)]].sum(axis=1)


#                 # restrictive_bound = fit_params_df[("p0","b0")] + a_sum * min(k, 1 / k)

#                 lb_mask = fit_params_df[("lb", "b0")].isna() & (a_sum < 0)
#                 ub_mask = fit_params_df[("ub", "b0")].isna() & (a_sum > 0)

#                 fit_params_df[("lb", "b0")] = fit_params_df[("lb", "b0")].where(
#                     ~lb_mask, fit_params_df[("p0","b0")] + a_sum * k
#                 )
#                 fit_params_df[("ub", "b0")] = fit_params_df[("ub", "b0")].where(
#                     ~ub_mask, fit_params_df[("p0","b0")] + a_sum * k
#                 )

#                 # form1_mask = (b0_df < 0)  & (restrictive_bound < b0_df)    # Negative upward
#                 # form2_mask = (b0_df > 0)  & (restrictive_bound > b0_df)    # Positive downward
#                 # form3_mask = (b0_df > 0)  & (restrictive_bound < b0_df)    # Positive upward

#                 # # Create Series to assign
#                 # # restrictive_bound_series = pd.Series(restrictive_bound, index=df.index)

#                 # # Apply to lower and upper bounds using .where() and .fillna()
#                 # fit_params_df[("lb", "b0")] = fit_params_df[("lb", "b0")].where(~(form1_mask | form3_mask),
#                 #                                                         restrictive_bound).fillna(fit_params_df[("lb", "b0")])

#                 # fit_params_df[("ub", "b0")] = fit_params_df[("ub", "b0")].where(~form2_mask,
#                 #                                                         restrictive_bound).fillna(fit_params_df[("ub", "b0")])

#         # --- Vectorized bounds for beta ---
#         if "beta0" in param_names:
#             fit_params_df[("lb", "beta0")] = fit_params_df[("lb", "beta0")].fillna(1e-32)
#             fit_params_df[("ub", "beta0")] = fit_params_df[("ub", "beta0")].fillna(1.0)

#         # --- Vectorized bounds for all other parameters ---
#         # other_idx = [p for p in param_names if p not in taus_idx + ["beta0"]]
#         # if other_idx:
#         lb_consts, ub_consts = zip(*[(lb_mod(col, p), ub_mod(col, p)) for p in param_names])

#         lb_other = (fit_params_df["p0"] / lb_consts).combine(
#             fit_params_df["p0"] * lb_consts, func=np.minimum
#         )
#         ub_other = (fit_params_df["p0"] / ub_consts).combine(
#             fit_params_df["p0"] * ub_consts, func=np.maximum
#         )
#         fit_params_df["lb"] = fit_params_df["lb"].fillna(lb_other)
#         fit_params_df["ub"] = fit_params_df["ub"].fillna(ub_other)

#             # p0_other = fit_params_df["p0"][other_idx]
#             # lb_other = np.minimum(
#             #     p0_other / np.array([lb_mod(col, p) for p in other_idx]),
#             #     p0_other * np.array([lb_mod(col, p) for p in other_idx]),
#             # )
#             # # fit_params_df["lb"][other_idx] = lb_other
#             # fit_params_df[[("lb", o) for o in other_idx]] = lb_other

#             # ub_other = np.maximum(
#             #     p0_other / np.array([ub_mod(col, p) for p in other_idx]),
#             #     p0_other * np.array([ub_mod(col, p) for p in other_idx]),
#             # )
#             # # fit_params_df["ub"][other_idx] = ub_other
#             # fit_params_df[[("ub", o) for o in other_idx]] = ub_other

#             # if not np.isnan(tau_min):
#             #     t_min_exp = min(t_min_exp, np.log10(tau_min))
#             # if not np.isnan(tau_max):
#             #     t_max_exp = max(t_max_exp, np.log10(tau_max))

#             # if r_vals.empty:
#             #     tau_vals = np.logspace(t_min_exp, t_max_exp, n_pairs)
#             # else:
#             #     tau_vals = np.array([r_vals.mean()])

#             # tau_lb_mod = -(t_max_exp - t_min_exp) - abs(lb_mod(col, "tau", override=5))
#             # tau_ub_mod = np.max(tau_vals) * ub_mod(col, "tau")

#         for n, param_name in enumerate(param_names):
#             # pair_num = n // 2
#             # --- Set scale ---
#             if isinstance(scale_val, (list, tuple)):
#                 fit_params_df[("scale", param_name)] = (
#                     scale_val[n] if n < len(scale_val) else "linear"
#                 )
#             elif isinstance(scale_val, str):
#                 fit_params_df[("scale", param_name)] = scale_val

#             # # --- Fill missing p0 values ---
#             # mask = fit_params_df[("p0", param_name)].isna()
#             # if mask.any():
#             #     if param_name == "b0":
#             #         fit_params_df.loc[mask, ("p0", param_name)] = means[mask]
#             #     elif param_name == f"a{pair_num}":
#             #         fit_params_df.loc[mask, ("p0", param_name)] = bases[mask] / (pair_num + 1)
#             #     elif param_name == f"tau{pair_num}":
#             #         fit_params_df.loc[mask, ("p0", param_name)] = tau_vals[pair_num]
#             #     else:
#             #         fit_params_df.loc[mask, ("p0", param_name)] = 1.0

#             # --- Bounds from p0 ---
#             # lb_mask = fit_params_df[("lb", param_name)].isna()
#             # ub_mask = fit_params_df[("ub", param_name)].isna()
#             p0_vals = fit_params_df[("p0", param_name)].astype(float)
#             # if param_name.startswith("tau"):
#             #     if lb_mask.any():
#             #         lb_calc = np.minimum(
#             #             p0_vals * 10**tau_lb_mod, p0_vals * lb_mod(col, param_name)
#             #         )
#             #         lb_calc = np.where(abs(lb_calc) < 1e-32, np.sign(lb_calc) * 1e-32, lb_calc)
#             #         fit_params_df.loc[lb_mask, ("lb", param_name)] = lb_calc[lb_mask]
#             #     if ub_mask.any():
#             #         ub_calc = np.maximum(p0_vals / tau_ub_mod, p0_vals * ub_mod(col, param_name))
#             #         ub_limit = max(10, ub_mod(col, param_name))
#             #         ub_calc = np.where(ub_calc > ub_limit, p0_vals + ub_limit, ub_calc)
#             #         fit_params_df.loc[ub_mask, ("ub", param_name)] = np.max(ub_calc)
#             # elif param_name.startswith("beta"):
#             #     if lb_mask.any():
#             #         # Lower bound just above zero
#             #         fit_params_df.loc[lb_mask, ("lb", param_name)] = 1e-32
#             #     if ub_mask.any():
#             #         # Upper bound at 1.0
#             #         fit_params_df.loc[ub_mask, ("ub", param_name)] = 1.0
#             # else:
#             #     if lb_mask.any():
#             #         lb_calc = np.minimum(
#             #             p0_vals / lb_mod(col, param_name), p0_vals * lb_mod(col, param_name)
#             #         )
#             #         fit_params_df.loc[lb_mask, ("lb", param_name)] = lb_calc[lb_mask]
#             #     if ub_mask.any():
#             #         ub_calc = np.maximum(
#             #             p0_vals / ub_mod(col, param_name), p0_vals * ub_mod(col, param_name)
#             #         )
#             #         fit_params_df.loc[ub_mask, ("ub", param_name)] = ub_calc[ub_mask]

#             res_df[param_name] = p0_vals
#             res_df[f"{param_name}_std"] = 1.0
#             if priors is not None:
#                 res_df[f"{param_name}_std"] = priors.get(f"{param_name}_std", 1.0)

#         fit_params_dict[col] = fit_params_df
#         fit_param_val_dict[col] = res_df

#     # ...existing code...
#     fit_params_dict["Equations"] = pd.DataFrame.from_dict(
#         {col: {"Equation": equations[col], "Eqn Name": equation_names[col]} for col in equations},
#         orient="index",
#         columns=["Equation", "Eqn Name"],
#     )
#     # ...existing code...
#     fit_param_val_dict["Equations"] = fit_params_dict["Equations"].copy()

#     return fit_map, fit_params_dict, fit_param_val_dict


# # %% Equation Functions
# def linear_func(t: np.ndarray, a: float, b: float) -> np.ndarray:
#     """Linear function for fitting."""
#     return a * t + b


# def pow_func(t: np.ndarray, A: float, tau: float, B: float = 0.0) -> np.ndarray:
#     """Power law decay/growth for fitting."""
#     t[t == 0] = 1e-32  # Avoid division by zero
#     return A * (t) ** (-tau) + B


# def exp_func(t: np.ndarray, A: float, tau: float, B: float) -> np.ndarray:
#     """Exponential decay/growth for fitting."""
#     try:
#         res = A * np.exp(-t / tau) + B
#     except FloatingPointError:
#         # Handle potential overflow in exp calculation
#         res = A * np.exp(-t / (tau + 1e-32)) + B
#     return res


# def double_exp_func(
#     t: np.ndarray, A1: float, tau1: float, A2: float, tau2: float, B: float
# ) -> np.ndarray:
#     """Double exponential decay/growth for fitting."""
#     try:
#         res = A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + B
#     except FloatingPointError:
#         # Handle potential overflow in exp calculation
#         res = A1 * np.exp(-t / (tau1 + 1e-32)) + A2 * np.exp(-t / (tau2 + 1e-32)) + B
#     return res


# def stretch_exp_func(t: np.ndarray, A: float, tau: float, beta: float, B: float) -> np.ndarray:
#     """Stretched exponential decay/growth for fitting."""
#     try:
#         res = A * np.exp(-((t / tau) ** beta)) + B
#     except FloatingPointError:
#         # Handle potential overflow in exp calculation
#         res = A * np.exp(-((t / (tau + 1e-32)) ** beta)) + B
#     return res


# def n_exp_func(t: np.ndarray, *params: float) -> np.ndarray:
#     """
#     N-exponential decay/growth for fitting.

#     Parameters
#     ----------
#     t : np.ndarray
#         Time array.
#     *params : float
#         Sequence of (A, tau) pairs, optionally followed by B if odd number of params.
#         If even, B is set to 0.

#     Returns
#     -------
#     np.ndarray
#         Evaluated sum of exponentials plus offset.
#     """
#     result = np.zeros_like(t)
#     for i in range(len(params) // 2):
#         try:
#             result += params[2 * i] * np.exp(-t / params[2 * i + 1])
#         except FloatingPointError:
#             result += params[2 * i] * np.exp(-t / (params[2 * i + 1] + 1e-32))
#     B = params[-1] if len(params) % 2 else 0.0
#     return result + B


# # %% Initial Guess Functions
# def initial_guess_poly_func(
#     t: np.ndarray | pd.DataFrame, y: np.ndarray | None = None, deg: int = 1
# ) -> list[float]:
#     """
#     Estimate initial parameters for linear fit: f(x) = a * x + b

#     Args:
#         x (np.ndarray): Independent variable (e.g., time).
#         y (np.ndarray): Dependent variable (e.g., signal).

#     Returns:
#         tuple: Initial guess for (a, b)
#     """
#     if isinstance(t, pd.DataFrame):
#         y = t.iloc[:, 1].to_numpy(copy=True) if y is None else y
#         t = t.iloc[:, 0].to_numpy(copy=True)
#     if y is None:
#         raise ValueError("y must be provided if t is a DataFrame or Series.")

#     return np.polyfit(t, y, deg).tolist()


# def initial_guess_pow_func(
#     t: np.ndarray | pd.DataFrame, y: np.ndarray | None = None
# ) -> list[float]:
#     if isinstance(t, pd.DataFrame):
#         y = t.iloc[:, 1].to_numpy(copy=True) if y is None else y
#         t = t.iloc[:, 0].to_numpy(copy=True)
#     if y is None:
#         raise ValueError("y must be provided if t is a DataFrame or Series.")
#     # Estimate baseline offset b
#     b_idx = np.argmin(abs(y))
#     if b_idx < len(y) * 3 / 4:
#         # Ensure b_idx is in the last quarter of y
#         b_idx = -1
#     b0 = y[b_idx]
#     mask = (t != 0) & (y != 0) & np.isfinite(t) & np.isfinite(y)
#     slope, intercept = np.polyfit(np.log(abs(t[mask])), np.log(abs(y[mask])), 1)
#     return [np.sign(b0) * np.exp(intercept), abs(slope), b0]


# def initial_guess_exp_func(
#     t: np.ndarray | pd.DataFrame, y: np.ndarray | None = None, n_exp: int = 1
# ) -> list[float]:
#     """
#     Estimate initial guesses for multi-exponential decay model parameters:
#     f(t) = a1·exp(-t/τ1) + a2·exp(-t/τ2) + ... + an·exp(-t/τn) + b

#     Args:
#         t (np.ndarray): Independent variable (e.g., time).
#         y (np.ndarray): Dependent variable (e.g., signal).
#         n_exp (int): Number of exponential terms.

#     Returns:
#         dict: Estimated a_i and tau_i parameters
#     """
#     assert n_exp >= 1, "Number of exponentials must be >= 1"
#     if isinstance(t, pd.DataFrame):
#         y = t.iloc[:, 1].to_numpy(copy=True) if y is None else y
#         t = t.iloc[:, 0].to_numpy(copy=True)
#     if y is None:
#         raise ValueError("y must be provided if t is a DataFrame or Series.")

#     # Estimate baseline offset b
#     b_idx = np.max([np.argmin(abs(y)), np.argmax(abs(y))])
#     if b_idx < len(y) * 3 / 4:
#         # Ensure b_idx is in the last quarter of y
#         b_idx = -1
#     b0 = y[b_idx]

#     # Adjust for baseline
#     y_adj = abs(y - b0 + 1e-32)
#     # sign also indicates direction as mean should be towards the center
#     # sign = int(np.sign(np.mean(y) - b0))

#     # Estimate tau3 from tail slope
#     N = max(2, int(len(t) * 0.05))
#     # sign also indicates direction as mean should be towards the center
#     sign = int(np.sign(np.mean(y[: N * 2]) - np.mean(y[-N * 2 :])))

#     # tail_N = max(5, int(len(t) * 0.5))
#     taus = [float(-1 / np.polyfit(t[-N * 12 :], np.log(y_adj[-N * 12 :]), 1)[0])]
#     taus[0] = max(1e-8, taus[0])
#     # tail_N = max(5, int(len(t) * 0.5))
#     # taus[0] = -1 / np.polyfit(t[-tail_N:], np.log(y_adj[-tail_N:]), 1)[0]

#     if n_exp == 1:
#         # a_val = np.mean(y_adj * np.exp(t / taus[0])) * sign
#         a_val = np.max(y_adj) if t[-1] / taus[0] > 500 else np.mean(y_adj * np.exp(t / taus[0]))
#         return [a_val * sign, taus[0], b0]

#     dt = float(np.median(np.diff(t)))
#     early_N = N * 6
#     slope1 = -np.gradient(y_adj[: N * 2], t[: N * 2]).mean()
#     taus.append(abs(y_adj[0] / slope1) if slope1 != 0 else dt)
#     if taus[0] != 1e-8:
#         tau_0 = taus[1]
#         while tau_0 >= taus[0] and early_N > 3:
#             early_N = max(3, early_N - N)
#             slope1 = -np.gradient(y_adj[:early_N], t[:early_N]).mean()
#             tau_0 = abs(y_adj[0] / slope1) if slope1 != 0 else dt
#         if tau_0 < taus[1]:
#             taus[1] = tau_0
#     taus[1] = max(1e-16, taus[1])

#     taus = sorted(taus)
#     taus = np.logspace(np.log10(taus[0]), np.log10(taus[1]), n_exp).tolist()

#     # Construct exponential basis matrix
#     E = np.vstack([np.exp(-t / tau) for tau in taus]).T

#     # Solve for amplitudes a_i via least squares
#     a_vals, *_ = np.linalg.lstsq(E, y_adj, rcond=None)
#     bad_a_count = sum(a_vals <= 0)
#     if bad_a_count > 0:
#         # Construct a data relavent "minimal" value
#         a_min = 10 ** (int(np.floor(np.log10(abs(np.ptp(y_adj))))) - 2)
#         if bad_a_count == len(a_vals):
#             a_vals[:-1] = a_min
#             a_vals[-1] = abs(np.mean(y_adj * np.exp(t / taus[-1])) - (bad_a_count - 1) * a_min)
#         else:
#             addative = (-sum(a_vals[a_vals <= 0]) - a_min * bad_a_count) / sum(a_vals > 0)
#             a_vals = np.where(a_vals <= 0, a_min, a_vals + addative)

#     a_vals = a_vals * sign
#     return sort_exp_params([p for i in range(n_exp) for p in (a_vals[i], taus[i])] + [b0])[0]
#     # return sort_exp_params([p for i in range(n_exp) for p in (a_vals[i] * sign, taus[i])] + [b0])[
#     #     0
#     # ]


# def initial_guess_str_exp_func(
#     t: np.ndarray | pd.DataFrame, y: np.ndarray | None = None, beta_min: float = 0.3
# ) -> list[float]:
#     """
#     Estimate initial parameters for stretched exponential:
#     f(t) = a * exp[-(t / tau)^beta] + b

#     Args:
#         t (np.ndarray): Independent variable.
#         y (np.ndarray): Dependent variable.
#         initial_guess_exp_func (callable): Function that returns [a1, tau1, a2, tau2, b] for n_exp=2.
#         beta_min (float): Minimum allowable beta value (defaults to 0.3).

#     Returns:
#         list: [a, tau, beta, b] initial parameter guess
#     """
#     if isinstance(t, pd.DataFrame):
#         y = t.iloc[:, 1].to_numpy(copy=True) if y is None else y
#         t = t.iloc[:, 0].to_numpy(copy=True)
#     if y is None:
#         raise ValueError("y must be provided if t is a DataFrame or Series.")

#     # Get estimates from 2-exponential model
#     a1, tau1, a2, tau2, b0 = initial_guess_exp_func(t, y, 2)

#     # Beta: based on tau spread
#     beta = np.log(2) / np.log(max(tau2 / tau1, tau1 / tau2))

#     return [
#         (abs(a1) + abs(a2)) * np.sign(b0),
#         np.sqrt(tau1 * tau2),
#         min(1.0, max(beta_min, beta)),
#         b0,
#     ]

# # Define how to process a group of data
# def _process_group(data, sorted_params, use_geo=True) -> pd.DataFrame:
#     # For each adjacent pair of parameters
#     # base_params = sorted(params, key=lambda p: data[('p0', p)].median())
#     # d0 = data.copy(deep=True)
#     for i in range(len(sorted_params) - 1):
#         min_param = sorted_params[i]
#         max_param = sorted_params[i + 1]

#         # Get the maximum upper bound for the lower parameter
#         min_abs = data.loc[:, (["lb", "ub"], min_param)].abs()
#         max_abs = data.loc[:, (["lb", "ub"], max_param)].abs()

#         # Check if there's an overlap
#         if min_abs.max(axis=1).max() > max_abs.min(axis=1).min():
#             # # Save original bound deltas before modification
#             min_signs = np.sign(data[("p0", min_param)].to_numpy(copy=True))
#             max_signs = np.sign(data[("p0", max_param)].to_numpy(copy=True))

#             min_mask = min_abs.eq(min_abs.max(axis=1), axis=0)
#             max_mask = max_abs.eq(max_abs.min(axis=1), axis=0)

#             # if use_geo:
#             #     min_delta = np.min(np.log10(np.clip(min_abs, 1e-32, np.inf)).diff(axis=1).abs())
#             #     max_delta = np.min(np.log10(np.clip(max_abs, 1e-32, np.inf)).diff(axis=1).abs())

#             # else:
#             #     min_delta = np.min(min_abs.diff(axis=1).abs())
#             #     max_delta = np.min(max_abs.diff(axis=1).abs())

#             if use_geo:
#                 min_delta = 10 ** (
#                     np.log10(np.clip(min_abs, 1e-32, np.inf))
#                     .diff(axis=1)
#                     .abs()
#                     .mul(-min_signs, axis=0)
#                 ).bfill(axis=1)
#                 max_delta = 10 ** (
#                     np.log10(np.clip(max_abs, 1e-32, np.inf))
#                     .diff(axis=1)
#                     .abs()
#                     .mul(-max_signs, axis=0)
#                 ).bfill(axis=1)
#                 # min_delta = 10 ** (-min_signs * (
#                 #     np.log10(np.clip(min_abs, 1e-32, np.inf)).diff(axis=1).abs()
#                 # ).T).T.bfill(axis=1)
#                 # max_delta = 10 ** (-max_signs * (
#                 #     np.log10(np.clip(max_abs, 1e-32, np.inf)).diff(axis=1).abs()
#                 # ).T).T.bfill(axis=1)
#                 # min_delta = 10 ** (np.log10(np.clip(min_abs, 1e-32, np.inf)).diff(axis=1).abs()).bfill(
#                 #     axis=1
#                 # )
#                 # max_delta = 10 ** (np.log10(np.clip(max_abs, 1e-32, np.inf)).diff(axis=1).abs()).bfill(
#                 #     axis=1
#                 # )
#                 min_delta["ub"] = 1 / min_delta["ub"]
#                 max_delta["ub"] = 1 / max_delta["ub"]
#             else:
#                 min_delta = min_abs.diff(axis=1).abs().bfill(axis=1)
#                 max_delta = max_abs.diff(axis=1).abs().bfill(axis=1)
#                 min_delta["ub"] = -min_delta["ub"]
#                 max_delta["ub"] = -max_delta["ub"]
#                 # min_delta[~min_mask] = 0
#                 # max_delta[~max_mask] = 0
#             min_delta[min_mask] = np.nan
#             max_delta[max_mask] = np.nan

#             min_p0_max = data[("p0", min_param)].abs().max()
#             max_p0_min = data[("p0", max_param)].abs().min()

#             if min_p0_max <= max_p0_min:
#                 # logic for when the p0 values do not overlap
#                 if use_geo:
#                     mid_point = np.sqrt(min_p0_max * max_p0_min)
#                 else:
#                     mid_point = (min_p0_max + max_p0_min) / 2
#             else:
#                 # mid_point = (
#                 #     data.loc[:, ("p0", [min_param, max_param])]
#                 #     .abs()
#                 #     .mean(axis=1)
#                 #     .median()
#                 # )
#                 if use_geo:
#                     mid_point = np.sqrt(
#                         min_abs.max(axis=1).median() * max_abs.min(axis=1).median()
#                     )
#                 else:
#                     mid_point = (min_abs.max(axis=1).median() + max_abs.min(axis=1).median()) / 2
#             # Apply new bounds to ALL rows in the group
#             data[min_mask] = list(mid_point * min_signs)
#             data[max_mask] = list(mid_point * max_signs)

#             # Check for invalid bounds
#             shift_low = data[("lb", min_param)] > data[("ub", min_param)]
#             shift_high = data[("lb", max_param)] > data[("ub", max_param)]

#             if any(shift_low):
#                 if use_geo:
#                     data.update(
#                         min_delta.mul(min_signs * mid_point, axis=0).loc[shift_low, ["lb", "ub"]]
#                     )
#                     # data.update((data * min_delta).loc[shift_low, ["lb", "ub"]])
#                     # data.loc[shift_low, ~min_mask] = (data * min_delta).loc[
#                     #     shift_low, ["lb", "ub"]
#                     # ]
#                 else:
#                     data.update(
#                         min_delta.add(min_signs * mid_point, axis=0).loc[shift_low, ["lb", "ub"]]
#                     )
#                     # data.update((data + min_delta).loc[shift_low, ["lb", "ub"]])
#                     # data.loc[shift_low, ~min_mask] = (data + min_delta).loc[
#                     #     shift_low, ["lb", "ub"]
#                     # ]
#             if any(shift_high):
#                 if use_geo:
#                     data.update(
#                         max_delta.mul(max_signs * mid_point, axis=0).loc[shift_high, ["lb", "ub"]]
#                     )
#                     # data.update((data * max_delta).loc[shift_high, ["lb", "ub"]])
#                     # data.loc[shift_high, ~max_mask] = (data * max_delta).loc[
#                     #     shift_high, ["lb", "ub"]
#                     # ]
#                 else:
#                     data.update(
#                         max_delta.add(max_signs * mid_point, axis=0).loc[shift_high, ["lb", "ub"]]
#                     )
#                     # data.update((data + max_delta).loc[shift_high, ["lb", "ub"]])
#                     # data.loc[shift_high, ~max_mask] = (data + max_delta).loc[
#                     #     shift_high, ["lb", "ub"]
#                     # ]

#             if any(data[("lb", min_param)] > data[("ub", min_param)]):
#                 raise ValueError("Lower parameter bounds overlap detected.")

#             if any(data[("lb", max_param)] > data[("ub", max_param)]):
#                 raise ValueError("Higher parameter bounds overlap detected.")

#     return data

# %%
