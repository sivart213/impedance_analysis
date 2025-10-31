import re
import time
import warnings
from typing import Any
from collections import defaultdict

import numpy as np
import pandas as pd

from eis_analysis.dc_fit.extract_tools import (
    DEFAULT_DIR,
    get_job,
    load_dc_data,  # noqa: F401
    save_results,  # noqa: F401
    partial_selection,
)
from eis_analysis.dc_fit.permitivity_calculations import (
    collect_peak_summary_df,
    transform_to_permittivity,
)

np.seterr(invalid="raise")


def get_target_curves(
    curves: dict,
    curve_selectors: list | tuple = (),
) -> dict:
    """
    Helper to select target curves using curve_selectors and partial_selection.

    Parameters
    ----------
    curves : dict
        Dictionary of curves.
    curve_selectors : list, optional
        List of selection argument tuples.
    partial_selection_func : callable, optional
        Function to perform partial selection.

    Returns
    -------
    target_curves : dict
        Dictionary of selected curves.
    """
    if not curve_selectors:
        return curves.copy()
    target_curves = defaultdict(dict)
    for selection in curve_selectors:
        if not selection:
            continue
        if selection[0] in curves:
            curve_keys = [selection[0]]
            sel_args = list(selection[1:])
        else:
            curve_keys = [k for k in curves]
            sel_args = list(selection)
        any_all = "any"
        if sel_args and sel_args[-1] in [any, all, "any", "all"]:
            any_all = sel_args.pop()
        for ck in curve_keys:
            target_curves[ck].update(partial_selection(curves[ck], *sel_args, any_all=any_all))
    return dict(target_curves)


def parse_perm_kwargs(
    kwargs: dict[str, Any],
    curve_key: str,
    k: Any,
) -> dict[str, Any]:
    """
    Parse and update kwargs for transform_to_permittivity, extracting required arguments
    for decay_from_fit from a nested 'params' dictionary in kwargs.

    Parameters
    ----------
    kwargs : dict
        Original keyword arguments.
    curve_key : str
        The key for the current curve group (e.g., "fit 1", "base").
    k : Any
        The subkey/index for the curve within the group.

    Returns
    -------
    dict
        Updated kwargs with 'params_current', 'params_voltage', and 'equation_df' if available.
    """
    updated_kwargs = kwargs.copy()
    params = updated_kwargs.pop("params", None)
    vals_key = updated_kwargs.pop("vals_key")
    if params is not None:
        try:
            subdict = params.get(f"{curve_key} vals", params.get(vals_key, {}))
            updated_kwargs["equation_df"] = subdict["Equations"]
            updated_kwargs["params_current"] = subdict["Current"].loc[k]
            updated_kwargs["params_voltage"] = subdict["Voltage"].loc[k]
        except KeyError:
            return updated_kwargs
    return updated_kwargs


def calculate_permittivity(
    curves: dict,
    *data_keys: str,
    perm_curves: dict | None = None,
    curve_selectors: list | tuple = (),
    suppress_warnings: bool = False,
    **kwargs,
) -> dict:
    """
    Calculate permittivity curves from input curve data.

    Parameters
    ----------
    curves : dict
        Dictionary of curve DataFrames, typically grouped by curve type (e.g., "base", "fit 1").
    *data_keys : str
        Optional. Names of specific curve groups to process (e.g., "cleaned", "fit 1").
        If not provided, all groups in `curves` are processed.
    perm_curves : dict, optional
        Dictionary to update with calculated permittivity curves. If None, a new dictionary is created.
    curve_selectors : list, optional
        List of selection argument tuples for filtering which curves to process.
        Each tuple specifies a curve group and selection criteria.
    **kwargs
        Additional keyword arguments passed to `transform_to_permittivity` and `parse_perm_kwargs`.
        Common keys include:
            - params: dict of fit parameters for each curve group.
            - vals_key: str, key for parameter lookup.
            - decay_mod, remove_mean, padding, dt_function, pre_spline, post_spline, max_f_max, etc.

    Returns
    -------
    perm_curves : dict
        Dictionary of calculated permittivity curves, grouped by curve type and subkey.

    Notes
    -----
    - For each selected curve group, applies `transform_to_permittivity` to each curve.
    - Handles both raw and fit curve groups.
    - Collects and warns about negative real permittivity values encountered during conversion.
    - Updates or creates `perm_curves` in-place.
    - Designed for batch processing of multiple curve groups and flexible selection.
    """
    if perm_curves is None:
        perm_curves = {}

    target_curves = get_target_curves(curves, curve_selectors)

    if not target_curves:
        target_curves = curves.copy()

    if not data_keys:
        datasets = set(target_curves.keys())
    else:
        datasets = set(data_keys)

    for curve_key, curve_dict in target_curves.items():
        percent_list = []
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            if curve_key in {"base", "full", "cleaned", "trimmed"} and curve_key in datasets:
                perm_curves.setdefault(curve_key, {})
                perm_curves[curve_key].update(
                    {
                        k: transform_to_permittivity(
                            v,
                            **parse_perm_kwargs(kwargs, curve_key, k),
                        )
                        for k, v in curve_dict.items()
                    }
                )
            elif "fit" in curve_key and curve_key in datasets:
                perm_curves.setdefault(curve_key, {})
                perm_curves[curve_key].update(
                    {
                        k: transform_to_permittivity(**parse_perm_kwargs(kwargs, curve_key, k))
                        for k in curve_dict
                    }
                )
            # Extract percent_negative from warning messages
            for warn in wlist:
                match = re.search(r"permittivity: ([0-9.]+)% < 0", str(warn.message))
                if match:
                    percent_list.append(float(match.group(1)))
        if percent_list and not suppress_warnings:
            avg_percent = np.mean(percent_list)
            n_conversions = len(percent_list) / len(curve_dict) * 100
            warnings.warn(
                f"Permittivity calculation for '{curve_key}' produced negative real permittivity in "
                f"{n_conversions:.2f}% of conversions with an average {avg_percent:.2f}% of points affected.",
                RuntimeWarning,
                stacklevel=2,
            )
    return perm_curves


def collect_permittivity_peaks(
    perm_curves: dict,
    perm_peaks: dict[str, pd.DataFrame] | None = None,
    *data_keys: str,
    curve_selectors: list | tuple = (),
    verbose: bool = True,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """
    Collect peak summary data from permittivity curves.

    Parameters
    ----------
    perm_curves : dict
        Dictionary of permittivity curves.

    Returns
    -------
    perm_peaks : dict
        Dictionary of permittivity peak summaries.
    """
    if perm_peaks is None:
        perm_peaks = {}

    if not data_keys:
        datasets = set(k for k in perm_curves.keys() if k in {"cleaned", "trimmed"} or "fit" in k)
    else:
        datasets = set(k for k in data_keys if k in {"cleaned", "trimmed"} or "fit" in k)

    target_curves = get_target_curves(perm_curves, curve_selectors)

    for curve_key in target_curves:
        if curve_key in datasets:
            if verbose:
                print(f"    Collecting peaks for {curve_key}")

            res = {}
            res["_imag"], res["_imag_all"] = collect_peak_summary_df(
                target_curves[curve_key],
                **kwargs,
            )
            res["_loss"], res["_loss_all"] = collect_peak_summary_df(
                target_curves[curve_key],
                column="loss tangent",
                **kwargs,
            )
            # Update only the relevant entries
            for suffix in ["_imag", "_imag_all", "_loss", "_loss_all"]:
                key = f"{curve_key}{suffix}"
                if key in perm_peaks and "all" in suffix:
                    mask = (
                        perm_peaks[key]
                        .index.droplevel(-1)
                        .isin(res[suffix].index.droplevel(-1).unique())
                    )
                    all_df = pd.concat([perm_peaks[key][~mask], res[suffix]])
                    all_df.attrs.update(perm_peaks[key].attrs)
                    all_df.attrs.update(res[suffix].attrs)
                    perm_peaks[key] = all_df.sort_index()
                elif key in perm_peaks:
                    perm_peaks[key].update(res[suffix])
                    perm_peaks[key].attrs.update(res[suffix].attrs)
                else:
                    perm_peaks[key] = res[suffix]

    return perm_peaks


def run_perm_calcs(
    curves: dict,
    perm_jobs: list[dict] | tuple[dict, ...] = (),
    perm_curves: dict | None = None,
    perm_peaks: dict | None = None,
    params: dict | None = None,
    suppress_warnings: bool = False,
    verbose: bool = True,
) -> tuple[dict, dict]:
    """
    Run permittivity calculations based on job configurations.

    Parameters
    ----------
    curves : dict
        Dictionary of curve DataFrames.
    params : dict, optional
        Dictionary of fit parameters needed for permittivity calculations.
    perm_curves : dict, optional
        Dictionary to store calculated permittivity curves. If None, a new one is created.
    perm_peaks : dict, optional
        Dictionary to store calculated permittivity peaks. If None, a new one is created.
    perm_jobs : list[dict], optional
        List of job configurations. If None, no jobs will be executed.

    Returns
    -------
    tuple[dict, dict]
        Tuple of (perm_curves, perm_peaks) dictionaries.
    """
    if perm_curves is None:
        perm_curves = {}
    if perm_peaks is None:
        perm_peaks = {}

    if verbose:
        print(f"{time.ctime()}: Running permittivity calculations...")

    for idx, job in enumerate(perm_jobs):
        if verbose:
            if "job_name" in job:
                print(f"{time.ctime()}: Job {idx} of {len(perm_jobs)-1}; {job['job_name']}")
            else:
                print(f"{time.ctime()}: Job {idx} of {len(perm_jobs)-1}")

        # Extract job settings
        data_keys = job.get("data_keys", [])

        curve_selectors = job.get("curve_selectors", ())

        # Run permittivity calculations
        if data_keys:
            perm_curves |= calculate_permittivity(
                curves,
                *data_keys,
                perm_curves=perm_curves,
                curve_selectors=curve_selectors,
                params=params,
                suppress_warnings=suppress_warnings,
                **job.get("calc_profile", PROFILES["calc_default"]),
            )

            # Collect permittivity peaks
            perm_peaks |= collect_permittivity_peaks(
                perm_curves,
                perm_peaks,
                *data_keys,
                curve_selectors=curve_selectors,
                verbose=verbose,
                **job.get("peak_profile", PROFILES["peak_default"]),
            )

    if verbose:
        print(f"{time.ctime()}: Calculations complete.")

    return perm_curves, perm_peaks


def get_perm_job(job_key: str, key_num: int = 1, **kwargs) -> dict:
    """
    Returns a copy of the job dict for permittivity calculations.

    Parameters
    ----------
    job_key : str
        The key for the job in all_jobs.
    **kwargs : dict
        Additional keyword arguments to override job settings.

    Returns
    -------
    dict
        The job dict with all settings resolved.
    """
    kwargs.setdefault("job_source", all_jobs)
    kwargs.setdefault("job_profiles", PROFILES)
    kwargs.setdefault("job_names", ["calc", "peak"])

    job = get_job(job_key, **kwargs)

    job.setdefault("data_keys", [])
    if isinstance(job["data_keys"], str):
        job["data_keys"] = [job["data_keys"]]

    n_key = job.pop("vals_key", job["calc_profile"].get("vals_key", key_num))
    job["calc_profile"]["vals_key"] = n_key if isinstance(n_key, str) else f"fit {n_key} vals"
    job["job_name"] = job.get("job_name", job_key)

    return job


PROFILES = {
    # Base calculation profiles
    "calc_default": {
        "decay_mod": None,
        "remove_mean": "b0",
        "padding": False,
        "dt_function": np.max,
        "pre_spline": "pchip",
        "post_spline": "pchip",
        "max_f_max": 1,
    },
    "locked_freq_range": {
        "min_f_max": 1,
        "max_f_min": 5e-4,
        "min_f_min": 1e-4,
    },
    "high_freq_range": {
        "max_f_max": 5,
        "max_f_min": 7.5e-2,
        "min_f_min": 5e-2,
    },
    "mid_freq_range": {
        "max_f_max": 1,
        "max_f_min": 8e-2,
        "min_f_min": 4e-2,
    },
    "low_freq_range": {
        "max_f_max": 1,
        "max_f_min": 5e-2,
        "min_f_min": 1e-2,
    },
    "low_freq_max": {
        "max_f_max": 0.5,
    },
    "smooth_v1": {
        "decay_mod": {"window_length": 0.015, "polyorder": 3},
    },
    # Peak collection profiles
    "peak_default": {
        "min_peaks": 3,
        "fit_step": 5,
        "slope_min": 0.00,
        "slope_max": 0.5,
        "fit_mask_mode": ("weight", "high_temp"),
        "fit_mode_use": "sequential",
        "fit_select_method": "median, min_std",
        "normalize_weight_mode": ("temp", "condition"),
    },
    "2_peak": {
        "min_peaks": 2,
    },
    "5_peak": {
        "min_peaks": 5,
    },
    "temp_norm": {
        "normalize_weight_mode": ("temp",),
    },
    "max_std": {
        "fit_select_method": "max, min_std",
    },
}

# Permittivity calculation jobs
all_jobs = {
    "init_cleaned": {
        "calc_profile": ["calc_default"],
        "peak_profile": ["peak_default"],
        "data_keys": ["cleaned"],
    },
    "init_trimmed": {
        "calc_profile": ["calc_default"],
        "peak_profile": ["peak_default"],
        "data_keys": ["trimmed"],
    },
    "init_fits": {
        "calc_profile": ["calc_default", "locked_freq_range"],
        "peak_profile": ["peak_default", "2_peak", "temp_norm"],
        "data_keys": ["fit 1", "fit 2", "fit 3", "fit 4", "fit 5"],
    },
    "cln2_pre": {
        "calc_profile": ["calc_default", "mid_freq_range"],  # , "smooth_v1"],
        "peak_profile": ["peak_default", "5_peak", "temp_norm"],
        "data_keys": ["cleaned"],
        "curve_selectors": [("cln2", "pre", "all")],
    },
    "200_pre": {
        "calc_profile": ["calc_default", "locked_freq_range"],  # ,"low_freq_max"],
        "peak_profile": ["peak_default", "5_peak", "temp_norm"],
        "data_keys": ["cleaned"],
        "curve_selectors": [("200", "pre", "all")],
    },
    "301_dh": {
        "calc_profile": ["calc_default", "high_freq_range"],
        "peak_profile": ["peak_default", "max_std", "temp_norm"],
        "data_keys": ["cleaned"],
        "curve_selectors": [("301", "dh", "all")],
        "calc_kwargs": {
            "min_f_min": 6e-2,
        },
    },
    "301_dry": {
        "calc_profile": ["calc_default", "high_freq_range"],
        "peak_profile": ["peak_default", "max_std", "temp_norm"],
        "data_keys": ["cleaned"],
        "curve_selectors": [("301", "dry", "all")],
        "calc_kwargs": {
            "max_f_min": 6e-2,
            "min_f_min": 3e-2,
        },
    },
}

perm_jobs = [
    # Primary jobs
    get_perm_job("init_trimmed"),
    get_perm_job("init_cleaned"),
    get_perm_job("init_fits"),
    # Targeted jobs (run on cleaned)
    get_perm_job("cln2_pre"),
    get_perm_job("200_pre"),
    get_perm_job("301_dh"),
    get_perm_job("301_dry"),
]

# %%
if __name__ == "__main__":
    try:
        loaded_data = globals()["loaded_data"]
    except Exception:
        loaded_data = load_dc_data(DEFAULT_DIR, "curves", "params", "fits")

    # Calculate permittivity curves from loaded curves
    perm_curves = loaded_data.get("perms", {})
    perm_peaks = loaded_data.get("perm_peaks", {})
    params = loaded_data.get("params")
    curves = loaded_data.get("curves", {})

    # %% run permittivity inits
    perm_curves, perm_peaks = run_perm_calcs(
        curves=curves,
        perm_jobs=perm_jobs,
        perm_curves=perm_curves,
        perm_peaks=perm_peaks,
        params=params,
    )

    # %% Save results
    save_results(
        DEFAULT_DIR,
        perm_peaks=perm_peaks,
        attrs=True,
    )
    # # %% Save results
    # save_results(
    #     DEFAULT_DIR,
    #     # "cleaned",
    #     # "fit 1",
    #     perm_curves=perm_curves,
    #     attrs=True,
    # )


# perm_kwargs = {
#     # "params": params,
#     "vals_key": "fit 1 vals",
#     "decay_mod": None,
#     "remove_mean": "b0",
#     "padding": False,
#     "dt_function": np.max,
#     "pre_spline": "pchip",
#     "post_spline": "pchip",
#     "max_f_max": 1,
# }

# peak_kwargs = {
#     "min_peaks": 3,
#     "fit_step": 5,
#     "slope_min": 0.00,
#     "slope_max": 0.5,
#     "fit_mask_mode": ("weight", "high_temp"),
#     "fit_mode_use": "sequential",
#     "fit_select_method": "median, min_std",
#     "normalize_weight_mode": ("temp", "condition"),
# }
# perm_kwargs["params"] = params

# perm_curves = calculate_permittivity(
#     curves,
#     "cleaned",
#     perm_curves=perm_curves,
#     **perm_kwargs,
# )

# perm_peaks = collect_permittivity_peaks(
#     perm_curves,
#     perm_peaks,
#     "cleaned",
#     **peak_kwargs,
# )

# perm_curves = calculate_permittivity(
#     curves,
#     "fit 1",
#     "fit 2",
#     "fit 3",
#     "fit 4",
#     "fit 5",
#     perm_curves=perm_curves,
#     **{
#         **perm_kwargs,
#         "max_f_max": 1,
#         "min_f_max": 1,
#         "max_f_min": 5e-4,
#         "min_f_min": 1e-4,
#     },
# )

# perm_peaks = collect_permittivity_peaks(
#     perm_curves,
#     perm_peaks,
#     "fit 1",
#     "fit 2",
#     "fit 3",
#     "fit 4",
#     "fit 5",
#     **{
#         **peak_kwargs,
#         "min_peaks": 2,
#         # "fit_mode_use": "parallel",
#         "normalize_weight_mode": "temp",
#     },
# )

# # %% run permittivity analysis
# # --- Calculate permittivity groups ---
# # {"window_length": 0.015, "polyorder": 5}, # {"mode": "gradient"},
# perm_curves = calculate_permittivity(
#     curves,
#     "cleaned",
#     perm_curves=perm_curves,
#     curve_selectors=[("cln2", "pre", "all")],
#     **{
#         **perm_kwargs,
#         "decay_mod": {"window_length": 0.015, "polyorder": 3},
#         "max_f_max": 5,
#         "max_f_min": 7.5e-2,
#         "min_f_min": 5e-2,
#     },
# )
# perm_peaks = collect_permittivity_peaks(
#     perm_curves,
#     perm_peaks,
#     "cleaned",
#     curve_selectors=[("cln2", "pre", "all")],
#     **{
#         **peak_kwargs,
#         "min_peaks": 5,
#         "fit_mode_use": "sequential",
#     },
# )

# perm_curves = calculate_permittivity(
#     curves,
#     "cleaned",
#     perm_curves=perm_curves,
#     curve_selectors=[("200", "pre", "all")],
#     **{
#         **perm_kwargs,
#         # "decay_mod": {"window_length": 0.025, "polyorder": 2},
#         "max_f_max": 0.5,
#     },
# )
# perm_peaks = collect_permittivity_peaks(
#     perm_curves,
#     perm_peaks,
#     "cleaned",
#     curve_selectors=[("200", "pre", "all")],
#     **{
#         **peak_kwargs,
#         "min_peaks": 5,
#         "fit_mode_use": "sequential",
#         "normalize_weight_mode": ("temp",),
#     },
# )

# perm_curves = calculate_permittivity(
#     curves,
#     "cleaned",
#     perm_curves=perm_curves,
#     curve_selectors=[("301", "dh", "all")],
#     **{
#         **perm_kwargs,
#         # "decay_mod": {"window_length": 0.010, "polyorder": 4},
#         "max_f_max": 5,
#         "max_f_min": 7.5e-2,
#         "min_f_min": 6e-2,
#     },
# )
# perm_peaks = collect_permittivity_peaks(
#     perm_curves,
#     perm_peaks,
#     "cleaned",
#     curve_selectors=[("301", "dh", "all")],
#     **{
#         **peak_kwargs,
#         "fit_mode_use": "sequential",
#         "fit_select_method": "max, min_std",
#         "normalize_weight_mode": ("temp",),
#     },
# )
