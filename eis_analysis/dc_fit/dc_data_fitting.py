# -*- coding: utf-8 -*- eis_analysis/**, local/*
import time
from copy import deepcopy
from typing import Any

import numpy as np

from eis_analysis.dc_fit.extract_tools import (
    DEFAULT_DIR,
    get_job,
    load_dc_data,  # noqa: F401
    save_results,  # noqa: F401
    create_std_df,
    bias_prior_fits,
)
from eis_analysis.dc_fit.segment_fitting import (
    fit_segments,
    fit_preparation,
    create_target_set,
    collect_segment_stats,
)
from eis_analysis.dc_fit.dc_data_cleaning import (
    sm_kwargs,
)
from eis_analysis.dc_fit.segment_cleaning import (
    smooth_segment,
)

# from eis_analysis.dc_fit.dc_data_processing import (
#     DEFAULT_DIR,
#     get_job,
#     load_dc_data,  # noqa: F401
#     save_results,  # noqa: F401
# )

np.seterr(invalid="raise")


SKIP_COLS = ["Current rms", "Voltage rms", "Current abs", "Voltage abs"]


def get_fit_job(job_key: str, prior_fit_num: int | None = None, **kwargs) -> dict:
    """
    Returns a copy of the job dict with prior_fit inserted if prior_fit_num is not None.

    Parameters
    ----------
    job_key : str
        The key for the job in ALL_JOBS dict.
    prior_fit_num : int or None
        The fit number to use for prior_fit, or None for no prior_fit.

    Returns
    -------
    dict
        The job dict, with prior_fit if specified.
    """
    kwargs.setdefault("job_source", ALL_JOBS)
    kwargs.setdefault("job_profiles", PROFILES)
    kwargs.setdefault("job_names", ["prep", "fit"])

    job = get_job(job_key, **kwargs)

    n_prior = job.get("prior_fit", prior_fit_num)
    if isinstance(n_prior, int):
        job["prior_fit"] = f"fit {n_prior}"

    return job


# %%
def run_fitting(
    points: dict,
    curves: dict,
    fit_jobs: list[dict[str, Any]],
    status: dict | None = None,
    run_smoothing: bool = False,
    # global_bias: float = 0.75,
    global_method_map: dict[str, Any] | tuple[str, ...] | list[str] = (),
    params: dict | None = None,
    verbose: bool = True,
    **kwargs: Any,
) -> tuple[dict, dict, dict]:
    """
    Run fitting sequences on cleaned DC data using named prep and fit profiles.

    Parameters
    ----------
    points : dict
        Loaded points dictionary.
    curves : dict
        Loaded curves dictionary.
    fit_jobs : list[dict]
        List of fit job dicts, each with 'prep_profile', 'fit_profile', and optional 'targets'.
    status : dict, optional
        Status dictionary for external monitoring.
    run_smoothing : bool, optional
        If True, re-run cleaning on loaded curves.
    global_bias : float, optional
        Bias for prior fits.
    params : dict, optional
        Optional params dictionary for prior fits.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    points : dict
        Updated points dictionary.
    curves : dict
        Updated curves dictionary.
    params : dict
        Updated params dictionary.
    """

    if run_smoothing:
        if verbose:
            print(f"{time.ctime()}: Re-smooth data...")
        curves["cleaned"] = smooth_segment(
            curves.get("trimmed", curves["cleaned"]),
            **kwargs,
        )
    if verbose:
        print(f"{time.ctime()}: Begin fitting...")

    if params is None:
        params = {}

    maps = {}

    stats_df = collect_segment_stats(curves["cleaned"])

    for count, job in enumerate(fit_jobs):
        fit_num = job.get("fit_num")
        if fit_num is None:
            fit_nums = [int(k.split()[1]) for k in points if k.startswith("fit ")]
            fit_num = max(fit_nums, default=0) + 1

        if verbose:
            print(f"{time.ctime()}: Job {count} of {len(fit_jobs)-1}, Fit {fit_num}")

        prior_fit_key = job.get("prior_fit")
        initial_guesses = params.get(f"{prior_fit_key} vals", None) if prior_fit_key else None
        if initial_guesses is not None and (job.get("bias", 0.0) or job.get("pre_bias", 0.0)):
            bias = job.get("bias", 0.0)
            agg_method = job.get("agg_method", "median")
            method_map = job.get("method_map", global_method_map)
            # Pre-bias: Intended to provide a minor shift to a more global relationship which might
            # distort the p0's if applied strongly. Since bias is typically > 0.5, it's default is 1-bias
            initial_guesses = bias_prior_fits(
                deepcopy(initial_guesses),
                *job.get("pre_bias_keys", ["condition", "temp"]),
                bias=job.get("pre_bias", 1 - bias),
                agg_method=job.get("pre_agg_method", agg_method),
                method_map=job.get("pre_method_map", method_map),
            )
            # Main bias: Intended to apply trending that is more directly related to the curve being fit.
            initial_guesses = bias_prior_fits(
                initial_guesses,
                *job.get("bias_keys", ["sample_name", "condition", "temp"]),  # default
                bias=bias,
                agg_method=agg_method,
                method_map=method_map,
            )

        targets = set()
        if job.get("targets") is not None:
            for target in job["targets"]:
                targets.update(
                    create_target_set(params.get(f"{prior_fit_key} vals", {}), **target)
                )

        # --- Preparation call ---
        (
            maps[f"fit {fit_num}"],
            init_cond,
            initial_guesses,
            # params[f"fit {fit_num} conditions"],
            # params[f"fit {fit_num} vals"],
        ) = fit_preparation(
            curves["cleaned"],
            # mean_df=points.get(f"{prior_fit_key}", points["updated means"]),
            initial_guesses=initial_guesses,
            prior_fit=params.get(f"{prior_fit_key} vals", None),
            skipcols=job.get("skipcols", SKIP_COLS),
            stats_df=stats_df.copy(),
            targets=targets,
            **job.get("prep_profile", PROFILES["exp_default"]),
        )

        points[f"fit {fit_num}"] = points.get(
            f"{prior_fit_key}", create_std_df(source_df=points["updated means"])
        ).copy()
        curves[f"fit {fit_num}"] = {
            k: v.copy() for k, v in curves.get(f"{prior_fit_key}", {}).items()
        }
        if targets and f"{prior_fit_key} conditions" in params:
            target_cols, _ = map(set, zip(*targets))
            fit_cond = params[f"{prior_fit_key} conditions"]
            for col in init_cond.keys():
                if col != "Equations" and col not in target_cols and col in fit_cond:
                    init_cond[col] = fit_cond[col]
                    initial_guesses[col] = params[f"{prior_fit_key} vals"][col]
                    init_cond["Equations"].loc[col] = fit_cond["Equations"].loc[col]
            initial_guesses["Equations"] = init_cond["Equations"].copy()

        params[f"fit {fit_num} conditions"] = init_cond
        params[f"fit {fit_num} vals"] = initial_guesses
        # # if prior_fit_key
        # params[f"fit {fit_num} vals"] = deepcopy(
        #     params.get(f"{prior_fit_key} vals", initial_guesses)
        # )

        # --- Fit call ---
        try:
            points[f"fit {fit_num}"], curves[f"fit {fit_num}"], params[f"fit {fit_num} vals"] = (
                fit_segments(
                    curves["cleaned"],
                    fit_map=maps[f"fit {fit_num}"],
                    fit_params_dict=params[f"fit {fit_num} conditions"],
                    fit_df=points[f"fit {fit_num}"],
                    fit_curves=curves[f"fit {fit_num}"],
                    fit_params=params[f"fit {fit_num} vals"],
                    targets=targets,
                    status=status,
                    verbose=verbose,
                    **job.get("fit_profile", PROFILES["de_default"]),
                )
            )
        except KeyboardInterrupt:
            if verbose:
                print()
            for col in params[f"fit {fit_num} vals"]:
                if col == "Equations":
                    continue
                params[f"fit {fit_num} vals"][col].drop(
                    "status", axis=1, errors="ignore", inplace=True
                )
            raise KeyboardInterrupt

        if status is not None and verbose:
            print(
                f"\r    Fit {fit_num} Complete.  "
                f"Mean Error: {status['ave_run_error']:.3e}; "
                f"Mean change: {status['ave_change']:.1f}%; "
                f"% Updated: {status['percent_updated']}%",
                flush=True,
            )

    if verbose:
        print(f"{time.ctime()}: Fitting complete.")

    return points, curves, params


# fmt: off
PROFILES = {
    # --- Preparation maps ---
    # Fit maps
    "exp_default": {
        "base_fit_map": {"Current": "exp_func", "Resistance": "exp_func"},
        },
    "2_exp": {
        "base_fit_map": {"Current": "2_exp_func", "Resistance": "2_exp_func"},
    },
    "3_exp": {
        "base_fit_map": {"Current": "3_exp_func", "Resistance": "3_exp_func"},
    },
    "pow_c": {
        "base_fit_map": {"Current": "pow_func"},
    },
    "str_exp_c": {
        "base_fit_map": {"Current": "stretch_exp_func"},
    },
    # Scaling maps
    "log_c": {
        "base_scale_map": {"Current": "log"},
    },
    "log_smix_c": {
        "base_scale_map": {"Current": ["lin", "log", "lin", "log"]},
    },
    "1_log_emix_c": {
        "base_scale_map": {"Current": ["lin", "log", "log"]},
    },
    "2_log_emix_c": {
        "base_scale_map": {"Current": ["lin", "lin", "lin", "log", "lin"]},
    },
    "3_log_emix_c": {
        "base_scale_map": {"Current": ["lin", "log", "lin", "lin", "lin", "log", "lin"]},
    },
    # Bound maps
    "beta_bound0": {
        "lb_map": {("*", "beta0"): 0.6},
        "p0_map": {("*", "beta0"): 0.9},
        "ub_map": {("*", "beta0"): 1},
    },
    "beta_bound": {
        "lb_map": {("*", "beta0"): 1e-32},
        "p0_map": {("*", "beta0"): 0.9},
        "ub_map": {("*", "beta0"): 1},
    },
    "t0min_bound": {
        "lb_map": {("*", "tau0"): 1e-4},
        "p0_map": {("*", "tau0"): 1e-3},
        "ub_map": {("*", "tau0"): 0.3},
    },
    "1_t_bound": {
        "lb_map": {("*", "tau0"): 1e-4},
        "p0_map": {("*", "tau0"): 1},
        "ub_map": {("*", "tau0"): 40},
    },
    "2_t_bound": {
        "lb_map": {("*", "tau0"): 1e-4, ("*", "tau1"): 1},
        "p0_map": {("*", "tau0"): 1e-3, ("*", "tau1"): 15},
        "ub_map": {("*", "tau0"): 0.3, ("*", "tau1"): 40},
    },
    "2_t_mid_bound": {
        "lb_map": {("*", "tau1"): 5},
        "ub_map": {("*", "tau0"): 5},
    },
    "3_t_bound": {
        "lb_map": {("*", "tau0"): 1e-4, ("*", "tau1"): 1e-1, ("*", "tau2"): 1},
        "p0_map": {("*", "tau0"): 1e-3, ("*", "tau1"): 1, ("*", "tau2"): 15},
        "ub_map": {("*", "tau0"): 0.3, ("*", "tau1"): 10, ("*", "tau2"): 40},
    },
    "t_mid_bound": {
        "check_overlaps": ["tau"],
        "overlap_group_keys": [], #["sample_name", "condition", "temp"],
    },
    "tk_mid_bound": {
        "check_overlaps": ["tau"],
        "overlap_group_keys": [ ["sample_name", "condition", "temp"]],
    },
    "at_mid_bound": {
        "check_overlaps": ["a", "tau"],
        "overlap_group_keys": [],
    },
    "atk_mid_bound": {
        "check_overlaps": ["a", "tau"],
        "overlap_group_keys": [["sample_name", "condition", "temp"], []],
    },
    "atkk_mid_bound": {
        "check_overlaps": ["a", "tau"],
        "overlap_group_keys": [["sample_name", "condition", "temp"], ["sample_name", "condition", "temp"]],
    },
    "mod_20": {"mod": 20},
    "mod_50": {"mod": 50},
    "mod_100": {"mod": 100},
    # --- Fit profiles---
    # Least squares profiles
    "ls_default": {
        "fit_type": "ls",
        "max_nfev": 5e3,
    },
    "ls_fast": {
        "fit_type": "ls",
        "max_nfev": 5e2,
    },
    "ls_long": {
        "fit_type": "ls",
    },
    # Differential evolution profiles
    "de_default": {
        "fit_type": "de",
        "init": "sobol",
    },
    "de_fast": {
        "fit_type": "de",
        "maxiter": 500,
        "init": "sobol",
    },
    "de_18_bin": {
        "fit_type": "de",
        "strategy": "best1bin",
        "popsize": 18,
        "init": "sobol",
    },
    "de_14_bin": {
        "fit_type": "de",
        "strategy": "best1bin",
        "popsize": 14,
        "init": "sobol",
    },
    "de_14_exp": {
        "fit_type": "de",
        "strategy": "best1exp",
        "popsize": 14,
        "init": "sobol",
    },
}

ALL_JOBS = {
        "fast_start_0":{
            "prep_profile": ["2_exp"],
            "fit_profile": ["de_fast"],
            "fit_kwargs": {
                "loss_func": "total_abs_perc_err",
                "retain_best": True,
            },
        },
        "fast_volt_temp":{
            "prep_profile": ["2_exp"],
            "fit_profile": ["de_fast"],
            "targets": [
                {"column": "Voltage", "expr": "=="},
                {"column": "Temperature", "expr": "=="},
            ],
            "fit_kwargs": {
                "loss_func": "total_abs_perc_err",
                "retain_best": True,
            },
        },
        "fast_curr_res":{
            "prep_profile": ["2_exp"],
            "fit_profile": ["de_fast"],
            "targets": [
                {"column": "Current", "expr": "=="},
                {"column": "Resistance", "expr": "=="},
            ],
            "fit_kwargs": {
                "loss_func": "total_abs_perc_err",
                "retain_best": False,
                # "weight_inflect": 0.25,
            },
        },
        "exp_0":{
            "prep_profile": ["2_exp", "atk_mid_bound", "mod_50"],
            "fit_profile": ["de_default"],
            "prep_kwargs": {
                # "lb_map": {("*", "tau0"): 0.3},
                "lb_map": {("*", "tau0"): 0.5, ("*", "tau1"): 6.0, ("Current", "a0"): 2e-12, ("Current", "a1"): 2e-12},
                "ub_map": {("*", "tau0"): 5.0, ("*", "tau1"): 50.0},
            },
            "fit_kwargs": {
                "loss_func": "total_abs_perc_err",
                "retain_best": False,
                # "weight_inflect": 0.25,
                # "maxiter": 2000,
                # "popsize": 18,
            },
        },
        "exp_1":{
            "prep_profile": ["2_exp", "t_mid_bound", "mod_50"],
            "fit_profile": ["de_default"],
            "targets": [
                {"column": "Current", "expr": "=="},
                {"column": "Resistance", "expr": "=="},
            ],
            # "prep_kwargs": {
            #     "lb_map": {("*", "tau0"): 0.5, ("*", "tau1"): 6.0, ("Current", "a0"): 2e-12, ("Current", "a1"): 2e-12},
            #     "ub_map": {("*", "tau0"): 4.0, ("*", "tau1"): 50.0},
            # },
            "fit_kwargs": {
                "loss_func": "total_abs_perc_err",
                "retain_best": True,
                "weight_inflect": 0.25,
                # "weight_inflect": 0.25,
                # "maxiter": 2000,
                # "popsize": 18,
            },
        },
        "exp_2":{
            "prep_profile": ["2_exp", "t_mid_bound", "mod_50"],
            "fit_profile": ["de_default"],
            "targets": [
                {"column": "Current", "expr": "=="},
                {"column": "Resistance", "expr": "=="},
            ],
            "prep_kwargs": {
                "lb_map": {("*", "tau0"): 0.3},# ("Current", "a0"): 9e-14, ("Current", "a1"): 9e-14},
                # "ub_map": {("*", "tau0"): 4.0, ("*", "tau1"): 50.0},
            },
            "fit_kwargs": {
                "loss_func": "total_abs_perc_err",
                "retain_best": True,
                "weight_inflect": 0.25,
                # "maxiter": 2000,
                # "popsize": 18,
            },
        },
        "exp_3":{
            "prep_profile": ["2_exp", "t_mid_bound", "mod_50"],
            "fit_profile": ["de_14_exp"],
            "targets": [
                {"column": "Current", "expr": "=="},
                {"column": "Resistance", "expr": "=="},
            ],
            "prep_kwargs": {
                "lb_map": {("*", "tau0"): 0.3},# ("Current", "a0"): 9e-14, ("Current", "a1"): 9e-14},
                # "ub_map": {("*", "tau0"): 4.0, ("*", "tau1"): 50.0},
            },
            "fit_kwargs": {
                "loss_func": "total_abs_perc_err",
                "retain_best": True,
                "weight_inflect": 0.25,
                # "maxiter": 2000,
                # "popsize": 18,
            },
        },
        "exp_exp_1":{
            "prep_profile": ["2_exp", "t_mid_bound", "mod_20"],
            "fit_profile": ["de_14_exp"],
            "targets": [
                {"column": "Current", "expr": "=="},
                {"column": "Resistance", "expr": "=="},
            ],
            "prep_kwargs": {
                "lb_map": {("*", "tau0"): 0.3},
                "ub_map": {("*", "tau0"): 5.0, ("*", "tau1"): 50.0},
            },
            "fit_kwargs": {
                "loss_func": "total_abs_perc_err",
                "retain_best": True,
                "weight_inflect": 0.25,
            },
        },
        "exp_exp_2":{
            "prep_profile": ["2_exp", "atk_mid_bound", "mod_50"],
            "fit_profile": ["de_14_exp"],
            "targets": [
                {"column": "Current", "expr": "=="},
                {"column": "Resistance", "expr": "=="},
            ],
            "prep_kwargs": {
                "lb_map": {("*", "tau0"): 0.3},
                "ub_map": {("*", "tau0"): 5.0},
            },
            "fit_kwargs": {
                "loss_func": "total_abs_perc_err",
                "retain_best": True,
                "weight_inflect": 0.25,
                "maxiter": 2000,
                "popsize": 18,
            },
        },
        "ls_exp_2":{
            "prep_profile": ["2_exp", "atk_mid_bound", "mod_100"],
            "fit_profile": ["ls_default"],
            "targets": [
                {"column": "Current", "expr": "=="},
                {"column": "Resistance", "expr": "=="},
            ],
            "prep_kwargs": {
                "lb_map": {("*", "tau0"): 0.5},
            },
            "fit_kwargs": {
                "loss_func": "abs_perc_err",
                "retain_best": False,
                "weight_inflect": 0.25,
            },
        },
        "exp_2exp_3":{
            "prep_profile": ["2_exp", "atk_mid_bound", "mod_50"],
            "fit_profile": ["de_14_exp"],
            "targets": [
                {"column": "Current", "expr": "=="},
                {"column": "Resistance", "expr": "=="},
            ],
            "prep_kwargs": {
                "lb_map": {("*", "tau0"): 0.5, ("*", "tau1"): 1.0},
                # "ub_map": {("*", "tau1"): 50.0},
            },
            "fit_kwargs": {
                "loss_func": "total_abs_perc_err",
                "retain_best": True,
                "weight_inflect": 0.25,
                "maxiter": 2000,
                "popsize": 18,
            },
        },
        "exp_exp_4":{
            "prep_profile": ["3_exp", "3_log_emix_c"],
            "fit_profile": ["de_default"],
            "targets": [
                {"column": "Current", "expr": "=="},
                {"column": "Resistance", "expr": "=="},
            ],
            "prep_kwargs": {
                "lb_map": {("*", "tau0"): 1e-3, ("*", "tau1"): 0.5, ("*", "tau2"): 100.0},
                "ub_map": {("*", "tau1"): 8.0, ("*", "tau2"): 500.0}
                # "lb_map": {("*", "tau0"): 0.6, ("Current", "a0"): 9e-14, ("Current", "a1"): 9e-14},
                # "ub_map": {("*", "tau0"): 4.0, ("*", "tau1"): 50.0},
            },
            "fit_kwargs": {
                "loss_func": "total_abs_perc_err",
                "retain_best": False,
                "weight_inflect": 0.4,
                # "maxiter": 2000,
                # "popsize": 18,
            },
        },
        "exp_exp_5":{
            "prep_profile": ["3_exp", "atk_mid_bound", "mod_50"],
            "fit_profile": ["de_14_exp"],
            "targets": [
                {"column": "Current", "expr": "=="},
                {"column": "Resistance", "expr": "=="},
            ],
            "prep_kwargs": {
                "lb_map": {("*", "tau0"): 1e-3, ("*", "tau1"): 0.5, ("*", "tau2"): 2.0},
                "ub_map": {("*", "tau1"): 8.0, ("*", "tau2"): 500.0}
                # "lb_map": {("*", "tau0"): 0.6, ("Current", "a0"): 9e-14, ("Current", "a1"): 9e-14},
                # "ub_map": {("*", "tau0"): 4.0, ("*", "tau1"): 50.0},
            },
            "fit_kwargs": {
                "loss_func": "total_abs_perc_err",
                "retain_best": True,
                "weight_inflect": 0.4,
                # "maxiter": 2000,
                # "popsize": 18,
            },
        },
        "str_exp_0":{
            "prep_profile": ["str_exp_c", "beta_bound0", "mod_50"],
            "fit_profile": ["de_default"],
            "targets": [
                {"column": "Current", "expr": "=="},
                {"column": "Resistance", "expr": "=="},
            ],
            "fit_kwargs": {
                "loss_func": "total_abs_perc_err",
                "retain_best": False,
                "weight_inflect": 0.25,
                # "maxiter": 2000,
                # "popsize": 18,
            },
        },
        "str_exp_1":{
            "prep_profile": ["str_exp_c"],
            "fit_profile": ["de_default"],
            "targets": [
                {"column": "Current", "expr": "=="},
                {"column": "Resistance", "expr": "=="},
                {"column": "Resistance", "expr": "=="},
            ],
            "fit_kwargs": {
                "loss_func": "total_abs_perc_err",
                "retain_best": True,
                "weight_inflect": 0.25,
                # "maxiter": 2000,
                # "popsize": 18,
            },
        },
        "str_exp_2":{
            "prep_profile": ["str_exp_c", "beta_bound"],
            "fit_profile": ["de_14_exp"],
            "targets": [
                {"column": "Current", "expr": "=="},
                {"column": "Resistance", "expr": "=="},
            ],
            "fit_kwargs": {
                "loss_func": "total_abs_perc_err",
                "retain_best": True,
                "weight_inflect": 0.25,
                # "maxiter": 2000,
                # "popsize": 18,
            },
        },
        "pow_1":{
            "prep_profile": ["pow_c", "log_current"],
            "fit_profile": ["de_default"],
            "targets": [
                {"column": "Current", "expr": "=="},
            ],
        },
        "pow_2":{
            "prep_profile": ["pow_c", "log_current"],
            "fit_profile": ["de_default"],
            "targets": [
                {"column": "Current", "expr": "=="},
            ],
        },
        "100_dh_exp_1":{
            "prep_profile": ["2_exp"],
            "fit_profile": ["de_default"],
            "targets": [
                {"column": "Current", "expr": "(sample_name == '100') & (condition == 'dh')"},
            ],
            "fit_kwargs": {
                "loss_func": "total_abs_perc_err",
                "retain_best": False,
            },
        },
        "exp_2exp_err":{
            "prep_profile": ["2_exp", "atk_mid_bound", "mod_100"],
            "fit_profile": ["de_14_exp"],
            "targets": [
                {"column": "Resistance", "expr": "Error > @quantile(Error, 0.8)"},
            ],
            "prep_kwargs": {
                "lb_map": {("*", "tau0"): 0.3},
                # "ub_map": {("*", "tau1"): 50.0},
            },
            "fit_kwargs": {
                "loss_func": "total_abs_perc_err",
                "retain_best": True,
                "weight_inflect": 0.25,
                "maxiter": 2000,
                "popsize": 18,
            },
        },
        "exp_3exp_err":{
            "prep_profile": ["3_exp", "atk_mid_bound", "mod_100"],
            "fit_profile": ["de_14_exp"],
            "targets": [
                {"column": "Current", "expr": "Error > @quantile(Error, 0.8)"},
                {"column": "Resistance", "expr": "Error > @quantile(Error, 0.8)"},
            ],
            "prep_kwargs": {
                "lb_map": {("*", "tau0"): 1e-12, ("*", "tau1"): 0.3},
                # "ub_map": {("*", "tau1"): 50.0},
            },
            "fit_kwargs": {
                "loss_func": "total_abs_perc_err",
                "retain_best": True,
                "weight_inflect": 0.25,
                "maxiter": 2000,
                "popsize": 18,
            },
        },
        "str_exp_err":{
            "prep_profile": ["str_exp_c", "beta_bound", "mod_100"],
            "fit_profile": ["de_14_exp"],
            "targets": [
                {"column": "Current", "expr": "Error > @quantile(Error, 0.8)"},
                {"column": "Resistance", "expr": "Error > @quantile(Error, 0.8)"},
            ],
            "fit_kwargs": {
                "loss_func": "total_abs_perc_err",
                "retain_best": True,
                "weight_inflect": 0.25,
                "maxiter": 2000,
                "popsize": 18,
            },
        },
        "exp_2exp_pre_dry1":{
            "prep_profile": ["2_exp", "tk_mid_bound", "mod_100"],
            "fit_profile": ["de_14_exp"],
            "targets": [
                {"column": "Current", "expr": "condition != 'dh'"},
                {"column": "Resistance", "expr": "condition != 'dh'"},
            ],
            "prep_kwargs": {
                "lb_map": {("*", "tau0"): 0.5, ("*", "tau1"): 6.0, ("Current", "a1"): 6e-13},
                "ub_map": {("*", "tau0"): 5.0, ("*", "tau1"): 35.0},
            },
            "fit_kwargs": {
                "loss_func": "total_abs_perc_err",
                "retain_best": False,
                "weight_inflect": 0.25,
            },
        },
        "exp_2exp_dh1":{
            "prep_profile": ["2_exp", "tk_mid_bound", "mod_100"],
            "fit_profile": ["de_14_exp"],
            "targets": [
                {"column": "Current", "expr": "condition == 'dh'"},
                {"column": "Resistance", "expr": "condition == 'dh'"},
            ],
            "prep_kwargs": {
                "lb_map": {("*", "tau0"): 0.3, ("*", "tau1"): 8.0, ("Current", "a1"): 6e-13},
                "ub_map": {("*", "tau0"): 4.0, ("*", "tau1"): 35.0},
            },
            "fit_kwargs": {
                "loss_func": "total_abs_perc_err",
                "retain_best": False,
                "weight_inflect": 0.25,
                # "maxiter": 2000,
                # "popsize": 18,
            },
        },
        "exp_2exp_pre_dry2":{
            "prep_profile": ["2_exp", "atk_mid_bound", "mod_50"],
            "fit_profile": ["de_14_exp"],
            "targets": [
                {"column": "Current", "expr": "condition != 'dh'"},
                {"column": "Resistance", "expr": "condition != 'dh'"},
            ],
            "prep_kwargs": {
                "lb_map": {("*", "tau0"): 0.5, ("*", "tau1"): 6.0, ("Current", "a1"): 6e-13},
                "ub_map": {("*", "tau0"): 5.0, ("*", "tau1"): 45.0},
            },
            "fit_kwargs": {
                "loss_func": "total_abs_perc_err",
                "retain_best": True,
                "weight_inflect": 0.25,
            },
        },
        "exp_2exp_dh2":{
            "prep_profile": ["2_exp", "atk_mid_bound", "mod_50"],
            "fit_profile": ["de_14_exp"],
            "targets": [
                {"column": "Current", "expr": "condition == 'dh'"},
                {"column": "Resistance", "expr": "condition == 'dh'"},
            ],
            "prep_kwargs": {
                "lb_map": {("*", "tau0"): 0.3, ("*", "tau1"): 8.0, ("Current", "a1"): 6e-13},
                "ub_map": {("*", "tau0"): 4.0, ("*", "tau1"): 45.0},
            },
            "fit_kwargs": {
                "loss_func": "total_abs_perc_err",
                "retain_best": True,
                "weight_inflect": 0.25,
                # "maxiter": 2000,
                # "popsize": 18,
            },
        },
    }


run_fit_kwargs = {}
run_fit_kwargs |= dict(
    fit_jobs=[
        get_fit_job("fast_start_0", fit_num=0),
        get_fit_job("fast_volt_temp", 0, fit_num=0, bias=0.6),
        get_fit_job("exp_2", 0, fit_num=0, bias=0.6),

        get_fit_job("exp_exp_2", 0, fit_num=1,
            pre_bias=0.5, bias=0.5,
            pre_bias_keys=["temp"],
            # bias_keys=["sample_name", "condition"],
        ),
        get_fit_job("exp_2exp_err", 1, fit_num=1, pre_bias=0.6, bias=0.6),

        get_fit_job("exp_2exp_dh1", 1, fit_num=2, pre_bias=0.25, bias=0.25),
        get_fit_job("exp_2exp_pre_dry1", 2, fit_num=2, pre_bias=0.25, bias=0.25),
        get_fit_job("exp_2exp_dh2", 2, fit_num=2, pre_bias=0.25, bias=0.75),
        get_fit_job("exp_2exp_pre_dry2", 2, fit_num=2, pre_bias=0.25, bias=0.75),
        get_fit_job("exp_2exp_dh2", 2, fit_num=2, pre_bias=0.25, bias=0.5),
        get_fit_job("exp_2exp_pre_dry2", 2, fit_num=2, pre_bias=0.25, bias=0.5),
        get_fit_job("exp_2exp_err", 2, fit_num=2, pre_bias=0.6, bias=0.6),

        get_fit_job("exp_2exp_3", 2, fit_num=3, pre_bias=0.25, bias=0.25),
        get_fit_job("exp_2exp_err", 3, fit_num=3, pre_bias=0.6, bias=0.6),

        get_fit_job("exp_exp_4", 3, fit_num=4, pre_bias=0.25, bias=0.75),
        get_fit_job("exp_exp_4", 4, fit_num=4, pre_bias=0.25, bias=0.75),
        # get_fit_job("exp_exp_4", 4, fit_num=4,
        #     pre_bias=0.75, bias=0.75,
        #     pre_bias_keys=["temp"],
        #     bias_keys=["sample_name", "condition"],
        # ),
        get_fit_job("exp_exp_5", 4, fit_num=4, pre_bias=0.5, bias=0.5),
        get_fit_job("exp_exp_5", 4, fit_num=4, pre_bias=0.25, bias=0.25),
        get_fit_job("exp_3exp_err", 4, fit_num=4, pre_bias=0.6, bias=0.6),

        get_fit_job("str_exp_0", 3, fit_num=5, pre_bias=0.25, bias=0.5),
        get_fit_job("str_exp_1", 5, fit_num=5, pre_bias=0.25, bias=0.5),
        get_fit_job("str_exp_1", 5, fit_num=5, pre_bias=0.25, bias=0.5),
        get_fit_job("str_exp_1", 5, fit_num=5, pre_bias=0.25, bias=0.5),
        get_fit_job("str_exp_err", 5, fit_num=5, pre_bias=0.6, bias=0.6),
    ],
    run_smoothing=False,
    # global_bias=0.6,  # 0.75,
    global_method_map={
        "b0": None,
        "tau0": lambda x: 0.25 * x.min() + 0.75 * x.median(),  # "min",
        "tau @ max": lambda x: 0.25 * x.max() + 0.75 * x.median(),  # "max",
        "a0": lambda x: 0.25 * x.max() + 0.75 * x.median(),  # "min",
        "a @ max": lambda x: 0.25 * x.min() + 0.75 * x.median(),  # "max",
    },
)
run_fit_kwargs |= sm_kwargs
# fmt: on

# %%
if __name__ == "__main__":

    try:
        # loaded_data = globals()["loaded_data"]
        points = globals()["points"]
        curves = globals()["curves"]
        params = globals()["params"] if "params" in globals() else {}

    except Exception:
        # "fits", "points", "grouped", "params"
        # loaded_data = load_dc_data(DEFAULT_DIR, "points", "curves")
        points = load_dc_data(DEFAULT_DIR, "points")["points"]
        curves = load_dc_data(DEFAULT_DIR, "curves")["curves"]
        params = load_dc_data(DEFAULT_DIR, "params")["params"]

    # points = loaded_data.get("points", {})
    # curves = loaded_data.get("curves", {})
    # params = loaded_data.get("params", {})
    status = {}

    # %%
    points, curves, params = run_fitting(
        points=points,
        curves=curves,
        status=status,
        params=params,
        **run_fit_kwargs,
    )

    # # %%
    # save_results(
    #     data_path=DEFAULT_DIR,
    #     points=points,
    #     # curves=curves,
    #     params=params,
    #     # save_mode="a",
    #     attrs=True,
    # )


# --- old fit jobs ---
# get_fit_job("fast_volt_temp", fit_num=0),
# get_fit_job("fast_curr_res",0, fit_num=0),
# get_fit_job("exp_0", 0, fit_num=0, pre_bias=0.25, bias=0.5),
# get_fit_job("exp_exp_2", 0, fit_num=1, pre_bias=0.4, bias=0.4),
# get_fit_job("exp_exp_dh", 2, fit_num=2, pre_bias=0.25, bias=0.25),
# get_fit_job("exp_exp_2", 0, fit_num=0),
# get_fit_job("exp_exp_2", 1, fit_num=2, pre_bias=0.4, bias=0.5),
# get_fit_job("exp_exp_2", 2, fit_num=2, pre_bias=0.3, bias=0.6),
# get_fit_job("exp_1", 1, fit_num=1),
# get_fit_job("exp_2exp_3", 0, fit_num=2, pre_bias=0.25, bias=0.6),
# get_fit_job("exp_2exp_3", 0, fit_num=3, pre_bias=0.25, bias=0.25),
# get_fit_job("exp_2exp_3", 0, fit_num=1,
#     pre_bias=0.4, bias=0.4,
#     pre_bias_keys=["temp"],
#     bias_keys=["sample_name", "condition"],
# ),
# get_fit_job("exp_2", 0, fit_num=0),
# get_fit_job("exp_exp_4", 1, fit_num=4, pre_bias=0.25, bias=0.25),
# get_fit_job("str_exp_2", 5, fit_num=5),
# get_fit_job("str_exp_2", 5, fit_num=5),
# get_fit_job("str_exp_2", 5, fit_num=5),

# fit_jobs = [
#     # get_fit_job("fast_start_0", fit_num=0),
#     get_fit_job("exp_0", fit_num=0),
#     get_fit_job("exp_0", 0, fit_num=1, pre_bias=0.25, bias=0.5),
#     get_fit_job("exp_1", 1, fit_num=1, pre_bias=0.25, bias=0.5),
#     # get_fit_job("exp_1", 1, fit_num=1),
#     get_fit_job("exp_2exp_3", 1, fit_num=1, pre_bias=0.25, bias=0.5),
#     get_fit_job("exp_2exp_3", 1, fit_num=2, pre_bias=0.4, bias=0.6),
#     get_fit_job("exp_2exp_3", 1, fit_num=3, pre_bias=0.25, bias=0.25),
#     get_fit_job("exp_exp_4", 1, fit_num=4, pre_bias=0.1, bias=0.1),
#     get_fit_job("str_exp_0", 1, fit_num=5),
#     get_fit_job("str_exp_1", 5, fit_num=5),
#     get_fit_job("str_exp_2", 5, fit_num=5),
#     # get_fit_job("str_exp_2", 5, fit_num=5),
#     # get_fit_job("str_exp_2", 5, fit_num=5),
# ]
