# # -*- coding: utf-8 -*-
# filepath: c:\Users\j2cle\Documents\Python\impedance_analysis\eis_analysis\dc_fit\dc_analysis.py

import numpy as np

from eis_analysis.dc_fit.dc_data_post import (
    run_point_fitting,
)
from eis_analysis.dc_fit.extract_tools import (
    DEFAULT_DIR,
    save_results,  # noqa: F401
)
from eis_analysis.dc_fit.dc_data_fitting import (
    run_fitting,
    run_fit_kwargs,
)
from eis_analysis.dc_fit.dc_data_cleaning import (
    pre_process,
    pre_process_kwargs,
)
from eis_analysis.dc_fit.dc_data_processing import (
    perm_jobs,
    run_perm_calcs,
)

# Set numpy error handling
np.seterr(invalid="raise")

if __name__ == "__main__":
    # Define data directory
    status = {}
    curves = {}
    points = {}
    params = {}
    perm_curves = {}
    perm_peaks = {}

    curves, rej_curves, points, rej_points = pre_process(
        **pre_process_kwargs,
    )

    # Save results first few curves
    save_results(
        data_path=DEFAULT_DIR,
        curves=curves,
        save_mode="w",
        attrs=True,
    )
    saved_curves = list(curves.keys())

    points, curves, params = run_fitting(
        points=points,
        curves=curves,
        status=status,
        params=params,
        **run_fit_kwargs,
    )

    # save_results(
    #     DEFAULT_DIR,
    #     points=points,
    #     params=params,
    #     save_mode="w",
    #     attrs=True,
    # )

    results = run_point_fitting(
        points=points,
        params=params,
    )

    save_results(
        DEFAULT_DIR,
        attrs=True,
        **results,  # type: ignore[call-arg]
    )

    # Save points and new/modified curves
    if run_fit_kwargs.get("run_smoothing", False) and "cleaned" in saved_curves:
        saved_curves.remove("cleaned")

    save_results(
        DEFAULT_DIR,
        *[k for k in curves if k not in saved_curves],
        curves=curves,
        save_mode="w",
        attrs=True,
    )

    perm_curves, perm_peaks = run_perm_calcs(
        curves=curves,
        perm_jobs=perm_jobs,
        perm_curves=perm_curves,
        perm_peaks=perm_peaks,
        params=params,
        suppress_warnings=True,  # Set to True to suppress warnings during permittivity calculations
    )

    # Save results
    save_results(
        data_path=DEFAULT_DIR,
        perm_curves=perm_curves,
        perm_peaks=perm_peaks,
        save_mode="w",
        attrs=True,
    )

    # Pre-process the data
    # sm_kwargs = {
    #     "columns_to_smooth": ["Current"],
    #     "window_length": 0.075,
    #     "polyorder": 1,
    #     "normalize": False,
    # }
    # curves, rej_curves, points, rej_points = pre_process(
    #     data_dir,
    #     ["converted_data.xlsx", "converted_100_and_301.xlsx"],
    #     ["cln2dh80c_r1", "cln2dh80c_r2", "cln2dh85c_r1", "cln2dh85c_r2", "cln2dh85c_r3"],
    #     min_slope=-1e-6,
    #     # outlier_eval=True,
    #     use_gradient_residual=True,
    #     reason="artifact",
    #     start_region=4,
    #     smooth=True,
    #     sm_kwargs=sm_kwargs,
    #     ensure_point=True,
    #     skew_threshold=3,
    #     diff_threshold=5e-12,
    # )

    # # Prepare fitting jobs
    # fit_jobs = [
    #     get_fit_job("exp_0", fit_num=0),
    #     get_fit_job("exp_0", 0, fit_num=1),
    #     get_fit_job("exp_1", 1, fit_num=1),
    #     get_fit_job("ls_exp_2", 1, fit_num=2),
    #     get_fit_job("exp_exp_3", 1, fit_num=3),
    #     get_fit_job("str_exp_0", 1, fit_num=4),
    #     get_fit_job("str_exp_1", 4, fit_num=4),
    #     get_fit_job("str_exp_2", 4, fit_num=5),
    #     get_fit_job("str_exp_2", 5, fit_num=5),
    #     get_fit_job("str_exp_2", 5, fit_num=5),
    # ]

    # # Run fitting
    # points, curves, params = run_fitting(
    #     points=points,
    #     curves=curves,
    #     fit_jobs=fit_jobs,
    #     status=status,
    #     run_smoothing=False,
    #     prior_bias=0.75,
    #     params=params,
    #     ignore=["b0"],
    #     **sm_kwargs,
    # )

    # # Calculate permittivity curves
    # perm_curves = calculate_permittivity(
    #     curves,
    #     "cleaned",
    #     perm_curves=perm_curves,
    #     **{
    #         "params": params,
    #         "vals_key": "fit 1 vals",
    #         "decay_mod": None,
    #         "remove_mean": "b0",
    #         "padding": False,
    #         "dt_function": np.max,
    #         "pre_spline": "pchip",
    #         "post_spline": "pchip",
    #         "max_f_max": 1,
    #     },
    # )

    # # %% run permittivity inits
    # perm_kwargs = {
    #     "params": params,
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


# import re
# import time
# from pathlib import Path

# import numpy as np

# from eis_analysis.system_utilities import (
#     save,
#     load_file,
# )
# from eis_analysis.dc_fit.extract_tools import (
#     bias_prior_fits,
#     reduce_df_index,
#     insert_polarity_column,
#     aggregate_primary_columns,
# )
# from eis_analysis.dc_fit.column_parsing import (
#     apply_col_mapping,
#     evaluate_segments,
# )
# from eis_analysis.dc_fit.segment_fitting import (
#     fit_segments,
#     create_std_df,
#     fit_preparation,
#     # plot_fit_curves,
#     create_target_set,
#     targeted_df_update,
# )
# from eis_analysis.dc_fit.segment_cleaning import (
#     clean_segment_data,
# )
# from eis_analysis.dc_fit.periodicity_detection import (
#     process_segments_mean,
# )
# from eis_analysis.dc_fit.permitivity_calculations import (
#     collect_peak_summary_df,
#     transform_to_permittivity,
# )

# # %% --- Main Section ---
# np.seterr(invalid="raise")
# if __name__ == "__main__":
#     data_file = Path(
#         r"D:\Online\ASU Dropbox\Jacob Clenney\Work Docs\Data\Analysis\IS\EVA\DC Analysis\rnd 2"
#     )
#     loaded_data = {}
#     points = {}
#     params = {}
#     curves = {}
#     maps = {}
#     grp_points = {}
#     pol_points = {}
#     params_flt = {}
#     perm_curves = {}
#     perm_peaks = {}
#     status = {}
#     results = tuple()
#     fit = 0
#     targets = None
#     target_val = 1e-12
#     prior_bias = 0.75

#     run_segmentation = True
#     run_parsing = True
#     run_builder = True
#     run_builder_full = True
#     run_cleaning = True
#     run_fit_sequence = {4, 5}  # {1, 2, 3, 4, 5}
#     run_targeting = False
#     run_conversions = True
#     run_grouping = False
#     run_save = False
#     run_save_full = False
#     run_plots = False

#     # %% Import and Load data
#     if not loaded_data:
#         print(f"{time.ctime()}: Loading data...")

#         for file in ["converted_data.xlsx", "converted_100_and_301.xlsx"]:
#             loaded_data.update(load_file(data_file / file)[0])

#     # %% Segmenting loop
#     if run_segmentation:
#         print(f"{time.ctime()}: Segmenting data...")

#         points["raw_means"], curves["raw"], curves["raw_full"] = process_segments_mean(
#             loaded_data,
#             "cln2dh80c_r1",
#             "cln2dh80c_r2",
#             "cln2dh85c_r1",
#             "cln2dh85c_r2",
#             "cln2dh85c_r3",
#             inclusive=False,
#         )

#     # %% Parsing section
#     if run_parsing:
#         print(f"{time.ctime()}: Parsing data...")

#         results = evaluate_segments(points["raw_means"])
#         points["means"] = results[0].copy()
#         points["updated means"] = results[0].copy(deep=True)

#     # %% Build segmented DataFrame from parsed data
#     if run_builder:
#         print(f"{time.ctime()}: Building segmented DataFrame...")

#         curves["base"] = apply_col_mapping(curves["raw"], results[0], results[1])

#     # %% Build full
#     if run_builder_full:
#         print(f"{time.ctime()}: Building full DataFrame...")
#         curves["full"] = apply_col_mapping(curves["raw_full"], results[0], results[1])

#     # %% Clean segments
#     if run_cleaning:
#         print(f"{time.ctime()}: Cleaning segment data...")

#         curves["cleaned"] = clean_segment_data(
#             curves["base"],
#             stable_voltage=True,
#             check_exp_start=True,
#             check_exp_end=True,
#             alpha=0.05,
#             smooth=False,
#             window_length=0.05,
#             polyorder=1,
#         )

#     # %% Fit Sequence 1
#     fit += 1
#     if fit in run_fit_sequence:
#         print(f"{time.ctime()}: Running fit sequence {fit}...")

#         targets = set()
#         if run_targeting:
#             targets = create_target_set(
#                 params.get(f"fit {fit - 1} vals", {}), "Current", value=target_val
#             )

#         maps[f"fit {fit}"], params[f"fit {fit} conditions"], params[f"fit {fit} vals"] = (
#             fit_preparation(
#                 curves["cleaned"],
#                 mean_df=points.get(f"fit {fit - 1}", points["updated means"]),
#                 prior_fits=bias_prior_fits(params.get(f"fit {fit - 1} vals"), bias=prior_bias),
#                 fit_raw=False,
#                 skipcols=[
#                     "Current abs",
#                     "Current rms",
#                     "Voltage abs",
#                     "Voltage rms",
#                 ],
#                 # base_scale_map={"Current": ["lin", "log", "log"]},
#                 base_fit_map={
#                     "Current": "3_exp_func",
#                     "Resistance": "3_exp_func",
#                 },
#                 base_scale_map={"Current": ["lin", "log", "lin", "log", "lin", "log", "log"]},
#                 lb_map={("*", "tau0"): 1e-4, ("*", "tau1"): 1e-1, ("*", "tau2"): 1},
#                 p0_map={("*", "tau0"): 1e-3, ("*", "tau1"): 1, ("*", "tau2"): 15},
#                 ub_map={("*", "tau0"): 0.1, ("*", "tau1"): 10, ("*", "tau2"): 40},
#             )
#         )

#         points[f"fit {fit}"] = points.get(
#             f"fit {fit - 1}", create_std_df(source_df=points["updated means"])
#         ).copy()
#         curves[f"fit {fit}"] = {k: v.copy() for k, v in curves.get(f"fit {fit - 1}", {}).items()}
#         # params[f"fit {fit} vals"] = {}
#         #     k: v.copy() for k, v in params.get(f"fit {fit - 1} vals", {}).items()
#         # }

#         points[f"fit {fit}"], curves[f"fit {fit}"], params[f"fit {fit} vals"] = fit_segments(
#             curves["cleaned"],
#             fit_map=maps[f"fit {fit}"],
#             fit_params_dict=params[f"fit {fit} conditions"],
#             weight_inflect=False,
#             fit_type="de",
#             targets=targets,
#             status=status,
#             fit_df=points[f"fit {fit}"],
#             fit_curves=curves[f"fit {fit}"],
#             fit_params=params[f"fit {fit} vals"],
#             # strategy="rand2bin",
#             init="sobol",
#             # maxiter=600,
#         )

#         if run_targeting:
#             points["updated means"] = targeted_df_update(
#                 points["updated means"],
#                 points[f"fit {fit}"],
#                 params[f"fit {fit} vals"]["Current"],
#                 value=target_val,
#             )

#     # %% Fit Sequence 2
#     fit += 1
#     if fit in run_fit_sequence:
#         print(f"{time.ctime()}: Running fit sequence {fit}...")

#         targets = create_target_set(params.get(f"fit {fit - 1} vals", {}), "Current", op="--")
#         targets.update(
#             create_target_set(params.get(f"fit {fit - 1} vals", {}), "Resistance", op="--")
#         )
#         if run_targeting:
#             targets = create_target_set(
#                 params.get(f"fit {fit - 1} vals", {}), "Current", value=target_val
#             )
#         priors = bias_prior_fits(
#             params.get(f"fit {fit - 1} vals"),
#             "sample_name",
#             "temp",
#             bias=1 - prior_bias,
#             agg_method="median",
#         )
#         priors = bias_prior_fits(priors, bias=prior_bias, agg_method="median")

#         maps[f"fit {fit}"], params[f"fit {fit} conditions"], params[f"fit {fit} vals"] = (
#             fit_preparation(
#                 curves["cleaned"],
#                 mean_df=points.get(f"fit {fit - 1}", points["updated means"]),
#                 prior_fits=priors,
#                 fit_raw=False,
#                 skipcols=[
#                     "Current abs",
#                     "Current rms",
#                     "Voltage abs",
#                     "Voltage rms",
#                 ],
#                 base_fit_map={
#                     "Current": "3_exp_func",
#                     "Resistance": "3_exp_func",
#                 },
#                 # base_scale_map={"Current": ["lin", "log", "lin", "log", "lin", "log", "log"]},
#                 # mod=5,
#                 # p0_map={("*", "tau0"): 0.01},
#                 # ub_map={("*", "tau0"): 1},
#             )
#         )

#         points[f"fit {fit}"] = points.get(
#             f"fit {fit - 1}", create_std_df(source_df=points["updated means"])
#         ).copy()
#         curves[f"fit {fit}"] = {k: v.copy() for k, v in curves.get(f"fit {fit - 1}", {}).items()}
#         # params[f"fit {fit} vals"] = {}
#         #     k: v.copy() for k, v in params.get(f"fit {fit - 1} vals", {}).items()
#         # }

#         points[f"fit {fit}"], curves[f"fit {fit}"], params[f"fit {fit} vals"] = fit_segments(
#             curves["cleaned"],
#             fit_map=maps[f"fit {fit}"],
#             fit_params_dict=params[f"fit {fit} conditions"],
#             weight_inflect=True,
#             fit_type="de",
#             targets=targets,
#             status=status,
#             fit_df=points[f"fit {fit}"],
#             fit_curves=curves[f"fit {fit}"],
#             fit_params=params[f"fit {fit} vals"],
#             strategy="best1bin",
#             init="sobol",
#             popsize=18,  # default is 15 higher is better but slower
#             # mutation=(0.75, 1.25),  # default is (0.5, 1) higher is better but slower
#             # recombination=0.6,  # default is 0.7 lower is better but slower
#             # maxiter=600,
#         )
#         if run_targeting:
#             points["updated means"] = targeted_df_update(
#                 points["updated means"],
#                 points[f"fit {fit}"],
#                 params[f"fit {fit} vals"]["Current"],
#                 value=target_val,
#             )

#     # %% Fit Sequence 3
#     fit += 1
#     if fit in run_fit_sequence:
#         print(f"{time.ctime()}: Running fit sequence {fit}...")

#         targets = create_target_set(params.get(f"fit {fit - 1} vals", {}), "Current", op="--")
#         targets.update(
#             create_target_set(params.get(f"fit {fit - 1} vals", {}), "Resistance", op="--")
#         )
#         if run_targeting:
#             targets = create_target_set(
#                 params.get(f"fit {fit - 1} vals", {}), "Current", value=target_val
#             )

#         priors = bias_prior_fits(
#             params.get(f"fit {fit - 1} vals"),
#             "sample_name",
#             "temp",
#             bias=1 - prior_bias,
#             agg_method="median",
#         )
#         priors = bias_prior_fits(priors, bias=prior_bias, agg_method="median")

#         maps[f"fit {fit}"], params[f"fit {fit} conditions"], params[f"fit {fit} vals"] = (
#             fit_preparation(
#                 curves["cleaned"],
#                 mean_df=points.get(f"fit {fit - 1}", points["updated means"]),
#                 prior_fits=priors,
#                 fit_raw=False,
#                 skipcols=[
#                     "Current abs",
#                     "Current rms",
#                     "Voltage abs",
#                     "Voltage rms",
#                 ],
#                 base_fit_map={
#                     "Current": "3_exp_func",
#                     "Resistance": "3_exp_func",
#                 },
#                 # base_scale_map={"Current": ["lin", "log", "lin", "log", "lin", "log", "log"]},
#                 # mod=5,
#                 # p0_map={("*", "tau0"): 0.01, ("*", "tau1"): 0.5},
#                 # ub_map={("*", "tau0"): 1, ("*", "tau1"): 1},
#             )
#         )

#         points[f"fit {fit}"] = points.get(
#             f"fit {fit - 1}", create_std_df(source_df=points["updated means"])
#         ).copy()
#         curves[f"fit {fit}"] = {k: v.copy() for k, v in curves.get(f"fit {fit - 1}", {}).items()}
#         # params[f"fit {fit} vals"] = {}
#         #     k: v["p0"].copy() if k != "Equations" else v.copy() for k, v in params[f"fit {fit} conditions"].items()
#         # }
#         #     k: v.copy() for k, v in params.get(f"fit {fit - 1} vals", {}).items()
#         # }

#         points[f"fit {fit}"], curves[f"fit {fit}"], params[f"fit {fit} vals"] = fit_segments(
#             curves["cleaned"],
#             fit_map=maps[f"fit {fit}"],
#             fit_params_dict=params[f"fit {fit} conditions"],
#             weight_inflect=True,
#             fit_type="de",
#             # workers=3,
#             # disp=1,
#             # verbose=0,
#             # max_nfev=1e3,
#             targets=targets,
#             status=status,
#             fit_df=points[f"fit {fit}"],
#             fit_curves=curves[f"fit {fit}"],
#             fit_params=params[f"fit {fit} vals"],
#             strategy="best1exp",
#             init="sobol",
#             popsize=14,  # default is 15 higher is better but slower
#             # mutation=(0.75, 1.25),  # default is (0.5, 1) higher is better but slower
#             # recombination=0.6,  # default is 0.7 lower is better but slower
#             # maxiter=600,
#         )
#         if run_targeting:
#             points["updated means"] = targeted_df_update(
#                 points["updated means"],
#                 points[f"fit {fit}"],
#                 params[f"fit {fit} vals"]["Current"],
#                 value=target_val,
#             )

#     # %% Fit Sequence 4
#     fit += 1
#     if fit in run_fit_sequence:
#         print(f"{time.ctime()}: Running fit sequence {fit}...")

#         targets = create_target_set(params.get(f"fit {fit - 1} vals", {}), "Current", op="--")
#         if run_targeting:
#             targets = create_target_set(
#                 params.get(f"fit {fit - 1} vals", {}), "Current", value=target_val
#             )

#         priors = bias_prior_fits(
#             params.get(f"fit {fit - 1} vals"),
#             "sample_name",
#             "temp",
#             bias=1 - prior_bias,
#             agg_method="median",
#         )
#         priors = bias_prior_fits(priors, bias=prior_bias, agg_method="median")

#         maps[f"fit {fit}"], params[f"fit {fit} conditions"], params[f"fit {fit} vals"] = (
#             fit_preparation(
#                 curves["cleaned"],
#                 mean_df=points.get(f"fit {fit - 1}", points["updated means"]),
#                 prior_fits=priors,
#                 fit_raw=False,
#                 skipcols=[
#                     "Current abs",
#                     "Current rms",
#                     "Voltage abs",
#                     "Voltage rms",
#                 ],
#                 base_fit_map={
#                     "Current": "pow_func",
#                     # "Resistance": "pow_func",
#                 },
#                 base_scale_map={"Current": "log"},
#                 # mod=5,
#                 # p0_map={("*", "tau0"): 0.01, ("*", "tau1"): 0.5},
#                 # ub_map={("*", "tau0"): 1, ("*", "tau1"): 1},
#             )
#         )

#         points[f"fit {fit}"] = points.get(
#             f"fit {fit - 1}", create_std_df(source_df=points["updated means"])
#         ).copy()
#         curves[f"fit {fit}"] = {k: v.copy() for k, v in curves.get(f"fit {fit - 1}", {}).items()}
#         # params[f"fit {fit} vals"] = {}
#         #     k: v.copy() for k, v in params.get(f"fit {fit - 1} vals", {}).items()
#         # }

#         points[f"fit {fit}"], curves[f"fit {fit}"], params[f"fit {fit} vals"] = fit_segments(
#             curves["cleaned"],
#             fit_map=maps[f"fit {fit}"],
#             fit_params_dict=params[f"fit {fit} conditions"],
#             weight_inflect=True,
#             fit_type="de",
#             # workers=3,
#             # disp=1,
#             # verbose=0,
#             # max_nfev=1e3,
#             targets=targets,
#             status=status,
#             fit_df=points[f"fit {fit}"],
#             fit_curves=curves[f"fit {fit}"],
#             fit_params=params[f"fit {fit} vals"],
#             strategy="best1bin",
#             init="sobol",
#             popsize=14,  # default is 15 higher is better but slower
#             # mutation=(0.75, 1.25),  # default is (0.5, 1) higher is better but slower
#             # recombination=0.6,  # default is 0.7 lower is better but slower
#         )
#         if run_targeting:
#             points["updated means"] = targeted_df_update(
#                 points["updated means"],
#                 points[f"fit {fit}"],
#                 params[f"fit {fit} vals"]["Current"],
#                 value=target_val,
#             )
#     # %% Fit Sequence 5
#     fit += 1
#     if fit in run_fit_sequence:
#         print(f"{time.ctime()}: Running fit sequence {fit}...")

#         targets = create_target_set(params.get(f"fit {fit - 1} vals", {}), "Current", op="--")
#         if run_targeting:
#             targets = create_target_set(
#                 params.get(f"fit {fit - 1} vals", {}), "Current", value=target_val
#             )

#         priors = bias_prior_fits(
#             params.get(f"fit {fit - 1} vals"),
#             "sample_name",
#             "temp",
#             bias=1 - prior_bias,
#             agg_method="median",
#         )
#         priors = bias_prior_fits(priors, bias=prior_bias, agg_method="median")

#         maps[f"fit {fit}"], params[f"fit {fit} conditions"], params[f"fit {fit} vals"] = (
#             fit_preparation(
#                 curves["cleaned"],
#                 mean_df=points.get(f"fit {fit - 1}", points["updated means"]),
#                 prior_fits=priors,
#                 fit_raw=False,
#                 skipcols=[
#                     "Current abs",
#                     "Current rms",
#                     "Voltage abs",
#                     "Voltage rms",
#                 ],
#                 base_fit_map={
#                     "Current": "pow_func",
#                     # "Resistance": "pow_func",
#                 },
#                 base_scale_map={"Current": "log"},
#                 # mod=5,
#                 # p0_map={("*", "tau0"): 0.01, ("*", "tau1"): 0.5},
#                 # ub_map={("*", "tau0"): 1, ("*", "tau1"): 1},
#             )
#         )

#         points[f"fit {fit}"] = points.get(
#             f"fit {fit - 1}", create_std_df(source_df=points["updated means"])
#         ).copy()
#         curves[f"fit {fit}"] = {k: v.copy() for k, v in curves.get(f"fit {fit - 1}", {}).items()}
#         # params[f"fit {fit} vals"] = {}
#         #     k: v.copy() for k, v in params.get(f"fit {fit - 1} vals", {}).items()
#         # }

#         points[f"fit {fit}"], curves[f"fit {fit}"], params[f"fit {fit} vals"] = fit_segments(
#             curves["cleaned"],
#             fit_map=maps[f"fit {fit}"],
#             fit_params_dict=params[f"fit {fit} conditions"],
#             weight_inflect=True,
#             fit_type="de",
#             # workers=3,
#             # disp=1,
#             # verbose=0,
#             # max_nfev=1e3,
#             status=status,
#             targets=targets,
#             fit_df=points[f"fit {fit}"],
#             fit_curves=curves[f"fit {fit}"],
#             fit_params=params[f"fit {fit} vals"],
#             strategy="best1bin",
#             init="sobol",
#             popsize=14,  # default is 15 higher is better but slower
#             # mutation=(0.75, 1.25),  # default is (0.5, 1) higher is better but slower
#             # recombination=0.6,  # default is 0.7 lower is better but slower
#         )
#         if run_targeting:
#             points["updated means"] = targeted_df_update(
#                 points["updated means"],
#                 points[f"fit {fit}"],
#                 params[f"fit {fit} vals"]["Current"],
#                 value=target_val,
#             )

#     # # %% Plotting
#     # if run_plots:
#     #     print(f"{time.ctime()}: Plotting results...")

#     #     target_curves = partial_selection(curves["cleaned"], "300", 60.0, any_all="all")
#     #     target_curves.update(partial_selection(curves["cleaned"], "200", 60.0, any_all="all"))

#     #     plot_fit_curves(target_curves, columns="Current", groupby=[0, 1], log_y=True)

#     # %% Post-processing permittivity
#     if run_conversions:
#         print(f"{time.ctime()}: Converting data...")
#         for curve_key in curves:
#             # print(curve_key)
#             if curve_key in ["base", "full", "cleaned"]:
#                 perm_curves[curve_key] = {
#                     k: transform_to_permittivity(
#                         v,
#                     )
#                     for k, v in curves[curve_key].items()
#                 }
#                 if curve_key == "cleaned":
#                     perm_peaks["cleaned_imag"], perm_peaks["cleaned_imag_all"] = (
#                         collect_peak_summary_df(
#                             perm_curves["cleaned"],
#                             min_peaks=4,
#                         )
#                     )
#                     perm_peaks["cleaned_loss"], perm_peaks["cleaned_loss_all"] = (
#                         collect_peak_summary_df(
#                             perm_curves["cleaned"],
#                             column="loss tangent",
#                             min_peaks=4,
#                         )
#                     )
#             elif "fit" in curve_key:
#                 perm_curves[curve_key] = {
#                     k: transform_to_permittivity(
#                         v,
#                         current_col="Current_fit",
#                         voltage_col="Voltage_fit",
#                     )
#                     for k, v in curves[curve_key].items()
#                 }
#                 perm_peaks[f"{curve_key}_imag"], perm_peaks[f"{curve_key}_imag_all"] = (
#                     collect_peak_summary_df(
#                         perm_curves[curve_key],
#                         min_peaks=4,
#                     )
#                 )

#                 perm_peaks[f"{curve_key}_loss"], perm_peaks[f"{curve_key}_loss_all"] = (
#                     collect_peak_summary_df(
#                         perm_curves[curve_key],
#                         column="loss tangent",
#                         min_peaks=4,
#                     )
#                 )

#         if run_save or run_save_full:
#             print(f"{time.ctime()}: Saving results...")

#             save(perm_peaks, data_file / "converted", "perm_peaks", file_type="xls", mode="a")

#         if run_save_full:
#             print(f"{time.ctime()}: Saving result arrays...")
#             for key in perm_curves:
#                 save(
#                     perm_curves[key],
#                     data_file / "converted",
#                     f"perms_{key}",
#                     file_type="xls",
#                     mode="a",
#                 )

#     # %% Post-processing
#     if run_grouping:
#         print(f"{time.ctime()}: Grouping data...")

#         for key in list(points):
#             pol_points[key] = insert_polarity_column(points[key].copy())
#             if re.match(r"^fit\s\d{1,2}$", key.lower().strip()):
#                 pol_points.update(
#                     {f"{key} {k}": v for k, v in reduce_df_index(pol_points[f"{key}"]).items()}
#                 )

#                 grp_points[f"{key} grouped"] = aggregate_primary_columns(
#                     points[key], split_by_polarity=True
#                 )
#                 grp_points.update(
#                     {
#                         f"{key} {k}": v
#                         for k, v in reduce_df_index(grp_points[f"{key} grouped"]).items()
#                     }
#                 )
#                 grp_points.update(
#                     {
#                         f"{key} {k}": v
#                         for k, v in reduce_df_index(grp_points[f"{key} grouped"], 1).items()
#                     }
#                 )
#                 grp_points.update(
#                     {
#                         f"{key} {int(k)}": v
#                         for k, v in reduce_df_index(grp_points[f"{key} grouped"], 2).items()
#                     }
#                 )

#         for fit_key, subdict in params.items():
#             if not fit_key.startswith("fit "):
#                 continue
#             parts = fit_key.split()
#             if len(parts) < 3:
#                 continue
#             fit_n = parts[1]
#             typ = parts[2]  # 'vals' or 'conditions'
#             for param, value in subdict.items():
#                 if param == "Equations":
#                     params_flt[f"fit {fit_n} Equations"] = value
#                 else:
#                     params_flt[f"fit {fit_n} {param} {typ}"] = value

#     # %% Save results
#     if run_save or run_save_full:
#         print(f"{time.ctime()}: Saving results...")

#         save(pol_points, data_file / "converted", "points", file_type="xls")

#         save(grp_points, data_file / "converted", "grouped", file_type="xls")

#         save(params_flt, data_file / "converted", "params", file_type="xls")

#     if run_save_full:
#         print(f"{time.ctime()}: Saving result arrays...")
#         for key in curves:
#             save(curves[key], data_file / "converted", f"curves_{key}", file_type="xls")

#     # #%%
#     # test_df00 = bias_prior_fits(params["fit 1 vals"], bias = 0.0)
#     # test_df05 = bias_prior_fits(params["fit 1 vals"], bias = 0.5)
#     # test_df08 = bias_prior_fits(params["fit 1 vals"], bias = 0.8)
#     # test_df10 = bias_prior_fits(params["fit 1 vals"], bias = 1.0)
