# -*- coding: utf-8 -*-
import time
from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd

from eis_analysis.system_utilities import (
    load_file,  # noqa: F401
)
from eis_analysis.dc_fit.extract_tools import (
    DEFAULT_DIR,
    save_results,  # noqa: F401
)
from eis_analysis.dc_fit.column_parsing import (
    apply_col_mapping,
    evaluate_segments,
)
from eis_analysis.dc_fit.segment_cleaning import (
    smooth_segment,
    filter_datasets,
    clean_segment_data,
)
from eis_analysis.dc_fit.periodicity_detection import (
    process_segments_mean,
)

np.seterr(invalid="raise")


# %%
def test_clean_segment_data(
    curves: dict,
    results: pd.DataFrame | None = None,
    col_name: str = "",
    **clean_args,
) -> pd.DataFrame:
    """
    Evaluate the impact of clean_segment_data parameters on segment lengths.

    Parameters
    ----------
    curves : dict
        Dictionary containing curve data, must include "base".
    results : pd.DataFrame or None, optional
        Existing results dataframe to append to.
    col_name : str or None, optional
        Name of the column for this run. If None, auto-generates as "run {n}".
    **clean_args
        Arguments to pass to clean_segment_data.

    Returns
    -------
    pd.DataFrame
        DataFrame with index as curves["base"] keys and columns for base and cleaned lengths.
    """

    # If no results, create new DataFrame
    if results is None:
        results = pd.DataFrame(index=curves["base"].keys())  # type: ignore
        # results["base len"] = [len(df) for df in curves["base"].values()]
    # Generate column name if not provided
    if not col_name:
        col_name = f"run {len(results.columns)}"
    # Run cleaning and collect lengths
    cleaned = clean_segment_data(curves["base"], **clean_args)
    # results[col_name] = [len(df) for df in cleaned.values()]
    results[col_name] = [df.iloc[0, 0] for df in cleaned.values()]
    return results


# %%
def pre_process(
    data_path: str | Path,
    file_list: list[str],
    skip_list: list[str] | tuple[str, ...] = (),
    stable_voltage: bool = True,
    check_exp_start: bool = True,
    check_exp_end: bool = True,
    alpha: float = 0.05,
    smooth: bool = False,
    # window_length: float = 0.05,
    # polyorder: int = 1,
    min_slope: int | float = 0.0,
    ensure_point: bool = True,
    verbose: bool = True,
    **kwargs: Any,
) -> tuple[dict, ...]:
    """
    Pre-process DC data and save results.

    Parameters
    ----------
    data_path : str
        Path to the data directory.
    file_list : Sequence[str]
        List of filenames to load.
    skip_list : Sequence[str]
        List of sample names to skip in process_segments_mean.
    """
    data_file = Path(data_path)

    loaded_data = {}
    points = {}
    curves = {}

    if verbose:
        print(f"{time.ctime()}: Loading data...")
    for file in file_list:
        loaded_data.update(load_file(data_file / file)[0])
    if verbose:
        print(f"{time.ctime()}: Segmenting data...")
    points["raw_means"], curves["raw"], curves["raw_full"] = process_segments_mean(
        loaded_data,
        *skip_list,
        inclusive=False,
    )

    if verbose:
        print(f"{time.ctime()}: Parsing data...")
    results = evaluate_segments(points["raw_means"])
    points["means"] = results[0].copy()
    points["updated means"] = results[0].copy(deep=True)

    if verbose:
        print(f"{time.ctime()}: Building segmented DataFrame...")
    curves["base"] = apply_col_mapping(curves["raw"], results[0], results[1])

    if verbose:
        print(f"{time.ctime()}: Building full DataFrame...")
    curves["full"] = apply_col_mapping(curves["raw_full"], results[0], results[1])

    if verbose:
        print(f"{time.ctime()}: Cleaning segment data...")
    curves["trimmed"] = clean_segment_data(
        curves["base"],
        stable_voltage=stable_voltage,
        check_exp_start=check_exp_start,
        check_exp_end=check_exp_end,
        alpha=alpha,
        # window_length=window_length,
        # polyorder=polyorder,
        **kwargs,
    )
    if smooth:
        if verbose:
            print(f"{time.ctime()}: Smoothing curves...")
        curves["cleaned"] = smooth_segment(
            curves["trimmed"],
            # window_length=window_length,
            # polyorder=polyorder,
            **kwargs.get("sm_kwargs", kwargs),
        )
    else:
        curves["cleaned"] = curves["trimmed"].copy()

    if verbose:
        print(f"{time.ctime()}: Filtering slopes...")
    return filter_datasets(curves, points, min_slope, ensure_point)


sm_kwargs = {
    "columns_to_smooth": ["Current"],
    "window_length": 0.05,
    "polyorder": 1,
    "normalize": False,
}

pre_process_kwargs = {}
pre_process_kwargs |= dict(
    data_path=DEFAULT_DIR.parent,
    file_list=["converted_data.xlsx", "converted_100_and_301.xlsx"],
    skip_list=[
        "cln2dh80c_r1",
        "cln2dh80c_r2",
        "cln2dh85c_r1",
        "cln2dh85c_r2",
        "cln2dh85c_r3",
        "300dry80c_r1",
    ],  # 300dry80c_r1
    min_slope=-1e-6,
    # outlier_eval=True,
    use_gradient_residual=True,
    reason="artifact",
    start_region=4,
    smooth=True,
    sm_kwargs=sm_kwargs,
    ensure_point=True,
    skew_threshold=3,
    diff_threshold=5e-12,
)


if __name__ == "__main__":
    # Default usage

    curves, rej_curves, points, rej_points = pre_process(
        **pre_process_kwargs,
    )

    # %% Save
    # save_results(
    #     DEFAULT_DIR,
    #     points=points,
    #     save_mode="a",
    #     attrs=True,
    # )
    save_results(
        DEFAULT_DIR,
        points=points,
        curves=curves,
        save_mode="w",
        attrs=True,
    )

    # # %% test cleaning
    # kwargs = {
    #     "stable_voltage": True,
    #     "check_exp_start": True,
    #     "check_exp_end": True,
    #     "alpha": 0.05,
    #     "smooth": False,
    #     "window_length": 0.05,
    #     "polyorder": 1,
    #     "start_region": 4,
    #     # "skew_threshold": 3,
    #     # "diff_threshold": 1e-12,
    # }
    # df = test_clean_segment_data(curves, **kwargs)
    # df = test_clean_segment_data(curves, df, "outlier", outlier_eval=True, **kwargs)
    # df = test_clean_segment_data(curves, df, "skew", skew_threshold=3, **kwargs)
    # df = test_clean_segment_data(curves, df, "diff", diff_threshold=1e-12, **kwargs)
    # df = test_clean_segment_data(curves, df, "skew diff", skew_threshold=3, diff_threshold=5e-12, **kwargs)

    # df = test_clean_segment_data(curves, df, "resid_1", use_gradient_residual=True, **kwargs)
    # df = test_clean_segment_data(
    #     curves, df, "resid_2", use_gradient_residual=True, reason="artifact", **kwargs
    # )
    # df = test_clean_segment_data(
    #     curves, df, "resid_skew", use_gradient_residual=True, reason="artifact", skew_threshold=3, **kwargs
    # )
    # df = test_clean_segment_data(
    #     curves, df, "resid_diff", use_gradient_residual=True, reason="artifact", diff_threshold=5e-12, **kwargs
    # )
    # df = test_clean_segment_data(
    #     curves, df, "resid_skew_diff", use_gradient_residual=True, reason="artifact", skew_threshold=3, diff_threshold=5e-12, **kwargs
    # )
    # # df = test_clean_segment_data(
    # #     curves,
    # #     df,
    # #     "resid_3",
    # #     use_gradient_residual=True,
    # #     **{**kwargs, "window_length": 0.3},
    # # )
    # # df = test_clean_segment_data(
    # #     curves,
    # #     df,
    # #     "resid_4",
    # #     use_gradient_residual=True,
    # #     **{**kwargs, "polyorder": 3},
    # # )
    # # df = test_clean_segment_data(
    # #     curves,
    # #     df,
    # #     "resid_1 outlier",
    # #     use_gradient_residual=True,
    # #     outlier_eval=True,
    # #     **kwargs,
    # # )
