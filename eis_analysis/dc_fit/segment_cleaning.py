# -*- coding: utf-8 -*-
from typing import overload
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter


def form_savgol_kwargs(
    length: int,
    window_length: int | float = 0.05,
    polyorder: int = 2,
    delta: float = 1,
    deriv: int = 0,
    axis: int = -1,
    mode: str = "interp",
    cval: float = 0.0,
    reason: str = "",
    **_,
) -> dict:
    """
    Construct keyword arguments for Savitzky-Golay filter.

    Parameters
    ----------
    length : int
        Length of the data array.
    window_length : int or float, optional
        Window length for smoothing. If float between 0 and 1, interpreted as fraction of length.
    polyorder : int, optional
        Polynomial order for smoothing.
    delta : float or None, optional
        Spacing of samples. If None, not included.
    **kwargs
        Additional keyword arguments for savgol_filter.

    Returns
    -------
    dict
        Dictionary of keyword arguments for savgol_filter.
    """
    wl = window_length
    po = max(0, int(polyorder))
    if reason == "artifact":
        wl = 0.05  # Use small fraction for steep event detection
        po = 2
    elif reason == "fit":  # fit preparation
        wl = 0.1  # Broader smoothing for curvature preservation
        po = 3

    if not isinstance(wl, int):
        if 0 < wl < 1:
            wl = max(int(length * wl), po + 1)
        else:
            wl = int(wl)
    if wl > length:
        wl = po + 1
    if wl % 2 == 0:
        wl += 1
    return {
        "window_length": int(wl),
        "polyorder": int(po),
        "delta": float(delta),
        "deriv": int(deriv),
        "axis": int(axis),
        "mode": mode,
        "cval": float(cval),
    }


def preprocess_gradient_residual(y, **kwargs):
    sv_kwargs = form_savgol_kwargs(len(y), **kwargs)
    grad_raw = np.gradient(y)
    y_smooth = savgol_filter(y, **sv_kwargs)
    grad_smooth = np.gradient(y_smooth)
    residual = grad_raw - grad_smooth
    return residual


@lru_cache(maxsize=256)
def get_g_crit(N, alpha=0.05):
    t = stats.t.ppf(1 - alpha / (2 * N), N - 2)
    return ((N - 1) / np.sqrt(N)) * np.sqrt(t**2 / (N - 2 + t**2))


def sequential_grubbs_clip(
    arr,
    alpha: float = 0.05,
    allowed_deviations: int = 2,
    # start_region: int = 3,
    # outlier_eval: bool = False,
    # use_gradient_residual: bool = False,
    # **kwargs,
):
    """
    Iteratively removes the largest outlier (by Grubbs/Z-score) and tracks original indices.
    Optionally uses the residual of the derivative and its smoothed version for outlier detection.
    After looping, finds the largest contiguous block of removed indices from 0.
    Stops after 'allowed_deviations' consecutive non-outlier iterations.
    Returns the clipped array and the start index of the good data.
    """
    arr = np.asarray(arr)

    active_idxs = np.arange(len(arr))  # CHANGED: use active_idxs instead of mask
    removed_idxs = set()
    provisional_idxs = set()
    outlier_idx = set()
    non_outlier_count = 0
    N = len(arr)

    while N > 3:
        arr_work = arr[active_idxs]
        mean = np.mean(arr_work)
        std = np.std(arr_work, ddof=1)
        if std == 0:
            break
        scores = abs(arr_work - mean) / std
        out_idx = np.argmax(scores)
        G = scores[out_idx]
        # Grubbs critical value
        G_crit = get_g_crit(N, alpha)
        if G <= G_crit:
            non_outlier_count += 1
            if non_outlier_count >= allowed_deviations:
                removed_idxs = removed_idxs - provisional_idxs
                break
            provisional_idxs.add(active_idxs[out_idx])
        else:
            non_outlier_count = 0
            provisional_idxs = set()  # Reset provisional indices
            outlier_idx.add(active_idxs[out_idx])
        removed_idxs.add(active_idxs[out_idx])
        active_idxs = np.delete(active_idxs, out_idx)
        N -= 1

    return removed_idxs, outlier_idx


def find_first_negative_slope(arr):
    """
    Finds the index of the first negative slope (local maximum) in the array.
    Returns the array starting from that index, and the index itself.
    """
    arr = np.asarray(arr)
    diffs = np.diff(abs(arr))
    neg_idxs = np.flatnonzero(diffs < 0)
    if len(neg_idxs) == 0:
        start_idx = 0
    else:
        start_idx = neg_idxs[0]
    return start_idx


def eval_arr_diffs(arr: np.ndarray, strict: bool = True) -> tuple[int, np.ndarray]:
    """
    Evaluate array for decay and first negative slope.

    Parameters
    ----------
    arr : np.ndarray
        Input 1D array.
    strict : bool, optional
        If True, requires strictly decreasing values (default: True).
        If False, allows non-increasing (flat or decreasing).

    Returns
    -------
    dict
    """
    arr = np.asarray(arr)
    if arr.size < 2:
        return 0, np.array([])

    diffs = np.diff(arr if sum(arr > 0) / arr.size > 0.5 else -arr)
    # Pad with False at both ends to catch runs at the edges
    neg_diffs = diffs < 0 if strict else diffs <= 0
    diff_decay = np.diff(np.concatenate(([False], neg_diffs, [False])).astype(int))
    run_starts = np.where(diff_decay == 1)[0]
    run_ends = np.where(diff_decay == -1)[0]

    if len(run_starts) <= 1:
        decay_start_idx = 0
    else:
        run_lengths = run_ends - run_starts
        max_idx = np.argmax(run_lengths)
        if max_idx == 0:
            max_idx = np.where(run_lengths == run_lengths[0])[0][-1]
        decay_start_idx = run_starts[max_idx]

    return int(decay_start_idx), diffs


class ArrayDiffEvaluator:
    """
    Analyze array for decay and first negative slope.
    Computes diffs on initialization.
    """

    def __init__(self, arr: np.ndarray, start: int = 0, end: int = 0):
        """
        Parameters
        ----------
        arr : np.ndarray
            Input 1D array.
        strict : bool, optional
            If True, requires strictly decreasing values (default: True).
            If False, allows non-increasing (flat or decreasing).
        """
        arr = np.asarray(arr)
        self.arr = arr if sum(arr > 0) / arr.size > 0.5 else -arr
        self.start = int(start)
        self.end = int(end)

        if self.arr.size < 3:
            self.diffs = np.array([])
        else:
            self.diffs = np.diff(self.arr)

    @property
    def start(self) -> int:
        return self._start

    @start.setter
    def start(self, value: int):
        self._start = min(max(0, int(value)), len(self.arr) - 2)

    @property
    def end(self) -> int:
        return self._end

    @end.setter
    def end(self, value: int):
        if value <= 0:
            value = len(self.arr) - 1
        else:
            value = min(len(self.arr) - 1, int(value))

        self._end = max(self.start + 1, value)

    def neg_start_idx(self, **kwargs) -> int:
        """
        Finds the index of the first negative slope (local maximum) in the array.
        Returns the index itself.
        """
        if self.diffs.size < 1:
            return self.start
        start = kwargs.get("start", self.start)
        end = kwargs.get("end", self.end)
        diffs = self.diffs[start:end]

        neg_idxs = np.flatnonzero(diffs < 0)
        if len(neg_idxs) == 0:
            start_idx = 0
        else:
            start_idx = neg_idxs[0]
        return start_idx

    def decay_start_idx(self, strict: bool = True, **kwargs) -> int:
        """
        Evaluate array for decay and first negative slope.

        Returns
        -------
        tuple[int, np.ndarray]
            decay_start_idx, diffs
        """
        if self.diffs.size < 1:
            return self.start

        start = kwargs.get("start", self.start)
        end = kwargs.get("end", self.end)
        diffs = self.diffs[start:end]

        neg_diffs = diffs < 0 if strict else diffs <= 0
        diff_decay = np.diff(np.concatenate(([False], neg_diffs, [False])).astype(int))
        run_starts = np.where(diff_decay == 1)[0]
        run_ends = np.where(diff_decay == -1)[0]

        if len(run_starts) <= 1:
            decay_start_idx = 0
        else:
            run_lengths = run_ends - run_starts
            max_idx = np.argmax(run_lengths)
            if max_idx == 0:
                max_idx = np.where(run_lengths == run_lengths[0])[0][-1]
            decay_start_idx = run_starts[max_idx]

        return int(decay_start_idx)

    def max_limit_idx(self, threshold: float, **kwargs) -> int:
        """
        Finds the first index where all diffs from that point onward are below the given threshold.

        Parameters
        ----------
        threshold : float
            Maximum allowed absolute diff.

        Returns
        -------
        int
            Start index where all diffs are below threshold.
        """
        if self.diffs.size < 1:
            return self.start
        start = kwargs.get("start", self.start)
        end = kwargs.get("end", self.end)
        diffs = np.abs(self.diffs[start:end])

        for idx in range(len(diffs)):
            if np.all(diffs[idx:] < threshold):
                return start + idx
        return end

    def skew_limit_idx(self, threshold: float, **kwargs) -> int:
        """
        Finds the first index where the ratio of mean/median of diffs from that point onward
        is below the given threshold.

        Parameters
        ----------
        threshold : float
            Maximum allowed mean/median ratio.

        Returns
        -------
        int
            Start index where mean/median ratio is below threshold.
        """
        if self.diffs.size < 1:
            return self.start
        start = kwargs.get("start", self.start)
        end = kwargs.get("end", self.end)
        diffs = np.abs(self.diffs[start:end])

        for idx in range(len(diffs) - 1):
            if np.mean(diffs[idx:]) / np.median(diffs[idx:]) < threshold:
                return self.start + idx
        return self.end


def eval_endpoints(
    arr: np.ndarray,
    start_region: int,
    removed_idxs: set,
    outlier_idx: set,
    diff_threshold: float = 0.0,
    skew_threshold: float = 0.0,
    trim_to_decay: bool = True,
) -> tuple[int, int]:
    """
    Evaluate endpoints and determine start and stop indices for sequential_grubbs_clip.

    Parameters
    ----------
    removed_idxs : set
        Set of removed indices.
    stop_idx : int
        Current stop index.
    arr : np.ndarray
        The processed array.
    start_region : int
        Region divisor for midpoint calculation.
    outlier_idx : set
        Set of outlier indices.
    outlier_eval : bool
        Whether to use secondary outlier evaluation.

    Returns
    -------
    tuple[int, int]
        (start_idx, stop_idx)
    """
    start_idx = 0
    stop_idx = len(arr)
    start_max = len(arr) // start_region
    e_diff = ArrayDiffEvaluator(arr)
    removed_idxs.add(0)
    skew_limit = 0
    max_limit = 0

    removed_sorted = np.array(sorted(removed_idxs))
    split_points = np.where(np.diff(removed_sorted) > 1)[0] + 1
    blocks = np.split(removed_sorted, split_points)
    if (stop_idx - 1) in removed_idxs:
        block = blocks.pop()
        stop_idx = block.min()
        e_diff.end = stop_idx
    try:
        if len(blocks) == 1 and np.array_equal(blocks[0], np.array([0])):
            start_idx = 0
        else:
            start_idx = np.max([min(b.max() + 1, start_max) for b in blocks])
    except Exception:
        start_idx = 0

    dec_idx = e_diff.decay_start_idx(True, end=start_idx + start_max)
    while dec_idx in outlier_idx and dec_idx < stop_idx:
        dec_idx += 1

    if dec_idx > start_max:
        dec_idx = 0

    min_pos = min(start_idx, dec_idx)
    max_pos = max(start_idx, dec_idx)

    if diff_threshold:
        max_limit = e_diff.max_limit_idx(diff_threshold, end=stop_idx)
        if max_limit > start_max:
            max_limit = 0

    if skew_threshold:
        skew_limit = e_diff.skew_limit_idx(skew_threshold, end=stop_idx)
        if skew_limit > start_max:
            skew_limit = 0

    low_limit = max([min_pos, max_limit, skew_limit])
    start_idx = min([max_pos, low_limit, start_max])

    if trim_to_decay:
        start_idx += e_diff.neg_start_idx(start=start_idx, end=stop_idx)

    return start_idx, stop_idx

    # # starts = []
    # # decay_ref = []
    # # for b in blocks:
    # #     starts.append(b.max() + 1)
    # #     decay_ref.append(e_diff.decay_start_idx(True, end=starts[-1] + start_max))

    # ret_idx = e_diff.decay_start_idx(True, end=start_max)
    # # if not blocks:
    # #     return start_idx, stop_idx

    # # starts = [b.max() + 1 for b in blocks if b.max() + 1 < start_max]
    # # start_idx = starts[-1]

    # # ret_idx, diffs = eval_arr_diffs(arr[: (start_idx + mid_point)], True)

    # if outlier_eval:
    #     while ret_idx in outlier_idx and ret_idx < len(arr):
    #         ret_idx += 1

    # if 0 <= ret_idx < start_max:
    #     start_idx = ret_idx
    # elif blocks and start_idx > blocks[0].max() + 1:
    #     # If the start_idx is beyond the first block, adjust it
    #     start_idx = blocks[0].max() + 1

    # return start_idx, stop_idx


@overload
def trim_current_region(
    data: dict[tuple, pd.DataFrame],
    min_length: int = ...,
    trim_w_grubbs: bool = ...,
    trim_to_decay: bool = ...,
    thresholds: float | list[float] = ...,
    diff_threshold: float = ...,
    skew_threshold: float = ...,
    outlier_eval: bool = ...,
    reset_time_zero: bool = ...,
    check_exp_end: bool = ...,
    **kwargs,
) -> dict[tuple, pd.DataFrame]: ...


@overload
def trim_current_region(
    data: pd.DataFrame,
    min_length: int = ...,
    trim_w_grubbs: bool = ...,
    trim_to_decay: bool = ...,
    thresholds: float | list[float] = ...,
    diff_threshold: float = ...,
    skew_threshold: float = ...,
    outlier_eval: bool = ...,
    reset_time_zero: bool = ...,
    check_exp_end: bool = ...,
    **kwargs,
) -> pd.DataFrame: ...


def trim_current_region(
    data: dict[tuple, pd.DataFrame] | pd.DataFrame,
    min_length: int = 3,
    trim_w_grubbs: bool = False,
    trim_to_decay: bool = True,
    thresholds: float | list[float] = 0.25,
    diff_threshold: float = 0.0,
    skew_threshold: float = 0.0,
    outlier_eval: bool = False,
    reset_time_zero: bool = True,
    check_exp_end: bool = True,
    **kwargs,
) -> dict[tuple, pd.DataFrame] | pd.DataFrame:
    """
    Trim segment(s) to exponential current region.
    Operates on dict of DataFrames or a single DataFrame.

    Parameters
    ----------
    data : dict[tuple, pd.DataFrame] or pd.DataFrame
        Input data.
    min_length : int, optional
        Minimum segment length after trimming.
    trim_w_grubbs : bool, optional
        If True, use Grubbs outlier test to trim start.
    trim_to_decay : bool, optional
        If True, use find_first_negative_slope after Grubbs test.
    thresholds : float or list[float], optional
        Threshold(s) for current endpoint trimming.
    shift_end : int, optional
        Additional rows to trim from end.
    reset_time_zero : bool, optional
        If True, reset time column to zero.
    **kwargs
        Additional keyword arguments, including:
        - alpha: float, Grubbs test significance level.
        - allowed_deviations: int, Grubbs test allowed deviations.
        - check_exp_end: bool, if True, trim end using Grubbs endpoint logic.

    Returns
    -------
    dict[tuple, pd.DataFrame] or pd.DataFrame
        Trimmed data.
    """

    min_length = max(abs(min_length), 1)
    if isinstance(thresholds, (int, float)):
        thresholds = [float(thresholds)] * 2

    if isinstance(data, dict):
        return {
            idx: trim_current_region(
                df,
                min_length=min_length,
                trim_w_grubbs=trim_w_grubbs,
                trim_to_decay=trim_to_decay,
                thresholds=thresholds,
                diff_threshold=diff_threshold,
                skew_threshold=skew_threshold,
                outlier_eval=outlier_eval,
                reset_time_zero=reset_time_zero,
                check_exp_end=check_exp_end,
                **kwargs,
            )
            for idx, df in data.items()
        }

    df = data.copy()
    if len(df) >= 2 + min_length:
        if trim_w_grubbs:
            current_arr = df["Current"].to_numpy(copy=True)
            n_arr = np.asarray(current_arr) / np.nanmedian(current_arr)

            if kwargs.get("use_gradient_residual", False):
                m_arr = preprocess_gradient_residual(n_arr, **kwargs)
            else:
                m_arr = np.sign(n_arr) * np.log(np.abs(n_arr) + 1e-24)

            removed_idxs, outlier_idx = sequential_grubbs_clip(
                m_arr,
                alpha=kwargs.get("alpha", 0.05),
                allowed_deviations=kwargs.get("allowed_deviations", 2),
            )

            start, end = eval_endpoints(
                n_arr,
                kwargs.get("start_region", 3),
                removed_idxs,
                outlier_idx if outlier_eval else set(),
                diff_threshold=float(diff_threshold / np.nanmedian(current_arr)),
                skew_threshold=skew_threshold,
                trim_to_decay=trim_to_decay,
            )
            end = max(end, start + min_length) if check_exp_end else len(df)

            start = min(start, end - min_length)  # Recheck for min_length
            df = df.iloc[start:end].copy()
        else:
            start = 0
            end = len(df)
            current_diff = (
                (df["Current"] / float(np.nanmedian(df["Current"])))
                .diff()
                .bfill()
                .to_numpy(copy=True)
            )
            while end > start + min_length and current_diff[end - 1] > thresholds[1]:
                end -= 1
            end = max(end, start + min_length)
            df = df.iloc[start:end].copy()
        if reset_time_zero and "time" in df.columns and not df.empty:
            df["time"] = df["time"] - df["time"].iloc[0]
    return df


@overload
def trim_voltage_region(
    data: dict[tuple, pd.DataFrame],
    min_length: int = ...,
    shift_start: int = ...,
    shift_end: int = ...,
    thresholds: float | list[float] = ...,
    reset_time_zero: bool = ...,
) -> dict[tuple, pd.DataFrame]: ...


@overload
def trim_voltage_region(
    data: pd.DataFrame,
    min_length: int = ...,
    shift_start: int = ...,
    shift_end: int = ...,
    thresholds: float | list[float] = ...,
    reset_time_zero: bool = ...,
) -> pd.DataFrame: ...


def trim_voltage_region(
    data: dict[tuple, pd.DataFrame] | pd.DataFrame,
    min_length: int = 3,
    shift_start: int = 0,
    shift_end: int = 0,
    thresholds: float | list[float] = 0.25,
    reset_time_zero: bool = True,
) -> dict[tuple, pd.DataFrame] | pd.DataFrame:
    """
    Trim segment(s) to stable voltage region.
    Operates on dict of DataFrames or a single DataFrame.
    """
    if isinstance(thresholds, (int, float)):
        thresholds = [float(thresholds)] * 2

    if isinstance(data, dict):
        return {
            idx: trim_voltage_region(
                df, min_length, shift_start, shift_end, thresholds, reset_time_zero
            )
            for idx, df in data.items()
        }

    min_length = max(abs(min_length), 1)
    df = data.copy()
    if len(df) >= 2 + min_length:
        deviation = abs(
            df["Voltage"].to_numpy(copy=True) / np.nanmedian(df["Voltage"].to_numpy(copy=True)) - 1
        )
        start = 0
        end = len(deviation)
        while start < end - min_length and deviation[start] > thresholds[0]:
            start += 1
        start = min(start + abs(shift_start), end - min_length)
        while end > start + min_length and deviation[end - 1] > thresholds[0]:
            end -= 1
        end = max(end - abs(shift_end), start + min_length)
        df = df.iloc[start:end].copy()
        if reset_time_zero and "time" in df.columns and not df.empty:
            df["time"] = df["time"] - df["time"].iloc[0]

    return df


@overload
def smooth_segment(
    data: dict[tuple, pd.DataFrame],
    columns_to_smooth: list[str] | None = ...,
    normalize: bool = ...,
    **kwargs,
) -> dict[tuple, pd.DataFrame]: ...


@overload
def smooth_segment(
    data: pd.DataFrame,
    columns_to_smooth: list[str] | None = ...,
    normalize: bool = ...,
    **kwargs,
) -> pd.DataFrame: ...


def smooth_segment(
    data: dict[tuple, pd.DataFrame] | pd.DataFrame,
    columns_to_smooth: list[str] | None = None,
    normalize: bool = False,
    **kwargs,
) -> dict[tuple, pd.DataFrame] | pd.DataFrame:
    """
    Apply Savitzky-Golay smoothing to segment(s).
    Operates on dict of DataFrames or a single DataFrame.
    """
    if isinstance(data, dict):
        return {idx: smooth_segment(df, columns_to_smooth, **kwargs) for idx, df in data.items()}

    df = data.copy()
    valid_cols = set(df.columns) - {"time"}
    if columns_to_smooth:
        valid_cols &= set(columns_to_smooth)
    if not valid_cols:
        return df

    kwargs.setdefault("delta", float(np.diff(df["time"]).mean()))
    sv_kwargs = form_savgol_kwargs(len(df), **kwargs)

    if len(df) > sv_kwargs["window_length"] > sv_kwargs["polyorder"]:
        for col in valid_cols:
            if np.issubdtype(str(df[col].dtype), np.number) and not df[col].isna().any():
                try:
                    norm = 1
                    if normalize:
                        col_min = df[col][df[col] != 0].abs()
                        norm = (
                            10 ** int(np.floor(np.log10(col_min.min())))
                            if not col_min.empty
                            else 1
                        )
                    df[col] = savgol_filter(
                        df[col].to_numpy(copy=True) / norm,
                        **sv_kwargs,
                    )
                    df[col] = df[col] * norm
                except Exception:
                    pass
    return df


def clean_segment_data(
    standardized_dict: dict[tuple, pd.DataFrame],
    min_length: int = 3,
    shift_start: int = 0,
    shift_end: int = 0,
    stable_voltage: bool = False,
    check_exp_start: bool = False,
    trim_to_decay: bool = True,
    check_exp_end: bool = True,
    thresholds: float | list[float] = 0.25,
    alpha: float = 0.05,
    allowed_deviations: int = 2,
    reset_time_zero: bool | str = "voltage",
    **kwargs,
) -> dict[tuple, pd.DataFrame]:
    """
    Clean segment data by trimming to stable voltage and/or exponential current regions,
    with optional smoothing.
    """
    cleaned_dict = {}

    if isinstance(thresholds, (int, float)):
        thresholds = [float(thresholds)] * 2

    if isinstance(reset_time_zero, str):
        if "volt" in reset_time_zero:
            reset_v0 = True
            reset_i0 = False
        else:
            reset_v0 = True
            reset_i0 = True
    else:
        reset_v0, reset_i0 = reset_time_zero, reset_time_zero

    for idx, df in standardized_dict.items():
        df_clean = df.copy()
        # Voltage trimming
        if stable_voltage or shift_start or shift_end:
            df_clean = trim_voltage_region(
                df_clean,
                min_length=min_length,
                shift_start=shift_start,
                shift_end=shift_end,
                thresholds=thresholds,
                reset_time_zero=reset_v0,
            )
        # Current trimming
        if check_exp_start or check_exp_end:
            df_clean = trim_current_region(
                df_clean,
                min_length=min_length,
                trim_w_grubbs=check_exp_start,
                trim_to_decay=trim_to_decay,
                thresholds=thresholds,
                reset_time_zero=reset_i0,
                alpha=alpha,
                allowed_deviations=allowed_deviations,
                **kwargs,
            )
        cleaned_dict[idx] = df_clean

    return cleaned_dict


def ensure_decay(
    data_dict: dict[str, pd.DataFrame],
    min_slope: int | float = 0.0,
    ensure_point: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    Filter datasets to retain only those where abs(Current) decays (slope < 0) in the last half.

    Parameters
    ----------
    data_dict : dict[str, pd.DataFrame]
        Dictionary of DataFrames to check.

    Returns
    -------
    tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]
        Tuple of (accepted, rejected) dictionaries. Each DataFrame has 'slope' in .attrs.
    """
    records = []
    for key, df in data_dict.items():
        n = len(df)
        if n < 4:
            slope = 0.0
            accepted = False
        else:
            vals = np.abs(df["Resistance"].to_numpy(copy=True)[n // 2 :])
            slope = np.polyfit(df["time"].to_numpy(copy=True)[n // 2 :], vals, 1)[0] / np.max(vals)
            # slope = fit_vals[0] / fit_vals[1]
            accepted = slope >= min_slope
        df.attrs["end slope"] = float(slope)
        df.attrs["accepted"] = accepted
        records.append(
            {
                "sample_name": key[0],
                "condition": key[1],
                "temp": key[2],
                "polarity": np.sign(df["Voltage"].iloc[n // 2]),
                "slope": float(slope),
                "slope diff": abs(float(min_slope - slope)),
                "accepted": accepted,
            }
        )
    df_status = pd.DataFrame.from_records(records)
    if ensure_point:
        # Ensure at least one dataset per group
        for _, group_df in df_status.groupby(
            ["sample_name", "condition", "temp", "polarity"], dropna=False, observed=True
        ):
            if not group_df["accepted"].any():
                # Restore the rejected with slope closest to zero
                df_status.loc[group_df["slope diff"].idxmin(), "accepted"] = True

    # Build accepted and rejected dicts
    accepted = {}
    rejected = {}
    for idx, key in enumerate(data_dict.keys()):
        if df_status.loc[idx, "accepted"]:
            accepted[key] = data_dict[key]
        else:
            rejected[key] = data_dict[key]
    return accepted, rejected


# def filter_datasets(curves: dict, points: dict) -> tuple[dict, ...]:
#     """
#     Filter the 'cleaned' sub-dict in parent_dict using ensure_decay,
#     update 'cleaned' and 'cleaned_rej', and filter all other sub-dicts to match.

#     Parameters
#     ----------
#     parent_dict : dict
#         Dictionary containing at least a 'cleaned' key (dict of DataFrames).
#         Updates 'cleaned' and adds/updates 'cleaned_rej'.
#         All other sub-dicts are filtered to only include keys present in 'cleaned' or 'cleaned_rej'.
#     """
#     # Filter the 'cleaned' dict
#     accepted, rejected = ensure_decay(curves["cleaned"])

#     # Get keys for accepted and rejected
#     rejected_keys = set(rejected.keys())

#     # Filter all other sub-dicts to match accepted or rejected keys
#     accepted_curves = {}
#     rejected_curves = {}
#     for key, d_set in curves.items():
#         if key == "cleaned":
#             accepted_curves[key] = accepted
#             rejected_curves[key] = rejected
#             continue
#         accepted_curves[key] = {k: v for k, v in d_set.items() if k not in rejected_keys}
#         rejected_curves[key] = {k: v for k, v in d_set.items() if k in rejected_keys}

#     return accepted_curves, rejected_curves


def filter_datasets(
    curves: dict, points: dict, min_slope: int | float = 0.0, ensure_point: bool = True
) -> tuple[dict, ...]:
    """
    Filter the 'cleaned' sub-dict in parent_dict using ensure_decay,
    update 'cleaned' and 'cleaned_rej', and filter all other sub-dicts to match.
    Also filters the points dict of DataFrames to match accepted/rejected keys,
    skipping DataFrames whose index has one less level.

    Parameters
    ----------
    curves : dict
        Dictionary containing at least a 'cleaned' key (dict of DataFrames).
        Updates 'cleaned' and adds/updates 'cleaned_rej'.
        All other sub-dicts are filtered to only include keys present in 'cleaned' or 'cleaned_rej'.
    points : dict
        Dictionary of DataFrames whose index typically matches accepted/rejected keys.
        DataFrames with index of one less level are skipped.

    Returns
    -------
    tuple[dict, ...]
        Tuple of (accepted_curves, rejected_curves, accepted_points, rejected_points).
    """
    # Filter the 'cleaned' dict
    accepted, rejected = ensure_decay(curves["cleaned"], min_slope, ensure_point)

    # Get keys for accepted and rejected
    rejected_keys = set(rejected.keys())

    # Filter all other sub-dicts in curves to match accepted or rejected keys
    accepted_curves = {}
    rejected_curves = {}
    for key, d_set in curves.items():
        if key == "cleaned":
            accepted_curves[key] = accepted
            rejected_curves[key] = rejected
            continue
        accepted_curves[key] = {k: v for k, v in d_set.items() if k not in rejected_keys}
        rejected_curves[key] = {k: v for k, v in d_set.items() if k in rejected_keys}

    # Filter points dict
    accepted_points = {}
    rejected_points = {}
    for key, df in points.items():
        # If index has one less level than keys, skip filtering
        if len(df.index.names) < len(next(iter(rejected_keys), ())):
            accepted_points[key] = df
            continue
        # Otherwise, filter rows by accepted/rejected keys
        accepted_points[key] = df[~df.index.isin(rejected_keys)]
        rejected_points[key] = df[df.index.isin(rejected_keys)]

    return accepted_curves, rejected_curves, accepted_points, rejected_points


# def find_decay_start(arr: np.ndarray, strict: bool = True, invalid: set | None = None) -> int:
#     """
#     Finds the start index of the longest sequentially decaying section in an array.

#     Parameters
#     ----------
#     arr : np.ndarray
#         Input 1D array.
#     strict : bool, optional
#         If True, requires strictly decreasing values (default: True).
#         If False, allows non-increasing (flat or decreasing).

#     Returns
#     -------
#     start_idx : int
#         Start index of the longest decaying section.
#     """
#     arr = np.asarray(arr)
#     if arr.size < 2:
#         return 0

#     diffs = np.diff(arr)

#     # Pad with False at both ends to catch runs at the edges
#     decaying = np.concatenate(([False], diffs < 0 if strict else diffs <= 0, [False]))

#     diff_decay = np.diff(decaying.astype(int))
#     run_starts = np.where(diff_decay == 1)[0]
#     run_ends = np.where(diff_decay == -1)[0]

#     if len(run_starts) <= 1:
#         return 0

#     run_lengths = run_ends - run_starts
#     max_idx = np.argmax(run_lengths)
#     if max_idx == 0:
#         max_idx = np.where(run_lengths == run_lengths[0])[0][-1]

#     ret_idx = run_starts[max_idx]
#     if invalid is not None:
#         while ret_idx in invalid and ret_idx < len(arr):
#             ret_idx += 1

#     return ret_idx


# @lru_cache(maxsize=256)
# def get_g_crit(N, alpha=0.05):
#     t = stats.t.ppf(1 - alpha / (2 * N), N - 2)
#     return ((N - 1) / np.sqrt(N)) * np.sqrt(t**2 / (N - 2 + t**2))


# def sequential_grubbs_clip(
#     arr,
#     alpha: float = 0.05,
#     allowed_deviations: int = 2,
#     start_region: int = 3,
#     eval_endpoint: bool = False,
#     outlier_eval: bool = False,
#     use_gradient_residual: bool = False,
#     **kwargs,
# ):
#     """
#     Iteratively removes the largest outlier (by Grubbs/Z-score) and tracks original indices.
#     Optionally uses the residual of the derivative and its smoothed version for outlier detection.
#     After looping, finds the largest contiguous block of removed indices from 0.
#     Stops after 'allowed_deviations' consecutive non-outlier iterations.
#     Returns the clipped array and the start index of the good data.
#     """
#     arr = np.asarray(arr)
#     if use_gradient_residual:
#         arr = preprocess_gradient_residual(arr, **kwargs)
#     else:
#         arr = np.sign(arr) * np.log(np.abs(arr) + 1e-24)
#     active_idxs = np.arange(len(arr))  # CHANGED: use active_idxs instead of mask
#     removed_idxs = set()
#     provisional_idxs = set()
#     outlier_idx = set()
#     non_outlier_count = 0
#     N = len(arr)

#     while N > 3:
#         arr_work = arr[active_idxs]
#         mean = np.mean(arr_work)
#         std = np.std(arr_work, ddof=1)
#         if std == 0:
#             break
#         scores = abs(arr_work - mean) / std
#         out_idx = np.argmax(scores)
#         G = scores[out_idx]
#         # Grubbs critical value
#         G_crit = get_g_crit(N, alpha)
#         if G <= G_crit:
#             non_outlier_count += 1
#             if non_outlier_count >= allowed_deviations:
#                 removed_idxs = removed_idxs - provisional_idxs
#                 break
#             provisional_idxs.add(active_idxs[out_idx])
#         else:
#             non_outlier_count = 0
#             provisional_idxs = set()  # Reset provisional indices
#             outlier_idx.add(active_idxs[out_idx])
#         removed_idxs.add(active_idxs[out_idx])
#         active_idxs = np.delete(active_idxs, out_idx)
#         N -= 1

#     start_idx = 0
#     stop_idx = len(arr)
#     if removed_idxs and min(removed_idxs) < len(arr) // start_region:
#         blocks = []
#         if eval_endpoint and (stop_idx - 1) in removed_idxs:
#             removed_sorted = np.array(sorted(removed_idxs))
#             split_points = np.where(np.diff(removed_sorted) > 1)[0] + 1
#             blocks = np.split(removed_sorted, split_points)
#             block = blocks.pop()
#             stop_idx = block.min()

#         start_idx = (
#             blocks[-1].max() + 1
#             if blocks
#             else max([x for x in removed_idxs if x < len(arr) // start_region]) + 1
#         )
#         ret_idx = -1
#         if not use_gradient_residual:
#             outlier_idx = outlier_idx if outlier_eval else None
#             ret_idx = find_decay_start(
#                 arr[: (start_idx + len(arr)) // start_region], True, outlier_idx
#             )

#         if 0 <= ret_idx < len(arr) / start_region:
#             start_idx = ret_idx
#         elif blocks and start_idx > blocks[0].max() + 1:
#             # If the start_idx is beyond the first block, adjust it
#             start_idx = blocks[0].max() + 1
#         elif sum(np.diff(sorted(removed_idxs)) != 1) >= 2:
#             removed_sorted = np.array(sorted(removed_idxs))
#             split_points = np.where(np.diff(removed_sorted) > 1)[0] + 1
#             blocks = np.split(removed_sorted, split_points)
#             start_idx = blocks[0].max() + 1

#     return start_idx, stop_idx

# def clip_initial_gradient_outliers(y, x, alpha=0.05, max_clip=10):
#     residual = preprocess_gradient_residual(y)

#     N = len(residual)
#     active_idxs = np.arange(N)
#     clipped = []
#     counter = 0

#     for i in active_idxs[:max_clip]:  # Only consider early region
#         subset = residual[i : i + 5]  # Small slice
#         std = np.std(subset, ddof=1)
#         mean = np.mean(subset)
#         if std == 0:
#             continue
#         G = abs(residual[i] - mean) / std
#         G_crit = get_g_crit(len(subset), alpha)

#         if G > G_crit:
#             clipped.append(i)
#         else:
#             break  # Stop at first good point

#     return clipped


# def find_decay_start(arr: np.ndarray, strict: bool = True) -> tuple:
#     """
#     Finds the start index of the longest sequentially decaying section in an array.

#     Parameters
#     ----------
#     arr : np.ndarray
#         Input 1D array.
#     strict : bool, optional
#         If True, requires strictly decreasing values (default: True).
#         If False, allows non-increasing (flat or decreasing).

#     Returns
#     -------
#     start_idx : int
#         Start index of the longest decaying section.
#     """
#     arr = np.asarray(arr)
#     if arr.size < 2:
#         return 0, 0

#     diffs = np.diff(arr)

#     # Pad with False at both ends to catch runs at the edges
#     decaying = np.concatenate(([False], diffs < 0 if strict else diffs <= 0, [False]))

#     diff_decay = np.diff(decaying.astype(int))
#     run_starts = np.where(diff_decay == 1)[0]
#     run_ends = np.where(diff_decay == -1)[0]

#     if len(run_starts) <= 1:
#         return 0, 0

#     run_lengths = run_ends - run_starts
#     max_idx = np.argmax(run_lengths)
#     if max_idx == 0:
#         max_idx = np.where(run_lengths == run_lengths[0])[0][-1]

#     ret_idx = run_starts[max_idx]

#     return (
#         ret_idx,
#         np.max(abs(diffs)),
#         np.mean(abs(diffs)),
#         np.median(abs(diffs)),
#         np.min(abs(diffs)),
#     )
