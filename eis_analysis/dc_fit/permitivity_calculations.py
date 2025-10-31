# -*- coding: utf-8 -*-
import re
import warnings
from typing import Any
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, find_peaks, get_window  # peak_widths
from numpy.exceptions import RankWarning
from scipy.interpolate import CubicSpline, PchipInterpolator

from eis_analysis.dc_fit.extract_tools import (
    DEFAULT_KEYS,
    form_std_df_index,
)
from eis_analysis.dc_fit.segment_fitting import (
    get_fit_func,
)
from eis_analysis.dc_fit.segment_cleaning import (
    smooth_segment,
)

warnings.filterwarnings("ignore", category=RankWarning)


VACUUM_PERMITTIVITY = 8.854187817e-14  # F/cm


def remove_low_freq_baseline(
    arr: np.ndarray, fs: float, cutoff: float = 0.01, order: int = 2
) -> np.ndarray:
    """
    Remove low-frequency baseline using a Butterworth high-pass filter.

    Parameters
    ----------
    arr : np.ndarray
        Input signal.
    fs : float
        Sampling frequency (Hz).
    cutoff : float
        Cutoff frequency for high-pass filter (Hz).
    order : int
        Filter order.

    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    b, a = butter(order, cutoff / (0.5 * fs), btype="high")  # type: ignore
    return filtfilt(b, a, arr)


def find_equation_row(col: str, eq_df: pd.DataFrame) -> tuple[Callable, str]:
    # Try direct match
    eqn = ""
    if col in eq_df.index:
        eqn = str(eq_df.loc[col, "Eqn Name"])  # type: ignore
    # Try substring match
    for idx in eq_df.index:
        if col in idx or idx in col:
            eqn = str(eq_df.loc[idx, "Eqn Name"])
            break
    if not eqn:
        raise KeyError(f"Column '{col}' not found in equation_df index or as substring.")
    func, _ = get_fit_func(eqn)
    return func, eqn


def align_arr(
    arr: np.ndarray,
    extension: np.ndarray,
    n_points: int,
    align_pts: int = 3,
) -> np.ndarray:
    """
    Align and extend an array with a simulated extension for smooth transition.

    Parameters
    ----------
    arr : np.ndarray
        The original array to be extended.
    extension : np.ndarray
        The array to use for extension (e.g., simulated fit data).
    n_points : int
        The length of the original array (where extension starts).
    align_pts : int, optional
        Number of points at the end of arr to use for mean alignment (default: 3).

    Returns
    -------
    np.ndarray
        The concatenated array, with extension shifted to align with arr.
    """
    if len(extension) <= n_points:
        return arr
    ext_aligned = extension[n_points:] + (np.mean(arr[-align_pts:]) - extension[n_points])
    return np.concatenate([arr, ext_aligned])


def val_round(value: float, decimals: int = 8) -> float:
    """Round a float to a specified number of decimals to reduce floating point errors."""
    if not value:
        return 0.0
    return float(np.round(value, decimals=decimals - int(np.log10(abs(value)))))


def arr_round(arr: np.ndarray | float, decimals: int = 8) -> np.ndarray | float:
    """
    Round a NumPy array to a specified number of decimals.

    Parameters
    ----------
    arr : ArrayLike
        Input array to round.
    decimals : int, optional
        Number of decimal places to round to (default: 8).

    Returns
    -------
    np.ndarray
        Rounded array.
    """
    med = abs(np.median(arr))
    if med == 0:
        return np.round(arr, decimals=decimals)
    return np.round(arr, decimals=decimals - int(np.log10(med)))


def time_freq_translate(
    *,
    dt: float = 0.0,
    t_min: float = 0.0,
    t_max: float = 0.0,
    f_min: float = 0.0,
    f_max: float = 0.0,
    shift_freq: float = 0.0,
    shift_time: float = 1.0,
) -> dict[str, float]:
    """
    Translate between time-domain (dt, t_max) and frequency-domain (f_min, f_max) parameters,
    with optional shifting of f_min/t_max by a number of decades.

    Parameters
    ----------
    dt : float
        Time step in seconds.
    t_min : float
        Start time in seconds.
    t_max : float
        Final time (duration) in seconds.
    f_min : float
        Minimum frequency in Hz.
    f_max : float
        Maximum frequency in Hz.
    shift_freq : float, optional
        Number of decades to shift f_min/f_max (default: 0.0).
    shift_time : float, optional
        Multiplier applied to modify dt/t_max (default: 1.0).

    Returns
    -------
    dict
        Dictionary with keys: 'dt', 't_max', 'f_min', 'f_max'.
    """
    # Compute missing values if possible
    if shift_time <= 0:
        raise ValueError("shift_time must be greater than 0.")

    # f_min = 1.0 / t_max
    if t_max:
        f_min = 1.0 / ((t_max - t_min) * shift_time) * 10**shift_freq
        t_max = 1.0 / f_min + t_min
    elif f_min:
        t_max = 1.0 / (f_min * 10**shift_freq) * shift_time + t_min
        f_min = 1.0 / t_max + t_min

    # f_max = 1.0 / (2.0 * dt)
    if dt:
        f_max = 1.0 / (2.0 * (dt * shift_time)) * 10**shift_freq
        dt = 1.0 / (2.0 * f_max)
    elif f_max:
        dt = 1.0 / (2.0 * (f_max * 10**shift_freq)) * shift_time
        f_max = 1.0 / (2.0 * dt)

    result = {
        "dt": val_round(dt, 3),
        "t_max": val_round(t_max),
        "f_min": val_round(f_min),
        "f_max": val_round(f_max),
    }
    return result


def decay_from_data(
    df: pd.DataFrame,
    current_col: str = "Current",
    voltage_col: str = "Voltage",
    time_col: str = "time",
    dims_cm: tuple[float, float] = (25.0, 0.045),
    dt_function: Callable[[np.ndarray], float] | None = None,
    pre_spline: str = "cubic",
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Transform current/voltage decay data to time-domain and frequency-domain permittivity.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns for time, current, and voltage.
    current_col : str, optional
        Name of the current column (default: "Current").
    voltage_col : str, optional
        Name of the voltage column (default: "Voltage").
    time_col : str, optional
        Name of the time column (default: "time").
    dims_cm : tuple of float, optional
        Sample geometry as (area_cm2, thickness_cm) (default: (25.0, 0.045)).
    dt_function : Callable or None, optional
        Function to compute dt from np.diff(time_array). Should accept the diff array and return a float.
        If None, defaults to np.median.
    pre_spline : str, optional
        Spline type for interpolation ("cubic" for CubicSpline, "pchip" for PchipInterpolator).
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - time, voltage, current, conductivity, decay_raw
    """
    kwargs.setdefault("extrapolate", True)
    kwargs.setdefault("bc_type", "natural")
    # --- Calculate decay implementation ---
    if dims_cm[1] >= 50:
        dims_cm = (dims_cm[0], float(dims_cm[1]) * 1e-4)

    # Use smallest dt for uniform grid
    t = df[time_col].to_numpy(copy=True)
    # Use CubicSpline for interpolation and optional extrapolation
    if pre_spline == "pchip":
        interp_current = PchipInterpolator(
            t, df[current_col].to_numpy(copy=True), extrapolate=kwargs.get("extrapolate", True)
        )
        interp_voltage = PchipInterpolator(
            t, df[voltage_col].to_numpy(copy=True), extrapolate=kwargs.get("extrapolate", True)
        )
    else:
        interp_current = CubicSpline(
            t,
            df[current_col].to_numpy(copy=True),
            extrapolate=kwargs.get("extrapolate", True),
            bc_type=kwargs.get("bc_type", "natural"),
        )
        interp_voltage = CubicSpline(
            t,
            df[voltage_col].to_numpy(copy=True),
            extrapolate=kwargs.get("extrapolate", True),
            bc_type=kwargs.get("bc_type", "natural"),
        )

    diffs = np.diff(t)
    if dt_function is None:
        dt = float(np.median(diffs))
    else:
        dt = dt_function(diffs)
    # dt = val_round(dt, 3)
    # max_dt = kwargs.get("max_dt", time_freq_translate(f_max=kwargs.get("min_f_max", 1))["dt"])
    min_dt = kwargs.get(
        "min_dt", time_freq_translate(t_min=t[0], f_max=kwargs.get("max_f_max", 100))["dt"]
    )
    # min_t_end = kwargs.get(
    #     "min_t_end", time_freq_translate(f_min=kwargs.get("max_f_min", 1e-2))["t_max"]
    # )
    max_t_end = kwargs.get(
        "max_t_end", time_freq_translate(t_min=t[0], f_min=kwargs.get("min_f_min", 1e-5))["t_max"]
    )
    dt = val_round(max(dt, min_dt), 3)
    t_end = np.round(min(t[-1], max_t_end))

    uniform_t = np.linspace(t[0], t_end, int(np.round((t_end - t[0]) / dt)) + 1)
    uniform_c = interp_current(uniform_t) / float(dims_cm[0])
    uniform_v = interp_voltage(uniform_t) / float(dims_cm[1])
    conductivity = uniform_c / uniform_v

    result = pd.DataFrame(
        {
            "time": uniform_t,
            "E field": uniform_v,
            "flux": uniform_c,
            "conductivity": conductivity,
            "decay_raw": conductivity / VACUUM_PERMITTIVITY,
        }
    )
    result.attrs |= df.attrs.copy()
    result.attrs["dt"] = dt

    return result


def decay_from_fit(
    params_current: pd.Series | list[float],
    params_voltage: pd.Series | list[float],
    equation_df: pd.DataFrame,
    current_col: str = "Current",
    voltage_col: str = "Voltage",
    time_array: np.ndarray | None = None,
    dims_cm: tuple[float, float] = (25.0, 0.045),
    dt: float = np.inf,
    t_end: float = 0.0,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Generate simulated decay data from fit parameters and equation mapping.

    Parameters
    ----------
    params_current : pd.Series or list
        Fit parameters for the current curve.
    params_voltage : pd.Series or list
        Fit parameters for the voltage curve.
    equation_df : pd.DataFrame
        DataFrame mapping column names to 'Equation' and 'Eqn Name'.
    current_col : str, optional
        Name of the current column (default: "Current").
    voltage_col : str, optional
        Name of the voltage column (default: "Voltage").
    time_array : np.ndarray or None, optional
        Time array to evaluate the fit function on. If None, generated from dt/n_points.
    dims_cm : tuple of float, optional
        Sample geometry as (area_cm2, thickness_cm).
    dt : float or None, optional
        Time step for simulation. Used if time_array is None.
    n_points : int or None, optional
        Number of points for simulation. Used if time_array is None.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - time, voltage, current, conductivity, decay_raw
    """

    fit_func_current, eqn_name_current = find_equation_row(current_col, equation_df)
    fit_func_voltage, eqn_name_voltage = find_equation_row(voltage_col, equation_df)

    # Convert parameters, filtering out "_std" and "Error" if Series
    timestamp = None
    if isinstance(params_current, pd.Series):
        c_params = np.asarray([params_current[k] for k in params_current.index if k[-1].isdigit()])
        timestamp = params_current.get("timestamp", timestamp)
    else:
        c_params = np.asarray(params_current)
    if isinstance(params_voltage, pd.Series):
        v_params = np.asarray([params_voltage[k] for k in params_voltage.index if k[-1].isdigit()])
        timestamp = params_voltage.get("timestamp", timestamp)
    else:
        v_params = np.asarray(params_voltage)

    dt = abs(dt)
    t_start = kwargs.get("t_start", 0.0)
    t_end = abs(t_end)
    max_dt = kwargs.get("max_dt", time_freq_translate(f_max=kwargs.get("min_f_max", 1))["dt"])
    min_dt = kwargs.get("min_dt", time_freq_translate(f_max=kwargs.get("max_f_max", 10))["dt"])
    min_t_end = kwargs.get(
        "min_t_end",
        time_freq_translate(t_min=t_start, f_min=kwargs.get("max_f_min", 1e-2))["t_max"],
    )
    max_t_end = kwargs.get(
        "max_t_end",
        time_freq_translate(t_min=t_start, f_min=kwargs.get("min_f_min", 1e-5))["t_max"],
    )

    if time_array is not None:
        t = np.asarray(time_array)
        t_end = t_end or t[-1]
        t_start = t_start if t_start and t_start < t_end else t[0]
        dt = dt if dt < (t_end - t_start) else float(np.median(np.diff(t)))
    elif t_end - t_start <= dt:
        taus = np.array(
            [c_params[i] for i in range(1, max(2, len(c_params) - 1), 2) if c_params[i] > 0]
        )
        if taus.size > 0:
            dt = min(np.min(taus) / 10, dt)
            t_end = max((7 if len(taus) == 1 else 5) * np.max(taus), t_end)

    dt = val_round(max(min(max_dt, dt), min_dt), 3)
    t_end = min(max(min_t_end, np.round(t_end)), max_t_end)
    t_start = 0.0 if t_end - t_start <= dt else t_start
    uniform_t = np.linspace(t_start, t_end, int(np.round((t_end - t_start) / dt)) + 1)

    if dims_cm[1] >= 50:
        dims_cm = (dims_cm[0], float(dims_cm[1]) * 1e-4)

    # Simulate current and voltage curves
    uniform_c = fit_func_current(uniform_t, *c_params) / float(dims_cm[0])
    uniform_v = fit_func_voltage(uniform_t, *v_params) / float(dims_cm[1])

    conductivity = uniform_c / uniform_v

    result = pd.DataFrame(
        {
            "time": uniform_t,
            "E field": uniform_v,
            "flux": uniform_c,
            "conductivity": conductivity,
            "decay_raw": conductivity / VACUUM_PERMITTIVITY,
        }
    )

    result.attrs = {
        "dt": dt,
        "fit_func_current": eqn_name_current,
        "fit_func_voltage": eqn_name_voltage,
        "params_current": c_params,
        "params_voltage": v_params,
    }
    if timestamp is not None:
        result.attrs["timestamp"] = timestamp

    return result


def prepare_decay_df(
    df: pd.DataFrame | None = None,
    params_current: pd.Series | list[float] | None = None,
    params_voltage: pd.Series | list[float] | None = None,
    equation_df: pd.DataFrame | None = None,
    current_col: str = "Current",
    voltage_col: str = "Voltage",
    time_col: str = "time",
    shift_freq: float | int = -1,
    dims_cm: tuple[float, float] = (25.0, 0.045),
    decay_mod: dict | None = None,
    window_type: str | tuple | None = None,
    remove_mean: int | float | Callable | str | None = 0.075,
    padding: bool = True,
    # filter_cutoff: float = 0.0,
    # filter_order: int = 2,
    dt_function: Callable[[np.ndarray], float] | None = None,
    pre_spline: str = "cubic",
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Prepare decay DataFrame by calling decay_from_data and/or decay_from_fit,
    then apply filtering, mean removal, and padding.

    Parameters
    ----------
    df : pd.DataFrame or None
        DataFrame with columns for time, current, and voltage.
    params_current : pd.Series or list, optional
        Fit parameters for the current curve.
    params_voltage : pd.Series or list, optional
        Fit parameters for the voltage curve.
    equation_df : pd.DataFrame, optional
        DataFrame mapping column names to 'Equation' and 'Eqn Name'.
    current_col : str, optional
        Name of the current column (default: "Current").
    voltage_col : str, optional
        Name of the voltage column (default: "Voltage").
    time_col : str, optional
        Name of the time column (default: "time").
    dims_cm : tuple of float, optional
        Sample geometry as (area_cm2, thickness_cm) (default: (25.0, 0.045)).
    decay_mod : dict or None, optional
        If True, apply gradient and smoothing to decay signal (default: True).
    window_type : str, tuple, or None, optional
        Window type for get_window (default: ("tukey", 0.5)).
    remove_mean : int, float, Callable, str, or None, optional
        Whether to subtract the steady-state mean before FFT (default: 0.075).
        If "b0", use b0 from fit parameters.
    padding : bool, optional
        Whether to zero-pad to next power of two for FFT (default: True).
        If "fit_extend", use simulated fit to extend data.
    filter_cutoff : float, optional
        Cutoff frequency for high-pass filter (Hz, default: 0.01).
    filter_order : int, optional
        Filter order for high-pass filter (default: 2).
    dt_function : Callable or None, optional
        Function to compute dt from np.diff(time_array). Should accept the diff array and return a float.
        If None, defaults to np.median.
    pre_spline : str, optional
        Spline type for interpolation ("cubic" for CubicSpline, "pchip" for PchipInterpolator).
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - time, voltage, current, decay_raw, decay, ss decay
    """
    # Determine which decay function(s) to call
    decay_df = pd.DataFrame()
    fit_df = pd.DataFrame()

    # If df is provided, call decay_from_data
    if df is not None:
        decay_df = decay_from_data(
            df,
            current_col=current_col,
            voltage_col=voltage_col,
            time_col=time_col,
            dims_cm=dims_cm,
            dt_function=dt_function,
            pre_spline=pre_spline,
            **kwargs,
        )

    # If fit args are provided, call decay_from_fit
    if params_current is not None and params_voltage is not None and equation_df is not None:
        # If df is provided, use its time column for alignment/padding
        time_array = None
        dt = np.inf
        t_end = 0.0
        if not decay_df.empty:
            time_array = decay_df["time"].to_numpy(copy=True)
            dt = decay_df.attrs.get("dt", np.inf)
            t_end = time_freq_translate(
                t_min=time_array[0], t_max=time_array[-1], shift_freq=shift_freq
            )["t_max"]

        fit_df = decay_from_fit(
            params_current,
            params_voltage,
            equation_df,
            current_col=current_col,
            voltage_col=voltage_col,
            time_array=time_array,
            dims_cm=dims_cm,
            dt=dt,
            t_end=t_end,
            **kwargs,
        )
        # if decay_df.empty:
        #     res_df = fit_df

    if decay_df.empty and fit_df.empty:
        raise ValueError("Either df or params_current/params_voltage must be provided.")

    res_df = decay_df if not decay_df.empty else fit_df
    res_df.attrs |= fit_df.attrs

    # Filtering and padding only apply to decay_df
    uniform_d = res_df["decay_raw"].to_numpy(copy=True)
    uniform_t = res_df["time"].to_numpy(copy=True)
    uniform_v = res_df["E field"].to_numpy(copy=True)
    uniform_c = res_df["flux"].to_numpy(copy=True)
    uniform_s = res_df["conductivity"].to_numpy(copy=True)
    n_points = len(uniform_d)
    dt = res_df.attrs.get("dt", val_round(float(np.median(np.diff(uniform_t)))))

    # # Filtering
    # if filter_cutoff > 0:
    #     fs = 1.0 / dt
    #     uniform_d = remove_low_freq_baseline(
    #         uniform_d, fs, cutoff=filter_cutoff, order=filter_order
    #     )

    # Decay modification
    if not decay_df.empty and decay_mod is not None:
        if decay_mod.get("mode", "interp") == "gradient":
            uniform_d = np.gradient(
                uniform_d, uniform_t, edge_order=decay_mod.get("edge_order", 1)
            )
        else:
            decay_mod["delta"] = dt
            decay_mod.setdefault("deriv", 0)
            decay_mod.setdefault("polyorder", 2)
            decay_mod.setdefault("window_length", 3)
            uniform_d = smooth_segment(
                pd.DataFrame({"time": uniform_t, "decay_raw": uniform_d}), **decay_mod
            )["decay_raw"].to_numpy(copy=True)

    # Mean removal
    final_mean = 0.0
    if not fit_df.empty and remove_mean == "b0":
        # Use b0 from fit parameters (last value in param arrays)
        final_mean = (fit_df.attrs["params_current"][-1] / float(dims_cm[0])) / (
            VACUUM_PERMITTIVITY * (fit_df.attrs["params_voltage"][-1] / float(dims_cm[1]))
        )
        uniform_d = uniform_d - final_mean
    elif callable(remove_mean):
        final_mean = remove_mean(uniform_d)
        uniform_d = uniform_d - final_mean
    elif isinstance(remove_mean, int) and remove_mean < 0:
        final_mean = np.nanmean(uniform_d[remove_mean:])
        uniform_d = uniform_d - final_mean
    elif isinstance(remove_mean, (int, float)):
        perc = remove_mean if 0 < remove_mean < 1 else abs(remove_mean) / 100.0
        final_mean = np.nanmean(uniform_d[-max(1, int(n_points * perc)) :])
        uniform_d = uniform_d - final_mean

    if window_type is not None:
        uniform_d = get_window(window_type, n_points) * uniform_d

    # Padding
    if not decay_df.empty and padding and final_mean:
        if len(fit_df) > n_points:
            uniform_t = fit_df["time"].to_numpy(copy=True)
            uniform_d = align_arr(uniform_d, fit_df["decay_raw"].to_numpy(copy=True), n_points)
            uniform_v = align_arr(uniform_v, fit_df["E field"].to_numpy(copy=True), n_points)
            uniform_c = align_arr(uniform_c, fit_df["flux"].to_numpy(copy=True), n_points)
            uniform_s = align_arr(uniform_s, fit_df["conductivity"].to_numpy(copy=True), n_points)
        else:
            pad_len = 2 ** int(np.ceil(np.log2(n_points))) - n_points
            uniform_d = np.pad(uniform_d, (0, pad_len), "constant")
            uniform_t = np.linspace(uniform_t[0], uniform_t[-1] + pad_len * dt, n_points + pad_len)
            uniform_v = np.pad(uniform_v, (0, pad_len), "median")
            uniform_c = np.pad(
                uniform_c, (0, pad_len), mode="median", stat_length=max(1, int(0.25 * n_points))
            )
            uniform_s = np.pad(
                uniform_s,
                (0, pad_len),
                mode="median",
                stat_length=max(1, int(0.25 * n_points)),
            )

    result = pd.DataFrame(
        {
            "time": uniform_t,
            "E field": uniform_v,
            "flux": uniform_c,
            "conductivity": uniform_s,
            "decay": uniform_d,
            "ss decay": final_mean,
        }
    )
    result.attrs = {
        "ss decay": final_mean,
        "DC conductivity": final_mean * VACUUM_PERMITTIVITY,
    }
    if not decay_df.empty:
        result.attrs |= decay_df.attrs
    if not fit_df.empty:
        result.attrs |= fit_df.attrs
    return result


def transform_to_permittivity(
    df: pd.DataFrame | None = None,
    current_col: str = "Current",
    voltage_col: str = "Voltage",
    time_col: str = "time",
    dims_cm: tuple[float, float] = (25.0, 0.045),
    decay_mod: dict | None = None,
    window_type: str | tuple | None = None,
    remove_mean: int | float | Callable | None = 0.075,
    padding: bool = True,
    logspace_freq: bool = True,
    equal_length: bool = True,
    # filter_cutoff: float = 0.0,
    # filter_order: int = 2,
    dt_function: Callable[[np.ndarray], float] | None = None,
    pre_spline: str = "cubic",
    post_spline: str = "cubic",
    max_points: int = 500,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Transform current/voltage decay data to time-domain and frequency-domain permittivity.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns for time, current, and voltage.
    current_col : str, optional
        Name of the current column (default: "Current").
    voltage_col : str, optional
        Name of the voltage column (default: "Voltage").
    time_col : str, optional
        Name of the time column (default: "time").
    dims_cm : tuple of float, optional
        Sample geometry as (area_cm2, thickness_cm) (default: (25.0, 0.045)).
    decay_mod : dict or None, optional
        If True, apply gradient and smoothing to decay signal (default: True).
    window_type : str, tuple, or None, optional
        Window type for get_window (default: ("tukey", 0.5)).
    remove_mean : int, float, Callable, or None, optional
        Whether to subtract the steady-state mean before FFT (default: 0.075).
    padding : bool, optional
        Whether to zero-pad to next power of two for FFT (default: True).
    logspace_freq : bool, optional
        Whether to use logspace for frequency axis (default: True).
    equal_length : bool, optional
        Whether to spline FFT result to output frequency axis (default: True).
    filter_cutoff : float, optional
        Cutoff frequency for high-pass filter (Hz, default: 0.01).
    filter_order : int, optional
        Filter order for high-pass filter (default: 2).
    dt_function : Callable or None, optional
        Function to compute dt from np.diff(time_array). Should accept the diff array and return a float.
        If None, defaults to np.median.
    pre_spline : str, optional
        Spline type for interpolation ("cubic" for CubicSpline, "pchip" for PchipInterpolator).
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - time, voltage, current, decay_raw, decay, ss decay
        - frequency, permittivity, real perm, imag perm, mag perm, phase perm, loss tangent
    """
    # Prepare decay and related columns using the existing function
    decay_df = prepare_decay_df(
        df,
        current_col=current_col,
        voltage_col=voltage_col,
        time_col=time_col,
        dims_cm=dims_cm,
        decay_mod=decay_mod,
        window_type=window_type,
        remove_mean=remove_mean,
        padding=padding,
        # filter_cutoff=filter_cutoff,
        # filter_order=filter_order,
        dt_function=dt_function,
        pre_spline=pre_spline,
        **kwargs,
    )

    # Use the decay and time arrays from the prepared DataFrame
    uniform_d = decay_df["decay"].to_numpy(copy=True)
    uniform_t = decay_df["time"].to_numpy(copy=True)
    n_points = len(uniform_d)
    dt = decay_df.attrs.get("dt", np.median(np.diff(uniform_t)))

    # FFT and frequency axis
    perm_fft = np.asarray(fft(uniform_d))
    freq = fftfreq(n_points, d=dt)

    pos_mask = freq >= 0
    freq = freq[pos_mask]
    p_compl = perm_fft[pos_mask]
    p_real = np.real(p_compl)
    p_imag = np.imag(p_compl)
    p_mag = np.abs(p_compl)
    p_phase = np.angle(p_compl, deg=True)

    # Interpolate to output frequency axis if requested
    if equal_length:
        freq_min = max(np.min(freq[freq > 0]), 1e-10)  # type: ignore
        freq_max = np.max(freq)
        if logspace_freq:
            freq_out = np.logspace(np.log10(freq_min), np.log10(freq_max), n_points)
        else:
            freq_out = np.linspace(freq_min, freq_max, n_points)

        s_kwargs = {}
        s_kwargs["extrapolate"] = kwargs.get("extrapolate", True)
        spline = PchipInterpolator
        if post_spline == "cubic":
            s_kwargs["bc_type"] = kwargs.get("bc_type", "natural")
            spline = CubicSpline

        s_real = spline(freq, p_real, **s_kwargs)(freq_out)
        s_imag = spline(freq, p_imag, **s_kwargs)(freq_out)
        s_mag = spline(freq, p_mag, **s_kwargs)(freq_out)
        s_phase = spline(freq, p_phase, **s_kwargs)(freq_out)
        s_fd = (
            spline(freq, p_compl, **s_kwargs)(freq_out)
            if post_spline == "cubic"
            else s_real + 1j * s_imag
        )

    else:
        delta_fd = [np.nan] * (n_points - len(freq)) if n_points > len(freq) else []
        freq_out = np.concatenate([freq, delta_fd])
        s_fd = np.concatenate([p_compl, delta_fd])
        s_real = np.concatenate([p_real, delta_fd])
        s_imag = np.concatenate([p_imag, delta_fd])
        s_mag = np.concatenate([p_mag, delta_fd])
        s_phase = np.concatenate([p_phase, delta_fd])

    try:
        loss = s_imag / s_real
    except (FloatingPointError, ValueError):
        loss = s_imag / np.where(s_real != 0, s_real, 1e-32)

    # Update the DataFrame in place
    decay_df["frequency"] = freq_out
    decay_df["permittivity"] = s_fd
    decay_df["real perm"] = s_real
    decay_df["imag perm"] = s_imag
    decay_df["mag perm"] = s_mag
    decay_df["phase perm"] = s_phase
    decay_df["loss tangent"] = loss

    # Update attrs
    decay_df.attrs["DC permittivity"] = p_real[0]
    decay_df.attrs["DC imag"] = p_real[0]
    decay_df.attrs["inf permittivity"] = decay_df["real perm"].dropna().iloc[-1]
    decay_df.attrs["inf imag"] = decay_df["imag perm"].dropna().iloc[-1]

    if max_points and (dec_len := len(decay_df) - 1) > max_points:
        max_points = max_points - 1 if max_points % 2 else max_points
        if (_dt := dec_len * dt / max_points) < 1:
            mult = 10 ** np.ceil(-np.log10(_dt))
            max_points = int(
                (dec_len * dt)
                / ([1, 2, 2.5, 5][np.argmin(np.abs([1, 2, 2.5, 5] - _dt * mult))] / mult)
            )
        decay_df = decay_df.iloc[np.linspace(0, dec_len, num=max_points + 1, dtype=int)]

    # Warn if negative permittivity
    if np.any(decay_df["real perm"] < 0):
        percent_negative = 100.0 * np.sum(decay_df["real perm"] < 0) / len(decay_df)
        warnings.warn(
            f"Negative values found in real permittivity: {percent_negative:.2f}% < 0. "
            "This may indicate a malformed FFT conversion.",
            RuntimeWarning,
            stacklevel=2,
        )

    return decay_df


def is_log_spaced(arr: np.ndarray, rtol: float = 1e-2) -> bool:
    """
    Check if an array is approximately logarithmically spaced.

    Parameters
    ----------
    arr : np.ndarray
        Input array (must be positive and 1D).
    rtol : float, optional
        Relative tolerance for uniformity of log spacing.

    Returns
    -------
    bool
        True if array is log-spaced, False otherwise.
    """
    arr = np.asarray(arr)
    if np.any(arr <= 0):
        return False
    log_diffs = np.diff(np.log10(arr))
    return np.allclose(log_diffs, log_diffs[0], rtol=rtol)


def find_wide_peaks(
    freq: np.ndarray,
    arr: np.ndarray,
    min_width: float = 5,
    min_prominence: float = 0.05,
    background_subtract: bool = True,
    min_peaks: int = 2,
    weighting_func: Callable | None = None,
) -> pd.DataFrame:
    """
    Find wide, prominent peaks in the input array, optionally using background subtraction.

    Parameters
    ----------
    freq : np.ndarray
        Frequency array.
    arr : np.ndarray
        Input array (e.g., 'mag perm', 'imag perm', etc.).
    min_width : float, optional
        Minimum width (in number of points) for a peak to be considered valid.
    min_prominence : float, optional
        Minimum prominence for a peak to be considered valid.
    background_subtract : bool, optional
        If True, also search for peaks after background subtraction or if no peaks found.
    min_peaks : int, optional
        Minimum number of peaks to find before reducing min_prominence recursively.
    weighting_func : callable(width, prominences) | None, optional
        If provided, a function that takes two arguments: the normalized width and prominences arrays,
        and returns a new weight array. If None, the default weighting is used.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: 'peak_freq', 'peak_arr', 'width', 'prominence', 'weight', 'bkg_subtracted'.
    """

    def calc_weights(wid, prom, ar):
        """
        Calculate weights based on widths and prominences.
        If weighting_func is provided, use it to calculate weights.
        """
        if not len(wid):
            return wid, prom, np.array([])
        n_widths = wid / len(ar)
        n_prominences = prom / (np.ptp(ar) if np.ptp(ar) > 0 else 1e-32)
        if weighting_func is not None:
            # weighting_func(width, prominences)
            return n_widths, n_prominences, weighting_func(n_widths, n_prominences)
        return n_widths, n_prominences, np.sqrt(n_widths * n_prominences)

    peaks, props = find_peaks(arr, width=min_width, prominence=min_prominence)
    widths, prominences, weights = calc_weights(props["widths"], props["prominences"], arr)
    result = pd.DataFrame(
        {
            "peak_freq": freq[peaks],
            "peak_tau": 1 / (2 * np.pi * freq[peaks]),
            "peak_arr": arr[peaks],
            "width": widths,
            "prominence": prominences,
            "weight": weights,
            "bkg_subtracted": 0.0,  # False
        },
        dtype=float,
    )

    if (background_subtract or result.empty) and len(arr) > 2:
        x = np.log10(freq) if is_log_spaced(freq) else freq
        coeffs = np.polyfit(x, arr, 1)
        baseline = np.polyval(coeffs, x)
        peaks, props = find_peaks(arr - baseline, width=min_width, prominence=min_prominence)
        widths, prominences, weights = calc_weights(props["widths"], props["prominences"], arr)
        result_bkg = pd.DataFrame(
            {
                "peak_freq": freq[peaks],
                "peak_tau": 1 / (2 * np.pi * freq[peaks]),
                "peak_arr": arr[peaks],
                "width": widths,
                "prominence": prominences,
                "weight": weights,
                "bkg_subtracted": 1.0,  # True
            },
            dtype=float,
        )

        result = pd.concat([result, result_bkg], ignore_index=True)
        result = result.drop_duplicates(subset="peak_freq", keep="first")

    # Recursively reduce min_prominence if not enough peaks found
    if len(result) < min_peaks and min_prominence > 1e-6:
        return find_wide_peaks(
            freq,
            arr,
            min_width=min_width,
            min_prominence=min_prominence / 2,
            background_subtract=background_subtract,
            min_peaks=min_peaks,
        )

    # If still no peaks found, use the max of arr as a fallback
    if result.empty:
        widths, prominences, weights = calc_weights(
            np.array([1.0]), np.array([max(np.min(np.abs(np.diff(arr))), 1e-32)]), arr
        )
        result = pd.DataFrame(
            {
                "peak_freq": freq[np.argmax(arr)],
                "peak_tau": 1 / (2 * np.pi * freq[np.argmax(arr)]),
                "peak_arr": np.max(arr),
                "width": widths,
                "prominence": prominences,
                "weight": weights,
                "bkg_subtracted": -1.0,
            },
            dtype=float,
        )

    return result


def _fit_peak_trend(
    all_peaks_df: pd.DataFrame,
    step_size: int = 10,
    slope_min: float = -np.inf,
    slope_max: float = np.inf,
    mask_mode: str | tuple[str, ...] = "quantile",
    mode_use: str = "backup",
    select_method: str = "median, abs_max",
) -> tuple[float, float]:
    """
    Fit a global trend (slope) of log10(peak_freq) vs temp, masking a fraction of edge peaks by value.
    Iteratively increases edge_frac until the std of the residuals stabilizes.
    Limits slope to [slope_min, slope_max] and prefers expected_slope if provided.

    Parameters
    ----------
    all_peaks_df : pd.DataFrame
        DataFrame with MultiIndex including 'temp' and a column for 'peak_freq'.
    step_size : int, optional
        Step size for increasing edge_frac (default: 10).
    slope_min : float, optional
        Minimum allowed slope value.
    slope_max : float, optional
        Maximum allowed slope value.
    mask_mode : str or tuple of str, optional
        Masking mode(s) to apply. Can be a single string or a tuple of strings.
    mode_use : str, optional
        Determines how mask_mode(s) are applied when multiple are provided:
            - "parallel": All mask_modes are applied at once, so the mask becomes more restrictive
              with each mode (mask is the intersection of all).
            - "sequential": Each mask_mode is tried independently, updating a list of valid fits.
            - "backup": Only the first mask_mode is used, however recursion is used to
              try the next mask_mode if not enough valid fits are found. Default case.
    select_method : str, optional
        String specifying how to select the "best" slope from valid fits. Multiple methods can be combined
        using spaces, commas, or both (e.g., "median, abs_max min_std"). Methods evaluate valid slopes
        or the standard error of the estimate (SEE) of the fit for a valid slope.
        Valid methods: [median, mean, min, max, abs_median, abs_mean, abs_min, abs_max, median_see,
            mean_see, min_see, max_see].
            Note: 'med' is a valid alias for 'median' and 'std' is a valid alias for 'see'.

    Returns
    -------
    tuple[float, float]
        Slope and intercept of the global fit (log10(peak_freq) vs temp).
    """
    df = all_peaks_df[all_peaks_df["peak_freq"].notna()].copy()

    if len(df) < 3:
        raise ValueError("Not enough points for global peak trend fit.")

    if slope_min > slope_max:
        slope_min, slope_max = slope_max, slope_min
    elif slope_min == slope_max:
        raise ValueError("slope_min and slope_max must be different values.")

    peak_df = pd.DataFrame()
    peak_df["temps"] = df.index.get_level_values("temp").astype(float).to_numpy(copy=True)
    # peak_df["peaks"] = np.log10(df["peak_freq"].to_numpy(copy=True))
    peak_df["peaks"] = np.log(df["peak_freq"].to_numpy(copy=True))
    peak_df["weights"] = df["weight"].to_numpy(copy=True)
    peak_df = peak_df.sort_values("temps").reset_index(drop=True)

    coeffs = list(np.polyfit(peak_df["temps"], peak_df["peaks"], 1, w=peak_df["weights"]))
    see_history = [np.std(peak_df["peaks"] - np.polyval(coeffs, peak_df["temps"]))]
    coeffs_history = [coeffs]
    valid_see = []
    valid_slopes = []
    if slope_min <= coeffs[0] <= slope_max:
        valid_slopes.append(coeffs[0])
        valid_see.append(see_history[-1])

    n_temps = peak_df["temps"].nunique()
    if len(peak_df) <= n_temps:
        return coeffs[0], coeffs[1]

    step_size = int(np.clip(step_size, 1, 16))

    if isinstance(mask_mode, str):
        mask_mode = (mask_mode,)

    remaining_masks = ()
    if mode_use == "parallel":
        # mode_groups: tuple[tuple[str, ...]]; Setup to iter in the inner loop
        mode_groups = (mask_mode,)
    elif mode_use == "sequential":
        # mode_groups: tuple[tuple[str], ...]; Setup to iter in the outer loop
        mode_groups = tuple((m,) for m in mask_mode)
    else:  # backup/default
        # mode_groups: tuple[tuple[str]]; Setup to recurse the function as needed
        remaining_masks = mask_mode[1:]
        mode_groups = ((mask_mode[0],),)

    min_valid = 3
    for m_group in mode_groups:
        # Only one iteration unless mode_use is "sequential"
        n_temps = peak_df["temps"].nunique()
        for frac in range(step_size, 50 // step_size * step_size, step_size):
            mask = np.ones(len(peak_df), dtype=bool)
            for m_mode in m_group:
                # Only one iteration unless mode_use is "parallel"
                if m_mode == "quantile":
                    lower_thresh = peak_df["peaks"].quantile(frac / 100)
                    upper_thresh = peak_df["peaks"].quantile((100 - frac) / 100)
                    mask *= (peak_df["peaks"] >= lower_thresh) & (peak_df["peaks"] <= upper_thresh)
                elif m_mode == "weight":
                    lower_thresh = peak_df["weights"].quantile(frac / 100)
                    mask *= peak_df["weights"] >= lower_thresh
                elif "temp" in m_mode:
                    n_temps -= 1
                    if n_temps < 2:
                        break
                    is_rev = True if "low" in m_mode else False
                    temp_list = sorted(peak_df["temps"].unique(), reverse=is_rev)
                    mask *= peak_df["temps"].isin(temp_list[: -int(frac / step_size)])

            if n_temps < 2:
                break

            temps_masked = peak_df.loc[mask, "temps"]
            if temps_masked.nunique() < n_temps:
                if step_size > 1:
                    return _fit_peak_trend(
                        all_peaks_df,
                        step_size=step_size // 2,
                        slope_min=slope_min,
                        slope_max=slope_max,
                        mask_mode=mask_mode,
                        mode_use=mode_use,
                        select_method=select_method,
                    )
                break

            coeffs = list(
                np.polyfit(
                    temps_masked, peak_df.loc[mask, "peaks"], 1, w=peak_df.loc[mask, "weights"]
                )
            )
            see_history.append(
                np.std(peak_df.loc[mask, "peaks"] - np.polyval(coeffs, temps_masked))
                / np.sqrt(len(temps_masked))
            )

            coeffs_history.append(coeffs)
            if slope_min <= coeffs[0] <= slope_max:
                valid_slopes.append(coeffs[0])
                valid_see.append(see_history[-1])

            if len(valid_slopes) >= min_valid and all(
                abs(n - np.median(valid_see[-3:])) < 1e-3 for n in valid_see[-3:]
            ):
                min_valid += 3
                break

    if remaining_masks and len(valid_slopes) < 3:
        # Only called as a backup and when mode_use is not "sequential" or "parallel"
        return _fit_peak_trend(
            all_peaks_df,
            step_size=step_size,
            slope_min=slope_min,
            slope_max=slope_max,
            mask_mode=remaining_masks,
            mode_use=mode_use,
            select_method=select_method,
        )
    elif valid_slopes:
        best_slopes = []
        methods = re.split(r"[,\s]+", select_method)
        if "med" in methods or "median" in methods:
            best_slopes.append(np.median(valid_slopes))
        if "mean" in methods:
            best_slopes.append(np.mean(valid_slopes))
        if "max" in methods:
            best_slopes.append(valid_slopes[np.argmax(valid_slopes)])
        if "min" in methods:
            best_slopes.append(valid_slopes[np.argmin(valid_slopes)])
        if "abs_med" in methods or "abs_median" in methods:
            best_slopes.append(np.median(np.abs(valid_slopes)))
        if "abs_mean" in methods:
            best_slopes.append(np.mean(np.abs(valid_slopes)))
        if "abs_max" in methods:
            best_slopes.append(valid_slopes[np.argmax(np.abs(valid_slopes))])
        if "abs_min" in methods:
            best_slopes.append(valid_slopes[np.argmin(np.abs(valid_slopes))])
        if "min_std" in methods or "min_see" in methods:
            best_slopes.append(valid_slopes[np.argmin(valid_see)])
        if "max_std" in methods or "max_see" in methods:
            best_slopes.append(valid_slopes[np.argmax(valid_see)])
        if "mean_std" in methods or "mean_see" in methods:
            weights = 1 / (np.array(valid_see) + 1e-32)
            weights /= weights.sum()
            best_slopes.append(np.sum(np.array(valid_slopes) * weights))
        if (
            "median_std" in methods
            or "med_std" in methods
            or "median_see" in methods
            or "med_see" in methods
        ):
            best_slopes.append(
                valid_slopes[np.argmin(np.abs(np.array(valid_see) - np.median(valid_see)))]
            )

        best_slope = np.mean(best_slopes) if best_slopes else np.median(valid_slopes)
        idx = np.argmin([abs(s[0] - best_slope) for s in coeffs_history])
        coeffs_history.append(coeffs_history[idx])
    else:
        # If no valid slopes found, return the best slope and intercept from the last coeffs
        if slope_min <= 0 <= slope_max:
            slope = 0.0
        else:
            slope = slope_min if abs(slope_min) < abs(slope_max) else slope_max
        intercept = np.average(
            peak_df["peaks"] - slope * peak_df["temps"], weights=peak_df["weights"]
        )
        coeffs_history.append([slope, intercept])

    return coeffs_history[-1][0], coeffs_history[-1][1]


def collect_peak_summary_df(
    perm_dict: dict,
    column: str = "imag perm",
    min_width: float = 0.05,
    min_prominence: float = 0.05,
    min_peaks: int = 2,
    weighting_func: Callable | None = None,
    fit_step: int = 10,
    slope_min: float = -np.inf,
    slope_max: float = np.inf,
    fit_mask_mode: str | tuple[str, ...] = "weight",
    fit_mode_use: str = "sequential",
    fit_select_method: str = "median, min_std",
    weight_diff: bool = False,
    normalize_weight_mode: str | tuple[str, ...] = "temp",
    min_weight: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarize the main peak for each entry in a permittivity dictionary,
    using all peaks and temperature trends to select the best peak for each entry.
    If group slope is too far from global, revert to global slope.

    Parameters
    ----------
    perm_dict : dict
        Dictionary of DataFrames from transform_to_permittivity, keys are MultiIndex tuples.
    column : str, optional
        Column to evaluate for peak finding (default: "imag perm").
    min_width : float, optional
        Minimum width (in number of points) for a peak to be considered valid.
    min_prominence : float, optional
        Minimum prominence for a peak to be considered valid.
    min_peaks : int, optional
        Minimum number of peaks to find before reducing min_prominence recursively.
    slope_min : float, optional
        Minimum allowed slope value.
    slope_max : float, optional
        Maximum allowed slope value.
    global_slope_tol : float, optional
        Relative tolerance for group slope vs global slope (default: 0.2).

    Returns
    -------
    pd.DataFrame
        DataFrame with MultiIndex (from keys + 'peaks') and columns:
        'peak_freq', 'peak_mag', 'width', 'prominence', 'bkg_subtracted'
    """
    peaks_df = pd.DataFrame(
        index=list(perm_dict.keys()),
        columns=[
            "peak_freq",
            "peak_tau",
            "peak_arr",
            "width",
            "prominence",
            "weight",
            "bkg_subtracted",
        ],
        dtype=float,
    )
    peaks_df = form_std_df_index(peaks_df)
    ts_series = pd.Series(index=peaks_df.index, dtype="datetime64[ns]")
    all_peaks_dict = {}
    for key, df in perm_dict.items():
        freq = df["frequency"].to_numpy(copy=True)
        arr = df[column].to_numpy(copy=True)
        if np.sum(arr < 0) > 0.5 * len(arr):
            arr = -1 * arr
        min_width_pts = max(1, int(np.round(len(arr) * min_width)))
        all_peaks_dict[key] = find_wide_peaks(
            freq,
            arr,
            min_width=min_width_pts,
            min_prominence=min_prominence,
            min_peaks=min_peaks,
            weighting_func=weighting_func,
        )
        if len(all_peaks_dict[key]) < min_peaks and min_width_pts > 3:
            all_peaks_dict[key] = find_wide_peaks(
                freq,
                arr,
                min_width=3,
                min_prominence=min_prominence,
                min_peaks=min_peaks,
                weighting_func=weighting_func,
            )
        if all_peaks_dict[key].empty:
            all_peaks_dict[key].loc[0] = np.nan
        ts_series[key] = df.attrs.get("timestamp", pd.Timestamp.now())

    all_peaks_df = form_std_df_index(
        pd.concat(all_peaks_dict, axis=0), names=DEFAULT_KEYS + ("peaks",)
    )

    all_peaks_df = _normalize_peak_weights(all_peaks_df, normalize_weight_mode, min_weight)

    # all_peaks_df["weight"] = all_peaks_df["weight"].where(all_peaks_df["weight"] > 0, 1e-32)

    # Compute global slope and intercept for all peaks
    global_coeffs = _fit_peak_trend(
        all_peaks_df,
        step_size=fit_step,
        slope_min=slope_min,
        slope_max=slope_max,
        mask_mode=fit_mask_mode,
        mode_use=fit_mode_use,
        select_method=fit_select_method,
    )

    # Add "m", "b", and "diffs" columns to all_peaks_df and peaks_df
    for df in (all_peaks_df, peaks_df):
        df["global_coeffs"] = 0.0
        for col in ["pred_fit", "pred_fit_tau", "m", "b", "fit", "diffs", "peak_quantile"]:
            df[col] = np.nan

    grouped = all_peaks_df.groupby(["sample_name", "condition"], observed=True)
    for key, group in grouped:
        # Fit the group to find the slope and intercept using the helper
        fit_df = group.dropna(subset=["peak_freq"]).copy()

        if fit_df.empty:
            continue

        # fit_df["peak_quantile"] = fit_df["peak_freq"].apply(np.log10).rank(pct=True)
        fit_df["peak_quantile"] = fit_df["peak_freq"].apply(np.log).rank(pct=True)

        if len(fit_df) < 3:
            coeffs = global_coeffs
            fit_df["global_coeffs"] = 1.0
        else:
            # Use the helper to get group slope and intercept
            coeffs = _fit_peak_trend(
                fit_df,
                step_size=fit_step,
                slope_min=slope_min,
                slope_max=slope_max,
                mask_mode=fit_mask_mode,
                mode_use=fit_mode_use,
                select_method=fit_select_method,
            )

        peaks_df.attrs[key] = coeffs
        fit_df["m"] = coeffs[0]
        fit_df["b"] = coeffs[1]

        fit_df["fit"] = coeffs[0] * fit_df.index.get_level_values("temp").astype(float) + coeffs[1]
        # fit_df["pred_fit"] = 10 ** fit_df["fit"]
        # fit_df["diffs"] = abs(np.log10(fit_df["peak_freq"]) - fit_df["fit"])
        fit_df["pred_fit"] = np.exp(fit_df["fit"])
        fit_df["pred_fit_tau"] = 1 / (2 * np.pi * fit_df["pred_fit"])
        fit_df["diffs"] = abs(np.log(fit_df["peak_freq"]) - fit_df["fit"])

        all_peaks_df.update(fit_df)

        for _, group in fit_df.groupby(level=list(fit_df.index.names)[:-1], observed=True):
            best_idx = group["diffs"].idxmin()
            if weight_diff:
                best_idx = (group["diffs"] / group["weight"]).idxmin()
            peaks_df.loc[best_idx[:-1]] = fit_df.loc[best_idx]  # type: ignore

    peaks_df["timestamp"] = ts_series

    # Add selection columns
    for col in [
        "peak_temp_quantile",
        "peak_local_quantile",
    ]:
        all_peaks_df[col] = np.nan

    for col in [
        "best_local_diff",
        "best_local_prominence",
        "best_local_width",
        "best_local_weight",
        "best_diff",
        "best_prominence",
        "best_width",
        "best_weight",
    ]:
        all_peaks_df[col] = 0.0

    # "bests" for each segment
    grouped = all_peaks_df.groupby(all_peaks_df.index.names[:-1], observed=True)
    for _, group in grouped:
        if not group["peak_freq"].isna().all():
            # all_peaks_df.loc[group.index, "peak_local_quantile"] = (
            #     group["peak_freq"].apply(np.log10).rank(pct=True)
            # )
            all_peaks_df.loc[group.index, "peak_local_quantile"] = (
                group["peak_freq"].apply(np.log).rank(pct=True)
            )
        # By minimum diff
        if not group["diffs"].isna().all():
            all_peaks_df.loc[group["diffs"].idxmin(), "best_local_diff"] = 1.0
        # By maximum prominence
        if not group["prominence"].isna().all():
            all_peaks_df.loc[group["prominence"].idxmax(), "best_local_prominence"] = 1.0
        # By maximum width
        if not group["width"].isna().all():
            all_peaks_df.loc[group["width"].idxmax(), "best_local_width"] = 1.0

        if not group["weight"].isna().all():
            all_peaks_df.loc[group["weight"].idxmax(), "best_local_weight"] = 1.0

    # "Bests" for each set
    grouped = all_peaks_df.groupby(["sample_name", "condition", "temp"], observed=True)
    for _, group in grouped:
        if not group["peak_freq"].isna().all():
            # all_peaks_df.loc[group.index, "peak_temp_quantile"] = (
            #     group["peak_freq"].apply(np.log10).rank(pct=True)
            # )
            all_peaks_df.loc[group.index, "peak_temp_quantile"] = (
                group["peak_freq"].apply(np.log).rank(pct=True)
            )
        # By minimum diff
        if not group["diffs"].isna().all():
            all_peaks_df.loc[group["diffs"].idxmin(), "best_diff"] = 1.0
        # By maximum prominence
        if not group["prominence"].isna().all():
            all_peaks_df.loc[group["prominence"].idxmax(), "best_prominence"] = 1.0
        # By maximum width
        if not group["width"].isna().all():
            all_peaks_df.loc[group["width"].idxmax(), "best_width"] = 1.0

        if not group["weight"].isna().all():
            all_peaks_df.loc[group["weight"].idxmax(), "best_weight"] = 1.0

    all_peaks_df["timestamp"] = ts_series

    return peaks_df, all_peaks_df


def norm_func(x, mw=1e-32):
    x = x.astype(float)
    x_min = np.nanmin(x)
    x_max = np.nanmax(x)
    if x_max == x_min:
        return np.full_like(x, mw)
    return np.clip((x - x_min) / (x_max - x_min), mw, 1.0)


def _normalize_peak_weights(
    peaks_df: pd.DataFrame,
    mode: str | tuple[str, ...] = "",
    min_weight: float = 1e-32,
) -> pd.DataFrame:
    """
    Normalize the 'weight' column in peaks_df according to the specified mode.

    Parameters
    ----------
    peaks_df : pd.DataFrame
        DataFrame with a 'weight' column to normalize.
    mode : str or None, optional
        Normalization mode:
        - None or "None": do not normalize.
        - "dataset": normalize weights within each DataFrame returned by find_wide_peaks.
        - "all": normalize weights across all entries in peaks_df.
        - "temp": normalize weights for each (sample_name, condition, temp) group.
        - "condition": normalize weights for each (sample_name, condition) group.
    min_weight : float, optional
        Minimum weight value after normalization (default: 1e-32).

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized weights.
    """
    df = peaks_df.copy()

    if df.empty or not mode:
        return df

    df["weight"] = df["weight"].where(df["weight"] > 0, 1e-32)

    if isinstance(mode, tuple):
        df["weight"] = 0.0
        for md in mode:
            df["weight"] += _normalize_peak_weights(peaks_df, md, min_weight)["weight"]
        df["weight"] /= len(mode)
        return df

    if mode == "all":
        df["weight"] = norm_func(df["weight"].values)
    elif mode == "dataset":
        group_levels = df.index.names[:-1]
        df["weight"] = df.groupby(group_levels, observed=True)["weight"].transform(
            norm_func, min_weight
        )
    elif mode == "temp":
        group_levels = ["sample_name", "condition", "temp"]
        df["weight"] = df.groupby(group_levels, observed=True)["weight"].transform(
            norm_func, min_weight
        )
    elif mode == "condition":
        group_levels = ["sample_name", "condition"]
        df["weight"] = df.groupby(group_levels, observed=True)["weight"].transform(
            norm_func, min_weight
        )
    elif mode == "sample":
        group_levels = ["sample_name"]
        df["weight"] = df.groupby(group_levels, observed=True)["weight"].transform(
            norm_func, min_weight
        )

    return df


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

#     uniform_t = np.linspace(t[0], t[-1], n_points)
#     uniform_c = interp_current(uniform_t)
#     uniform_v = interp_voltage(uniform_t)

#     uniform_d = (uniform_c / float(dims_cm[0])) / (
#         VACUUM_PERMITTIVITY * (uniform_v / float(dims_cm[1]))
#     )

#     if filter_cutoff > 0:
#         fs = 1.0 / dt  # Sampling frequency in Hz
#         uniform_d = remove_low_freq_baseline(
#             uniform_d, fs, cutoff=filter_cutoff, order=filter_order
#         )

#     n_fft = n_points
#     uniform_d = uniform_d.copy()
#     if decay_mod is not None:
#         if decay_mod.get("mode", "interp") == "gradient":
#             uniform_d = np.gradient(
#                 uniform_d, uniform_t, edge_order=decay_mod.get("edge_order", 1)
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
#             uniform_d = savgol_filter(uniform_d, **decay_mod)

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
#         final_mean = remove_mean(uniform_d)
#         uniform_d = uniform_d - final_mean
#     elif isinstance(remove_mean, int) and remove_mean < 0:
#         final_mean = np.nanmean(uniform_d[remove_mean:])
#         uniform_d = uniform_d - final_mean
#     elif isinstance(remove_mean, (int, float)):
#         perc = remove_mean if 0 < remove_mean < 1 else abs(remove_mean) / 100.0
#         final_mean = np.nanmean(uniform_d[-max(1, int(n_points * perc)) :])
#         uniform_d = uniform_d - final_mean

#     decay_uniform_p_windowed = uniform_d * window
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
#                 "time": uniform_t,
#                 "voltage": uniform_v,
#                 "current": uniform_c,
#                 "decay_raw": uniform_d,
#                 "decay": uniform_d,
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
#                 "time": uniform_t,
#                 "voltage": uniform_v,
#                 "current": uniform_c,
#                 "decay_raw": uniform_d,
#                 "decay": uniform_d,
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
