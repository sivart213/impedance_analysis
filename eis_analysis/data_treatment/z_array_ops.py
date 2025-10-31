# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
import itertools

import numpy as np

try:
    from ..impedance_supplement import get_impedance
except ImportError:
    from eis_analysis.impedance_supplement import get_impedance


def find_f_peak_idx(Z: np.ndarray) -> int:
    """
    Find the frequency at which the imaginary part of the impedance is maximum.

    Parameters
    ----------
    Z : np.ndarray
        Array of complex impedance values.

    Returns
    -------
    float
        Frequency at which the imaginary part of the impedance is maximum.
    """
    Z = np.asarray(Z)
    # Find the index of the maximum imaginary part of Z
    max_index = np.argmax(abs(np.imag(Z)))

    if max_index == 0 or max_index == len(Z) - 1:
        # If the maximum is at the boundary, use the maximum of the imaginary part of the admittance
        Z = 1 / Z
        max_index = np.argmax(abs(np.imag(Z)))
        if max_index == 0 or max_index == len(Z) - 1:
            return 0

    # Return the corresponding frequency idx
    return int(max_index)


def find_peak_vals(
    f: np.ndarray | None = None,
    values: str | tuple[str, ...] | list[str] = "fpeak",
    **kwargs,
) -> list[float]:
    """
    Find the relavent parameter values at the RC peak.
    Parameters
    ----------
    params : tuple | list
        List of parameter values
    model : str
        Circuit model string
    f : np.ndarray
        Frequency array for impedance calculation
    values : str | tuple[str, ...] | list[str]
        Values to return. Can be one of:
        - "f" or "fpeak" for frequency at peak (Hz)
        - "RC" or "tau" for time constant at peak (s)
        - "R" for resistance at peak (Ohm)
        - "C" for capacitance at peak (F)
        Can be a single string or a list/tuple of strings.
    Returns
    -------
    list[float]
        List of requested values in the order they were requested.
    """
    if not values:
        values = ["fpeak"]
    if isinstance(values, str):
        values = [values]

    Z = kwargs.get("Z", np.array([]))

    if Z.size == 0:
        params = kwargs.get("params", ())
        model = kwargs.get("model", "")
        if len(params) <= 1 or not model:
            return [0.0] * len(values)
        Z = get_impedance(f, *params, model=model)

    if f is None:
        # return [0.0] * len(values)
        f = np.zeros_like(Z, dtype=float)

    max_index = find_f_peak_idx(Z)

    res = {}
    for val in values:
        if val.lower()[0] == "f":
            res["f"] = float(f[max_index])
        elif "RC" in val or "tau" in val.lower():
            res["RC"] = f_r_c_conversion(res.get("f", f[max_index]), default=0)
        elif "R" in val:
            # res.append(max(np.real(Z)))
            res["R"] = float(2 * (max(np.real(Z)) - np.real(Z)[max_index]))
        elif "C" in val:
            # res.append(f_r_c_conversion(f[max_index], max(np.real(Z)), default=0))
            r_val = res.get("R", float(2 * (max(np.real(Z)) - np.real(Z)[max_index])))
            res["C"] = f_r_c_conversion(res.get("f", f[max_index]), r_val, default=0)
    return list(res.values())


def f_peak_stats(
    values: list[float], stdevs: list[float], model: str, freq: np.ndarray
) -> tuple[float, float, float, float]:
    """
    Calculate the bounds of frequency and time constant by testing all combinations
    of parameter variations.

    Parameters
    ----------
    values : list[float]
        List of parameter values
    stdevs : list[float]
        List of parameter standard deviations
    model : str
        Circuit model string
    freq : np.ndarray
        Frequency array for impedance calculation

    Returns
    -------
    tuple[float, float, float, float]
        Minimum fpeak, maximum fpeak, minimum tau, maximum tau
    """
    # Create variations of each parameter (min and max)
    param_variations = []
    for val, std in zip(values, stdevs):
        param_variations.append([val - std, val + std])

    # Generate all combinations of parameter variations
    combinations = list(itertools.product(*param_variations))

    # Calculate fpeak for each combination
    fpeaks = []
    for params in combinations:
        # fpeak = find_f_peak(params, model, freq)
        fpeak = find_peak_vals(f=freq, values="fpeak", params=params, model=model)[0]
        if fpeak != 0:
            fpeaks.append(fpeak)

    # Return min and max values
    if not fpeaks:
        return 0.0, 0.0, 0.0, 0.0

    # Convert fpeaks to tau values
    taus = [f_r_c_conversion(f, default=0) for f in fpeaks]

    return min(fpeaks), max(fpeaks), min(taus), max(taus)


def threshold_intersect(x_arr: np.ndarray, y_arr: np.ndarray, threshold=np.nan):
    """
    Find the x value where y_arr crosses a given threshold via linear interpolation.

    Parameters
    ----------
    x_arr : np.ndarray
        Array of x values.
    y_arr : np.ndarray
        Array of y values.
    threshold : float, optional
        The threshold value to find the intersection. Default is NaN, which uses the mean of y_arr.

    Returns
    -------
    float
        The x value where y_arr crosses the threshold. Returns NaN if no crossing is found.

    """
    # indices where y_arr crosses thresh
    thresh = float(threshold)
    if np.isnan(thresh):
        thresh = np.nanmean(y_arr)

    sign = np.sign(y_arr - thresh)
    cs = np.where(sign[:-1] * sign[1:] < 0)[0]
    if len(cs) == 0:
        return np.nan
    k = cs[0]
    # linear interpolation in x_arr
    x0, x1 = x_arr[k], x_arr[k + 1]
    y0, y1 = y_arr[k], y_arr[k + 1]
    t = (thresh - y0) / (y1 - y0)
    res = x0 + t * (x1 - x0)
    return res if np.isfinite(res) else np.nan


def find_fwhm_points(x_arr, y_arr, peak_idx=None):
    """
    Find the full width at half maximum (FWHM) of a peak in y_arr.

    Parameters
    ----------
    x_arr : np.ndarray
        Array of x values.
    y_arr : np.ndarray
        Array of y values.
    peak_idx : int, optional
        Index of the peak in y_arr. If None, it will be determined.

    Returns
    -------
    tuple[float, float, float]
        half maximum value, lower x value at half maximum, upper x value at half maximum.
    """
    # indices where y_arr crosses half_max
    peak = int(np.nanargmax(y_arr)) if peak_idx is None else peak_idx
    half_max = y_arr[peak] / 2
    lo = threshold_intersect(x_arr[: peak + 1], y_arr[: peak + 1], half_max)
    hi = threshold_intersect(x_arr[peak:], y_arr[peak:], half_max)

    return half_max, lo, hi


def arc_quality(
    freq: np.ndarray, Z: np.ndarray, peak_idx: int | None = None
) -> tuple[float, float, float, float]:
    """
    Calculate the peak frequency, quality factor in frequency and time constant domains.

    Parameters
    ----------
    freq : np.ndarray
        Array of frequencies.
    Z : np.ndarray
        Array of complex impedance values.
    peak_idx : int or None
        Index of the peak in the Z array. If None, it will be determined.
    Returns
    -------
    tuple[float, float, float, float]
        Peak frequency (Hz), quality factor in frequency domain,
        peak time constant (s), quality factor in time constant domain.
    """
    # Peak index
    peak_idx = find_f_peak_idx(Z) if peak_idx is None else peak_idx

    # Find crossings on each side via linear interpolation in logf
    f_p = freq[peak_idx]

    _, f_lo, f_hi = find_fwhm_points(np.log10(freq), abs(np.imag(Z)), peak_idx)
    f_lo = 10**f_lo
    f_hi = 10**f_hi

    Q_f = f_p / (f_hi - f_lo)

    tau_p = 1.0 / (2 * np.pi * f_p)

    tau_lo = 1.0 / (2 * np.pi * f_hi)
    tau_hi = 1.0 / (2 * np.pi * f_lo)

    Q_tau = tau_p / (tau_hi - tau_lo)

    return f_p, Q_f, tau_p, Q_tau


def f_r_c_conversion(*vals: float, default: float = 0.0) -> float:
    """
    Convert between f, RC, R, and C utilyzing the equation 1=2*pi*f*R*C.

    Parameters
    ----------
    vals : float
        Values to convert to. Can be one of:
        - f (Hz)
        - RC (s)
        - R (Ohm)
        - C (F)

    Returns
    -------
    float
        Converted value.
    """
    if any(v == 0 for v in vals) or not vals:
        return default
    return float(1 / (2 * np.pi * np.prod(vals)))


def omega_r_c_conversion(*vals: float, default: float = 0.0) -> float:
    """
    Convert between omega, RC, R, and C utilyzing the equation 1=omega*R*C.

    Parameters
    ----------
    vals : float
        Values to convert to. Can be one of:
        - omega (rad/s)
        - RC (s)
        - R (Ohm)
        - C (F)

    Returns
    -------
    float
        Converted value.
    """
    if any(v == 0 for v in vals) or not vals:
        return default
    return float(1 / np.prod(vals))
