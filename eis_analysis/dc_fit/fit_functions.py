from typing import Any
from collections import defaultdict
from collections.abc import Callable

import numpy as np
import pandas as pd

from eis_analysis.data_treatment import FittingMethods  # Add this import


def linear_func(t: np.ndarray, a: float, b: float) -> np.ndarray:
    """Linear function for fitting."""
    return a * t + b


def pow_func(t: np.ndarray, A: float, tau: float, B: float = 0.0) -> np.ndarray:
    """Power law decay/growth for fitting."""
    t[t == 0] = 1e-32  # Avoid division by zero
    return A * (t) ** (-tau) + B


def exp_func(t: np.ndarray, A: float, tau: float, B: float) -> np.ndarray:
    """Exponential decay/growth for fitting."""
    try:
        res = A * np.exp(-t / tau) + B
    except FloatingPointError:
        # Handle potential overflow in exp calculation
        res = A * np.exp(-t / (tau + 1e-32)) + B
    return res


def double_exp_func(
    t: np.ndarray, A1: float, tau1: float, A2: float, tau2: float, B: float
) -> np.ndarray:
    """Double exponential decay/growth for fitting."""
    try:
        res = A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + B
    except FloatingPointError:
        # Handle potential overflow in exp calculation
        res = A1 * np.exp(-t / (tau1 + 1e-32)) + A2 * np.exp(-t / (tau2 + 1e-32)) + B
    return res


def stretch_exp_func(t: np.ndarray, A: float, tau: float, beta: float, B: float) -> np.ndarray:
    """Stretched exponential decay/growth for fitting."""
    try:
        res = A * np.exp(-((t / tau) ** beta)) + B
    except FloatingPointError:
        # Handle potential overflow in exp calculation
        res = A * np.exp(-((t / (tau + 1e-32)) ** beta)) + B
    return res


def n_exp_func(t: np.ndarray, *params: float) -> np.ndarray:
    """
    N-exponential decay/growth for fitting.

    Parameters
    ----------
    t : np.ndarray
        Time array.
    *params : float
        Sequence of (A, tau) pairs, optionally followed by B if odd number of params.
        If even, B is set to 0.

    Returns
    -------
    np.ndarray
        Evaluated sum of exponentials plus offset.
    """
    result = np.zeros_like(t)
    for i in range(len(params) // 2):
        try:
            result += params[2 * i] * np.exp(-t / params[2 * i + 1])
        except FloatingPointError:
            result += params[2 * i] * np.exp(-t / (params[2 * i + 1] + 1e-32))
    B = params[-1] if len(params) % 2 else 0.0
    return result + B


# %% Initial Guess Functions
def initial_guess_poly_func(
    t: np.ndarray | pd.DataFrame, y: np.ndarray | None = None, deg: int = 1, **_
) -> list[float]:
    """
    Estimate initial parameters for linear fit: f(x) = a * x + b

    Args:
        x (np.ndarray): Independent variable (e.g., time).
        y (np.ndarray): Dependent variable (e.g., signal).

    Returns:
        tuple: Initial guess for (a, b)
    """
    if isinstance(t, pd.DataFrame):
        y = t.iloc[:, 1].to_numpy(copy=True) if y is None else y
        t = t.iloc[:, 0].to_numpy(copy=True)
    if y is None:
        raise ValueError("y must be provided if t is a DataFrame or Series.")

    return np.polyfit(t, y, deg).tolist()


def initial_guess_pow_func(
    t: np.ndarray | pd.DataFrame, y: np.ndarray | None = None, **kwargs
) -> list[float]:
    if isinstance(t, pd.DataFrame):
        y = t.iloc[:, 1].to_numpy(copy=True) if y is None else y
        t = t.iloc[:, 0].to_numpy(copy=True)
    if y is None:
        raise ValueError("y must be provided if t is a DataFrame or Series.")
    # Estimate baseline offset b
    # b_idx = np.argmin(abs(y))
    # if b_idx < len(y) * 3 / 4:
    #     # Ensure b_idx is in the last quarter of y
    #     b_idx = -1
    # b0 = y[b_idx]
    b0 = float(kwargs.get("b0", np.nan))
    if np.isnan(b0):
        b_idx = np.max([np.argmin(abs(y)), np.argmax(abs(y))])
        if b_idx < len(y) * 3 / 4:
            # Ensure b_idx is in the last quarter of y
            b_idx = -1
        b0 = float(y[b_idx])
    mask = (t != 0) & (y != 0) & np.isfinite(t) & np.isfinite(y)
    slope, intercept = np.polyfit(np.log(abs(t[mask])), np.log(abs(y[mask])), 1)
    return [np.sign(b0) * np.exp(intercept), abs(slope), b0]


def initial_guess_exp_func(
    t: np.ndarray | pd.DataFrame,
    y: np.ndarray | None = None,
    n_exp: int = 1,
    **kwargs,
) -> list[float]:
    """
    Estimate initial guesses for multi-exponential decay model parameters:
    f(t) = a1·exp(-t/τ1) + a2·exp(-t/τ2) + ... + an·exp(-t/τn) + b

    Args:
        t (np.ndarray): Independent variable (e.g., time).
        y (np.ndarray): Dependent variable (e.g., signal).
        n_exp (int): Number of exponential terms.

    Returns:
        dict: Estimated a_i and tau_i parameters
    """
    assert n_exp >= 1, "Number of exponentials must be >= 1"
    if isinstance(t, pd.DataFrame):
        y = t.iloc[:, 1].to_numpy(copy=True) if y is None else y
        t = t.iloc[:, 0].to_numpy(copy=True)
    if y is None:
        raise ValueError("y must be provided if t is a DataFrame or Series.")

    # Estimate baseline offset b
    b0 = float(kwargs.get("b0", np.nan))
    if np.isnan(b0):
        b_idx = np.max([np.argmin(abs(y)), np.argmax(abs(y))])
        if b_idx < len(y) * 3 / 4:
            # Ensure b_idx is in the last quarter of y
            b_idx = -1
        b0 = float(y[b_idx])

    # Adjust for baseline
    # y_adj = abs(y - b0 + 1e-32)
    y_adj = abs(y - b0)
    y_adj[y_adj == 0] = 1e-32  # Avoid log(0) or division by zero
    # sign also indicates direction as mean should be towards the center
    # sign = int(np.sign(np.mean(y) - b0))

    # Estimate tau3 from tail slope
    N = max(2, int(len(t) * 0.05))
    # sign also indicates direction as mean should be towards the center
    sign = int(np.sign(np.mean(y[: N * 2]) - np.mean(y[-N * 2 :])))

    # tail_N = max(5, int(len(t) * 0.5))
    taus = [float(-1 / np.polyfit(t[-N * 12 :], np.log(y_adj[-N * 12 :]), 1)[0])]
    taus[0] = max(1e-8, taus[0])
    # tail_N = max(5, int(len(t) * 0.5))
    # taus[0] = -1 / np.polyfit(t[-tail_N:], np.log(y_adj[-tail_N:]), 1)[0]

    if n_exp == 1:
        # a_val = np.mean(y_adj * np.exp(t / taus[0])) * sign
        a_val = np.max(y_adj) if t[-1] / taus[0] > 500 else np.mean(y_adj * np.exp(t / taus[0]))
        return [float(a_val * sign), taus[0], b0]

    dt = float(np.median(np.diff(t)))
    early_N = N * 6
    slope1 = -np.gradient(y_adj[: N * 2], t[: N * 2]).mean()
    taus.append(abs(y_adj[0] / slope1) if slope1 != 0 else dt)
    if taus[0] != 1e-8:
        tau_0 = taus[1]
        while tau_0 >= taus[0] and early_N > 3:
            early_N = max(3, early_N - N)
            slope1 = -np.gradient(y_adj[:early_N], t[:early_N]).mean()
            tau_0 = abs(y_adj[0] / slope1) if slope1 != 0 else dt
        if tau_0 < taus[1]:
            taus[1] = tau_0
    taus[1] = max(1e-16, taus[1])

    taus = sorted(taus)
    taus = np.logspace(np.log10(taus[0]), np.log10(taus[1]), n_exp).tolist()

    # Construct exponential basis matrix
    E = np.vstack([np.exp(-t / tau) for tau in taus]).T

    # Solve for amplitudes a_i via least squares
    a_vals, *_ = np.linalg.lstsq(E, y_adj, rcond=None)
    bad_a_count = sum(a_vals <= 0)
    if bad_a_count > 0:
        # Construct a data relavent "minimal" value
        a_min = 10 ** (int(np.floor(np.log10(abs(np.ptp(y_adj))))) - 2)
        if bad_a_count == len(a_vals):
            a_vals[:-1] = a_min
            a_vals[-1] = abs(np.mean(y_adj * np.exp(t / taus[-1])) - (bad_a_count - 1) * a_min)
        else:
            addative = (-sum(a_vals[a_vals <= 0]) - a_min * bad_a_count) / sum(a_vals > 0)
            a_vals = np.where(a_vals <= 0, a_min, a_vals + addative)

    a_vals = a_vals * sign  # type: ignore
    return sort_exp_params([p for i in range(n_exp) for p in (a_vals[i], taus[i])] + [b0])[0]
    # return sort_exp_params([p for i in range(n_exp) for p in (a_vals[i] * sign, taus[i])] + [b0])[
    #     0
    # ]


def initial_guess_str_exp_func(
    t: np.ndarray | pd.DataFrame,
    y: np.ndarray | None = None,
    beta_min: float = 0.3,
    **kwargs,
) -> list[float]:
    """
    Estimate initial parameters for stretched exponential:
    f(t) = a * exp[-(t / tau)^beta] + b

    Args:
        t (np.ndarray): Independent variable.
        y (np.ndarray): Dependent variable.
        initial_guess_exp_func (callable): Function that returns [a1, tau1, a2, tau2, b] for n_exp=2.
        beta_min (float): Minimum allowable beta value (defaults to 0.3).

    Returns:
        list: [a, tau, beta, b] initial parameter guess
    """
    if isinstance(t, pd.DataFrame):
        y = t.iloc[:, 1].to_numpy(copy=True) if y is None else y
        t = t.iloc[:, 0].to_numpy(copy=True)
    if y is None:
        raise ValueError("y must be provided if t is a DataFrame or Series.")

    # Get estimates from 2-exponential model
    a1, tau1, a2, tau2, b0 = initial_guess_exp_func(t, y, 2, **kwargs)

    # Beta: based on tau spread
    beta = np.log(2) / np.log(max(tau2 / tau1, tau1 / tau2))

    return [
        (abs(a1) + abs(a2)) * np.sign(b0),
        np.sqrt(tau1 * tau2),
        min(1.0, max(beta_min, beta)),
        b0,
    ]


def sort_exp_params(
    *values: list[float] | np.ndarray,
    offset: int = 0,
    grp_len: int = 2,
) -> list[list[float]]:
    """
    Sorts exponential fit parameters so that (a, tau) pairs are ordered by tau ascending.
    All additional arrays (i.e. stds or names) in *values are sorted in the same order as values[0].

    Parameters
    ----------
    *values : list[float] or np.ndarray
        Flat list/array of parameters: [a0, tau0, a1, tau1, ..., (b0)]
    offset : int, optional
        Extra offset to account for additional parameters (e.g., b0) at the end of the list.
    grp_len : int, optional
        Size of the group. Last value in group will be sorted.

    Returns
    -------
    list[list[float], ...]
        Sorted arrays, each with (a, tau) pairs ordered by tau ascending, offset (if present) last.
    """
    if not values or len(values[0]) <= grp_len * 2 - 1:
        return [list(v) for v in values]

    a0_n = int((len(values[0]) - offset) // grp_len * grp_len)

    arrays = [np.pad(np.asarray(v), (0, max(0, a0_n - len(v)))) for v in values]

    core, offsets = zip(*[[arr[:a0_n], arr[a0_n:].tolist()] for arr in arrays])

    reshaped = [arr.reshape(-1, grp_len) for arr in core]
    sort_idx = np.argsort(reshaped[0][:, grp_len - 1])  # sort by tau from first array

    return [r[sort_idx].flatten().tolist() + o for r, o in zip(reshaped, offsets)]


def run_segment_fit(
    t: np.ndarray,
    y: np.ndarray,
    fit_func: Callable,
    p0,
    fit_type: str = "ls",
    **kwargs,
) -> dict | None:
    """
    Fit a segment to the specified function.

    Parameters
    ----------
    t : np.ndarray
        Time array.
    y : np.ndarray
        Data array.
    fit_func : Callable
        Function to fit.
    p0 : tuple, optional
        Initial guess for parameters.
    fit_type : str, optional
        Fitting method. Default is "ls".
    **kwargs
        Additional keyword arguments passed to ls_fit.

    Returns
    -------
    fit_result : dict
        Dictionary with fit parameters, covariance, and fitted curve.
    """
    mask = np.isfinite(t) & np.isfinite(y)
    if np.sum(mask) < 3:
        return None

    t_fit = t[mask]
    y_fit = y[mask]

    bounds = kwargs.pop("bounds", None)

    fm = FittingMethods()
    if fit_type == "de" or "diff" in fit_type:
        func = fm.de_fit
    elif "basin" in fit_type:
        func = fm.basin_fit
    else:
        func = fm.ls_fit

    try:
        popt, perror = func(t_fit, y_fit, p0, fit_func, bounds=bounds, **kwargs)
        if popt is None:
            return None
        # y_fit_curve = fit_func(t_fit, *popt)
        return {
            "params": np.array(popt),
            "cov": perror,
            # "y_fit": y_fit_curve,
            "t": t_fit,
            "y": y_fit,
        }
    except Exception as exc:
        print(f"Error fitting segment: {exc}")
        raise exc


def calc_aicc(rss, n, k):
    """
    Compute AICc for least-squares fits.

    Parameters
    ----------
    rss : float
        Residual sum of squares.
    n : int
        Number of data points.
    k : int
        Number of fitted parameters.

    Returns
    -------
    float
        AICc value (lower is better).
    """
    if rss <= 0:
        return np.inf
    aic = n * np.log(rss / n) + 2 * k
    return aic + (2 * k * (k + 1)) / (n - k - 1) if n - k - 1 > 0 else aic


def eval_sum_of_squares(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """
    Evaluate the sum of squares between true and predicted values.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    tuple
        (ss_residual, ss_total, r_squared)
        - ss_residual: Residual sum of squares
        - ss_total: Total sum of squares
        - r_squared: R-squared value
    """
    ss_residual = float(np.sum((y_true - y_pred) ** 2))
    ss_total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else np.nan
    return ss_residual, ss_total, r_squared


def perform_arrhenius_fit(
    temps: np.ndarray,
    y_vals: np.ndarray,
    weights: np.ndarray | None = None,
    celsius_to_kelvin: bool = True,
    **kwargs,
) -> dict:
    """
    Perform Arrhenius fit on data (y = A * exp(-Ea / (kB*T))).

    Parameters
    ----------
    temps : np.ndarray
        Temperature values (K by default, or °C if celsius_to_kelvin=True).
    y_vals : np.ndarray
        Dependent variable values to fit (must be positive).
    weights : array-like, optional
        Weights to use for the fit.
    celsius_to_kelvin : bool, default False
        If True, convert input temps from °C to K.
    **kwargs : dict
        Passed through to perform_exponential_fit.

    Returns
    -------
    dict
        Dictionary with fit parameters including:
        - 'A': Pre-exponential factor
        - 'Ea': Activation energy in eV
        - 'R2': R-squared value of fit
        - 'fit': Fitted y_vals values
        - 'AICc': if requested
    """
    K_B_EV = 8.617333262145e-5  # eV/K

    x_in = np.asarray(temps, dtype=float)
    y_in = np.asarray(y_vals, dtype=float)

    res = {
        "A": np.nan,
        "Ea": np.nan,
        "R2": np.nan,
        "fit": np.full_like(y_in, np.nan, dtype=float),
    }
    if kwargs.get("aicc", False):
        res["AICc"] = np.inf
    try:
        x_in = 1.0 / (x_in + 273.15) if celsius_to_kelvin else 1.0 / x_in

        if "Ea" in kwargs:
            kwargs["beta"] = -kwargs.pop("Ea") / K_B_EV
        elif "A" in kwargs:
            kwargs["alpha"] = kwargs.pop("A")
        # Call the general exponential fitter
        exp_res = perform_exponential_fit(x_in, y_in, weights=weights, **kwargs)

        # Map slope β back to Ea
        res["A"] = exp_res["alpha"]
        res["Ea"] = -exp_res["beta"] * K_B_EV
        res["R2"] = exp_res["R2"]
        res["fit"] = exp_res["fit"]

        if "AICc" in exp_res:
            res["AICc"] = exp_res["AICc"]

    except Exception as e:
        print(f"Error in arrhenius fit: {e}")
        # raise e
    return res


def perform_linear_fit(
    x_vals: np.ndarray, y_vals: np.ndarray, weights: np.ndarray | None = None, **kwargs
) -> dict:
    """
    Perform linear fit on data (y = a*x + b).

    Parameters
    ----------
    x_vals : np.ndarray
        Independent variable values.
    y_vals : np.ndarray
        Dependent variable values to fit.
    weights : array-like, optional
        Weights to use for the fit.
    **kwargs : dict
        Additional parameters:
        - 'a': float, optional
            Pre-defined slope
        - 'b': float, optional
            Pre-defined intercept

    Returns
    -------
    dict
        Dictionary with fit parameters including:
        - 'a': Slope
        - 'b': Intercept
        - 'R2': R-squared value of fit
        - 'fit': Fitted y_vals values
        - 'model': Always 'linear'
    """
    x_in = np.asarray(x_vals, dtype=float)
    y_in = np.asarray(y_vals, dtype=float)

    res = {
        "m": np.nan,
        "b": np.nan,
        "R2": np.nan,
        "fit": np.full_like(y_in, np.nan, dtype=float),
    }
    if kwargs.get("aicc", False):
        res["AICc"] = np.inf
    try:
        mask = np.isfinite(x_in) & np.isfinite(y_in)
        x_arr = x_in[mask]
        if (length := len(x_arr)) >= 2:
            # Take log of y_vals for linear regression
            y_arr = y_in[mask]

            # Use predefined parameters if provided
            if "slope" in kwargs or "m" in kwargs:
                slope = kwargs.get("slope", kwargs["m"])
                intercept = np.mean(y_arr - slope * x_arr)
            elif "intercept" in kwargs or "b" in kwargs:
                intercept = kwargs.get("intercept", kwargs["b"])
                slope = np.mean((y_arr - intercept) / x_arr)
            else:
                weight = weights
                if weight is not None:
                    weight = np.asarray(weights, dtype=float)[mask]
                    if len(weight) != length or np.ptp(weight) == 0:
                        weights = None  # Reset weights if not diverse

                slope, intercept = np.polyfit(x_arr, y_arr, 1, w=weight)

            res["b"] = intercept
            res["m"] = slope

            res["fit"] = intercept + slope * x_in

            ss_residual, _, res["R2"] = eval_sum_of_squares(y_arr, intercept + slope * x_arr)

            if kwargs.get("aicc", False):
                res["AICc"] = calc_aicc(ss_residual, length, 2)  # 2 params: alpha, beta

    except Exception as e:
        print(f"Error in linear fit: {e}")
        # raise e
    return res


def perform_exponential_fit(
    x_vals: np.ndarray, y_vals: np.ndarray, weights: np.ndarray | None = None, **kwargs
) -> dict:
    """
    Perform exponential fit on data (y = 𝛼*exp(β*x)).

    Parameters
    ----------
    x_vals : np.ndarray
        Independent variable values.
    y_vals : np.ndarray
        Dependent variable values to fit (must be positive).
    weights : array-like, optional
        Weights to use for the fit.
    **kwargs : dict
        Additional parameters:
        - 'alpha': float, optional
            Pre-defined pre-exponential (scale) factor
        - 'beta': float, optional
            Pre-defined exponent (rate) factor

    Returns
    -------
    dict
        Dictionary with fit parameters including:
        - 'alpha': Pre-exponential factor
        - 'beta': Exponent factor
        - 'R2': R-squared value of fit
        - 'fit': Fitted y_vals values
    """
    x_in = np.asarray(x_vals, dtype=float)
    y_in = np.asarray(y_vals, dtype=float)

    res = {
        "alpha": np.nan,
        "beta": np.nan,
        "R2": np.nan,
        "fit": np.full_like(y_in, np.nan, dtype=float),
    }
    if kwargs.get("aicc", False):
        res["AICc"] = np.inf
    try:
        mask = np.isfinite(x_in) & np.isfinite(y_in)
        x_arr = x_in[mask]
        if (length := len(x_arr)) >= 2:
            # Take log of y_vals for linear regression
            y_arr = y_in[mask]
            sign = int(np.median(np.sign(y_arr))) or int(kwargs.get("sign", 1))
            y_arr[y_arr == 0] = kwargs.get("min_val", 1e-32) * sign  # Avoid log(0)

            # Use predefined parameters if provided
            if "b" in kwargs or "beta" in kwargs:
                slope = kwargs.get("b", kwargs["beta"])
                intercept = np.mean(np.log(np.abs(y_arr)) - slope * x_arr)
            elif "a" in kwargs or "alpha" in kwargs:
                intercept = np.log(abs(kwargs.get("a", kwargs["alpha"])))
                slope = np.mean((np.log(np.abs(y_arr)) - intercept) / x_arr)
            else:
                weight = weights
                if weight is not None:
                    weight = np.asarray(weights, dtype=float)[mask]
                    if len(weight) != length or np.ptp(weight) == 0:
                        weights = None  # Reset weights if not diverse

                slope, intercept = np.polyfit(x_arr, np.log(np.abs(y_arr)), 1, w=weight)

            res["alpha"] = sign * np.exp(min(intercept, 700))
            res["beta"] = slope

            res["fit"] = res["alpha"] * np.exp(np.clip(slope * x_in, None, 700))

            ss_residual, _, res["R2"] = eval_sum_of_squares(
                y_arr, res["alpha"] * np.exp(np.clip(slope * x_arr, None, 700))
            )

            if kwargs.get("aicc", False):
                res["AICc"] = calc_aicc(ss_residual, length, 2)  # 2 params: alpha, beta

    except Exception as e:
        print(f"Error in exponential fit: {e}")
        # raise e
    return res


def perform_general_fit(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    weights: np.ndarray | None = None,
    mode: str = "",
    arrh_form: bool = False,
    **kwargs,
):
    """
    Try both linear and exponential fits, return the better one
    based on AICc.

    Parameters
    ----------
    x_vals : np.ndarray
        Independent variable values.
    y_vals : np.ndarray
        Dependent variable values.
    weights : array-like, optional
        Weights for fitting.
    mode : str, default ""
        If "lin", only linear fit is performed.
        If "exp", only exponential fit is performed.
        Otherwise, both are tried and the best is returned.
    arrh_form : bool, default False
        If True, use Arrhenius form for exponential fit.
    **kwargs : dict
        Passed through to underlying fitters.

    Returns
    -------
    dict
        Dictionary with:
        - 'alpha'/'A'/'b': Pre-exponential factor or intercept
        - 'beta'/'Ea'/'m': Exponent factor or slope
        - 'fit': fitted y-values
        - 'R2': R-squared of fit
    """
    # Call your existing fitters
    if "exp" in mode.lower() or "arrh" in mode.lower():
        if arrh_form or "arrh" in mode.lower():
            return perform_arrhenius_fit(x_vals, y_vals, weights=weights, **kwargs)
        return perform_exponential_fit(x_vals, y_vals, weights=weights, **kwargs)
    elif "lin" in mode.lower():
        return perform_linear_fit(x_vals, y_vals, weights=weights, **kwargs)

    kwargs["aicc"] = True

    lin_result = perform_linear_fit(x_vals, y_vals, weights=weights, **kwargs)
    if arrh_form:
        exp_result = perform_arrhenius_fit(x_vals, y_vals, weights=weights, **kwargs)
    else:
        exp_result = perform_exponential_fit(x_vals, y_vals, weights=weights, **kwargs)

    # Extract AICc values
    aicc_lin = lin_result.pop("AICc", np.inf)
    aicc_exp = exp_result.pop("AICc", np.inf)

    # Choose model with lower AICc
    if aicc_exp < aicc_lin:
        return exp_result
    return lin_result


def extract_x_values(
    ref_df: pd.DataFrame,
    x_name: str | list[str] | tuple[str, ...] = (),
    min_unique: int = 2,
    dtype: type = float,
) -> pd.Series:
    """
    Extract x values (e.g., temperatures) from a DataFrame.

    Parameters
    ----------
    ref_df : pd.DataFrame
        DataFrame containing the x values
    x_name : str or list[str] or tuple[str, ...], default ()
        Name(s) of the x variable to search for in priority order
    min_unique : int, default 2
        Minimum number of unique values required for selection when x_name is a list/tuple
    dtype : type, default float
        Data type to cast the extracted values to

    Returns
    -------
    pd.Series
        Extracted x values

    Raises
    ------
    KeyError
        If none of the provided names are found in DataFrame columns or index

    Notes
    -----
    The function searches for x_name in DataFrame columns and index.
    For multiple x_name options, it selects the first one found that has
    at least min_unique unique values.
    """
    # Reset index to have all data as columns for uniform access
    df = ref_df.reset_index(drop=False, inplace=False)
    # Handle single name case
    if isinstance(x_name, str) and x_name:
        x_name = [x_name]

    # Track the first valid result we find
    values = None

    # Try each name in order
    for name in x_name or df.columns:
        # Try to extract values from this name
        if name not in df.columns:
            continue

        # if dtype is numeric and x_name was empty, ensure we only consider numeric columns
        if pd.api.types.is_numeric_dtype(dtype) != pd.api.types.is_numeric_dtype(df[name]):
            continue

        if df[name].nunique(dropna=True) >= min_unique:
            values = pd.Series(df[name], dtype=dtype)
            values.attrs["name"] = name
            return values

        if values is None:
            values = pd.Series(df[name], dtype=dtype)
            values.attrs["name"] = name

    # If we get here, no name had enough unique values
    if values is not None:
        return values

    # If no valid result, raise an error
    raise KeyError(f"None of the provided names {x_name} found in DataFrame columns or index")


def data_trend_eval(
    df: pd.DataFrame,
    x_data: str | list[str] | tuple[str, ...] | np.ndarray,
    fit_cols: set[str] | list[str] | tuple[str, ...] = (),
    fit_func: Callable | str = "",
    sign_default: int = 0,
    fit_resid: bool = False,
    pass_kwargs_eval: Callable[[str], bool] = lambda col: False,
    **kwargs,
) -> pd.DataFrame:
    """
    Apply fits to specified columns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to process
    x_data : str | list[str] | tuple[str, ...] | np.ndarray
        Independent variable name(s) or values
    fit_cols : set[str] | list[str] | tuple[str, ...], default ()
        Columns to fit, uses all columns if empty
    fit_func : Callable | str, default "arrhenius"
        Function to use for fitting or name ("arrhenius", "exponential", "linear", "general")
    sign_default : int, default 0
        If non-zero and data contains mixed signs, converts to abs(data)*sign_default
    fit_resid : bool, default False
        Whether to calculate and store residuals
    pass_kwargs_eval : Callable[[str], bool], default lambda col: False
        Function to determine if kwargs should be passed for a column
    **kwargs : dict
        Additional arguments passed to fitting function

    Returns
    -------
    pd.DataFrame
        Modified DataFrame with fitting results in columns and attrs
    """
    df0 = df.copy()
    cond_kwargs = kwargs.pop("cond_kwargs", {})

    if not fit_cols:
        fit_cols = df0.columns.to_list()

    if isinstance(fit_func, str):
        if fit_func:
            fit_func = fit_func.lower()
            if "arrh" in fit_func:
                fit_func = perform_arrhenius_fit
            elif "exp" in fit_func and "lin" not in fit_func:
                fit_func = perform_exponential_fit
            elif "lin" in fit_func and "exp" not in fit_func:
                fit_func = perform_linear_fit
            else:
                fit_func = perform_general_fit
        else:
            fit_func = perform_general_fit

    if not isinstance(x_data, np.ndarray):
        x_arr = extract_x_values(df0, x_data, dtype=kwargs.get("dtype", float))
        x_data = x_arr.to_numpy(copy=True)
        kwargs["arrh_form"] = "temp" in x_arr.attrs.get("name", "").lower()

    for col in fit_cols:
        if col not in df0.columns:
            continue

        # Handle mixed sign data
        if sign_default and any(df0[col] < 0) and any(df0[col] > 0):
            df0[col] = np.abs(df0[col]) * sign_default

        # Calculate weights from Error column if available
        e_weights = df0.get("Error", None)
        if e_weights is not None:
            e_weights = e_weights.max() - e_weights
            e_weights = np.clip(e_weights / e_weights.max(), 1e-32, 1)

        # Apply Arrhenius fit with proper parameters
        fit_kwargs = kwargs.copy()
        if pass_kwargs_eval(col):
            fit_kwargs |= cond_kwargs

        res = fit_func(x_data, df0[col].to_numpy(), weights=e_weights, **fit_kwargs)
        # Process results
        res = {f"{col}_{k}": v for k, v in res.items()}
        df0[f"{col}_fit"] = res.pop(f"{col}_fit")

        if fit_resid:
            df0[f"{col}_fit_resid"] = df0[col] - df0[f"{col}_fit"]

        df0.attrs |= res

    return df0


def data_group_trend_eval(
    data: dict[str, pd.DataFrame],
    x_data: str | list[str] | tuple[str, ...] = (),
    columns: set[str] | list[str] | tuple[str, ...] = (),
    skip: Callable[[Any], bool] = lambda x: False,
    col_selector: Callable[[str], bool] = lambda col: col[-1].isdigit(),
    pass_kwargs_eval: Callable[[str, str], bool] = lambda key, col: False,
    **kwargs,
) -> tuple[dict, dict]:
    """
    Apply fits to specified columns in grouped DataFrames.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        Dictionary of DataFrames to process
    x_data : str | list[str] | tuple[str, ...], default ()
        Independent variable name(s) or values
    columns : set[str] | list[str] | tuple[str, ...], default ()
        Specific columns to fit, empty uses col_selector to determine columns
    skip : Callable[[Any], bool], default lambda x: False
        Function to determine if a DataFrame should be skipped
    col_selector : Callable[[str], bool], default lambda col: col[-1].isdigit()
        Function to select which columns to fit when columns is empty
    pass_kwargs_eval : Callable[[str, str], bool], default lambda key, col: False
        Function to determine if kwargs should be passed for a column given the key and column
    **kwargs : dict
        Additional arguments passed to data_trend_eval

    Returns
    -------
    tuple[dict, dict]
        Tuple containing:
        - Dictionary with the same structure as input, with fit results in DataFrame attrs
        - Dictionary of extracted fit results keyed by the original dictionary keys
    """
    revised_data = {}
    fit_results = {}

    for key, df0 in data.items():
        if skip(df0):
            revised_data[key] = df0
            continue

        # Get unique temperatures and sort them
        fit_cols = columns if columns else [s for s in df0.columns if col_selector(s)]
        df = df0[[s for s in df0.columns if "_fit" not in s]].copy()
        df.attrs = {}

        # Apply Arrhenius fits using the helper function
        df = data_trend_eval(
            df,
            x_data,
            fit_cols,
            pass_kwargs_eval=lambda col: pass_kwargs_eval(key, col),
            **kwargs,
        )

        fit_results[key] = df.attrs

        revised_data[key] = df

    return revised_data, fit_results


def nested_data_group_trend_eval(
    grouped_data: dict[str, dict[str, pd.DataFrame]],
    x_data: str | list[str] | tuple[str, ...] = (),
    data_sets: set[str] | list[str] | tuple[str, ...] = (),
    skip: Callable = lambda x: "vals" not in x,
    col_selector: Callable = lambda col: col[-1].isdigit(),
    pass_kwargs_eval: Callable[[str, str], bool] = lambda key, col: False,
    **kwargs,
) -> tuple[dict, dict]:
    """
    Apply fits to specified columns in nested grouped DataFrames.

    Parameters
    ----------
    grouped_data : dict[str, dict[str, pd.DataFrame]]
        Dictionary of dictionaries of DataFrames to process
    x_data : str | list[str] | tuple[str, ...], default ()
        Independent variable name(s) or values
    data_sets : set[str] | list[str] | tuple[str, ...], default ()
        Data set keys to include in fitting, empty includes all
    skip : Callable, default lambda x: "vals" not in x
        Function to determine if a top-level entry should be skipped
    col_selector : Callable, default lambda col: col[-1].isdigit()
        Function to select which columns to fit
    pass_kwargs_eval : Callable[[str, str], bool], default lambda key, col: False
        Function to determine if kwargs should be passed for a column given the key and column
    **kwargs : dict
        Additional arguments passed to data_trend_eval

    Returns
    -------
    tuple[dict, dict]
        Tuple containing:
        - Dictionary with the same structure as input, with fit results in DataFrame attrs
        - Dictionary of extracted fit results keyed by "{top_key} {key}"
    """
    revised_data = defaultdict(dict)
    fit_results = defaultdict(dict)

    for top_key, vals in grouped_data.items():
        if skip(top_key):
            revised_data[top_key] = vals
            continue

        for key, df0 in vals.items():
            if data_sets and key not in data_sets:
                revised_data[top_key][key] = df0
                continue

            fit_cols = [s for s in df0.columns if col_selector(s)]
            df = df0[[s for s in df0.columns if "_fit" not in s]].copy()
            df.attrs = {}

            # Apply Arrhenius fits using the helper function
            df = data_trend_eval(
                df,
                x_data,
                fit_cols,
                pass_kwargs_eval=lambda col: pass_kwargs_eval(key, col),
                **kwargs,
            )

            fit_results[f"{top_key} {key}"] |= df.attrs
            df.attrs = df0.attrs.copy() | df.attrs
            revised_data[top_key][key] = df

    return dict(revised_data), dict(fit_results)

    # kwargs.setdefault("Ea", 0.5914)


# def perform_arrhenius_fit(temps, y_vals, celsius_to_kelvin=True, weights=None, **kwargs):
#     """
#     Perform Arrhenius fit on data (ln(y) vs 1/T).

#     Parameters
#     ----------
#     temps : array-like
#         Temperature values (in Celsius by default).
#     y_vals : array-like
#         values to fit (Current or Resistance).
#     celsius_to_kelvin : bool, optional
#         Whether to convert temperatures from Celsius to Kelvin, default is True.

#     Returns
#     -------
#     dict
#         Dictionary with fit parameters including:
#         - 'Ea': Activation energy in kJ/mol
#         - 'A': Pre-exponential factor
#         - 'R2': R-squared value of fit
#         - 'slope': Slope of the linear fit
#         - 'intercept': Intercept of the linear fit
#     """
#     # Convert to numpy arrays and handle NaN y_vals
#     temps = np.asarray(temps, dtype=float)
#     y_vals = np.asarray(y_vals, dtype=float)

#     if pd.Series(weights).nunique() < 2:
#         weights = None  # Reset weights if not diverse enough

#     activation_energy = np.nan
#     pre_exp_factor = np.nan
#     r_squared = np.nan
#     y_pred = np.full_like(y_vals, np.nan, dtype=float)
#     signs = np.sign(y_vals)
#     try:
#         if len(temps) >= 2:
#             # Transform for Arrhenius fit
#             x_arr = 1.0 / (temps + 273.15) if celsius_to_kelvin else 1.0 / temps
#             y_arr = np.abs(y_vals)

#             y_arr[y_arr <= 0] = np.nan  # Avoid log(0) or log(negative)
#             y_arr = np.log(np.nan_to_num(y_arr, nan=1e-16))  # Replace NaN with a small number

#             # Perform linear fit
#             if "Ea" in kwargs:
#                 slope = -kwargs["Ea"] / 8.617e-5  # Convert Ea from kJ/mol to eV
#                 intercept = np.mean(y_arr - slope * x_arr)
#             elif "A" in kwargs:
#                 intercept = np.log(kwargs["A"])
#                 slope = np.mean((y_arr - intercept) / x_arr)
#             else:
#                 slope, intercept = np.polyfit(x_arr, y_arr, 1, w=weights)

#             # Calculate R²
#             y_pred = slope * x_arr + intercept
#             ss_total = np.sum((y_arr - np.mean(y_arr)) ** 2)
#             ss_residual = np.sum((y_arr - y_pred) ** 2)
#             r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0

#             # Calculate Arrhenius parameters
#             activation_energy = -slope * 8.617e-5  # Ea in eV
#             pre_exp_factor = np.exp(intercept)  # A

#             y_pred = np.exp(y_pred) * signs  # Convert back to original scale

#     except Exception as e:
#         print(f"Error in Arrhenius fit: {e}")
#     return {
#         "Ea": activation_energy,
#         "A_val": pre_exp_factor,
#         "A_sign": np.median(signs),
#         "R2": r_squared,
#         "fit": y_pred,
#     }
