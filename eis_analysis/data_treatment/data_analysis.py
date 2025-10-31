# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 22:12:37 2022.

@author: j2cle
"""
import warnings
from typing import Any, Literal, TypeAlias, overload
from collections.abc import Callable

import numpy as np
import scipy
import pandas as pd
from scipy import stats
from numpy.typing import ArrayLike
from scipy.optimize import (
    Bounds,
    OptimizeResult,
    curve_fit,
    basinhopping,
    least_squares,
    differential_evolution,
)

IsFalse: TypeAlias = Literal[False]
IsTrue: TypeAlias = Literal[True]


def calculate_rc_freq(
    r_value: int | float | None = None,
    c_value: int | float | None = None,
    freq_value: int | float | None = None,
) -> float | None:
    """Calculate R, C, or freq based on the other two values."""
    if r_value is None:
        if c_value is not None and freq_value is not None:
            res = 1 / (2 * np.pi * freq_value * c_value)
            return res
        else:
            print("Please provide values for C and freq to calculate R.")
    elif c_value is None:
        if r_value is not None and freq_value is not None:
            res = 1 / (2 * np.pi * freq_value * r_value)
            return res
        else:
            print("Please provide values for R and freq to calculate C.")
    elif freq_value is None:
        if r_value is not None and c_value is not None:
            res = 1 / (2 * np.pi * r_value * c_value)
            return res
        else:
            print("Please provide values for R and C to calculate freq.")
    else:
        print("Please leave one field empty to calculate its value.")
    return None


# Unused
def single_rc(data: dict) -> dict:
    res = {}
    try:
        res["RC1"] = (
            calculate_rc_freq(data["R1_values"], data["CPE1_0_values"])
            if "CPE1_0_values" in data
            else calculate_rc_freq(data["R1_values"], data["C1_values"])
        )
        res["RC2"] = (
            calculate_rc_freq(data["R2_values"], data["CPE2_0_values"])
            if "CPE2_0_values" in data
            else calculate_rc_freq(data["R2_values"], data["C2_values"])
        )
        res["RC3"] = (
            calculate_rc_freq(data["R3_values"], data["CPE3_0_values"])
            if "CPE3_0_values" in data
            else calculate_rc_freq(data["R3_values"], data["C3_values"])
        )
    except Exception:
        pass
    return res


# Unused
def quick_rc(dataset):

    reslist = []
    for _, data in dataset.iterrows():
        try:
            res = {}
            res["RC1"] = (
                calculate_rc_freq(data["R1_values"], data["CPE1_0_values"])
                if "CPE1_0_values" in data
                else calculate_rc_freq(data["R1_values"], data["C1_values"])
            )
            res["RC2"] = (
                calculate_rc_freq(data["R2_values"], data["CPE2_0_values"])
                if "CPE2_0_values" in data
                else calculate_rc_freq(data["R2_values"], data["C2_values"])
            )
            res["RC3"] = (
                calculate_rc_freq(data["R3_values"], data["CPE3_0_values"])
                if "CPE3_0_values" in data
                else calculate_rc_freq(data["R3_values"], data["C3_values"])
            )
            reslist.append(res)
        except Exception:
            continue
    return reslist


# Unused
def calculate_rc_pairs(names, values):
    res = {}
    for i in range(len(names) - 1):
        if names[i].startswith("R") and names[i + 1].startswith("C"):
            rc_name = f"{names[i]}_{names[i + 1]}"
            res[rc_name] = calculate_rc_freq(values[i][0], values[i + 1][0])
    return res


# Unused
def parse_rc_row(rc_s):
    return [f"RC1: {row['RC1']:.3e}, RC2: {row['RC2']:.3e}, RC3: {row['RC3']:.3e}" for row in rc_s]


class ConfidenceAnalysis:
    """Generic confidence analysis that does not assume impedance-specific structures.

    Contains only numpy / pandas friendly logic and the basic properties used by
    impedance-specific subclasses. Designed to be independent of ComplexSystem.
    """

    NP_2EPS: float = 10 ** int(np.floor(np.log10(np.finfo(np.float64).eps)) * 2)

    def __init__(
        self,
        percentile: int | float = 5,
        popt: ArrayLike | None = None,
        std: ArrayLike = 5,
        # func: Callable | None = None,
    ):
        self._popt = np.array([1.0])
        self._std = np.array([0.1])
        self._percentile = (0.025, 0.975)
        self.percentile = percentile
        self.popt = popt
        self.std = std

    @property
    def popt(self) -> np.ndarray:
        """Contains the best-fit parameters."""
        return self._popt

    @popt.setter
    def popt(self, value: ArrayLike | None):
        if value is not None:
            self._popt = np.array(value)

    @property
    def std(self) -> np.ndarray:
        """Contains the standard deviation of the parameters."""
        return self._std

    @std.setter
    def std(self, value: ArrayLike | None):
        if isinstance(value, (int, float)):
            val_max = 100 if value >= 1 else 1
            self._std = np.array(self.popt) * value / val_max
        elif value is not None:
            self._std = np.array(value)

    @property
    def percentile(self) -> tuple[float, float]:
        """Gets the percentile data which is a list of two floats"""
        return self._percentile

    @percentile.setter
    def percentile(self, value: Any):
        if isinstance(value, (int, float)):
            percent_max = 100 if value >= 1 else 1
            value = [value / 2, (percent_max - value / 2)]
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            percent_max = 100 if max(value) >= 1 else 1
        else:
            return
        self._percentile = (
            float(min(value) / percent_max),
            float(max(value) / percent_max),
        )

    @property
    def param_bounds(self) -> list[tuple[float, float]]:
        """Generate parameter bounds based on popt and std."""
        return [(float(p - s), float(p + s)) for p, s in zip(self.popt, self.std)]

    def ci_from_data(self, data):
        """Generate confidence intervals from data."""
        perc = self.percentile[0] * 100
        ci_lower = np.percentile(data, perc, axis=0)
        ci_upper = np.percentile(data, 100 - perc, axis=0)
        return (ci_lower + ci_upper) / 2

    def ci_from_std(self, data, popt, perr):
        """Generate confidence intervals from standard deviations (uses t distribution)."""
        n = len(data)  # number of data points
        p = len(popt)  # number of parameters
        dof = max(0, n - p)  # degrees of freedom
        tval = scipy.stats.t.ppf(1.0 - self.percentile[0] / 2.0, dof)
        return tval * perr

    def safe_invert(self, mat: np.ndarray, threshold: float = 1e12) -> np.ndarray:
        """Safely invert a matrix, applying regularization if necessary."""
        cond_num = 0.0
        lambda_mod = 0.0
        try:
            cond_num = np.linalg.cond(mat)
            if cond_num >= threshold:
                cond_exp = int(np.floor(np.log10(cond_num)))
                lambda_mod = cond_num * np.eye(mat.shape[0]) / 10**cond_exp
            return np.linalg.inv(mat + lambda_mod)
        except (np.linalg.LinAlgError, FloatingPointError, OverflowError):
            if not lambda_mod and cond_num and threshold != -np.inf:
                # Fallback: inversion failed without regularization applied
                return self.safe_invert(mat, threshold=-np.inf)
            # Return a scaled identity matrix if inversion fails
            return np.eye(mat.shape[0]) * self.NP_2EPS

    def estimate_param_jacobian(
        self,
        func: Callable,
        params: list | np.ndarray,
        x_data: tuple | list | np.ndarray,
        epsilon: float = 1e-8,
    ) -> np.ndarray:
        """Estimate the Jacobian matrix using finite differences."""

        f0 = func(x_data, *params)
        N = f0.size
        P = len(params)
        J = np.zeros((N, P))
        fnull = np.full(N, self.NP_2EPS)
        for i in range(P):
            try:
                eps = epsilon * max(np.abs(params[i]), self.NP_2EPS)  # Avoid division by zero

                params1, params2 = params.copy(), params.copy()
                params1[i] += eps
                params2[i] -= eps

                f1 = func(x_data, *params1).reshape(N, order="F")
                f2 = func(x_data, *params2).reshape(N, order="F")
                deriv = (f1 - f2) / (2 * eps)

                deriv = np.nan_to_num(deriv, nan=self.NP_2EPS, posinf=1e300, neginf=-1e300)
            except FloatingPointError:
                deriv = fnull.copy()  # Fallback for numerical issues
            J[:, i] = deriv

        return J

    def estimate_dep_jacobian(
        self,
        func: Callable,
        params: list | np.ndarray,
        x_data: tuple | list | np.ndarray,
        epsilon: float = 1e-8,
    ) -> np.ndarray:
        """
        Estimate the Jacobian of dependent outputs wrt parameters using finite differences.

        This assumes func(x_data, *params) returns the full stacked output vector
        (same contract as estimate_param_jacobian). The result is reshaped into
        (N, K, P) where N = len(x_data), K = outputs per input, P = number of params.
        """

        # Evaluate once to get shape
        P = len(params)
        f0 = np.asarray(func(x_data, *params))
        # Infer number of outputs per input
        N = len(x_data)
        if f0.size % N != 0:
            raise ValueError("Output length not divisible by number of inputs")
        K = f0.size // N

        # Allocate
        J_dep = np.zeros((N, K, P))
        fnull = np.full((N, K), self.NP_2EPS)

        for i in range(P):
            try:
                eps = epsilon * max(np.abs(params[i]), self.NP_2EPS)

                params1, params2 = params.copy(), params.copy()
                params1[i] += eps
                params2[i] -= eps

                f1 = func(x_data, *params1).reshape(N, K, order="F")
                f2 = func(x_data, *params2).reshape(N, K, order="F")

                deriv = (f1 - f2) / (2 * eps)
                deriv = np.nan_to_num(deriv, nan=self.NP_2EPS, posinf=1e300, neginf=-1e300)
            except FloatingPointError:
                deriv = fnull.copy()  # Fallback for numerical issues

            J_dep[:, :, i] = deriv

        return J_dep

    def get_param_cov(
        self,
        func: Callable,
        params: list | np.ndarray,
        x_data: tuple | list | np.ndarray,
        epsilon: float = 1e-8,
        sigma2: float = 1.0,
    ) -> np.ndarray:
        """
        Compute parameter covariance matrix.
        If pcov not provided, estimate from Jacobian.
        """
        J = self.estimate_param_jacobian(func, params, x_data, epsilon=epsilon)
        JTJ = J.T @ J
        JTJ_inv = self.safe_invert(JTJ)
        return sigma2 * JTJ_inv

    def get_dep_cov(
        self,
        func: Callable,
        params: list | np.ndarray,
        x_data: tuple | list | np.ndarray,
        pcov: np.ndarray | None = None,
        epsilon: float = 1e-8,
        sigma2: float = 1.0,
    ) -> np.ndarray:
        """
        Compute dependent-output covariance matrices.
        If pcov is None, compute it internally from param Jacobian.
        Returns array of shape (N, K, K).
        """
        if pcov is None:
            pcov = self.get_param_cov(func, params, x_data, sigma2=sigma2, epsilon=epsilon)

        J_dep = self.estimate_dep_jacobian(func, params, x_data, epsilon=epsilon)
        Sigma_y = np.einsum("nkp,pq,nlq->nkl", J_dep, pcov, J_dep)
        return Sigma_y

    def gen_conf_band(
        self,
        x_array: ArrayLike,
        func: Callable,
        popt: ArrayLike | None = None,
        std: ArrayLike | None = None,
        percentile=None,
        target_form: str | list | tuple = "Z",
        **kwargs,
    ):
        """
        Generate confidence bands for both Nyquist and Bode plots.

        Parameters:
        x_array (array): Array of frequencies.
        popt (array): Best-fit parameters.
        std (array or float): Standard deviations of the parameters.
        func (function): Function to simulate the circuit.
        percentile (float): Percentile for confidence bands.
        target_form (str): Frequency data.

        Returns:
        bounds (dict): Dictionary containing the confidence bands.
        """

        if popt is not None:
            self.popt = popt
        if std is not None:
            self.std = std
        if percentile is not None:
            self.percentile = percentile

        # Evaluate model prediction
        x_array = np.asarray(x_array)
        y_pred = func(x_array, *self.popt)
        N = len(x_array)
        y_pred = np.asarray(y_pred).reshape(N, -1, order="F")

        # Compute parameter covariance
        pcov = self.get_param_cov(func, self.popt, x_array)

        # Compute dependent covariance
        dep_cov = self.get_dep_cov(func, self.popt, x_array, pcov=pcov)

        # Extract standard deviations for each output channel
        dep_std = np.sqrt(np.einsum("nii->n", dep_cov))

        # Percentile bounds
        lower_q, upper_q = self.percentile
        lower = y_pred - dep_std[:, None] * abs(lower_q)
        upper = y_pred + dep_std[:, None] * abs(upper_q)

        # Normalize target_form into iterable
        if isinstance(target_form, (list, tuple)):
            keys = target_form
        else:
            keys = [target_form]

        # Build DataFrames
        bounds: dict[str, pd.DataFrame] = {}
        for j, key in enumerate(keys):
            df = pd.DataFrame(
                {
                    "x": x_array,
                    "y_pred": y_pred[:, j],
                    "lower": lower[:, j],
                    "upper": upper[:, j],
                }
            )
            bounds[key] = df

        return bounds


class Statistics:
    """Return sum of squared errors (pred vs actual)."""

    NP_EPS_EXP = int(np.floor(np.log10(np.finfo(np.float64).eps)))
    NP_SYS_FMAX = 10 ** np.floor(np.log10(np.finfo(np.float64).max))
    SCIPY_SCALE_75 = stats.norm.ppf(0.75)

    def __init__(self):
        pass

    def __getitem__(self, key) -> Any:
        if hasattr(self, key):
            return getattr(self, key)

    def as_array(self, key):
        """Return the array of the given key."""
        if key == "r2_score":
            return self.squared_err
        parts = key.split("_")
        n = 0
        while n < len(parts) and "_".join(parts[n:]) not in self.array_method_list:
            n += 1
        # new_key = "_".join(key.split("_")[1:])
        new_key = "_".join(parts[n:])
        if hasattr(self, new_key):
            return getattr(self, new_key)
        return self.residuals

    @property
    def array_method_list(self):
        """Return the list of array method names."""
        return [
            "residuals",
            "squared_err",
            "max_err",
            "abs_err",
            "abs_perc_err",
            "log_error",
            "squared_log_error",
            "log10_error",
            "squared_log10_error",
        ]

    @property
    def single_method_list(self):
        """Return the list of single method names."""
        return [
            "r2_score",
            "mean_abs_err",
            "median_abs_err",
            "total_abs_err",
            "mean_squared_err",
            "total_squared_err",
            "root_mean_abs_err",
            "root_mean_squared_err",
            "mean_abs_perc_err",
            "total_abs_perc_err",
            "mean_log_error",
            "total_log_error",
            "mean_squared_log_error",
            "total_squared_log_error",
            "mean_log10_error",
            "total_log10_error",
            "mean_squared_log10_error",
            "total_squared_log10_error",
            "mean_poisson_dev",
            "mean_gamma_dev",
        ]

    @property
    def array_method_abbr(self):
        """Return a dictionary mapping abbreviated names to array method names."""
        return {
            "RES": "residuals",
            "SQE": "squared_err",
            "MaxE": "max_err",
            "AbsE": "abs_err",
            "APE": "abs_perc_err",
            "LE": "log_error",
            "SLE": "squared_log_error",
            "L10E": "log10_error",
            "SL10E": "squared_log10_error",
        }

    @property
    def single_method_abbr(self):
        """Return a dictionary mapping abbreviated names to single method names."""
        return {
            "R2": "r2_score",
            "MAE": "mean_abs_err",
            "MedAE": "median_abs_err",
            "TAE": "total_abs_err",
            "MSE": "mean_squared_err",
            "TSE": "total_squared_err",
            "RMAE": "root_mean_abs_err",
            "RMSE": "root_mean_squared_err",
            "MAPE": "mean_abs_perc_err",
            "TAPE": "total_abs_perc_err",
            "MLE": "mean_log_error",
            "TLE": "total_log_error",
            "MSLE": "mean_squared_log_error",
            "TSLE": "total_squared_log_error",
            "ML10E": "mean_log10_error",
            "TL10E": "total_log10_error",
            "MSL10E": "mean_squared_log10_error",
            "TSL10E": "total_squared_log10_error",
            "MPD": "mean_poisson_dev",
            "MGD": "mean_gamma_dev",
        }

    @property
    def array_methods(self):
        """Return the list of array method functions."""
        return [
            self.residuals,
            self.squared_err,
            self.max_err,
            self.abs_err,
            self.abs_perc_err,
            self.log_error,
            self.squared_log_error,
            self.log10_error,
            self.squared_log10_error,
        ]

    @property
    def single_methods(self):
        """Return the list of single method functions."""
        return [
            self.r2_score,
            self.mean_abs_err,
            self.median_abs_err,
            self.total_abs_err,
            self.mean_squared_err,
            self.total_squared_err,
            self.root_mean_squared_err,
            self.root_mean_abs_err,
            self.mean_abs_perc_err,
            self.total_abs_perc_err,
            self.mean_log_error,
            self.total_log_error,
            self.mean_squared_log_error,
            self.total_squared_log_error,
            self.mean_log10_error,
            self.total_log10_error,
            self.mean_squared_log10_error,
            self.total_squared_log10_error,
            self.mean_poisson_dev,
            self.mean_gamma_dev,
        ]

    @staticmethod
    @overload
    def parse_weights(
        weight: np.ndarray | None, length: int, for_numpy: IsTrue, nan_to_num: bool = ...
    ) -> np.ndarray | None: ...

    @staticmethod
    @overload
    def parse_weights(
        weight: np.ndarray | None, length: int, for_numpy: IsFalse = False, nan_to_num: bool = ...
    ) -> np.ndarray: ...

    @staticmethod
    def parse_weights(
        weight: np.ndarray | None, length: int, for_numpy: bool = False, nan_to_num: bool = False
    ) -> np.ndarray | None:
        """
        Normalize or generate weights for error calculations.

        If valid weights are provided, returns weights scaled by length/sum.
        Otherwise, returns an array of ones.

        Parameters
        ----------
        weight : np.ndarray | None
            Array of weights or None.
        length : int
            Length of the data arrays.
        for_numpy : bool, optional
            If True, returns validated unnormalized weights (or None) for NumPy functions (see Notes).
            If False (default), returns normalized weights intended for custom calculations.

        Returns
        -------
        np.ndarray
            Normalized weight array.

        Notes
        ----
        For NumPy functions that accept a `weights` argument (e.g., `np.average`), passing normalized
        weights (e.g., from `parse_weights`) is not necessary and may result in small differences
        (~1e-16 from mean) due to floating-point precision.
        """
        if weight is not None and len(weight) == length and np.sum(weight) != 0:
            weight = np.nan_to_num(np.asarray(weight)) if nan_to_num else np.asarray(weight)
            return weight if for_numpy else weight * length / np.sum(weight)
        return None if for_numpy else np.ones(length)

    @staticmethod
    @overload
    def parse_arrays(
        true: np.ndarray,
        pred: np.ndarray | None = ...,
        as_residual: IsTrue = True,
        null_val: int | float = ...,
        nan_to_num: bool = ...,
    ) -> np.ndarray: ...

    @staticmethod
    @overload
    def parse_arrays(
        true: np.ndarray,
        pred: np.ndarray | None,
        as_residual: IsFalse,
        null_val: int | float = ...,
        nan_to_num: bool = ...,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    @staticmethod
    @overload
    def parse_arrays(
        true: np.ndarray,
        pred: np.ndarray | None = ...,
        *,
        as_residual: IsFalse,
        null_val: int | float = ...,
        nan_to_num: bool = ...,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    @staticmethod
    def parse_arrays(
        true: np.ndarray,
        pred: np.ndarray | None = None,
        as_residual: bool = True,
        null_val: int | float = 0,
        nan_to_num: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Dynamically calculate residuals or weighted residuals, allowing pred and/or weight to be None.

        If pred is None, returns true.
        If weight is None, returns (true - pred) if pred is not None, else true.
        If both pred and weight are provided, returns (true - pred) * weight.

        Parameters
        ----------
        true : np.ndarray
            Array of true values.
        pred : np.ndarray | None, optional
            Array of predicted values. If None, returns true or weighted true.
        as_residual : bool, optional
            If True, returns residuals (true - pred). If False, returns (true, pred).
        null_val : int | float, optional
            Value to use if pred is None.
        nan_to_num : bool, optional
            If True, applies np.nan_to_num to true and pred arrays.

        Returns
        -------
        np.ndarray or tuple[np.ndarray, np.ndarray]
            The calculated array (residuals, weighted residuals, or true/weighted true).
        """
        true = np.asarray(true)
        pred = np.asarray(pred) if pred is not None else np.full_like(true, null_val)
        if nan_to_num:
            true = np.nan_to_num(true)
            pred = np.nan_to_num(pred)

        if as_residual:
            return np.nan_to_num(true - pred) if nan_to_num else true - pred
        return true, pred

    @staticmethod
    def parse_log_arrays(
        true: np.ndarray,
        pred: np.ndarray | None = None,
        min_val: float = -1,
        null_val: int | float | None = None,
        nan_to_num: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare arrays for log-based error calculations.

        Handles negative values by flipping sign if more than 50% are negative,
        and clips values below min_val to just above min_val for numerical stability.

        Parameters
        ----------
        *arrays : np.ndarray
            Arrays of values to process.
        min_val : float, optional
            Minimum allowed value (default: -1).

        Returns
        -------
        tuple[np.ndarray, ...]
            Cleaned arrays, in the same order as input.
        """
        true = np.nan_to_num(np.asarray(true)) if nan_to_num else np.asarray(true)
        arrays = (true,)
        if pred is not None:
            pred = np.nan_to_num(np.asarray(pred)) if nan_to_num else np.asarray(pred)
            arrays += (pred,)

        # Combine all arrays for the mask check
        combined = np.concatenate(arrays)
        if np.count_nonzero(combined < min_val) > 0.5 * combined.size:
            arrays = tuple(-1 * arr for arr in arrays)
        try:
            abs_arr = np.abs(combined[combined != 0])
            perturb = 10 ** (int(np.floor(np.log10(np.min(abs_arr)))) + Statistics.NP_EPS_EXP)
            arrays = tuple(np.where(arr < min_val, min_val + perturb, arr) for arr in arrays)
        except ValueError:
            warnings.warn(
                "All input arrays contain only zeros. Returning arrays unchanged.",
                UserWarning,
                stacklevel=2,
            )

        if len(arrays) == 1:
            filler = null_val if null_val is not None else -1 * min_val - 1
            arrays += (np.full_like(true, filler),)

        if nan_to_num:
            return np.nan_to_num(arrays[0]), np.nan_to_num(arrays[1])
            # return cast(np.ndarray, np.nan_to_num(arrays[0])), cast(
            #     np.ndarray, np.nan_to_num(arrays[1])
            # )
        return arrays[0], arrays[1]

    @staticmethod
    def median(
        data: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> float:
        """
        Compute the median or weighted median of data.

        Parameters
        ----------
        data : np.ndarray
            Input data array.
        weights : np.ndarray | None, optional
            Weights for each data point. If None, computes the regular median.

        Returns
        -------
        float
            Median or weighted median value.
        """
        if weights is None:
            return float(np.median(data))

        data = np.asarray(data)
        weights = np.asarray(weights)

        sorter = np.argsort(data)
        weight_sums = np.cumsum(weights[sorter])
        idx = np.searchsorted(weight_sums, 0.5 * weight_sums[-1])
        return data[sorter][min(idx, len(weight_sums) - 1)]

    # --- Multi Value Statistics ---
    @staticmethod
    def residuals(true, pred=None, weight=None):
        """Return calculated external standardized residual.."""
        array = Statistics.parse_arrays(true, pred)
        return array * Statistics.parse_weights(weight, len(true))

    @staticmethod
    def squared_err(true, pred=None, weight=None):
        """Return sum of squared errors (pred vs actual)."""
        array = np.square(Statistics.parse_arrays(true, pred))
        return array * Statistics.parse_weights(weight, len(true))

    @staticmethod
    def max_err(true, pred=None, weight=None):
        """Return calculated max_error."""
        array = abs(Statistics.parse_arrays(true, pred))
        return array * Statistics.parse_weights(weight, len(true))

    @staticmethod
    def abs_err(true, pred=None, weight=None):
        """Return calculated max_error."""
        array = abs(Statistics.parse_arrays(true, pred))
        return array * Statistics.parse_weights(weight, len(true))

    @staticmethod
    def abs_perc_err(true, pred=None, weight=None):
        """Return calculated mean_absolute_percentage_error."""
        true, pred = Statistics.parse_arrays(true, pred, False)
        weight = Statistics.parse_weights(weight, len(true))
        array = abs(true - pred) / abs(np.where(true != 0, true, 10**Statistics.NP_EPS_EXP))
        return array * weight

    @staticmethod
    def log_error(true, pred=None, weight=None):
        """Return calculated mean_squared_log_error."""
        true, pred = Statistics.parse_log_arrays(true, pred)
        weight = Statistics.parse_weights(weight, len(true))
        array = np.log(1 + true) - np.log(1 + pred)
        return array * weight

    @staticmethod
    def squared_log_error(true, pred=None, weight=None):
        """Return calculated mean_squared_log_error."""
        true, pred = Statistics.parse_log_arrays(true, pred)
        weight = Statistics.parse_weights(weight, len(true))
        array = np.square(np.log(1 + true) - np.log(1 + pred))
        return array * weight

    @staticmethod
    def log10_error(true, pred=None, weight=None):
        """Return calculated mean_squared_log_error."""
        true, pred = Statistics.parse_log_arrays(true, pred)
        weight = Statistics.parse_weights(weight, len(true))
        array = np.log10(1 + true) - np.log10(1 + pred)
        return array * weight

    @staticmethod
    def squared_log10_error(true, pred=None, weight=None):
        """Return calculated mean_squared_log_error."""
        true, pred = Statistics.parse_log_arrays(true, pred)
        weight = Statistics.parse_weights(weight, len(true))
        array = np.square(np.log10(1 + true) - np.log10(1 + pred))
        return array * weight

    @staticmethod
    def z_scores(
        true: np.ndarray,
        pred: np.ndarray | None = None,
        weight: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the Z-score (standardized residuals) for the input arrays."""
        array = Statistics.parse_arrays(true, pred)
        weight = Statistics.parse_weights(weight, len(true), True)
        mean = np.average(array, weights=weight)
        std = np.sqrt(np.average(np.square(array - mean), weights=weight))
        # Avoid division by zero
        if std == 0:
            return np.zeros_like(array)
        return (array - mean) / std

    @staticmethod
    def abs_z_scores(
        true: np.ndarray,
        pred: np.ndarray | None = None,
        weight: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the Z-score (standardized residuals) for the input arrays."""
        return abs(Statistics.z_scores(true, pred, weight))

    @staticmethod
    def mod_z_scores(
        true: np.ndarray,
        pred: np.ndarray | None = None,
        weight: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the modified Z-score (using median and MAD) for the input arrays."""
        arr = Statistics.parse_arrays(true, pred)
        weight = Statistics.parse_weights(weight, len(true), True)
        median = Statistics.median(arr, weight)
        mad = Statistics.median(np.abs(arr - median), weight)
        # Avoid division by zero
        if mad == 0:
            return np.zeros_like(arr)
        return Statistics.SCIPY_SCALE_75 * (arr - median) / mad

    # --- Single Value Statistics ---
    @staticmethod
    def r2_score(true, pred=None, weight=None):
        """Return calculated mean_absolute_error."""
        true, pred = Statistics.parse_arrays(true, pred, False)
        weight = Statistics.parse_weights(weight, len(true))
        mean = np.average(true, weights=weight, axis=0)
        sse = np.sum(np.square(true - pred) * weight)  # Sum of squared errors
        sst = np.sum(np.square(true - mean) * weight)  # Total sum of squares
        try:
            if sst == 0:
                return 1.0 if sse == 0 else 0
            return 1 - sse / sst
        except FloatingPointError:
            return -1 * Statistics.NP_SYS_FMAX

    @staticmethod
    def mean_abs_err(true, pred=None, weight=None):
        """Return calculated mean_absolute_error."""
        try:
            array = abs(Statistics.parse_arrays(true, pred))
            weight = Statistics.parse_weights(weight, len(true), True)
            return np.average(array, weights=weight, axis=0)
        except FloatingPointError:
            array = abs(Statistics.parse_arrays(true, pred, nan_to_num=True))
            weight = Statistics.parse_weights(weight, len(true), True, nan_to_num=True)
            return np.average(array, weights=weight, axis=0)

    @staticmethod
    def median_abs_err(true, pred=None, weight=None):
        """Return calculated median_absolute_error."""
        try:
            array = abs(Statistics.parse_arrays(true, pred))
            weight = Statistics.parse_weights(weight, len(true), True)
            return Statistics.median(array, weight)
        except FloatingPointError:
            array = abs(Statistics.parse_arrays(true, pred, nan_to_num=True))
            weight = Statistics.parse_weights(weight, len(true), True, nan_to_num=True)
            return Statistics.median(array, weight)

    @staticmethod
    def total_abs_err(true, pred=None, weight=None):
        """Return calculated total_absolute_error."""
        try:
            array = abs(Statistics.parse_arrays(true, pred))
            weight = Statistics.parse_weights(weight, len(true))
            return np.sum(array * weight)
        except FloatingPointError:
            array = abs(Statistics.parse_arrays(true, pred, nan_to_num=True))
            weight = Statistics.parse_weights(weight, len(true), nan_to_num=True)
            return np.sum(np.nan_to_num(array) * weight)

    @staticmethod
    def mean_squared_err(true, pred=None, weight=None):
        """Return calculated mean_squared_error."""
        try:
            array = np.square(Statistics.parse_arrays(true, pred))
            weight = Statistics.parse_weights(weight, len(true), True)
            return np.average(array, weights=weight, axis=0)
        except FloatingPointError:
            array = np.square(Statistics.parse_arrays(true, pred, nan_to_num=True))
            weight = Statistics.parse_weights(weight, len(true), True, nan_to_num=True)
            return np.average(array, weights=weight, axis=0)

    @staticmethod
    def total_squared_err(true, pred=None, weight=None):
        """Return calculated total_squared_error."""
        try:
            array = np.square(Statistics.parse_arrays(true, pred))
            weight = Statistics.parse_weights(weight, len(true))
            return np.sum(array * weight)
        except FloatingPointError:
            array = np.square(Statistics.parse_arrays(true, pred, nan_to_num=True))
            weight = Statistics.parse_weights(weight, len(true), nan_to_num=True)
            return np.sum(np.nan_to_num(array) * weight)

    @staticmethod
    def root_mean_squared_err(true, pred=None, weight=None):
        """Return calculated root_mean_squared_error."""
        try:
            array = np.square(Statistics.parse_arrays(true, pred))
            weight = Statistics.parse_weights(weight, len(true), True)
            return np.sqrt(np.average(array, weights=weight, axis=0))
        except FloatingPointError:
            array = np.square(Statistics.parse_arrays(true, pred, nan_to_num=True))
            weight = Statistics.parse_weights(weight, len(true), True, nan_to_num=True)
            return np.sqrt(np.average(array, weights=weight, axis=0))

    @staticmethod
    def root_mean_abs_err(true, pred=None, weight=None):
        """Return calculated root_mean_absolute_error."""
        try:
            array = abs(Statistics.parse_arrays(true, pred))
            weight = Statistics.parse_weights(weight, len(true), True)
            return np.sqrt(np.average(array, weights=weight, axis=0))
        except FloatingPointError:
            array = abs(Statistics.parse_arrays(true, pred, nan_to_num=True))
            weight = Statistics.parse_weights(weight, len(true), True, nan_to_num=True)
            return np.sqrt(np.average(array, weights=weight, axis=0))

    @staticmethod
    def mean_abs_perc_err(true, pred=None, weight=None):
        """Return calculated mean_absolute_percentage_error."""
        try:
            true, pred = Statistics.parse_arrays(true, pred, False)
            weight = Statistics.parse_weights(weight, len(true), True)
            array = abs(true - pred) / abs(np.where(true != 0, true, 10**Statistics.NP_EPS_EXP))
            return np.average(array, weights=weight, axis=0)
        except FloatingPointError:
            true, pred = Statistics.parse_arrays(true, pred, False, nan_to_num=True)
            weight = Statistics.parse_weights(weight, len(true), True, nan_to_num=True)
            array = abs(true - pred) / abs(np.where(true != 0, true, 10**Statistics.NP_EPS_EXP))
            return np.average(np.nan_to_num(array), weights=weight, axis=0)

    @staticmethod
    def total_abs_perc_err(true, pred=None, weight=None):
        """Return calculated mean_absolute_percentage_error."""
        try:
            true, pred = Statistics.parse_arrays(true, pred, False)
            weight = Statistics.parse_weights(weight, len(true))
            array = abs(true - pred) / abs(np.where(true != 0, true, 10**Statistics.NP_EPS_EXP))
            return np.sum(array * weight)
        except FloatingPointError:
            true, pred = Statistics.parse_arrays(true, pred, False, nan_to_num=True)
            weight = Statistics.parse_weights(weight, len(true), nan_to_num=True)
            array = abs(true - pred) / abs(np.where(true != 0, true, 10**Statistics.NP_EPS_EXP))
            return np.sum(np.nan_to_num(array) * weight)

    @staticmethod
    def mean_log_error(true, pred=None, weight=None):
        """Return calculated mean_squared_log_error."""
        true, pred = Statistics.parse_log_arrays(true, pred)
        array = np.log(1 + true) - np.log(1 + pred)
        weight = Statistics.parse_weights(weight, len(true), True)
        return np.average(array, weights=weight, axis=0)

    @staticmethod
    def total_log_error(true, pred=None, weight=None):
        """Return calculated mean_squared_log_error."""
        true, pred = Statistics.parse_log_arrays(true, pred)
        weight = Statistics.parse_weights(weight, len(true))
        array = np.log(1 + true) - np.log(1 + pred)
        return np.sum(array * weight)

    @staticmethod
    def mean_squared_log_error(true, pred=None, weight=None):
        """Return calculated mean_squared_log_error."""
        true, pred = Statistics.parse_log_arrays(true, pred)
        weight = Statistics.parse_weights(weight, len(true), True)
        array = np.square(np.log(1 + true) - np.log(1 + pred))
        return np.average(array, weights=weight, axis=0)

    @staticmethod
    def total_squared_log_error(true, pred=None, weight=None):
        """Return calculated mean_squared_log_error."""
        true, pred = Statistics.parse_log_arrays(true, pred)
        weight = Statistics.parse_weights(weight, len(true))
        array = np.square(np.log(1 + true) - np.log(1 + pred))
        return np.sum(array * weight)

    @staticmethod
    def mean_log10_error(true, pred=None, weight=None):
        """Return calculated mean_squared_log_error."""
        true, pred = Statistics.parse_log_arrays(true, pred)
        weight = Statistics.parse_weights(weight, len(true), True)
        array = np.log10(1 + true) - np.log10(1 + pred)
        return np.average(array, weights=weight, axis=0)

    @staticmethod
    def total_log10_error(true, pred=None, weight=None):
        """Return calculated mean_squared_log_error."""
        true, pred = Statistics.parse_log_arrays(true, pred)
        weight = Statistics.parse_weights(weight, len(true))
        array = np.log10(1 + true) - np.log10(1 + pred)
        return np.sum(array * weight)

    @staticmethod
    def mean_squared_log10_error(true, pred=None, weight=None):
        """Return calculated mean_squared_log_error."""
        true, pred = Statistics.parse_log_arrays(true, pred)
        weight = Statistics.parse_weights(weight, len(true), True)
        array = np.square(np.log10(1 + true) - np.log10(1 + pred))
        return np.average(array, weights=weight, axis=0)

    @staticmethod
    def total_squared_log10_error(true, pred=None, weight=None):
        """Return calculated mean_squared_log_error."""
        true, pred = Statistics.parse_log_arrays(true, pred)
        weight = Statistics.parse_weights(weight, len(true))
        array = np.square(np.log10(1 + true) - np.log10(1 + pred))
        return np.sum(array * weight)

    @staticmethod
    def mean_poisson_dev(true, pred=None, weight=None):
        """Return calculated mean_poisson_deviance."""
        if pred is None:
            warnings.warn(
                "Calling mean_poisson_dev without passing pred likely to result in a statistically meaningless value.",
                UserWarning,
                stacklevel=2,
            )
        true, pred = Statistics.parse_log_arrays(true, pred, 0.0, 10**Statistics.NP_EPS_EXP)
        weight = Statistics.parse_weights(weight, len(true), True)
        array = 2 * (true * np.log(true / pred) + pred - true)
        return np.average(array, weights=weight, axis=0)

    @staticmethod
    def mean_gamma_dev(true, pred=None, weight=None):
        """Return calculated mean_gamma_deviance."""
        if pred is None:
            warnings.warn(
                "Calling mean_gamma_dev without passing pred likely to result in a statistically meaningless value.",
                UserWarning,
                stacklevel=2,
            )
        true, pred = Statistics.parse_log_arrays(true, pred, 0.0, 10**Statistics.NP_EPS_EXP)
        weight = Statistics.parse_weights(weight, len(true), True)
        array = 2 * (np.log(pred / true) + true / pred - 1)
        return np.average(array, weights=weight, axis=0)


class FitCancelledException(Exception):
    """Exception raised when a fit is cancelled."""

    pass


# Evaluate kwargs for simplification/sanitation
class FittingMethods:
    """Class to create a fitting method manager."""

    NP_2EPS: float = 10 ** int(np.floor(np.log10(np.finfo(np.float64).eps)) * 2)  # 1e-16

    @property
    def lsq_kwargs(self):
        return {
            "method": "trf",
            "jac": "3-point",
            "x_scale": "jac",
            "ftol": 1e-14,
            "xtol": 1e-15,
            "gtol": 1e-8,
            "loss": "cauchy",
            "diff_step": None,
            "tr_solver": None,
            "tr_options": {},
            "jac_sparsity": None,
            "verbose": 0,
            "max_nfev": 1e6,
        }

    @property
    def de_kwargs(self):
        return {
            "strategy": "best1bin",
            "maxiter": 1000,
            "popsize": 15,
            "tol": 0.01,
            # "mutation": (0.5, 1),
            "recombination": 0.7,
        }

    @property
    def basin_kwargs(self):
        return {
            "niter": 100,
            "T": 1.0,
            "stepsize": 0.5,
            "minimizer_kwargs": {
                "method": "L-BFGS-B",
                "jac": "3-point",
            },
        }

    def circuit_fit(
        self,
        x_data: list | tuple | np.ndarray,
        y_data: list | tuple | np.ndarray,
        initial_guess: list | tuple | np.ndarray,
        model_func: Callable,
        bounds: list | tuple | Bounds | None = None,
        weights: list | tuple | np.ndarray | None = None,
        scale: list | tuple | str | None = None,
        **kwargs,
    ):
        """
        Perform a least-squares fit using the provided data and model function.

        Parameters
        ----------
        x_data : list | tuple | np.ndarray
            Dependent variable data.
        y_data : list | tuple | np.ndarray
            Independent variable data.
        initial_guess : list | tuple | np.ndarray
            Initial guess for the parameters to be optimized. Correlates to x0 in least_squares.
        model_func : Callable
            The model function to fit to the data. It should take the independent variable as the first argument,
            followed by the parameters to be optimized.
        bounds : list | tuple | Bounds | None, optional
            Bounds for the parameters to be optimized. If None, no bounds are applied.
        weights : list | tuple | np.ndarray | None, optional
            Weights for the data points. If None, no weights are applied.
        scale : list | tuple | str | None, optional
            Scaling for the parameters. If None, no scaling is applied.
            Options are `lin` or `log`. Passing as a list applies scaling to each parameter.
        **kwargs
            Additional keyword arguments to be passed to `scipy.optimize.least_squares`.
            kill_operation : Callable | None, optional
                A function that can be used to cancel the fit operation, used for thread operations
            loss_func : Callable, optional
                A function to calculate the loss in the obj_func, defaults to `Statistics.squared_err`.
            x_scale : str | np.ndarray, optional
                Scaling for the independent variable. If 'calc', it will calculate a scale based on the initial guess.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - popt: Optimized parameters as a NumPy array.
            - perror: Estimated errors of the optimized parameters as a NumPy array.
        """
        initial_guess, bounds, model_func, scale_list = self._convert_param_scale(
            initial_guess, bounds, scale, model_func
        )

        kwargs = {**self.lsq_kwargs, **kwargs}
        # kwargs.pop("weights", None)
        kwargs.pop("loss_func", None)
        kwargs.pop("obj_func", None)
        kwargs["max_nfev"] = kwargs.pop("maxfev", self.lsq_kwargs["max_nfev"])
        kwargs.pop("kill_operation", None)
        default_err = kwargs.pop("default_err", 1)

        if isinstance(kwargs.get("x_scale"), str) and "calc" in kwargs["x_scale"]:
            kwargs["x_scale"] = np.array(
                [10 ** (int(np.log10(abs(ig))) - 1) for ig in initial_guess]
            )

        # weighting scheme for fitting
        if weights is not None and len(weights) != len(y_data):
            weights = None

        self.result = curve_fit(
            model_func,
            x_data,
            y_data,
            p0=initial_guess,
            bounds=bounds,
            **kwargs,
        )

        return self.parse_result(self.result, scale_list, model_func, x_data, y_data, default_err)

    def ls_fit(
        self,
        x_data: list | tuple | np.ndarray,
        y_data: list | tuple | np.ndarray,
        initial_guess: list | tuple | np.ndarray,
        model_func: Callable,
        obj_func: Callable | None = None,
        bounds: list | tuple | Bounds | None = None,
        weights: list | tuple | np.ndarray | None = None,
        scale: list | tuple | str | None = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """
        Perform a least-squares fit using the provided data and model function.

        Parameters
        ----------
        x_data : list | tuple | np.ndarray
            Dependent variable data.
        y_data : list | tuple | np.ndarray
            Independent variable data.
        initial_guess : list | tuple | np.ndarray
            Initial guess for the parameters to be optimized. Correlates to x0 in least_squares.
        model_func : Callable
            The model function to fit to the data. It should take the independent variable as the first argument,
            followed by the parameters to be optimized.
        obj_func : Callable | None, optional
            Objective function which performs the minimization. Correlates to fun in least_squares.
            If None, defaults to `self.obj_func` , by default None
        bounds : list | tuple | Bounds | None, optional
            Bounds for the parameters to be optimized. If None, no bounds are applied.
        weights : list | tuple | np.ndarray | None, optional
            Weights for the data points. If None, no weights are applied.
        scale : list | tuple | str | None, optional
            Scaling for the parameters. If None, no scaling is applied.
            Options are `lin` or `log`. Passing as a list applies scaling to each parameter.
        **kwargs
            Additional keyword arguments to be passed to `scipy.optimize.least_squares`.
            kill_operation : Callable | None, optional
                A function that can be used to cancel the fit operation, used for thread operations
            loss_func : Callable, optional
                A function to calculate the loss in the obj_func, defaults to `Statistics.squared_err`.
            x_scale : str | np.ndarray, optional
                Scaling for the independent variable. If 'calc', it will calculate a scale based on the initial guess.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - popt: Optimized parameters as a NumPy array.
            - perror: Estimated errors of the optimized parameters as a NumPy array.
        """
        initial_guess, bounds, model_func, scale_list = self._convert_param_scale(
            initial_guess, bounds, scale, model_func
        )

        kwargs = {**self.lsq_kwargs, **kwargs}
        default_err = kwargs.pop("default_err", 1)
        loss = kwargs.pop("loss_func", Statistics.squared_err)
        kill_operation = kwargs.pop("kill_operation", None)
        if isinstance(loss, str):
            loss = getattr(Statistics, loss, Statistics.squared_err)

        if isinstance(kwargs.get("x_scale"), str) and "calc" in kwargs["x_scale"]:
            kwargs["x_scale"] = np.array(
                [10 ** (int(np.log10(abs(ig))) - 1) for ig in initial_guess]
            )

        obj_func = obj_func if callable(obj_func) else self.obj_func

        if weights is not None and len(weights) != len(y_data):
            weights = None

        try:
            self.result = least_squares(
                obj_func,
                initial_guess,
                args=(
                    x_data,
                    y_data,
                    model_func,
                    weights,
                    loss,
                    kill_operation,
                ),
                bounds=bounds,
                **kwargs,
            )
        except FitCancelledException:
            # print("Fit cancelled by user.")
            return None, None

        return self.parse_result(self.result, scale_list, model_func, x_data, y_data, default_err)

    def de_fit(
        self,
        x_data: list | tuple | np.ndarray,
        y_data: list | tuple | np.ndarray,
        initial_guess: list | tuple | np.ndarray,
        model_func: Callable,
        obj_func: Callable | None = None,
        bounds: list | tuple | Bounds | None = None,
        weights: list | tuple | np.ndarray | None = None,
        scale: list | tuple | str | None = None,
        **kwargs,
    ):
        """
        Perform a least-squares fit using the provided data and model function.

        Parameters
        ----------
        x_data : list | tuple | np.ndarray
            Dependent variable data.
        y_data : list | tuple | np.ndarray
            Independent variable data.
        initial_guess : list | tuple | np.ndarray
            Initial guess for the parameters to be optimized. Correlates to x0 in least_squares.
        model_func : Callable
            The model function to fit to the data. It should take the independent variable as the first argument,
            followed by the parameters to be optimized.
        obj_func : Callable | None, optional
            Objective function which performs the minimization. Correlates to fun in least_squares.
            If None, defaults to `self.obj_func` , by default None
        bounds : list | tuple | Bounds | None, optional
            Bounds for the parameters to be optimized. If None, no bounds are applied.
        weights : list | tuple | np.ndarray | None, optional
            Weights for the data points. If None, no weights are applied.
        scale : list | tuple | str | None, optional
            Scaling for the parameters. If None, no scaling is applied.
            Options are `lin` or `log`. Passing as a list applies scaling to each parameter.
        **kwargs
            Additional keyword arguments to be passed to `scipy.optimize.least_squares`.
            kill_operation : Callable | None, optional
                A function that can be used to cancel the fit operation, used for thread operations
            loss_func : Callable, optional
                A function to calculate the loss in the obj_func, defaults to `Statistics.squared_err`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - popt: Optimized parameters as a NumPy array.
            - perror: Estimated errors of the optimized parameters as a NumPy array.
        """

        initial_guess, bounds, model_func, scale_list = self._convert_param_scale(
            initial_guess, bounds, scale, model_func
        )

        # Convert bounds to the format required by differential_evolution
        if bounds is None:
            bounds = [(None, None)] * len(initial_guess)
        else:
            if isinstance(bounds, Bounds):
                bounds = [bounds.lb, bounds.ub]
            bounds = [
                tuple(1e300 if b == np.inf else -1e300 if b == -np.inf else b for b in bound_pair)
                for bound_pair in zip(*bounds)
            ]

        kwargs = {**self.de_kwargs, **kwargs}
        default_err = kwargs.pop("default_err", 1)
        loss = kwargs.pop("loss_func", Statistics.total_squared_err)
        kill_operation = kwargs.pop("kill_operation", None)
        if isinstance(loss, str):
            loss = getattr(Statistics, loss, Statistics.total_squared_err)
        if kill_operation is not None and callable(kill_operation):
            kwargs["callback"] = lambda xk, convergence: kill_operation()

        if weights is not None and len(weights) != len(y_data):
            weights = None

        obj_func = obj_func if callable(obj_func) else self.obj_func
        try:
            self.result = differential_evolution(
                obj_func,
                bounds,
                args=(
                    x_data,
                    y_data,
                    model_func,
                    weights,
                    loss,
                    # kill_operation,
                ),
                x0=initial_guess,
                **kwargs,
            )
        except FitCancelledException:
            # print("Fit cancelled by user.")
            return None, None

        return self.parse_result(self.result, scale_list, model_func, x_data, y_data, default_err)

    def basin_fit(
        self,
        x_data: list | tuple | np.ndarray,
        y_data: list | tuple | np.ndarray,
        initial_guess: list | tuple | np.ndarray,
        model_func: Callable,
        obj_func: Callable | None = None,
        bounds: list | tuple | Bounds | None = None,
        weights: list | tuple | np.ndarray | None = None,
        scale: list | tuple | str | None = None,
        **kwargs,
    ):
        """
        Perform a least-squares fit using the provided data and model function.

        Parameters
        ----------
        x_data : list | tuple | np.ndarray
            Dependent variable data.
        y_data : list | tuple | np.ndarray
            Independent variable data.
        initial_guess : list | tuple | np.ndarray
            Initial guess for the parameters to be optimized. Correlates to x0 in least_squares.
        model_func : Callable
            The model function to fit to the data. It should take the independent variable as the first argument,
            followed by the parameters to be optimized.
        obj_func : Callable | None, optional
            Objective function which performs the minimization. Correlates to fun in least_squares.
            If None, defaults to `self.obj_func` , by default None
        bounds : list | tuple | Bounds | None, optional
            Bounds for the parameters to be optimized. If None, no bounds are applied.
        weights : list | tuple | np.ndarray | None, optional
            Weights for the data points. If None, no weights are applied.
        scale : list | tuple | str | None, optional
            Scaling for the parameters. If None, no scaling is applied.
            Options are `lin` or `log`. Passing as a list applies scaling to each parameter.
        **kwargs
            Additional keyword arguments to be passed to `scipy.optimize.least_squares`.
            kill_operation : Callable | None, optional
                A function that can be used to cancel the fit operation, used for thread operations
            loss_func : Callable, optional
                A function to calculate the loss in the obj_func, defaults to `Statistics.squared_err`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - popt: Optimized parameters as a NumPy array.
            - perror: Estimated errors of the optimized parameters as a NumPy array.
        """

        initial_guess, bounds, model_func, scale_list = self._convert_param_scale(
            initial_guess, bounds, scale, model_func
        )

        # Convert bounds to the format required by minimize
        if bounds is None:
            bounds = [(None, None)] * len(initial_guess)
        else:
            if isinstance(bounds, Bounds):
                bounds = [bounds.lb, bounds.ub]
            bounds = [
                (None if abs(b) == np.inf else b for b in bound_pair)
                for bound_pair in zip(*bounds)
            ]

        kwargs = {**self.basin_kwargs, **kwargs}
        default_err = kwargs.pop("default_err", 1)
        loss = kwargs.pop("loss_func", Statistics.total_squared_err)
        kill_operation = kwargs.pop("kill_operation", None)
        if isinstance(loss, str):
            loss = getattr(Statistics, loss, Statistics.total_squared_err)

        if weights is not None and len(weights) != len(y_data):
            weights = None

        kwargs["minimizer_kwargs"]["bounds"] = bounds
        kwargs["minimizer_kwargs"]["args"] = (
            x_data,
            y_data,
            model_func,
            weights,
            loss,
            kill_operation,
        )

        obj_func = obj_func if callable(obj_func) else self.obj_func
        try:
            self.result = basinhopping(
                obj_func,
                initial_guess,
                **kwargs,
            )
        except FitCancelledException:
            # print("Fit cancelled by user.")
            return None, None

        return self.parse_result(self.result, scale_list, model_func, x_data, y_data, default_err)

    def parse_result(
        self,
        result: Any,
        scale_list: list | None,
        func: Callable | None = None,
        x_data: tuple | list | np.ndarray = (),
        y_data: tuple | list | np.ndarray = (),
        default_err: int | float | str = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Parse the result of a fitting operation to extract optimized parameters and their errors.

        Parameters
        ----------
        result : Any
            The result of the fitting operation. Supported types include outputs from
            `scipy.optimize.curve_fit`, `scipy.optimize.least_squares`, or similar optimizers.
        scale_list : list | None
            List indicating the scaling applied to each parameter, e.g., ['log', 'lin'].
            If 'log', the parameter was fit in log10 space and will be exponentiated back.
        func : Callable | None, optional
            The model function used for fitting. If provided, it is used to estimate the Jacobian
            for error calculation when the optimizer does not provide a covariance.
        x_data : tuple | list | np.ndarray, optional
            Independent variable data used in the fitting. Required if `func` is used for Jacobian estimation.
        y_data : tuple | list | np.ndarray, optional
            Dependent variable data used in the fitting. Used to estimate residual variance for error scaling.
        default_err : int | float | str, optional
            Value to use for errors if they cannot be estimated. If a string, uses a multiple of the parameter
            or the residual variance.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - popt: Optimized parameters as a NumPy array, rescaled to original space if needed.
            - perror: Estimated standard errors of the optimized parameters as a NumPy array,
            rescaled to original space if needed. For log10-scaled parameters, error is propagated as
            `perror = ln(10) * popt * perror_log`, where `perror_log` is the error in log10 space.

        Notes
        -----
        - For log10-scaled parameters, both the parameter and its error are transformed back to linear space.
        - If the covariance matrix cannot be estimated, errors are set to NaN or replaced by `default_err`.
        - The method supports results from both curve fitting and general optimization routines.
        """
        # Handle curve_fit results
        x_data = np.asarray(x_data)
        y_data = np.asarray(y_data)

        if isinstance(result, tuple) and result[1].shape == (
            len(result[0]),
            len(result[0]),
        ):
            popt, cov, *_ = result
            popt = np.asarray(popt)
        else:
            popt = (
                np.asarray(result[0])
                if isinstance(result, (tuple, list))
                else np.asarray(result.x)
            )
            cov = np.eye(len(popt)) * self.NP_2EPS
            if (
                isinstance(result, OptimizeResult)
                and hasattr(result, "jac")
                and result.jac.ndim == 2
            ):
                # result is from least_squares
                cov = self.safe_invert(result.jac.T @ result.jac)
            elif callable(func) and x_data.size != 0:
                # Estimate the Jacobian using numerical differentiation
                J = self._estimate_jacobian(func, popt, x_data)
                cov = self.safe_invert(J.T @ J)

        diag = np.diag(cov)
        s_sq = 1
        if all(diag < 0) or np.all(diag == self.NP_2EPS):
            perror = np.full_like(diag, np.nan)
        else:
            if callable(func) and x_data.size != 0 and y_data.size != 0:
                s_sq = np.mean((y_data - func(x_data, *popt)) ** 2)
            try:
                perror = np.sqrt(np.where((diag < 0), np.nan, diag * s_sq))
            except (ValueError, RuntimeWarning):
                perror = np.full_like(diag, np.nan)

        if scale_list is not None:
            s_popt = []
            s_perror = []
            for p, pe, scale in zip(popt, perror, scale_list):
                if scale == "log":
                    p = np.clip(p, -300, 300)  # Avoid overflow in log scale
                    s_popt.append(10.0**p)
                    s_perror.append(10.0**p * np.log(10) * pe)
                else:
                    s_popt.append(p)
                    s_perror.append(pe)

            popt = np.asarray(s_popt)
            perror = np.asarray(s_perror)

        if not isinstance(default_err, (int, float)):
            try:
                # Assumes default_err is a multiple to be applied rather than a fixed value
                perror[np.isnan(perror)] = abs(popt * float(default_err))
            except ValueError:
                # Assumes default_err is a string requesting the value to be p * s_sq
                perror[np.isnan(perror)] = abs(popt * s_sq)
        else:
            # Assumes default_err is a fixed value to replace NaNs
            perror = np.nan_to_num(perror, nan=default_err)

        return popt, perror

    def safe_invert(self, mat: np.ndarray, threshold: float = 1e12) -> np.ndarray:
        """Safely invert a matrix, applying regularization if necessary."""
        cond_num = 0.0
        lambda_mod = 0.0
        try:
            cond_num = np.linalg.cond(mat)
            if cond_num >= threshold:
                cond_exp = int(np.floor(np.log10(cond_num)))
                lambda_mod = cond_num * np.eye(mat.shape[0]) / 10**cond_exp
            return np.linalg.inv(mat + lambda_mod)
        except (np.linalg.LinAlgError, FloatingPointError, OverflowError):
            if not lambda_mod and cond_num and threshold != -np.inf:
                # Fallback: inversion failed without regularization applied
                return self.safe_invert(mat, threshold=-np.inf)
            # Return a scaled identity matrix if inversion fails
            return np.eye(mat.shape[0]) * self.NP_2EPS

    def _estimate_jacobian(
        self,
        func: Callable,
        params: list | np.ndarray,
        x_data: tuple | list | np.ndarray,
        epsilon: float = 1e-8,
    ) -> np.ndarray:
        """Estimate the Jacobian matrix using finite differences."""
        # breakpoint()

        J = np.array([])
        for i in range(len(params)):
            try:
                params1 = params.copy()
                params2 = params.copy()
                # eps = epsilon * (np.abs(params[i]) + self.NP_2EPS)  # Avoid division by zero
                eps = epsilon * max(np.abs(params[i]), self.NP_2EPS)  # Avoid division by zero
                params1[i] += eps
                params2[i] -= eps

                f1 = func(x_data, *params1)
                f2 = func(x_data, *params2)

                if J.size == 0:
                    J = np.zeros((len(f1), len(params)))
                deriv = (f1 - f2) / (2 * eps)
                if not np.all(np.isfinite(deriv)):
                    deriv = np.nan_to_num(deriv, nan=self.NP_2EPS, posinf=1e300, neginf=-1e300)
            except FloatingPointError:
                deriv = np.full(len(x_data), self.NP_2EPS)  # Fallback for numerical issues
            J[:, i] = deriv

        return J

    def _convert_param_scale(
        self: "FittingMethods",
        initial_guess: list | tuple | np.ndarray,
        bounds: list | tuple | Bounds | None,
        scale: str | list | tuple | None,
        model_func: Callable,
    ) -> tuple[list | tuple | np.ndarray, list | tuple | Bounds | None, Callable, list | None]:
        """Convert initial guess and bounds to log scale if specified."""
        if (
            bounds is not None
            and isinstance(bounds, (list, tuple))
            and len(bounds) == 2
            and all(isinstance(b, (int, float, complex, np.number)) for b in bounds)
        ):
            # bounds is a list of 2 numbers
            bounds = ([bounds[0]] * len(initial_guess), [bounds[1]] * len(initial_guess))

        if scale is None or "log" not in scale:
            return initial_guess, bounds, model_func, None

        scale_list = (
            list(map(str, scale))
            if isinstance(scale, (list, tuple)) and len(scale) == len(initial_guess)
            else ["log"] * len(initial_guess)
        )
        new_initial_guess = []
        new_bounds = [[], []]  # if bounds is not None else None

        if isinstance(bounds, Bounds):
            bounds = [bounds.lb, bounds.ub]

        for i, (guess, scale_item) in enumerate(zip(initial_guess, scale_list)):
            if (
                scale_item == "log"
                and guess > 0
                and (bounds is None or (bounds[0][i] >= 0 and bounds[1][i] >= 0))
            ):
                new_initial_guess.append(np.log10(guess))
                if bounds is not None:
                    new_bounds[0].append(np.log10(bounds[0][i] + 1e-300))
                    new_bounds[1].append(np.log10(bounds[1][i] + 1e-300))
            else:
                new_initial_guess.append(guess)
                if bounds is not None:
                    new_bounds[0].append(bounds[0][i])
                    new_bounds[1].append(bounds[1][i])
                scale_list[i] = "lin"

        if bounds is not None:
            new_bounds = (new_bounds[0], new_bounds[1])
        else:
            new_bounds = None

        # return new_initial_guess, new_bounds, scale_list
        new_model_func = self.log_func_wrap(model_func, scale_list)
        return new_initial_guess, new_bounds, new_model_func, scale_list

    @staticmethod
    def obj_func(
        params,
        x_data,
        y_data,
        model_func,
        weights=None,
        loss=None,
        kill_operation=None,
    ):
        """obj_func for least_squares with adjustable error calculation."""
        # Check for cancellation
        if callable(kill_operation) and kill_operation():
            raise FitCancelledException("Fit cancelled by user.")
        # Generate the data from the params
        sim_data = model_func(x_data, *params)

        if callable(loss):
            return loss(y_data, sim_data, weights)
        else:
            return y_data - sim_data

    @staticmethod
    def log_func_wrap(func, scale_list):
        """Wrap the function to convert parameters from log scale."""

        def wrapped_func(x, *params):
            params = [10**p if scale == "log" else p for p, scale in zip(params, scale_list)]
            return func(x, *params)

        return wrapped_func
