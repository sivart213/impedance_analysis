# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 22:12:37 2022.

@author: j2cle
"""
import itertools
import scipy
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, least_squares

from .complex_data import ComplexSystem

class ConfidenceAnalysis:
    """Class to generate confidence bands for EIS data."""

    def __init__(self, percentile=5, popt=None, std=5, func=None):
        self.percentile = percentile
        self.popt = popt
        self.std = std
        self.func = func
        self.df_lists = None

    def __getstate__(self):
        state = self.__dict__.copy()
        # Exclude the un-pickleable attribute
        state["func"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._func = None

    def __setitem__(self, key, value):
        self.__dict__[key] = value
        if key in ["func", "std"]:
            self.df_lists = None

    def __eq__(self, other):
        if not isinstance(other, ConfidenceAnalysis):
            return NotImplemented
        return (
            self.popt == other.popt
            and self.std == other.std
            and self.percentile == other.percentile
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def popt(self):
        """Contains the best-fit parameters."""
        if not hasattr(self, "_popt"):
            return None
        return self._popt

    @popt.setter
    def popt(self, value):
        self._popt = value
        self.df_lists = None

    @property
    def std(self):
        """Contains the standard deviation of the parameters."""
        return self._std

    @std.setter
    def std(self, value):
        if isinstance(value, (int, float)) and self.popt is not None:
            val_max = 100 if value >= 1 else 1
            self._std = np.array(self.popt) * value / val_max
        else:
            self._std = value
        self.df_lists = None

    @property
    def percentile(self):
        """Gets the percentile data which is a list of two floats"""
        return self._percentile

    @percentile.setter
    def percentile(self, value):
        if isinstance(value, (int, float)):
            percent_max = 100 if value >= 1 else 1
            percents = [value / 2, (percent_max - value / 2)]
            self._percentile = [
                min(percents) / percent_max,
                max(percents) / percent_max,
            ]

    def ci_from_data(self, data):
        """Generate confidence intervals from data."""
        perc = self.percentile[0] * 100
        ci_lower = np.percentile(data, perc, axis=0)
        ci_upper = np.percentile(data, 100 - perc, axis=0)
        return (ci_lower + ci_upper) / 2

    def ci_from_std(self, Z_data, popt, perr):
        """Generate confidence intervals from standard deviations."""
        n = len(Z_data)  # number of data points
        p = len(popt)  # number of parameters
        dof = max(0, n - p)  # degrees of freedom
        tval = scipy.stats.t.ppf(1.0 - self.percentile[0] / 2.0, dof)
        return tval * perr

    def _gen_df(
        self, freq, popt, std, func, num_freq_points, main_col, target_form, **kwargs
    ):
        """
        Generate dataframes for the given parameters.

        Parameters:
        freq (array): Array of frequencies.
        popt (array): Best-fit parameters.
        std (array or float): Standard deviations of the parameters.
        func (function): Function to simulate the circuit.
        num_freq_points (int): Number of frequency points.
        main_col (str): Column to stack.
        target_form (str): Type of complex data to generate

        Returns:
        zdf_list (list): List of dataframes for Z data.
        fdf_list (list): List of dataframes for frequency data.
        """

        if isinstance(std, (int, float)):
            std = np.array(popt) * std

        param_bounds = [(popt[i] - std[i], popt[i] + std[i]) for i in range(len(popt))]
        all_combinations = list(itertools.product(*param_bounds))

        zdf_list = []
        fdf_list = []

        if main_col is not None:
            comp_col = "imag" if main_col == "real" else "real"

        for params in all_combinations:
            if main_col is not None:
                freq_exp = np.logspace(
                    np.log10(freq.min()), np.log10(freq.max()), num=num_freq_points
                )
                Z = np.array(np.hsplit(func(freq_exp, *params), 2)).T
                independant = Z[:, 0] if main_col == "real" else -Z[:, 1]
                dependant = -Z[:, 1] if main_col == "real" else Z[:, 0]
                temp_df = pd.DataFrame(
                    {main_col: independant, f"Z_{comp_col}_{params}": dependant}
                )
                zdf_list.append(temp_df.copy())
            if target_form is not None:
                Z = np.array(np.hsplit(func(freq, *params), 2)).T
                data_system = ComplexSystem(
                    np.array([freq, Z[:, 0], Z[:, 1]]).T, **kwargs
                )
                if isinstance(target_form, str):
                    temp_df = data_system.base_df(target_form, "frequency")
                elif isinstance(target_form, (tuple, list)):
                    temp_df = data_system.get_custom_df(*target_form)
                temp_df = temp_df.rename(
                    columns={
                        col: f"{col}_{params}"
                        for col in temp_df.columns
                        if col != "freq"
                    }
                )
                fdf_list.append(temp_df.copy())

        self.df_lists = (zdf_list, fdf_list)

        return zdf_list, fdf_list

    def gen_conf_band_nyquist(
        self,
        freq,
        popt=None,
        std=None,
        func=None,
        num_freq_points=500,
        multiplier=50,
        main_col="real",
        percentile=None,
    ):
        """
        Generate confidence bands for Nyquist plots.

        Parameters:
        popt (array): Best-fit parameters.
        std (array or float): Standard deviations of the parameters.
        func (function): Function to simulate the circuit.
        freq (array): Array of frequencies.
        main_col (str): Column to stack.

        Returns:
        bounds (dict): Dictionary containing the confidence bands.
        """

        if popt is not None:
            self.popt = popt
        if std is not None:
            self.std = std
        if func is not None:
            self.func = func

        if self.df_lists is None:
            self._gen_df(
                freq, self.popt, self.std, self.func, num_freq_points, main_col, None
            )

        zdf_list, _ = self.df_lists

        self.percentile = percentile

        df = pd.concat(zdf_list, axis=0)

        # Calculate the decade range and number of bins
        decade_range = np.log10(abs(df[main_col].max())) - np.log10(
            abs(df[main_col].min())
        )
        num_bins = int(decade_range * multiplier)

        df = df.sort_values(by=main_col, ascending=True)
        df = df.interpolate(method="linear", axis=0).ffill().bfill()
        df = df.assign(
            binned_main_col=pd.cut(df[main_col], bins=num_bins, labels=False)
        )

        # Group by binned_main_col and calculate descriptive statistics for each group
        def calc_group_stats(group):
            # Exclude binned_main_col and main_col from the calculations
            group = group.drop(columns=["binned_main_col", main_col])
            # Flatten the DataFrame
            flattened = group.values.flatten()
            # Calculate descriptive statistics
            stats = pd.Series(flattened).describe(percentiles=self.percentile)
            return stats

        grouped_stats = (
            df.groupby("binned_main_col").apply(calc_group_stats).reset_index()
        )
        grouped_stats = grouped_stats.rename(columns={"binned_main_col": main_col})
        grouped_stats[main_col] = (
            df.groupby("binned_main_col").mean()[main_col].reset_index(drop=True)
        )

        return {"nyquist": grouped_stats}

    def gen_conf_band_bode(
        self,
        freq,
        popt=None,
        std=None,
        func=None,
        target_form="Z",
        percentile=None,
        **kwargs,
    ):
        """
        Generate confidence bands for Bode plots.

        Parameters:
        popt (array): Best-fit parameters.
        std (array or float): Standard deviations of the parameters.
        func (function): Function to simulate the circuit.
        freq (array): Array of frequencies.
        target_form (str): Frequency data.

        Returns:
        bounds (dict): Dictionary containing the confidence bands.
        """

        if popt is not None:
            self.popt = popt
        if std is not None:
            self.std = std
        if func is not None:
            self.func = func

        if self.df_lists is None:
            self._gen_df(freq, self.popt, self.std, self.func, None, None, target_form)

        _, fdf_list = self.df_lists

        self.percentile = percentile
        bounds = {}

        combined_df = pd.concat(fdf_list, axis=1)
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
        combined_df.set_index("freq", inplace=True)
        stats_dict = {}
        columns = ["real", "imag", "inv_imag", "mag", "phase", "inv_phase", "tan"]
        if isinstance(target_form, (tuple, list)):
            columns = target_form[1:]
        for col in columns:
            col_prefix = f"{col}_"
            matching_cols = [c for c in combined_df.columns if c.startswith(col_prefix)]
            if matching_cols:
                stats_dict[col] = (
                    combined_df[matching_cols]
                    .T.describe(percentiles=self.percentile)
                    .T.reset_index()
                    .rename(columns={"index": "freq"})
                )
        bounds.update(stats_dict)

        return bounds

    def gen_conf_band(
        self,
        freq,
        popt=None,
        std=None,
        func=None,
        num_freq_points=500,
        multiplier=50,
        main_col="real",
        target_form="Z",
        percentile=None,
        **kwargs,
    ):
        """
        Generate confidence bands for both Nyquist and Bode plots.

        Parameters:
        popt (array): Best-fit parameters.
        std (array or float): Standard deviations of the parameters.
        func (function): Function to simulate the circuit.
        freq (array): Array of frequencies.
        main_col (str): Column to stack.
        target_form (str): Frequency data.

        Returns:
        bounds (dict): Dictionary containing the confidence bands.
        """

        if popt is not None:
            self.popt = popt
        if std is not None:
            self.std = std
        if func is not None:
            self.func = func

        if self.df_lists is None:
            self._gen_df(
                freq,
                self.popt,
                self.std,
                self.func,
                num_freq_points,
                main_col,
                target_form,
                **kwargs,
            )

        bounds = {}
        self.percentile = percentile
        bounds.update(
            self.gen_conf_band_nyquist(
                freq,
                main_col=main_col,
                num_freq_points=num_freq_points,
                multiplier=multiplier,
            )
        )
        bounds.update(
            self.gen_conf_band_bode(
                freq, target_form=target_form, num_freq_points=num_freq_points
            )
        )

        return bounds


class Statistics:
    """Return sum of squared errors (pred vs actual)."""

    def __init__(self):
        pass

    def __getitem__(self, key):
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
            "mean_squared_err",
            "root_mean_abs_err",
            "root_mean_squared_err",
            "mean_abs_perc_err",
            "mean_log_error",
            "mean_squared_log_error",
            "mean_log10_error",
            "mean_squared_log10_error",
            "mean_poisson_dev",
            "mean_gamma_dev",
        ]

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
            self.mean_squared_err,
            self.root_mean_squared_err,
            self.root_mean_abs_err,
            self.mean_abs_perc_err,
            self.mean_log_error,
            self.mean_squared_log_error,
            self.mean_log10_error,
            self.mean_squared_log10_error,
            self.mean_poisson_dev,
            self.mean_gamma_dev,
        ]

    @staticmethod
    def residuals(true, pred, weight=None):
        """Return calculated external standardized residual.."""
        if weight is not None and len(weight) == len(true):
            return (true - pred) * weight * len(weight) / sum(weight)
        return true - pred

    @staticmethod
    def squared_err(true, pred, weight=None):
        """Return sum of squared errors (pred vs actual)."""
        if weight is not None and len(weight) == len(true):
            return np.square(true - pred) * weight * len(weight) / sum(weight)
        return np.square(true - pred)

    @staticmethod
    def max_err(true, pred, weight=None):
        """Return calculated max_error."""
        if weight is not None and len(weight) == len(true):
            return abs(true - pred) * weight * len(weight) / sum(weight)
        return abs(true - pred)

    @staticmethod
    def abs_err(true, pred, weight=None):
        """Return calculated max_error."""
        if weight is not None and len(weight) == len(true):
            return abs(true - pred) * weight * len(weight) / sum(weight)
        return abs(true - pred)

    @staticmethod
    def abs_perc_err(true, pred, weight=None):
        """Return calculated mean_absolute_percentage_error."""
        array = abs(true - pred) / abs(
            np.where(true != 0, true, np.finfo(np.float64).eps)
        )
        if weight is not None and len(weight) == len(true):
            return array * weight * len(weight) / sum(weight)
        return array

    @staticmethod
    def log_error(true, pred, weight=None):
        """Return calculated mean_squared_log_error."""
        true = true.copy()
        pred = pred.copy()
        mask = (true < -1) | (pred < -1)
        if sum(mask) / len(mask) > 0.5:
            true = -1 * true
            pred = -1 * pred
            mask = (true < -1) | (pred < -1)
        true[mask] = -1 + 1e-12
        pred[mask] = -1 + 1e-12
        array = np.log(1 + true) - np.log(1 + pred)
        if weight is not None and len(weight) == len(true):
            return array * weight * len(weight) / sum(weight)
        return array

    @staticmethod
    def squared_log_error(true, pred, weight=None):
        """Return calculated mean_squared_log_error."""
        true = true.copy()
        pred = pred.copy()
        mask = (true < -1) | (pred < -1)
        if sum(mask) / len(mask) > 0.5:
            true = -1 * true
            pred = -1 * pred
            mask = (true < -1) | (pred < -1)
        true[mask] = -1 + 1e-12
        pred[mask] = -1 + 1e-12
        array = np.square(np.log(1 + true) - np.log(1 + pred))
        if weight is not None and len(weight) == len(true):
            return array * weight * len(weight) / sum(weight)
        return array

    @staticmethod
    def log10_error(true, pred, weight=None):
        """Return calculated mean_squared_log_error."""
        true = true.copy()
        pred = pred.copy()
        mask = (true < -1) | (pred < -1)
        if sum(mask) / len(mask) > 0.5:
            true = -1 * true
            pred = -1 * pred
            mask = (true < -1) | (pred < -1)
        true[mask] = -1 + 1e-12
        pred[mask] = -1 + 1e-12
        array = np.log10(1 + true) - np.log10(1 + pred)
        if weight is not None and len(weight) == len(true):
            return array * weight * len(weight) / sum(weight)
        return array

    @staticmethod
    def squared_log10_error(true, pred, weight=None):
        """Return calculated mean_squared_log_error."""
        true = true.copy()
        pred = pred.copy()
        mask = (true < -1) | (pred < -1)
        if sum(mask) / len(mask) > 0.5:
            true = -1 * true
            pred = -1 * pred
            mask = (true < -1) | (pred < -1)
        true[mask] = -1 + 1e-12
        pred[mask] = -1 + 1e-12
        array = np.square(np.log10(1 + true) - np.log10(1 + pred))
        if weight is not None and len(weight) == len(true):
            return array * weight * len(weight) / sum(weight)
        return array

    ### single value results
    @staticmethod
    def r2_score(true, pred, weight=None):
        """Return calculated mean_absolute_error."""
        sse = np.sum(np.square(true - pred))
        sst = np.sum((true - np.average(true, weights=weight, axis=0)) ** 2)
        return 1 - sse / sst

    @staticmethod
    def mean_abs_err(true, pred, weight=None):
        """Return calculated mean_absolute_error."""
        return np.average(abs(true - pred), weights=weight, axis=0)

    @staticmethod
    def median_abs_err(true, pred, weight=None):
        """Return calculated median_absolute_error."""
        return np.median(abs(true - pred))

    @staticmethod
    def mean_squared_err(true, pred, weight=None):
        """Return calculated mean_squared_error."""
        return np.average(np.square(true - pred), weights=weight, axis=0)

    @staticmethod
    def root_mean_squared_err(true, pred, weight=None):
        """Return calculated mean_squared_error."""
        return np.sqrt(np.average(np.square(true - pred), weights=weight, axis=0))

    @staticmethod
    def root_mean_abs_err(true, pred, weight=None):
        """Return calculated root mean_squared_error."""
        return np.sqrt(np.average(abs(true - pred), weights=weight, axis=0))

    @staticmethod
    def mean_abs_perc_err(true, pred, weight=None):
        """Return calculated mean_absolute_percentage_error."""
        array = abs(true - pred) / abs(
            np.where(true != 0, true, np.finfo(np.float64).eps)
        )
        return np.average(array, weights=weight, axis=0)

    @staticmethod
    def mean_log_error(true, pred, weight=None):
        """Return calculated mean_squared_log_error."""
        true = true.copy()
        pred = pred.copy()
        mask = (true < -1) | (pred < -1)
        if sum(mask) / len(mask) > 0.5:
            true = -1 * true
            pred = -1 * pred
            mask = (true < -1) | (pred < -1)
        true[mask] = -1 + 1e-12
        pred[mask] = -1 + 1e-12
        array = np.log(1 + true) - np.log(1 + pred)
        return np.average(array, weights=weight, axis=0)

    @staticmethod
    def mean_squared_log_error(true, pred, weight=None):
        """Return calculated mean_squared_log_error."""
        true = true.copy()
        pred = pred.copy()
        mask = (true < -1) | (pred < -1)
        if sum(mask) / len(mask) > 0.5:
            true = -1 * true
            pred = -1 * pred
            mask = (true < -1) | (pred < -1)
        true[mask] = -1 + 1e-12
        pred[mask] = -1 + 1e-12
        array = np.square(np.log(1 + true) - np.log(1 + pred))
        return np.average(array, weights=weight, axis=0)

    @staticmethod
    def mean_log10_error(true, pred, weight=None):
        """Return calculated mean_squared_log_error."""
        true = true.copy()
        pred = pred.copy()
        mask = (true < -1) | (pred < -1)
        if sum(mask) / len(mask) > 0.5:
            true = -1 * true
            pred = -1 * pred
            mask = (true < -1) | (pred < -1)
        true[mask] = -1 + 1e-12
        pred[mask] = -1 + 1e-12
        array = np.log10(1 + true) - np.log10(1 + pred)
        return np.average(array, weights=weight, axis=0)

    @staticmethod
    def mean_squared_log10_error(true, pred, weight=None):
        """Return calculated mean_squared_log_error."""
        true = true.copy()
        pred = pred.copy()
        mask = (true < -1) | (pred < -1)
        if sum(mask) / len(mask) > 0.5:
            true = -1 * true
            pred = -1 * pred
            mask = (true < -1) | (pred < -1)
        true[mask] = -1 + 1e-12
        pred[mask] = -1 + 1e-12
        array = np.square(np.log10(1 + true) - np.log10(1 + pred))
        return np.average(array, weights=weight, axis=0)

    @staticmethod
    def mean_poisson_dev(true, pred, weight=None):
        """Return calculated mean_poisson_deviance."""
        true = true.copy()
        pred = pred.copy()
        mask = (true < -1) | (pred < -1)
        if sum(mask) / len(mask) > 0.5:
            true = -1 * true
            pred = -1 * pred
            mask = (true < -1) | (pred < -1)
        true[mask] = -1 + 1e-12
        pred[mask] = -1 + 1e-12
        array = 2 * (true * np.log(true / pred) + pred - true)
        return np.average(array, weights=weight, axis=0)

    @staticmethod
    def mean_gamma_dev(true, pred, weight=None):
        """Return calculated mean_gamma_deviance."""
        true = true.copy()
        pred = pred.copy()
        mask = (true < -1) | (pred < -1)
        if sum(mask) / len(mask) > 0.5:
            true = -1 * true
            pred = -1 * pred
            mask = (true < -1) | (pred < -1)
        true[mask] = -1 + 1e-12
        pred[mask] = -1 + 1e-12
        array = 2 * (np.log(pred / true) + true / pred - 1)
        return np.average(array, weights=weight, axis=0)


# noinspection PyTupleAssignmentBalance
class FittingMethods:
    """Class to create a fitting method manager."""

    @property
    def lsq_kwargs(self):
        return {
            "method": "trf",
            "jac": "3-point",
            "x_scale": "jac",
            "ftol": 1e-14,
            "xtol": 1e-8,
            "gtol": 1e-8,
            "loss": "cauchy",
            "diff_step": None,
            "tr_solver": None,
            "tr_options": {},
            "jac_sparsity": None,
            "verbose": 1,
            "max_nfev": 1e6,
        }

    def circuit_fit(
        self,
        impedances,
        func,
        initial_guess,
        cols,
        bounds=None,
        weight_by_modulus=False,
        **kwargs,
    ):

        kwargs = {**self.lsq_kwargs, **kwargs}
        kwargs.pop("weights", None)
        kwargs.pop("loss_func", None)
        kwargs["max_nfev"] = kwargs.pop("maxfev", self.lsq_kwargs["max_nfev"])

        # weighting scheme for fitting
        if weight_by_modulus:
            abs_Z = np.abs(impedances.Z)
            kwargs["sigma"] = np.hstack([abs_Z] * len(cols))

        self.result = curve_fit(
            func,
            impedances["freq"],
            np.hstack([impedances[c] for c in cols]),
            p0=initial_guess,
            bounds=bounds,
            **kwargs,
        )

        popt, pcov = self.result
        perror = np.sqrt(np.diag(pcov))

        return popt, perror

    def ls_fit(
        self,
        impedances,
        minimizer,
        initial_guess,
        func,
        cols,
        bounds=None,
        weight_by_modulus=False,
        **kwargs,
    ):

        kwargs = {**self.lsq_kwargs, **kwargs}
        weights = kwargs.pop("weights", None)
        loss = kwargs.pop("loss_func", None)

        # weighting scheme for fitting
        if weight_by_modulus:
            weights = np.hstack([np.abs(impedances.Z)] * len(cols))

        if "sq" in minimizer:
            minimizer = getattr(self, "sq_minimizer")
        else:
            minimizer = getattr(self, "base_minimizer")

        self.result = least_squares(
            minimizer,
            initial_guess,
            args=(func, impedances, cols, weights, loss),
            bounds=bounds,
            **kwargs,
        )

        popt = self.result.x
        J = self.result.jac
        cov = np.linalg.inv(J.T @ J)
        perr = np.sqrt(np.diag(cov))

        return popt, perr

    @staticmethod
    def base_minimizer(params, circuit_func, Z_data, cols, weights=None, loss=None):
        Z0 = np.array(np.hsplit(circuit_func(Z_data["freq"], *params), 2)).T
        Z_fit = ComplexSystem(
            Z0,
            Z_data["freq"],
            thickness=Z_data.thickness,
            area=Z_data.area,
        )

        Z_fit = np.hstack([Z_fit[c] for c in cols])
        Z_data = np.hstack([Z_data[c] for c in cols])

        if weights is not None and len(weights) != len(Z_data):
            weights = None  # Set to None if there is a mismatch

        residuals = Z_data - Z_fit

        if callable(loss):
            residuals = loss(Z_data, Z_fit, weights)

        return residuals

    @staticmethod
    def sq_minimizer(params, circuit_func, Z_data, cols, *args):
        """Minimizer for least_squares with squared error."""
        Z0 = np.array(np.hsplit(circuit_func(Z_data["freq"], *params), 2)).T
        Z_fit = ComplexSystem(
            Z0,
            Z_data["freq"],
            thickness=Z_data.thickness,
            area=Z_data.area,
        )

        Z_fit = np.hstack([Z_fit[c] for c in cols])
        Z_data = np.hstack([Z_data[c] for c in cols])
        return (Z_data - Z_fit) ** 2

    @staticmethod
    def wrapSystem(circuit_func, thickness, area, cols):
        """Used primarily for circuitfit"""

        def wrapped(freq, *params):
            """Wrap the circuit function."""
            Z0 = np.array(np.hsplit(circuit_func(freq, *params), 2)).T
            Z_fit = ComplexSystem(
                Z0,
                freq,
                thickness=thickness,
                area=area,
            )
            return np.hstack([Z_fit[c] for c in cols])

        return wrapped
