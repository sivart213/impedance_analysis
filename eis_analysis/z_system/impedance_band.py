# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 22:12:37 2022.

@author: j2cle
"""
import itertools
from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .system import ComplexSystem
from ..data_treatment.data_analysis import ConfidenceAnalysis


class ImpedanceConfidence(ConfidenceAnalysis):
    """Class to generate confidence bands for EIS data."""

    def _gen_df(
        self,
        x_array,
        popt,
        std,
        func,
        num_x_points,
        main_col,
        target_form,
        **kwargs,
    ):
        """
        Generate dataframes for the given parameters.

        Parameters:
        x_array (array): Array of frequencies.
        popt (array): Best-fit parameters.
        std (array or float): Standard deviations of the parameters.
        func (function): Function to simulate the circuit.
        num_x_points (int): Number of frequency points.
        main_col (str): Column to stack.
        target_form (str): Type of complex data to generate

        Returns:
        zdf_list (list): List of dataframes for Z data.
        fdf_list (list): List of dataframes for frequency data.
        """

        if isinstance(std, (int, float)):
            std = np.array(popt) * std

        zdf_list = []
        fdf_list = []

        # if main_col is not None:
        comp_col = "imag" if main_col == "real" else "real"

        for params in list(itertools.product(*self.param_bounds)):
            if main_col is not None:
                freq_exp = np.logspace(
                    np.log10(x_array.min()),
                    np.log10(x_array.max()),
                    num=num_x_points,
                )
                Z = np.array(np.hsplit(func(freq_exp, *params), 2)).T
                independant = Z[:, 0] if main_col == "real" else -Z[:, 1]
                dependant = -Z[:, 1] if main_col == "real" else Z[:, 0]
                temp_df = pd.DataFrame(
                    {
                        main_col: independant,
                        f"Z_{comp_col}_{params}": dependant,
                    }
                )
                zdf_list.append(temp_df.copy())
            if target_form is not None:
                Z = np.array(np.hsplit(func(x_array, *params), 2)).T
                data_system = ComplexSystem(np.array([x_array, Z[:, 0], Z[:, 1]]).T, **kwargs)
                if isinstance(target_form, str):
                    temp_df = data_system.get_df("freq", target_form)
                elif isinstance(target_form, (tuple, list)):
                    temp_df = data_system.get_df(*target_form)
                else:
                    raise ValueError("target_form must be a string or a tuple/list of strings.")
                temp_df = temp_df.rename(
                    columns={col: f"{col}_{params}" for col in temp_df.columns if col != "freq"}
                )
                fdf_list.append(temp_df.copy())

        return zdf_list, fdf_list

    def gen_conf_band_nyquist(
        self,
        x_array,
        popt=None,
        std=None,
        func: Callable | None = None,
        num_x_points=500,
        multiplier=50,
        main_col="real",
        percentile=None,
        df_list: list[pd.DataFrame] | None = None,
    ):
        """
        Generate confidence bands for Nyquist plots.

        Parameters:
        popt (array): Best-fit parameters.
        std (array or float): Standard deviations of the parameters.
        func (function): Function to simulate the circuit.
        x_array (array): Array of frequencies.
        main_col (str): Column to stack.

        Returns:
        bounds (dict): Dictionary containing the confidence bands.
        """

        if popt is not None:
            self.popt = popt
        if std is not None:
            self.std = std
        if percentile is not None:
            self.percentile = percentile

        if df_list is None:
            zdf_list, _ = self._gen_df(
                x_array,
                self.popt,
                self.std,
                func,
                num_x_points,
                main_col,
                None,
            )
        else:
            zdf_list = df_list

        df = pd.concat(zdf_list, axis=0)

        # Calculate the decade range and number of bins
        decade_range = np.log10(abs(df[main_col].max())) - np.log10(abs(df[main_col].min()))
        num_bins = int(decade_range * multiplier)

        df = df.sort_values(by=main_col, ascending=True)
        df = df.interpolate(method="linear", axis=0).ffill().bfill()
        df = df.assign(binned_main_col=pd.cut(df[main_col], bins=num_bins, labels=False))

        # Group by binned_main_col and calculate descriptive statistics for each group
        def calc_group_stats(group):
            # Exclude binned_main_col and main_col from the calculations
            group = group.drop(columns=["binned_main_col", main_col])
            # Flatten the DataFrame
            flattened = group.values.flatten()
            # Calculate descriptive statistics
            stats = pd.Series(flattened).describe(percentiles=list(self.percentile))
            return stats

        grouped_stats = df.groupby("binned_main_col").apply(calc_group_stats).reset_index()
        grouped_stats = grouped_stats.rename(columns={"binned_main_col": main_col})
        grouped_stats[main_col] = (
            df.groupby("binned_main_col").mean()[main_col].reset_index(drop=True)
        )

        return {"nyquist": grouped_stats}

    def gen_conf_band_bode(
        self,
        x_array: ArrayLike,
        popt: ArrayLike | None = None,
        std: ArrayLike | None = None,
        func: Callable | None = None,
        target_form: str | list | tuple = "Z",
        percentile=None,
        df_list: list[pd.DataFrame] | None = None,
        **kwargs,
    ):
        """
        Generate confidence bands for Bode plots.

        Parameters:
        popt (array): Best-fit parameters.
        std (array or float): Standard deviations of the parameters.
        func (function): Function to simulate the circuit.
        x_array (array): Array of frequencies.
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

        if df_list is None:
            _, fdf_list = self._gen_df(x_array, self.popt, self.std, func, None, None, target_form)
        else:
            fdf_list = df_list

        bounds = {}

        combined_df = pd.concat(fdf_list, axis=1)
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
        combined_df.set_index("freq", inplace=True)
        stats_dict = {}
        columns = ["real", "imag", "mag", "phase", "slope"]
        if isinstance(target_form, (tuple, list)):
            columns = target_form[1:]
        for col in columns:
            col_prefix = f"{col}_"
            matching_cols = [c for c in combined_df.columns if c.startswith(col_prefix)]
            if matching_cols:
                stats_dict[col] = (
                    combined_df[matching_cols]
                    .T.describe(percentiles=list(self.percentile))
                    .T.reset_index()
                    .rename(columns={"index": "freq"})
                )
        bounds.update(stats_dict)

        return bounds

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

        zdf_list, fdf_list = self._gen_df(
            x_array,
            self.popt,
            self.std,
            func,
            kwargs.get("num_x_points", 500),
            kwargs.get("main_col", "real"),
            target_form,
            **kwargs,
        )

        bounds = {}

        bounds.update(
            self.gen_conf_band_nyquist(
                x_array,
                main_col=kwargs.get("main_col", "real"),
                num_x_points=kwargs.get("num_x_points", 500),
                multiplier=kwargs.get("multiplier", 50),
                df_list=zdf_list,
            )
        )
        bounds.update(
            self.gen_conf_band_bode(
                x_array,
                target_form=target_form,
                num_x_points=kwargs.get("num_x_points", 500),
                df_list=fdf_list,
            )
        )

        return bounds
