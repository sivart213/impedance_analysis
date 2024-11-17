# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""


import re
from pathlib import Path

import numpy as np
import pandas as pd


from PyQt5.QtCore import pyqtSignal, QObject

from PyQt5.QtWidgets import (
    QMessageBox,
)


from impedance.models.circuits.fitting import (
    wrapCircuit,
    extract_circuit_elements,
    calculateCircuitLength,
)
from impedance.validation import linKK

from ..data_treatment.data_analysis import FittingMethods, Statistics, ComplexSystem
from ..equipment.mfia_ops import convert_mfia_df_for_fit
from ..system_utilities.file_io import load_file, save

class DataHandler:
    """Class to store data for plotting graphs."""

    def __init__(self):
        self.model = "p(R1,C1)-p(R2,C2)-p(R3,C3)"  # "R0-p(R1,CPE1,R2-CPE2)"  # "R0-p(R1,C1)"
        self.raw = {}
        self.linkk = None

        self.var_val = {
            "Z": "impedance",
            "Y": "admittance",
            "M": "modulus",
            "ε": "permittivity",
            "εᵣ": "relative_permittivity",
            "σ": "conductivity",
            "ρ": "resistivity",
            "Real": "real",
            "Imag": "imag",
            "+Imag": "pos_imag",
            "Mag": "mag",
            "Θ": "phase",
            "tan(δ)": "tan",
        }

        self.var_scale = {
            "Real": "lin",
            "Imag": "lin",
            "+Imag": "log",
            "Mag": "log",
            "Θ": "deg",
            "tan(δ)": "lin",
        }

        self.var_units = {
            "Z": r"[$\Omega$]",
            "Y": "[S]",
            "M": "[cm/F]",
            "ε": "[F/cm]",
            "εᵣ": "[1]",
            "σ": "[S/cm]",
            "ρ": r"[$\Omega$ cm]",
        }

        self.error_methods = {
            key.replace("_", " ").title(): key
            for key in Statistics().single_method_list.copy()
        }

        # Store the initialized values in a separate dictionary for resetting
        self.option_inits = {
            "simulation": {
                "limit_error": True,
                "freq_start": -4.5,
                "freq_stop": 7,
                "freq_num": 200,
                "area (cm^2)": 5 * 5,  # cm
                "thickness (cm)": 450e-4,
                "interval": 0.1,
            },
            "bands": {
                "band_color": "gray",
                "band_alpha": 0.2,
                "band_freq_num": 250,
                "band_mult": 50,
                "percentile": 5,
                "std_devs": 0.2,  # if string, and result use that
                "conf_band_upper": "97.5%",  # if min or max use that
                "conf_band_lower": "2.5%",  # if min or max use that
            },
            "fit": {
                "function": "least_squares",
                "type": "impedance",
                "modes": ["real", "imag", "mag"],
                "f_max": 1e7,
                "f_min": 1e-6,
                "weight_by_modulus": True,
                "bootstrap_percent": 95,
            },
            "curve_fit": {
                "absolute_sigma": False,
                "check_finite": None,
                "method": None,
                "jac": "3-point",
                "x_scale": "jac",
                "ftol": 1e-14,
                "xtol": 1e-6,
                "gtol": 1e-8,
                "loss": "cauchy",
                "diff_step": None,
                "tr_solver": None,
                "tr_options": {},
                "jac_sparsity": None,
                "verbose": 0,
                "maxfev": 1e6,
            },
            "least_sq": {
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
                "verbose": 0,
                "max_nfev": 1e6,
            },
            "linkk": {
                "f_max": 1e6,
                "f_min": 1e-6,
                "c": 0.5,
                "max_M": 200,
                "add_cap": False,
            },
            "element": {
                "R": 1e10,
                "C": 1e-10,
                "CPE": (1e-10, 1),
                "L": 100,
            },
            "element_range": {
                "R": [0, np.inf],
                "C": [0, 1],
                "CPE": [[0, 1], [-1, 1]],
                "L": [0, np.inf],
            },
        }

    def parse_default(
        self,
        name,
        defaults,
        override=None,
    ):
        """Parse the default value for the parameter."""
        if override is not None:
            if name in override.keys():
                return override[name]

        if self.model.lower() == "linkk":
            return 1

        if name in defaults.keys():
            return defaults[name]
        split_name = re.findall(r"(^[a-zA-Z]+)_?([0-9_]+)", name)[0]
        if split_name[0] not in defaults.keys():
            return 1
        if "_" in split_name[1]:
            index = int(
                eval(split_name[1].split("_")[-1], {}, {"inf": np.inf})
            )
            return defaults[split_name[0]][index]
        return defaults[split_name[0]]

    def parse_label(self, main, mode):
        """Parse the label based on the selected option."""
        units = self.var_units[main]
        if mode.lower() == "real":
            return f"{main}' {units}"
        elif mode.lower() == "imag":
            return f"{main}'' {units}"
        elif mode.lower() == "+imag":
            return f"{main}'' {units}"
        elif mode.lower() == "mag":
            return f"|{main}| {units}"
        elif mode == "phase" or self.var_val[mode] == "phase":
            return "θ [deg]"
        elif mode == "tan" or self.var_val[mode] == "tan":
            return "tan(δ) [1]"

    def parse_parameters(self, model=None):
        """Get the parameters of the model."""
        if model is None:
            model = self.model
        if model.lower() == "linkk":
            return ["M", "mu"]
        params = extract_circuit_elements(model)
        if len(params) != calculateCircuitLength(model):
            all_params = []
            for param in params:
                length = calculateCircuitLength(param)
                if length >= 2:
                    all_params.append(f"{param}_0")
                    for i in range(1, length):
                        all_params.append(f"{param}_{i}")
                else:
                    all_params.append(param)
            params = all_params
        return params

    def generate(
        self,
        params_values,
        model=None,
        freq=None,
        freq_start=-4.5,
        freq_stop=7,
        freq_num=200,
        area=25,
        thickness=450e-4,
        **kwargs,
    ):
        """Create the fit data based on the current parameter values."""
        if model is None:
            model = self.model

        if model.lower() == "linkk":
            return self.linkk

        circuit_func = wrapCircuit(model, {})

        if freq is None:
            freq = np.logspace(freq_start, freq_stop, freq_num)

        if not params_values:
            return
        try:
            Z = np.array(np.hsplit(circuit_func(freq, *params_values), 2)).T
        except (IndexError, AssertionError) as exc:
            raise IndexError("List index out of range") from exc

        return ComplexSystem(
            Z[:, 0] + 1j * Z[:, 1],
            frequency=freq,
            thickness=thickness,
            area=area,
        )

    def generate_linkk(
        self,
        df,
        c=0.5,
        max_M=200,
        add_cap=False,
        f_min=-4.5,
        f_max=7,
        area=25,
        thickness=450e-4,
        **kwargs,
    ):
        """Run the Lin-KK fit based on the selected model."""
        # Filter the scatter_data based on f_min and f_max
        df = df[(df["freq"] >= f_min) & (df["freq"] <= f_max)]
        f = df["freq"].to_numpy()
        Z = ComplexSystem(df[["real", "imag"]]).Z

        M, mu, Z_linKK, _, _ = linKK(
            f,
            Z,
            c=c,
            max_M=max_M,
            fit_type="complex",
            add_cap=add_cap,
        )

        # Create a DataFrame from Z_linKK
        self.linkk = ComplexSystem(
            Z_linKK,
            frequency=f,
            thickness=thickness,
            area=area,
        )

        return (
            M,
            mu,
        )


class FittingWorker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, main, iterations=1):
        super().__init__()
        self.main = main
        self.iterations = iterations
        self.cancelled = False

        # Data parsing moved back to init
        main.print_window.write("Preparing to fit...\n")

        # Retrieve f_min and f_max from options
        f_min = main.options["fit"]["f_min"]
        f_max = main.options["fit"]["f_max"]

        self.dataset_var = main.dataset_var.currentText()
        self.filtered_data = main.data.raw[self.dataset_var].base_df(
            None, "frequency"
        )

        # Filter the scatter_data based on f_min and f_max
        self.filtered_data = self.filtered_data[
            (self.filtered_data["freq"] >= f_min)
            & (self.filtered_data["freq"] <= f_max)
        ]

        # Check if filtered_data is empty
        if self.filtered_data.empty:
            QMessageBox.warning(
                main.root,
                "Warning",
                "No data points within the specified frequency range.",
            )
            return

        checked_params = np.array(main.parameters.checks)[:, 0]
        bounds_arr = np.array(main.bounds.values)[~checked_params]
        self.bounds = (bounds_arr[:, 0], bounds_arr[:, 1])

        self.initial_guess = []
        self.param_names = []
        self.constants = {}

        for param, check in zip(main.parameters, checked_params):
            if check:
                self.constants[param.name] = param.values[0]
            else:
                self.initial_guess.append(param.values[0])
                self.param_names.append(param.name)

        if isinstance(main.options["fit"]["type"], (list, tuple)):
            self.columns = [
                fit_type + "." + mode
                for mode in main.options["fit"]["modes"]
                for fit_type in main.options["fit"]["type"]
            ]
        else:
            self.columns = [
                main.options["fit"]["type"] + "." + mode
                for mode in main.options["fit"]["modes"]
            ]

        self.model = main.data.model
        self.options = main.options
        self.loss_func = None
        if "sq" in self.options["fit"]["function"]:
            self.loss_func = Statistics().as_array(
                main.data.error_methods[main.error_var.currentText()]
            )

    def run(self):
        try:
            if self.iterations > 1:
                self.perform_bootstrap_fit()

            else:
                fit_results = self.perform_fit(self.filtered_data)
                self.finished.emit(fit_results)
        except (TypeError, ValueError) as exc:
            self.error.emit(str(exc))

    def perform_fit(self, data):
        """Perform a single round of fitting."""
        if "circuit" in self.options["fit"]["function"]:
            fit_results, std_results = FittingMethods().circuit_fit(
                ComplexSystem(
                    data[["freq", "real", "imag"]],
                    thickness=self.options["simulation"]["thickness (cm)"],
                    area=self.options["simulation"]["area (cm^2)"],
                ),
                FittingMethods.wrapSystem(
                    wrapCircuit(self.model, self.constants),
                    thickness=self.options["simulation"]["thickness (cm)"],
                    area=self.options["simulation"]["area (cm^2)"],
                    cols=self.columns,
                ),
                self.initial_guess,
                cols=self.columns,
                bounds=self.bounds,
                weight_by_modulus=self.options["fit"]["weight_by_modulus"],
                **self.options["curve_fit"],
            )
        elif "sq" in self.options["fit"]["function"]:
            fit_results, std_results = FittingMethods().ls_fit(
                ComplexSystem(
                    data[["freq", "real", "imag"]],
                    thickness=self.options["simulation"]["thickness (cm)"],
                    area=self.options["simulation"]["area (cm^2)"],
                ),
                "base_minimizer",
                self.initial_guess,
                wrapCircuit(self.model, self.constants),
                self.columns,
                bounds=self.bounds,
                weight_by_modulus=self.options["fit"]["weight_by_modulus"],
                loss_func=self.loss_func,
                **self.options["least_sq"],
            )
        else:
            fit_results, std_results = FittingMethods().circuit_fit(
                ComplexSystem(
                    data[["freq", "real", "imag"]],
                    thickness=self.options["simulation"]["thickness (cm)"],
                    area=self.options["simulation"]["area (cm^2)"],
                ),
                wrapCircuit(self.model, self.constants),
                self.initial_guess,
                cols=["real", "imag"],
                bounds=self.bounds,
                weight_by_modulus=self.options["fit"]["weight_by_modulus"],
                **self.options["least_sq"],
            )
        # return fit_results, std_results
        fit_df = pd.DataFrame(
            [fit_results, std_results],
            index=["mean", "std"],
            columns=self.param_names,
        )
        fit_df.attrs["model"] = self.model
        fit_df.attrs["dataset_var"] = self.dataset_var
        fit_df.attrs["area"] = self.options["simulation"]["area (cm^2)"]
        fit_df.attrs["thickness"] = self.options["simulation"][
            "thickness (cm)"
        ]

        return fit_df

    def perform_bootstrap_fit(self):
        """Perform a bootstrap fit."""
        bootstrap_results = []
        for i in range(self.iterations):
            resampled_data = self.filtered_data.copy().sample(
                frac=1, replace=True
            )
            fit_results = self.perform_fit(resampled_data)
            # Collect results
            bootstrap_results.append(fit_results.loc["mean"])
            self.progress.emit(int((i + 1) / self.iterations * 100))
            if self.main.kill_operation:  # Check if cancelled
                break

        bootstrap_df = pd.DataFrame(
            bootstrap_results, columns=self.param_names
        )
        bootstrap_df.attrs["model"] = self.model
        bootstrap_df.attrs["dataset_var"] = self.dataset_var
        bootstrap_df.attrs["area"] = self.options["simulation"]["area (cm^2)"]
        bootstrap_df.attrs["thickness"] = self.options["simulation"][
            "thickness (cm)"
        ]

        self.finished.emit(bootstrap_df)


class LoadDataWorker(QObject):
    finished = pyqtSignal(dict, object)  # Signal to emit the results
    error = pyqtSignal(str)  # Signal to emit errors

    def __init__(self, file_path, options):
        super().__init__()
        self.file_path = file_path
        self.options = options

    def run(self):
        try:
            data_in = load_file(self.file_path)[0]
            valid_sheets = {}
            df_in = None

            if isinstance(data_in, dict):
                for sheet_name, df in data_in.items():
                    if (
                        isinstance(df.columns, pd.MultiIndex)
                        and any(
                            "realz" in level for level in df.columns.levels
                        )
                    ) or "realz" in df.columns:
                        df = convert_mfia_df_for_fit(df)
                    if sheet_name == "fit results":
                        df_in = df.copy()
                    elif all(
                        col in df.columns for col in ["freq", "real", "imag"]
                    ):
                        valid_sheets[sheet_name] = ComplexSystem(
                            df[["freq", "real", "imag"]],
                            thickness=self.options["simulation"][
                                "thickness (cm)"
                            ],
                            area=self.options["simulation"]["area (cm^2)"],
                        )
                if df_in is not None:
                    name_translation = dict(
                        zip(df_in["Name"].str.lower(), df_in["Dataset"])
                    )
                    valid_sheets = {
                        name_translation.get(k.lower(), k): v
                        for k, v in valid_sheets.items()
                    }
            else:
                if all(
                    col in data_in.columns for col in ["freq", "real", "imag"]
                ):
                    valid_sheets[self.file_path.stem] = ComplexSystem(
                        data_in[["freq", "real", "imag"]],
                        thickness=self.options["simulation"]["thickness (cm)"],
                        area=self.options["simulation"]["area (cm^2)"],
                    )

            self.finished.emit(valid_sheets, df_in)
        except Exception as e:
            self.error.emit(str(e))


class SaveResultsWorker(QObject):
    finished = pyqtSignal()  # Signal to emit when done
    error = pyqtSignal(str)  # Signal to emit errors

    def __init__(
        self, file_path, data, options, pinned, parameters, dataset_var
    ):
        super().__init__()
        self.file_path = file_path
        self.data = data
        self.options = options
        self.pinned = pinned
        self.parameters = parameters
        self.dataset_var = dataset_var

    def run(self):
        try:
            res = {}
            if not self.pinned.df.empty:
                filtered_cols = self.pinned.df_base_cols.copy()
                for col in self.pinned.df_sort_cols:
                    filtered_cols += [
                        c for c in self.pinned.df.columns if col in c
                    ]

                res_df = self.pinned.df[filtered_cols].copy()
                res_df.columns = [
                    (
                        col.replace(
                            f"_{self.pinned.df_sort_cols[0].lower()}", ""
                        )
                        if self.pinned.df_sort_cols[0].lower() in col
                        else col
                    )
                    for col in res_df.columns
                ]
                res["fit results"] = res_df
                for _, row in res_df.iterrows():
                    params_values = [
                        row[name]
                        for name in self.data.parse_parameters(
                            model=row["Model"]
                        )
                    ]
                    if (
                        self.data.raw != {}
                        and row["Dataset"] in self.data.raw.keys()
                    ):
                        local_data = self.data.raw[row["Dataset"]].base_df(
                            None, "frequency"
                        )
                    else:
                        local_data = self.data.generate(
                            params_values, **self.options["simulation"]
                        ).base_df(None, "frequency")

                    generated_data = self.data.generate(
                        params_values=params_values,
                        model=row["Model"],
                        freq=local_data["freq"].to_numpy(),
                        **self.options["simulation"],
                    ).base_df(None, "frequency")

                    for col in generated_data.columns:
                        local_data[f"pr_{col}"] = generated_data[col]

                    res[row["Name"]] = local_data
            else:
                params_values = [entry.values[0] for entry in self.parameters]
                params_names = [entry.name for entry in self.parameters]
                if self.data.raw != {}:
                    res[self.dataset_var.currentText()] = self.data.raw[
                        self.dataset_var.currentText()
                    ].base_df(None, "frequency")
                    generated_data = self.data.generate(
                        params_values,
                        freq=res[self.dataset_var.currentText()][
                            "freq"
                        ].to_numpy(),
                        **self.options["simulation"],
                    ).base_df(None, "frequency")
                    for col in generated_data.columns:
                        res[self.dataset_var.currentText()][f"pr_{col}"] = (
                            generated_data[col]
                        )
                else:
                    generated_data = self.data.generate(
                        params_values, **self.options["simulation"]
                    ).base_df(None, "frequency")
                    res["fit profile"] = generated_data

                res["fit results"] = pd.DataFrame(
                    [params_values], columns=params_names
                )

            for key, value in self.data.raw.items():
                if (
                    key not in res.keys()
                    and key
                    not in res["fit results"]["Dataset"].to_numpy().tolist()
                ):
                    res[key] = value.base_df(None, "frequency")

            save(
                res,
                Path(self.file_path).parent,
                name=Path(self.file_path).stem,
                file_type=Path(self.file_path).suffix,
            )
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class SaveFiguresWorker(QObject):
    finished = pyqtSignal()  # Signal to emit when done
    error = pyqtSignal(str)  # Signal to emit errors

    def __init__(self, file_path, fig1, fig2):
        super().__init__()
        self.file_path = file_path
        self.fig1 = fig1
        self.fig2 = fig2

    def run(self):
        try:
            base_path = Path(self.file_path)
            nyquist_file_path = base_path.with_name(
                base_path.stem + "_nyquist"
            ).with_suffix(base_path.suffix)
            bode_file_path = base_path.with_name(
                base_path.stem + "_bode"
            ).with_suffix(base_path.suffix)

            self.fig1.savefig(nyquist_file_path)
            self.fig2.savefig(bode_file_path)

            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
