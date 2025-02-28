# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""


import re
from pathlib import Path
from typing import Optional
import json
# from datetime import datetime
import numpy as np
import pandas as pd


from PyQt5.QtCore import pyqtSignal, QObject, QThread, Qt

from PyQt5.QtWidgets import (
    QMessageBox,
    QProgressDialog,
    # QInputDialog,
)


from impedance.models.circuits.fitting import (
    wrapCircuit,
    extract_circuit_elements,
    calculateCircuitLength,
)
from impedance.validation import linKK

from ..data_treatment import get_valid_keys
from ..data_treatment.data_analysis import FittingMethods, Statistics, ComplexSystem
from ..dict_ops import update_dict, filter_dict, check_dict
from ..equipment.mfia_ops import convert_mfia_df_for_fit
from ..system_utilities.file_io import load_file, save

CommonExceptions = (
    TypeError,
    ValueError,
    IndexError,
    KeyError,
    AttributeError,
)
# Path(__file__).parent / "settings.json"

class WorkerError(Exception):
    """Custom exception class for handling unexpected errors."""

    
def post_decoder(obj):
    try:
        if "__path__" in obj:
            return Path(obj["__path__"]).expanduser()
        if "__complex__" in obj:
            return complex(*obj["__complex__"])
        if "__invalid_float__" in obj:
            return eval(obj["__invalid_float__"], {}, {"inf": np.inf, "nan": np.nan})
    except (KeyError, TypeError, ValueError, SyntaxError):
        pass
    return obj

def pre_encoder(data):
    """Recursively preprocess data to convert Infinity and NaN values."""
    if isinstance(data, Path):
        return {"__path__": str(data)}
    elif isinstance(data, complex):
        return {"__complex__": [data.real, data.imag]}
    elif isinstance(data, float) and (np.isnan(data) or np.isinf(data)):
        return {"__invalid_float__": str(data)}
    elif isinstance(data, dict):
        return {k: pre_encoder(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple, set)):
        # Handle lists, tuples, and sets
        return type(data)([pre_encoder(i) for i in data])
    return data


class JSONSettings:
    """Class to store data for plotting graphs."""

    def __init__(self, defaults_path=None, settings_path=None):
        self.defaults_path = Path(defaults_path) if defaults_path else Path(__file__).parent / "defaults.json"
        self.settings_path = Path(settings_path) if settings_path else Path(__file__).parent / "settings.json"

        # Check if default settings file exists
        if not self.defaults_path.exists():
            raise FileNotFoundError(f"Default settings file not found: {self.defaults_path}")

        # Copy default settings to local settings if local settings file doesn't exist
        if not self.settings_path.exists():
            self.restore_defaults()

        # self.settings = self.load_settings()
    def from_json(self, file_path):
        """Load JSON file and return the data."""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return json.load(file, object_hook=post_decoder)
        except (json.JSONDecodeError, FileNotFoundError) as exc:
            return self.json_loading_error(file_path, exc)

    def json_loading_error(self, file_path, exc):
        """Handle JSON decode error."""
        if file_path == self.defaults_path:
            if isinstance(exc, FileNotFoundError):
                raise FileNotFoundError(f"Default file not found: {file_path}")
            raise ValueError(f"Error decoding JSON from default path: {Path(self.defaults_path).name}")
        else:
            print(f"{type(exc).__name__} when decoding JSON from {file_path}. Attempting to load defaults.")
            res = self.from_json(self.defaults_path)
            if res:
                print(f"Successfully loaded defaults from {Path(self.defaults_path).name}. Setting {Path(file_path).name} to defaults.")
                self.to_json(res, file_path)
                return res
        
    def load_settings(self, **kwargs):
        """Load settings from JSON files and return the settings dictionary."""
        settings = self.from_json(self.defaults_path)
        local_settings = self.from_json(self.settings_path)

        update_dict(settings, local_settings)

        if kwargs:
            kwargs = check_dict(kwargs, settings)
            settings = filter_dict(settings, kwargs)

        return settings

    def to_json(self, settings, file_path):
        """Save settings to a JSON file."""
        if file_path != self.defaults_path:
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(pre_encoder(settings), file, indent=4)

    def save_settings(self, **kwargs):
        """Save the local modified values to JSON file."""
        if not kwargs:
            return
        #     kwargs = self.option_inits
        settings = self.from_json(self.settings_path)

        # Update the local settings with the current settings
        kwargs = check_dict(kwargs, settings)
        update_dict(settings, kwargs)

        self.to_json(settings, self.settings_path)

    def restore_defaults(self, **kwargs):
        """Restore the default settings."""
        settings = self.from_json(self.defaults_path)
        
        if kwargs:
            kwargs = check_dict(kwargs, settings)
            kwargs = filter_dict(settings, kwargs)
            settings = self.from_json(self.settings_path)
            update_dict(settings, kwargs)

        self.to_json(settings, self.settings_path)
        return settings

# class SettingsHandler(JSONSettings):
#     """Class to store data for plotting graphs."""

#     def __init__(self):
#         super().__init__()  # Initialize JSONSettings
#         self.model = "p(R1,C1)"
#         self.raw = {}
#         self.raw_archive = {}
#         self.linkk = None
#         self.defaults = {}
#         self.var_val = {}
#         self.var_scale = {}
#         self.var_units = {}
#         self.load_dir = None
#         self.save_dir = None
#         self.export_dir = None
#         self.plot_types = []
#         self.option_inits = {}

#         self.load_settings()

#         self.error_methods = {
#             key.replace("_", " ").title(): key
#             for key in Statistics().single_method_list.copy()
#         }

#     def save_settings(self, **kwargs):
#         """Save the local modified values to JSON file."""
#         if not kwargs:
#             return
#         #     kwargs = self.option_inits
#         for key, value in kwargs.items():
#             if hasattr(self, key):
#                 setattr(self, key, value)
#         super().save_settings(**kwargs)

#     def load_settings(self, **kwargs):
#         """Load settings from JSON files and set attributes."""
#         settings = super().load_settings(**kwargs)  # Use JSONSettings method

#         # Set attributes directly from the loaded settings
#         for key, value in settings.items():
#             setattr(self, key, value)

#     def restore_defaults(self, **kwargs):
#         """Restore the default settings."""
#         settings = super().restore_defaults(**kwargs)  # Use JSONSettings method

#         # Set attributes directly from the restored settings
#         for key, value in settings.items():
#             setattr(self, key, value)

    
# class ModelHandler(JSONSettings):
#     """Class to store data for plotting graphs."""

#     def __init__(self):
#         super().__init__()  # Initialize JSONSettings
#         self.model = "p(R1,C1)"
#         self.raw = {}
#         self.raw_archive = {}
#         self.linkk = None
#         self.defaults = {}
#         self.var_val = {}
#         self.var_scale = {}
#         self.var_units = {}
#         self.load_dir = None
#         self.save_dir = None
#         self.export_dir = None
#         self.plot_types = []
#         self.option_inits = {}

#         self.load_settings()

#         self.error_methods = {
#             key.replace("_", " ").title(): key
#             for key in Statistics().single_method_list.copy()
#         }

#     def save_settings(self, **kwargs):
#         """Save the local modified values to JSON file."""
#         if not kwargs:
#             return
#         #     kwargs = self.option_inits
#         for key, value in kwargs.items():
#             if hasattr(self, key):
#                 setattr(self, key, value)
#         super().save_settings(**kwargs)

#     def load_settings(self, **kwargs):
#         """Load settings from JSON files and set attributes."""
#         settings = super().load_settings(**kwargs)  # Use JSONSettings method

#         # Set attributes directly from the loaded settings
#         for key, value in settings.items():
#             setattr(self, key, value)

#     def restore_defaults(self, **kwargs):
#         """Restore the default settings."""
#         settings = super().restore_defaults(**kwargs)  # Use JSONSettings method

#         # Set attributes directly from the restored settings
#         for key, value in settings.items():
#             setattr(self, key, value)


#     def parse_default(self, name, defaults, override=None):
#         """Parse the default value for the parameter."""
#         if override is not None:
#             if name in override.keys():
#                 return override[name]

#         if self.model.lower() == "linkk":
#             return 1

#         if name in defaults.keys():
#             return defaults[name]
#         split_name = re.findall(r"(^[a-zA-Z]+)_?([0-9_]+)", name)[0]
#         if split_name[0] not in defaults.keys():
#             return 1
#         if "_" in split_name[1]:
#             index = int(
#                 eval(split_name[1].split("_")[-1], {}, {"inf": np.inf})
#             )
#             return defaults[split_name[0]][index]
#         return defaults[split_name[0]]

#     def parse_label(self, main, mode):
#         """Parse the label based on the selected option."""
#         units = self.var_units[main]
#         if mode.lower() == "real":
#             return f"{main}' {units}"
#         elif mode.lower() == "imag":
#             return f"{main}'' {units}"
#         elif mode.lower() == "+imag":
#             return f"{main}'' {units}"
#         elif mode.lower() == "mag":
#             return f"|{main}| {units}"
#         elif mode == "phase" or self.var_val[mode] == "phase":
#             return "θ [deg]"
#         elif mode == "tan" or self.var_val[mode] == "tan":
#             return "tan(δ) [1]"

#     def parse_parameters(self, model=None):
#         """Get the parameters of the model."""
#         if model is None:
#             model = self.model
#         if not isinstance(model, str):
#             QMessageBox.warning(None, "Warning", f"Model must be a string. Current value is {model} of type {type(model)}")
#             model = self.model
#         if model.lower() == "linkk":
#             return ["M", "mu"]
#         # try:
#         params = extract_circuit_elements(model)
#         if len(params) != calculateCircuitLength(model):
#             all_params = []
#             for param in params:
#                 length = calculateCircuitLength(param)
#                 if length >= 2:
#                     all_params.append(f"{param}_0")
#                     for i in range(1, length):
#                         all_params.append(f"{param}_{i}")
#                 else:
#                     all_params.append(param)
#             params = all_params
#         return params

#     def generate(
#         self,
#         params_values,
#         model=None,
#         freq=None,
#         freq_start=-4.5,
#         freq_stop=7,
#         freq_num=200,
#         area=25,
#         thickness=450e-4,
#         dx=0,
#         **kwargs,
#     ):
#         """Create the fit data based on the current parameter values."""
#         if model is None:
#             model = self.model

#         if model.lower() == "linkk":
#             return self.linkk

#         circuit_func = wrapCircuit(model, {})

#         if freq is None:
#             freq = np.logspace(freq_start, freq_stop, freq_num)
#         elif kwargs.get("interp") and not kwargs.get("sim_param_freq", True) and freq_num > 2 * (u_num := len(np.unique(freq)) - 1):
#             num = (freq_num // u_num) * u_num + 1
#             freq = np.logspace(min(np.log10(freq)), max(np.log10(freq)), num)
        

#         if not params_values:
#             return
#         try:
#             Z = np.array(np.hsplit(circuit_func(freq, *params_values), 2)).T
#         except (IndexError, AssertionError) as exc:
#             raise IndexError("List index out of range") from exc

#         return ComplexSystemDx(
#             Z[:, 0] + 1j * Z[:, 1],
#             frequency=freq,
#             thickness=thickness,
#             area=area,
#             dx=dx,
#         )

#     def generate_linkk(
#         self,
#         df,
#         c=0.5,
#         max_M=200,
#         add_cap=False,
#         f_min=-4.5,
#         f_max=7,
#         area=25,
#         thickness=450e-4,
#         dx=0,
#         **kwargs,
#     ):
#         """Run the Lin-KK fit based on the selected model."""
#         # Filter the scatter_data based on f_min and f_max
#         df = df[(df["freq"] >= f_min) & (df["freq"] <= f_max)]
#         f = df["freq"].to_numpy()
#         Z = ComplexSystemDx(df[["real", "imag"]], dx=dx).Z
#         # Direct Access to ComplexSystemDx
#         M, mu, Z_linKK, _, _ = linKK(
#             f,
#             Z,
#             c=c,
#             max_M=max_M,
#             fit_type="complex",
#             add_cap=add_cap,
#         )

#         # Create a DataFrame from Z_linKK
#         self.linkk = ComplexSystemDx(
#             Z_linKK,
#             frequency=f,
#             thickness=thickness,
#             area=area,
#             dx=dx,
#         )

#         return (
#             M,
#             mu,
#         )
    
    
class DataHandler(JSONSettings):
    """Class to store data for plotting graphs."""

    def __init__(self):
        super().__init__()  # Initialize JSONSettings
        self.model = "p(R1,C1)"
        self.raw = {}
        self.raw_archive = {}
        self.linkk = None
        self.defaults = {}
        self.var_val = {}
        self.var_scale = {}
        self.var_units = {}
        self.load_dir = None
        self.save_dir = None
        self.export_dir = None
        self.plot_types = []
        self.option_inits = {}

        self.load_settings()

        self.error_methods = {
            key.replace("_", " ").title(): key
            for key in Statistics().single_method_list.copy()
        }

    def save_settings(self, **kwargs):
        """Save the local modified values to JSON file."""
        if not kwargs:
            return
        #     kwargs = self.option_inits
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        super().save_settings(**kwargs)

    def load_settings(self, **kwargs):
        """Load settings from JSON files and set attributes."""
        settings = super().load_settings(**kwargs)  # Use JSONSettings method

        # Set attributes directly from the loaded settings
        for key, value in settings.items():
            setattr(self, key, value)

    def restore_defaults(self, **kwargs):
        """Restore the default settings."""
        settings = super().restore_defaults(**kwargs)  # Use JSONSettings method

        # Set attributes directly from the restored settings
        for key, value in settings.items():
            setattr(self, key, value)


    def parse_default(self, name, defaults, override=None):
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
        if not isinstance(model, str):
            QMessageBox.warning(None, "Warning", f"Model must be a string. Current value is {model} of type {type(model)}")
            model = self.model
        if model.lower() == "linkk":
            return ["M", "mu"]
        # try:
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
        # dx=0,
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
        elif kwargs.get("interp") and not kwargs.get("sim_param_freq", True) and freq_num > 2 * (u_num := len(np.unique(freq)) - 1):
            num = (freq_num // u_num) * u_num + 1
            freq = np.logspace(min(np.log10(freq)), max(np.log10(freq)), num)
        

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
        # dx=0,
        **kwargs,
    ):
        """Run the Lin-KK fit based on the selected model."""
        # Filter the scatter_data based on f_min and f_max
        df = df[(df["freq"] >= f_min) & (df["freq"] <= f_max)]
        f = df["freq"].to_numpy()
        Z = ComplexSystem(df[["real", "imag"]]).Z
        # Direct Access to ComplexSystemDx
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
    
    def update_system(self, key, data, form=None, thickness=None, area=None, **kwargs):
        """Update the system based on the selected key."""
        if form is None and isinstance(data, pd.DataFrame) and "form" in data.attrs:
            form = data.attrs["form"]
        
        data = ComplexSystem(data, kwargs.get("frequency", None), thickness, area, form)
        if isinstance(key, str):
            if key in self.raw:
                self.raw[key].update(data)
            else:
                self.raw[key] = data
            self.raw_archive[key] = self.raw.copy()[key]
        return

    def base_df(self, key, form, sort_val, thickness=None, area=None, **kwargs):
        """Get the base DataFrame for the selected key."""    
        if key in self.raw:
            # if dx is not None:
            #     self.raw[key].dx = dx
            if thickness is not None:
                self.raw[key].thickness = thickness
            if area is not None:
                self.raw[key].area = area

            return self.raw[key].base_df(form, sort_val)
        return
    
    def custom_df(self, key, *args, thickness=None, area=None, **kwargs):
        """Get the base DataFrame for the selected key.""" 
        if key in self.raw:
            # if dx is not None:
            #     self.raw[key].dx = dx
            if thickness is not None:
                self.raw[key].thickness = thickness
            if area is not None:
                self.raw[key].area = area
            return self.raw[key].get_df(*args)
        return


class WorkerFunctions:
    """Mix-in class for GUI classes to handle worker functions."""

    worker: Optional[object] = None
    thread: Optional[QThread] = None
    progress_dialog: Optional[object] = None
    kill_operation: bool = False

    def create_progress_dialog(
        self,
        parent,
        title="Progress",
        label_text="Processing...",
        cancel=None,
        minimum=0,
        maximum=100,
        cancel_func=None,
    ):
        """Create and return a QProgressDialog."""
        self.progress_dialog = QProgressDialog(
            label_text, cancel, minimum, maximum, parent
        )
        self.progress_dialog.setWindowTitle(title)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        if cancel:
            cancel_func = (
                self.cancel_operation if cancel_func is None else cancel_func
            )
            self.progress_dialog.canceled.connect(cancel_func)
        self.progress_dialog.show()

    def run_in_thread(
        self,
        finished_slot=None,
        error_slot=None,
        progress_slot=None,
        progress_dialog=None,
    ):
        """Helper function to run a worker in a separate thread with optional progress dialog."""
        if finished_slot is None:
            finished_slot = self.thread_finished
        if error_slot is None:
            error_slot = self.on_worker_error

        self.thread = QThread()
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(finished_slot)
        self.worker.error.connect(error_slot)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Connect progress signal if provided
        if progress_slot:
            self.worker.progress.connect(progress_slot)

        self.thread.start()
        return progress_dialog

    def update_progress(self, value):
        """Update the progress bar."""
        self.progress_dialog.setValue(value)

    def on_worker_error(self, error_message):
        """Handle errors from worker functions."""
        self.progress_dialog.close()
        self.progress_dialog.deleteLater()
        self.kill_operation = False
        QMessageBox.critical(
            self, "Error", f"Operation failed: {error_message}"
        )

    def cancel_operation(self):
        """Cancel the bootstrap fit."""
        self.kill_operation = True  # Set cancellation flag

    def thread_finished(self, *_, **__):
        """Handle the completion of data I/O operations."""
        self.progress_dialog.close()
        self.progress_dialog.deleteLater()
        self.kill_operation = False

class FittingWorker(QObject):
    """Worker class to handle fitting operations."""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, main, iterations=1, run_type="fit"):
        """Initialize the worker with the main window and fitting parameters."""
        super().__init__()
        self.main = main
        self.iterations = iterations
        self.run_type = run_type
        self.cancelled = False

        # Data parsing moved back to init
        main.print_window.write("Preparing to fit...\n")

        # Retrieve f_min and f_max from options
        f_min = main.options["fit"]["f_min"]
        f_max = main.options["fit"]["f_max"]

        self.dataset_var = main.data.var.currentText()
        self.filtered_data = main.data.raw[self.dataset_var].get_df("freq", "real", "imag", "e_r") # As impedance dataframe

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

        max_exp = np.floor(np.log10(np.max(np.finfo(np.float64).max))) / 4

        bounds_arr[bounds_arr == np.inf] = 10**max_exp  # Replace inf with a large positive number
        bounds_arr[bounds_arr == -np.inf] = -10**max_exp  # Replace -inf with a large negative number
        bounds_arr[bounds_arr == 0] = 10**-max_exp  # Replace 0 with a small positive number

        self.bounds = (bounds_arr[:, 0], bounds_arr[:, 1])

        bound_checks = np.array(main.bounds.checks)[~checked_params]
        self.bound_bools = [bound_checks[:, 0], bound_checks[:, 1]]

        self.bound_range = np.array((main.quick_bound_vals["low"], main.quick_bound_vals["high"]))

        self.initial_guess = []
        self.param_names = []
        self.constants = {}

        for param, check in zip(main.parameters, checked_params):
            if check:
                self.constants[param.name] = param.values[0]
            else:
                self.initial_guess.append(param.values[0])
                self.param_names.append(param.name)

        modes = main.options["fit"]["modes"] if isinstance(main.options["fit"]["modes"], (list, tuple)) else [main.options["fit"]["modes"]]
        if isinstance(main.options["fit"]["type"], (list, tuple)):
            self.columns = [
                mode if "." in mode else fit_type + "." + mode
                for mode in modes
                for fit_type in main.options["fit"]["type"]
            ]
        else:
            self.columns = [
                mode if "." in mode else main.options["fit"]["type"] + "." + mode
                for mode in modes
            ]

        self.model = main.data.model
        self.options = main.options
        self.loss_func = None
        if "sq" in self.options["fit"]["function"]:
            self.loss_func = Statistics().as_array(
                main.data.error_methods[main.error_var.currentText()]
            )
        else:
            self.loss_func = Statistics()[
                main.data.error_methods[main.error_var.currentText()]
            ]

    def run(self):
        """Run the fitting operation."""
        try:
            if "boot" in self.run_type:
                self.perform_bootstrap_fit()
            elif "iter" in self.run_type:
                self.perform_iterative_fit()
            else:
                # convert data to system
                fit_results = self.perform_fit()
                self.finished.emit(fit_results)
        except WorkerError as exc:
            self.error.emit(str(exc))
            self.finished.emit(None)

    def perform_fit(self, data=None, weights=None):
        """Perform a single round of fitting."""
        try:
            if data is None:
                data = ComplexSystem(
                        self.filtered_data[["freq", "real", "imag"]],
                        thickness=self.options["simulation"]["thickness (cm)"],
                        area=self.options["simulation"]["area (cm^2)"],
                        # dx=self.options["simulation"]["dx"],
                    )

            if self.options["fit"]["weight_by_modulus"] and weights is None:
                weights = np.hstack([np.abs(data[self.options["fit"]["type"]])] * len(self.columns))

            # get x and y data here
            x_data = data["freq"]
            y_data = np.hstack([data[c] for c in self.columns])

            inputs = dict(
                x_data=x_data,
                y_data=y_data,
                initial_guess=self.initial_guess,
                model_func=self.model_func_sys_wrap(
                    wrapCircuit(self.model, self.constants),
                    thickness=self.options["simulation"]["thickness (cm)"],
                    area=self.options["simulation"]["area (cm^2)"],
                    # dx=self.options["simulation"]["dx"],
                    y_cols=self.columns,
                ),
                bounds=self.bounds,
                weights=weights,
                scale=self.options["fit"]["scale"],
                loss_func=self.loss_func,
                kill_operation=lambda: self.main.kill_operation,  # Pass kill_operation
            )
        except CommonExceptions as exc:
            raise WorkerError("Error occurred while preparing the data for fitting.") from exc

        try:
            if "circuit" in self.options["fit"]["function"]:
                inputs.pop("loss_func")
                fit_results, std_results = FittingMethods().circuit_fit(
                    **{**inputs, **self.options["curve_fit"]},
                )
            elif "diff" in self.options["fit"]["function"]:
                fit_results, std_results = FittingMethods().de_fit(
                    **{**inputs, **self.options["diff_evolution"]},
                )
            elif "basin" in self.options["fit"]["function"]:
                fit_results, std_results = FittingMethods().basin_fit(
                    **{**inputs, **self.options["basinhopping"]},
                )
            else:
                fit_results, std_results = FittingMethods().ls_fit(
                    **{**inputs, **self.options["least_sq"]},
                )
        except CommonExceptions as exc:
            raise WorkerError("Error occurred while fitting the data.") from exc
        
        try:
            if fit_results is None:
                return None
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
            fit_df.attrs["dx"] = self.options["simulation"]["dx"]

            return fit_df
        except CommonExceptions as exc:
            raise WorkerError("Error occurred while parsing the fit results.") from exc

    def perform_iterative_fit(self):
        """Perform an iterative fit."""
        try:
            data = ComplexSystem(
                    self.filtered_data[["freq", "real", "imag"]],
                    thickness=self.options["simulation"]["thickness (cm)"],
                    area=self.options["simulation"]["area (cm^2)"],
                    # dx=self.options["simulation"]["dx"],
                )
            weights = None
            if self.options["fit"]["weight_by_modulus"]:
                weights = np.hstack([np.abs(data[self.options["fit"]["type"]])] * len(self.columns))
        except CommonExceptions as exc:
            raise WorkerError("Error occurred while preparing the data for iterative fitting.") from exc    
        
        fit_results = None
        for i in range(self.iterations):
            new_results = self.perform_fit(data, weights)
            try:
                if new_results is None:
                    break
                else:
                    fit_results = new_results

                # Update initial guess for the next iteration
                self.initial_guess = fit_results.loc["mean"].tolist()
                
                # Update bounds based on the new initial guess
                new_bounds = np.array(self.initial_guess)[:, None] * self.bound_range
                
                # Ensure locked bounds are not changed
                new_bounds[:, 0] = np.where(self.bound_bools[0], self.bounds[0], new_bounds[:, 0])
                new_bounds[:, 1] = np.where(self.bound_bools[1], self.bounds[1], new_bounds[:, 1])
                
                self.bounds = (new_bounds[:, 0], new_bounds[:, 1])
                
                self.progress.emit(int((i + 1) / self.iterations * 100))
                if self.main.kill_operation:  # Check if cancelled
                    break
            except CommonExceptions as exc:
                raise WorkerError(f"Error occurred during iterative fit on level {i}") from exc

        self.finished.emit(fit_results)

    def perform_bootstrap_fit(self):
        """Perform a bootstrap fit."""
        bootstrap_results = []
        for i in range(self.iterations):
            try:
                resampled_data = self.filtered_data.copy().sample(
                    frac=1, replace=True
                )
                data_system = ComplexSystem(
                    resampled_data[["freq", "real", "imag"]],
                    thickness=self.options["simulation"]["thickness (cm)"],
                    area=self.options["simulation"]["area (cm^2)"],
                    # dx=self.options["simulation"]["dx"],
                )
            except CommonExceptions as exc:
                raise WorkerError("Error occurred while resampling the data.") from exc
                # convert resampled_data to system
            fit_results = self.perform_fit(data_system)
            try:
                if fit_results is None:
                    break
                # Collect results
                bootstrap_results.append(fit_results.loc["mean"])
                self.progress.emit(int((i + 1) / self.iterations * 100))
                if self.main.kill_operation:  # Check if cancelled
                    break
            except CommonExceptions as exc:
                raise WorkerError(f"Error occurred during bootstrap fit on level {i}") from exc
        try:
            bootstrap_df = pd.DataFrame(
                bootstrap_results, columns=self.param_names
            )
            bootstrap_df.attrs["model"] = self.model
            bootstrap_df.attrs["dataset_var"] = self.dataset_var
            bootstrap_df.attrs["area"] = self.options["simulation"]["area (cm^2)"]
            bootstrap_df.attrs["thickness"] = self.options["simulation"][
                "thickness (cm)"
            ]
            bootstrap_df.attrs["dx"] = self.options["simulation"]["dx"]

            self.finished.emit(bootstrap_df)
        except CommonExceptions as exc:
            raise WorkerError("Error occurred while parsing the bootstrap results") from exc
    
    @staticmethod
    def model_func_sys_wrap(eq_func, thickness, area, y_cols):
        """Used primarily for circuitfit"""

        def wrapped(x_data, *params):
            """Wrap the circuit function."""
            sim_data = eq_func(x_data, *params)
            sim_data = ComplexSystem(
                np.array(np.hsplit(sim_data, 2)).T,
                x_data,
                thickness=thickness,
                area=area,
                # dx=dx,
            )
            return np.hstack([sim_data[c] for c in y_cols])
            # return sim_data

        return wrapped
    
    @staticmethod
    def data_parse(data, y_cols):
        """Convert data for residual calculation."""
        y_cols = y_cols if isinstance(y_cols, (list, tuple)) else [y_cols]
        return np.hstack([data[c] for c in y_cols])



class LoadDataWorker(QObject):
    """Worker class to handle data loading operations."""
    finished = pyqtSignal(dict, object)  # Signal to emit the results
    error = pyqtSignal(str)  # Signal to emit errors

    def __init__(self, file_path, options):
        """Initialize the worker with the file path and options."""
        super().__init__()
        self.file_path = file_path
        self.options = options

    def run(self):
        """Run the data loading operation."""
        try:
            try:
                data_in = load_file(self.file_path)[0]
                valid_sheets = {}
                df_in = None
            except CommonExceptions as e:
                raise WorkerError("Error occurred while loading the file.") from e
            
            if isinstance(data_in, dict):
                for sheet_name, df in data_in.items():
                    try:
                        if sheet_name == "fit results":
                            df_in = df.copy()
                        elif (
                            isinstance(df.columns, pd.MultiIndex)
                            and any(
                                "realz" in level for level in df.columns.levels
                            )
                        ) or "realz" in df.columns:
                            df = convert_mfia_df_for_fit(df)
                        else:
                            valid_keys = get_valid_keys(df.columns, ["freq", "real", "imag"])
                            if valid_keys:
                                df = df[valid_keys]
                                df.columns = ["freq", "real", "imag"]
                        
                        if all(
                            col in df.columns for col in ["freq", "real", "imag"]
                        ):
                            # breakpoint()
                            valid_sheets[sheet_name] = ComplexSystem(
                                df[["freq", "real", "imag"]].sort_values("freq", ignore_index=True),
                                thickness=self.options["simulation"][
                                    "thickness (cm)"
                                ],
                                area=self.options["simulation"]["area (cm^2)"],
                                # dx=self.options["simulation"]["dx"],
                            )
                    except CommonExceptions as e:
                        raise WorkerError(f"Error occurred while parsing the data for {sheet_name}") from e

                if df_in is not None:
                    try:
                        name_translation = dict(
                            zip(df_in["Name"].str.lower(), df_in["Dataset"])
                        )
                        valid_sheets = {
                            name_translation.get(k.lower(), k): v
                            for k, v in valid_sheets.items()
                        }
                    except CommonExceptions as e:
                        raise WorkerError("Error occurred while translating the names.") from e
            else:
                try:
                    if all(
                        col in data_in.columns for col in ["freq", "real", "imag"]
                    ):
                        valid_sheets[self.file_path.stem] = ComplexSystem(
                            data_in[["freq", "real", "imag"]],
                            thickness=self.options["simulation"]["thickness (cm)"],
                            area=self.options["simulation"]["area (cm^2)"],
                            # dx=self.options["simulation"]["dx"],
                        )
                except CommonExceptions as e:
                    raise WorkerError("Error occurred while parsing the data.") from e

            self.finished.emit(valid_sheets, df_in)
        except WorkerError as e:
            self.error.emit(str(e))


class SaveResultsWorker(QObject):
    """Worker class to handle saving results."""
    finished = pyqtSignal()  # Signal to emit when done
    error = pyqtSignal(str)  # Signal to emit errors

    def __init__(
        self, file_path, data, options, pinned, parameters, dataset_var
    ):
        """Initialize the worker with the file path and data."""
        super().__init__()
        self.file_path = file_path
        self.data = data
        self.options = options
        self.pinned = pinned
        self.parameters = parameters
        self.dataset_var = dataset_var

    def run(self):
        """Run the saving operation."""
        try:
            res = {}
            if not self.pinned.df.empty:
                # Check if the pinned data is available, will result in saving the fits
                try:
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
                except KeyError as e:
                    raise WorkerError("Error occurred while parsing the pinned data into a dataframe.") from e
                
                for _, row in res_df.iterrows():
                    try:
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
                            # get the data from the raw data
                            local_data = self.data.raw[row["Dataset"]].get_df("freq", "real", "imag", "e_r")
                        else:
                            # use simulation settings to generate the data
                            local_data = self.data.generate(
                                params_values, **self.options["simulation"]
                            ).get_df("freq", "real", "imag", "e_r")

                        generated_data = self.data.generate(
                            params_values=params_values,
                            model=row["Model"],
                            freq=local_data["freq"].to_numpy(),
                            **{**self.options["simulation"], **{"interp": False}},
                            # **self.options["simulation"],
                        ).get_df("freq", "real", "imag", "e_r")

                        for col in generated_data.columns:
                            local_data[f"pr_{col}"] = generated_data[col]

                        res[row["Name"]] = local_data
                    except CommonExceptions as e:
                        raise WorkerError(f"Error occurred while parsing the data for {row['Name']}") from e
            else:
                try:
                    params_values = [entry.values[0] for entry in self.parameters]
                    params_names = [entry.name for entry in self.parameters]
                    if self.data.raw != {}:
                        res[self.dataset_var.currentText()] = self.data.raw[
                            self.dataset_var.currentText()
                        ].get_df("freq", "real", "imag", "e_r")
                        generated_data = self.data.generate(
                            params_values,
                            freq=res[self.dataset_var.currentText()][
                                "freq"
                            ].to_numpy(),
                            **{**self.options["simulation"], **{"interp": False}},
                            # **self.options["simulation"],
                        ).get_df("freq", "real", "imag", "e_r")
                        for col in generated_data.columns:
                            res[self.dataset_var.currentText()][f"pr_{col}"] = (
                                generated_data[col]
                            )
                    else:
                        generated_data = self.data.generate(
                            params_values, **self.options["simulation"]
                        ).get_df("freq", "real", "imag", "e_r")
                        res["fit profile"] = generated_data

                    res["fit results"] = pd.DataFrame(
                        [params_values], columns=params_names
                    )
                except CommonExceptions as e:
                    raise WorkerError("Error occurred while parsing the data.") from e
            try:
                for key, value in self.data.raw.items():
                    if key not in res.keys():
                        if (
                            "Dataset" not in res["fit results"].columns or key
                            not in res["fit results"]["Dataset"].to_numpy().tolist()
                        ):
                            res[key] = value.get_df("freq", "real", "imag", "e_r")
            except CommonExceptions as e:
                raise WorkerError("Error occurred while parsing the raw data.") from e
            
            try:
                save(
                    res,
                    Path(self.file_path).parent,
                    name=Path(self.file_path).stem,
                    file_type=Path(self.file_path).suffix,
                )
                self.finished.emit()
            except (TypeError, ValueError, IndexError, KeyError, AttributeError, PermissionError) as e:
                raise WorkerError("Error occurred while saving the results.") from e
        except WorkerError as e:
            self.error.emit(str(e))


class SaveFiguresWorker(QObject):
    """Worker class to handle saving figures."""
    finished = pyqtSignal()  # Signal to emit when done
    error = pyqtSignal(str)  # Signal to emit errors

    def __init__(self, file_path, fig1, fig2):
        """Initialize the worker with the file path and figures."""
        super().__init__()
        self.file_path = file_path
        self.fig1 = fig1
        self.fig2 = fig2

    def run(self):
        """Run the saving operation."""
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
        except (TypeError, ValueError, IndexError, KeyError, AttributeError, PermissionError) as e:
            self.error.emit(str(e))


    # def __init__(self):
    #     self.model = "p(R1,C1)-p(R2,C2)-p(R3,C3)"  # "R0-p(R1,CPE1,R2-CPE2)"  # "R0-p(R1,C1)"
    #     self.raw = {}
    #     self.raw_archive = {}
    #     self.linkk = None

    #     self.defaults = {
    #         "var_val" : {
    #             "Z": "impedance",
    #             "Y": "admittance",
    #             "M": "modulus",
    #             "C": "capacitance",
    #             "ε": "permittivity",
    #             "εᵣ": "relative_permittivity",
    #             "σ": "conductivity",
    #             "ρ": "resistivity",
    #             "Real": "real",
    #             "Imag": "imag",
    #             "+Imag": "pos_imag",
    #             "Mag": "mag",
    #             "Θ": "phase",
    #             "tan(δ)": "tan",
    #         },
    #         "var_scale" : {
    #             "Real": "lin",
    #             "Imag": "lin",
    #             "+Imag": "log",
    #             "Mag": "log",
    #             "Θ": "deg",
    #             "tan(δ)": "lin",
    #         },
    #         "var_units" : {
    #             "Z": r"[$\Omega$]",
    #             "Y": "[S]",
    #             "M": "[cm/F]",
    #             "C": "[F]",
    #             "ε": "[F/cm]",
    #             "εᵣ": "[1]",
    #             "σ": "[S/cm]",
    #             "ρ": r"[$\Omega$ cm]",
    #         },
    #         "option_inits" : {
    #             "simulation": {
    #                 "limit_error": True,
    #                 "sim_param_freq": True,
    #                 "interp": True,
    #                 "freq_start": -4.5,
    #                 "freq_stop": 7,
    #                 "freq_num": 200,
    #                 "area (cm^2)": 5 * 5,  # cm
    #                 "thickness (cm)": 450e-4,
    #                 "interval": 0.1,
    #             },
    #             "bands": {
    #                 "band_color": "gray",
    #                 "band_alpha": 0.2,
    #                 "band_freq_num": 250,
    #                 "band_mult": 50,
    #                 "percentile": 5,
    #                 "std_devs": 0.2,  # if string, and result use that
    #                 "conf_band_upper": "97.5%",  # if min or max use that
    #                 "conf_band_lower": "2.5%",  # if min or max use that
    #             },
    #             "fit": {
    #                 "function": "least_squares",
    #                 "type": "impedance",
    #                 "modes": ["real", "imag", "mag"],
    #                 "f_max": 1e7,
    #                 "f_min": 1e-6,
    #                 "scale": "linear",
    #                 "weight_by_modulus": True,
    #                 "bootstrap_percent": 95,
    #             },
    #             "curve_fit": {
    #                 "absolute_sigma": False,
    #                 "check_finite": None,
    #                 "method": None,
    #                 "jac": "3-point",
    #                 "x_scale": "jac",
    #                 "ftol": 1e-14,
    #                 "xtol": 1e-6,
    #                 "gtol": 1e-8,
    #                 "loss": "cauchy",
    #                 "diff_step": None,
    #                 "tr_solver": None,
    #                 "tr_options": {},
    #                 "jac_sparsity": None,
    #                 "verbose": 0,
    #                 "maxfev": 1e6,
    #             },
    #             "least_sq": {
    #                 "method": "trf",
    #                 "jac": "3-point",
    #                 "x_scale": "calc",
    #                 "ftol": 1e-14,
    #                 "xtol": 1e-15,
    #                 "gtol": 1e-8,
    #                 "loss": "cauchy",
    #                 "diff_step": None,
    #                 "tr_solver": None,
    #                 "tr_options": {},
    #                 "jac_sparsity": None,
    #                 "verbose": 0,
    #                 "max_nfev": 1e3,
    #             },
    #             "diff_evolution": {
    #                 "strategy": "best1bin",
    #                 "maxiter": 1000,
    #                 "popsize": 15,
    #                 "tol": 0.01,
    #                 # "mutation": (0.5, 1),
    #                 "recombination": 0.7,
    #                 "seed": None,
    #                 "disp": False,
    #                 "polish": True,
    #                 "init": "latinhypercube",
    #                 "atol": 0,
    #                 "updating": "immediate",
    #                 "workers": 1,

    #             },
    #             "basinhopping" :{
    #                 "niter": 100,
    #                 "T": 1.0,
    #                 "stepsize": 0.5,
    #                 "minimizer_kwargs": {
    #                     "method": "L-BFGS-B",
    #                     "jac": "3-point",
    #                 },
    #                 "interval": 50,
    #                 "disp": False,
    #                 "niter_success": None,
    #                 "seed": None,
    #                 "target_accept_rate": 0.5,
    #                 "stepwise_factor": 0.9,
    #             },
    #             "linkk": {
    #                 "f_max": 1e6,
    #                 "f_min": 1e-6,
    #                 "c": 0.5,
    #                 "max_M": 200,
    #                 "add_cap": False,
    #             },
    #             "element": {
    #                 "R": 1e9,
    #                 "C": 1e-10,
    #                 "CPE": (1e-10, 1),
    #                 "L": 100,
    #             },
    #             "element_range": {
    #                 "R": [0, np.inf],
    #                 "C": [0, 1],
    #                 "CPE": [[0, 1], [-1, 1]],
    #                 "L": [0, np.inf],
    #             },
    #         }
    #     }

    #     self.error_methods = {
    #         key.replace("_", " ").title(): key
    #         for key in Statistics().single_method_list.copy()
    #     }
    
    # @property
    # def var_val(self):
    #     return self.defaults["var_val"].copy()
    
    # @property
    # def var_scale(self):
    #     return self.defaults["var_scale"].copy()
    
    # @property
    # def var_units(self):
    #     return self.defaults["var_units"].copy()
    
    # @property
    # def option_inits(self):
    #     return {k: v.copy() for k, v in self.defaults["option_inits"].copy().items()}
    

    # def parse_default(
    #     self,
    #     name,
    #     defaults,
    #     override=None,
    # ):
    #     """Parse the default value for the parameter."""
    #     if override is not None:
    #         if name in override.keys():
    #             return override[name]

    #     if self.model.lower() == "linkk":
    #         return 1

    #     if name in defaults.keys():
    #         return defaults[name]
    #     split_name = re.findall(r"(^[a-zA-Z]+)_?([0-9_]+)", name)[0]
    #     if split_name[0] not in defaults.keys():
    #         return 1
    #     if "_" in split_name[1]:
    #         index = int(
    #             eval(split_name[1].split("_")[-1], {}, {"inf": np.inf})
    #         )
    #         return defaults[split_name[0]][index]
    #     return defaults[split_name[0]]

    
# class DataHandler(JSONSettings):
#     """Class to store data for plotting graphs."""

#     def __init__(self):
#         super().__init__()  # Initialize JSONSettings
#         self.model = "p(R1,C1)"
#         self.raw = {}
#         self.raw_archive = {}
#         self.linkk = None
#         self.defaults = {}
#         self.var_val = {}
#         self.var_scale = {}
#         self.var_units = {}
#         self.load_dir = None
#         self.save_dir = None
#         self.export_dir = None
#         self.plot_types = []
#         self.option_inits = {}

#         self.load_settings()

#         self.error_methods = {
#             key.replace("_", " ").title(): key
#             for key in Statistics().single_method_list.copy()
#         }

#     def save_settings(self, **kwargs):
#         """Save the local modified values to JSON file."""
#         if not kwargs:
#             return
#         #     kwargs = self.option_inits
#         for key, value in kwargs.items():
#             if hasattr(self, key):
#                 setattr(self, key, value)
#         super().save_settings(**kwargs)

#     def load_settings(self, **kwargs):
#         """Load settings from JSON files and set attributes."""
#         settings = super().load_settings(**kwargs)  # Use JSONSettings method

#         # Set attributes directly from the loaded settings
#         for key, value in settings.items():
#             setattr(self, key, value)

#     def restore_defaults(self, **kwargs):
#         """Restore the default settings."""
#         settings = super().restore_defaults(**kwargs)  # Use JSONSettings method

#         # Set attributes directly from the restored settings
#         for key, value in settings.items():
#             setattr(self, key, value)


#     def parse_default(self, name, defaults, override=None):
#         """Parse the default value for the parameter."""
#         if override is not None:
#             if name in override.keys():
#                 return override[name]

#         if self.model.lower() == "linkk":
#             return 1

#         if name in defaults.keys():
#             return defaults[name]
#         split_name = re.findall(r"(^[a-zA-Z]+)_?([0-9_]+)", name)[0]
#         if split_name[0] not in defaults.keys():
#             return 1
#         if "_" in split_name[1]:
#             index = int(
#                 eval(split_name[1].split("_")[-1], {}, {"inf": np.inf})
#             )
#             return defaults[split_name[0]][index]
#         return defaults[split_name[0]]

#     def parse_label(self, main, mode):
#         """Parse the label based on the selected option."""
#         units = self.var_units[main]
#         if mode.lower() == "real":
#             return f"{main}' {units}"
#         elif mode.lower() == "imag":
#             return f"{main}'' {units}"
#         elif mode.lower() == "+imag":
#             return f"{main}'' {units}"
#         elif mode.lower() == "mag":
#             return f"|{main}| {units}"
#         elif mode == "phase" or self.var_val[mode] == "phase":
#             return "θ [deg]"
#         elif mode == "tan" or self.var_val[mode] == "tan":
#             return "tan(δ) [1]"

#     def parse_parameters(self, model=None):
#         """Get the parameters of the model."""
#         if model is None:
#             model = self.model
#         if not isinstance(model, str):
#             QMessageBox.warning(None, "Warning", f"Model must be a string. Current value is {model} of type {type(model)}")
#             model = self.model
#         if model.lower() == "linkk":
#             return ["M", "mu"]
#         # try:
#         params = extract_circuit_elements(model)
#         if len(params) != calculateCircuitLength(model):
#             all_params = []
#             for param in params:
#                 length = calculateCircuitLength(param)
#                 if length >= 2:
#                     all_params.append(f"{param}_0")
#                     for i in range(1, length):
#                         all_params.append(f"{param}_{i}")
#                 else:
#                     all_params.append(param)
#             params = all_params
#         return params

#     def generate(
#         self,
#         params_values,
#         model=None,
#         freq=None,
#         freq_start=-4.5,
#         freq_stop=7,
#         freq_num=200,
#         area=25,
#         thickness=450e-4,
#         dx=0,
#         **kwargs,
#     ):
#         """Create the fit data based on the current parameter values."""
#         if model is None:
#             model = self.model

#         if model.lower() == "linkk":
#             return self.linkk

#         circuit_func = wrapCircuit(model, {})

#         if freq is None:
#             freq = np.logspace(freq_start, freq_stop, freq_num)
#         elif kwargs.get("interp") and not kwargs.get("sim_param_freq", True) and freq_num > 2 * (u_num := len(np.unique(freq)) - 1):
#             num = (freq_num // u_num) * u_num + 1
#             freq = np.logspace(min(np.log10(freq)), max(np.log10(freq)), num)
        

#         if not params_values:
#             return
#         try:
#             Z = np.array(np.hsplit(circuit_func(freq, *params_values), 2)).T
#         except (IndexError, AssertionError) as exc:
#             raise IndexError("List index out of range") from exc

#         return ComplexSystemDx(
#             Z[:, 0] + 1j * Z[:, 1],
#             frequency=freq,
#             thickness=thickness,
#             area=area,
#             dx=dx,
#         )

#     def generate_linkk(
#         self,
#         df,
#         c=0.5,
#         max_M=200,
#         add_cap=False,
#         f_min=-4.5,
#         f_max=7,
#         area=25,
#         thickness=450e-4,
#         dx=0,
#         **kwargs,
#     ):
#         """Run the Lin-KK fit based on the selected model."""
#         # Filter the scatter_data based on f_min and f_max
#         df = df[(df["freq"] >= f_min) & (df["freq"] <= f_max)]
#         f = df["freq"].to_numpy()
#         Z = ComplexSystemDx(df[["real", "imag"]], dx=dx).Z
#         # Direct Access to ComplexSystemDx
#         M, mu, Z_linKK, _, _ = linKK(
#             f,
#             Z,
#             c=c,
#             max_M=max_M,
#             fit_type="complex",
#             add_cap=add_cap,
#         )

#         # Create a DataFrame from Z_linKK
#         self.linkk = ComplexSystemDx(
#             Z_linKK,
#             frequency=f,
#             thickness=thickness,
#             area=area,
#             dx=dx,
#         )

#         return (
#             M,
#             mu,
#         )
    
#     def update_system(self, key, data, form=None, thickness=None, area=None, dx=None,**kwargs):
#         """Update the system based on the selected key."""
#         if form is None and isinstance(data, pd.DataFrame) and "form" in data.attrs:
#             form = data.attrs["form"]
        
#         data = ComplexSystemDx(data, kwargs.get("frequency", None), thickness, area, form, dx=dx)
#         if isinstance(key, str):
#             if key in self.raw:
#                 self.raw[key].update(data)
#             else:
#                 self.raw[key] = data
#             self.raw_archive[key] = self.raw.copy()[key]
#         return

#     def base_df(self, key, form, sort_val, thickness=None, area=None, dx=None, **kwargs):
#         """Get the base DataFrame for the selected key."""    
#         if key in self.raw:
#             if dx is not None:
#                 self.raw[key].dx = dx
#             if thickness is not None:
#                 self.raw[key].thickness = thickness
#             if area is not None:
#                 self.raw[key].area = area

#             return self.raw[key].base_df(form, sort_val)
#         return
    
#     def custom_df(self, key, *args, thickness=None, area=None, dx=None, **kwargs):
#         """Get the base DataFrame for the selected key.""" 
#         if key in self.raw:
#             if dx is not None:
#                 self.raw[key].dx = dx
#             if thickness is not None:
#                 self.raw[key].thickness = thickness
#             if area is not None:
#                 self.raw[key].area = area
#             return self.raw[key].get_df(*args)
#         return
