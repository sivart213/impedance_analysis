import re
# import sys

# from copy import deepcopy

import numpy as np
import pandas as pd

import sip
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import (
    # QApplication,
    # QDialog,
    QVBoxLayout,
    # QHBoxLayout,
    # QLabel,
    QWidget,
    # QTabWidget,
    # QTableWidget,
    # QLineEdit,
    # QPushButton,
    # QFrame,
    QMessageBox,
    # QFormLayout,
    # QDialogButtonBox,
    # QTreeWidget,
    # QTreeWidgetItem,
    # QAbstractItemView,
    QMainWindow,
    # QTableWidgetItem,
    # QTextEdit,
    # QProgressDialog,
    # QListWidget,
    # QListWidgetItem,
    # QInputDialog,
)
# from PyQt5.QtGui import QBrush, QColor, QFontMetrics

# from ..string_ops import format_number, common_substring
# from ..string_ops import format_number, find_common_str, safe_eval

# from .gui_widgets import MultiEntryManager

# from .gui_workers import DataHandler, JSONSettings

from .gui_windows import TableFrame
# import re
from pathlib import Path
# from typing import Optional
import json
# from datetime import datetime
# import numpy as np
# import pandas as pd


# from PyQt5.QtCore import pyqtSignal, QObject, QThread, Qt

# from PyQt5.QtWidgets import (
#     # QMessageBox,
#     # QProgressDialog,
#     # QInputDialog,
# )


from impedance.models.circuits.fitting import (
    wrapCircuit,
    extract_circuit_elements,
    calculateCircuitLength,
)
from impedance.validation import linKK

from ..data_treatment import get_valid_keys, ComplexSystem#, ComplexSystemDx
from ..data_treatment.data_analysis import FittingMethods, Statistics
from ..dict_ops import update_dict, filter_dict, check_dict
# from ..equipment.mfia_ops import convert_mfia_df_for_fit
# from ..system_utilities.file_io import load_file, save

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

    
# def post_decoder(obj):
#     try:
#         if "__path__" in obj:
#             return Path(obj["__path__"]).expanduser()
#         if "__complex__" in obj:
#             return complex(*obj["__complex__"])
#         if "__invalid_float__" in obj:
#             return eval(obj["__invalid_float__"], {}, {"inf": np.inf, "nan": np.nan})
#     except (KeyError, TypeError, ValueError, SyntaxError):
#         pass
#     return obj

# def pre_encoder(data):
#     """Recursively preprocess data to convert Infinity and NaN values."""
#     if isinstance(data, Path):
#         return {"__path__": str(data)}
#     elif isinstance(data, complex):
#         return {"__complex__": [data.real, data.imag]}
#     elif isinstance(data, float) and (np.isnan(data) or np.isinf(data)):
#         return {"__invalid_float__": str(data)}
#     elif isinstance(data, dict):
#         return {k: pre_encoder(v) for k, v in data.items()}
#     elif isinstance(data, (list, tuple, set)):
#         # Handle lists, tuples, and sets
#         return type(data)([pre_encoder(i) for i in data])
#     return data


# class JSONSettings:
#     """
#     Class to store data for plotting graphs.

#     Responsibilities:
#     - Load and save settings from/to JSON files.
#     - Restore default settings if local settings are not available.
#     - Handle JSON loading errors and provide default settings.

#     Integration:
#     - Use this class to manage application settings.
#     - Extend this class to create specific settings managers like `SettingsManager`.
#     """

#     def __init__(self, defaults_path=None, settings_path=None):
#         self.defaults_path = Path(defaults_path) if defaults_path else Path(__file__).parent / "defaults.json"
#         self.settings_path = Path(settings_path) if settings_path else Path(__file__).parent / "settings.json"

#         # Check if default settings file exists
#         if not self.defaults_path.exists():
#             raise FileNotFoundError(f"Default settings file not found: {self.defaults_path}")

#         # Copy default settings to local settings if local settings file doesn't exist
#         if not self.settings_path.exists():
#             self.restore_defaults()

#         # self.settings = self.load_settings()
#     def from_json(self, file_path):
#         """Load JSON file and return the data."""
#         try:
#             with open(file_path, "r", encoding="utf-8") as file:
#                 return json.load(file, object_hook=post_decoder)
#         except (json.JSONDecodeError, FileNotFoundError) as exc:
#             return self.json_loading_error(file_path, exc)

#     def json_loading_error(self, file_path, exc):
#         """Handle JSON decode error."""
#         if file_path == self.defaults_path:
#             if isinstance(exc, FileNotFoundError):
#                 raise FileNotFoundError(f"Default file not found: {file_path}")
#             raise ValueError(f"Error decoding JSON from default path: {Path(self.defaults_path).name}")
#         else:
#             print(f"{type(exc).__name__} when decoding JSON from {file_path}. Attempting to load defaults.")
#             res = self.from_json(self.defaults_path)
#             if res:
#                 print(f"Successfully loaded defaults from {Path(self.defaults_path).name}. Setting {Path(file_path).name} to defaults.")
#                 self.to_json(res, file_path)
#                 return res
        
#     def load_settings(self, **kwargs):
#         """Load settings from JSON files and return the settings dictionary."""
#         settings = self.from_json(self.defaults_path)
#         local_settings = self.from_json(self.settings_path)

#         update_dict(settings, local_settings)

#         if kwargs:
#             kwargs = check_dict(kwargs, settings)
#             settings = filter_dict(settings, kwargs)

#         return settings

#     def to_json(self, settings, file_path):
#         """Save settings to a JSON file."""
#         if file_path != self.defaults_path:
#             with open(file_path, "w", encoding="utf-8") as file:
#                 json.dump(pre_encoder(settings), file, indent=4)

#     def save_settings(self, **kwargs):
#         """Save the local modified values to JSON file."""
#         if not kwargs:
#             return
#         #     kwargs = self.option_inits
#         settings = self.from_json(self.settings_path)

#         # Update the local settings with the current settings
#         kwargs = check_dict(kwargs, settings)
#         update_dict(settings, kwargs)

#         self.to_json(settings, self.settings_path)

#     def restore_defaults(self, **kwargs):
#         """Restore the default settings."""
#         settings = self.from_json(self.defaults_path)
        
#         if kwargs:
#             kwargs = check_dict(kwargs, settings)
#             kwargs = filter_dict(settings, kwargs)
#             settings = self.from_json(self.settings_path)
#             update_dict(settings, kwargs)

#         self.to_json(settings, self.settings_path)
#         return settings

# class SettingsManager(JSONSettings):
#     """
#     Class to manage settings for the application.

#     Responsibilities:
#     - Load settings from JSON files and set attributes.
#     - Save modified settings to JSON files.
#     - Restore default settings.

#     Integration:
#     - Instantiate this class to manage application-wide settings.
#     - Use the `load_settings`, `save_settings`, and `restore_defaults` methods to handle settings.
#     """

#     def __init__(self):
#         super().__init__()
#         self.load_settings()

#     def save_settings(self, **kwargs):
#         """Save the local modified values to JSON file."""
#         if not kwargs:
#             return
#         for key, value in kwargs.items():
#             if hasattr(self, key):
#                 setattr(self, key, value)
#         super().save_settings(**kwargs)

#     def load_settings(self, **kwargs):
#         """Load settings from JSON files and set attributes."""
#         settings = super().load_settings(**kwargs)
#         for key, value in settings.items():
#             setattr(self, key, value)

#     def restore_defaults(self, **kwargs):
#         """Restore the default settings."""
#         settings = super().restore_defaults(**kwargs)
#         for key, value in settings.items():
#             setattr(self, key, value)

class DerivativeCalculator:
    """
    Class to handle derivative calculations.

    Responsibilities:
    - Calculate the derivative of data based on a specified level.
    - Handle edge cases and ensure numerical stability.

    Integration:
    - Use this class within `DataTransformer` to calculate derivatives of transformed data.
    - Instantiate with the desired derivative level (dx) and call `calculate_derivative` with data and x_data.
    """

    def __init__(self, dx=0):
        self.dx = dx

    def calculate_derivative(self, data, x_data):
        """Calculate the derivative of the data based on the specified level."""
        level = self.dx
        if level == 0:
            return data

        epsilon = 1e-10
        x_vals = []
        for f in x_data:
            if f not in x_vals:
                x_vals.append(f)
            else:
                x_vals.append(x_vals[-1] + f * epsilon)

        data = np.log10(data)
        x_vals = np.log10(x_vals)
        for _ in range(level):
            data = np.gradient(data, x_vals, edge_order=2)

        return data
    
# class DataParser:
#     """
#     Class to handle data parsing and labeling.

#     Responsibilities:
#     - Parse default values for parameters.
#     - Generate labels based on selected options.
#     - Extract and parse model parameters.

#     Integration:
#     - Use this class to parse and label data for visualization and analysis.
#     - Instantiate with model, var_units, and var_val, and call methods like `parse_default`, `parse_label`, and `parse_parameters`.
#     """

#     def __init__(self, model, var_units, var_val):
#         self.model = model
#         self.var_units = var_units
#         self.var_val = var_val

#     def parse_default(self, name, defaults, override=None):
#         """Parse the default value for the parameter."""
#         if override is not None and name in override.keys():
#             return override[name]
#         if self.model.lower() == "linkk":
#             return 1
#         if name in defaults.keys():
#             return defaults[name]
#         split_name = re.findall(r"(^[a-zA-Z]+)_?([0-9_]+)", name)[0]
#         if split_name[0] not in defaults.keys():
#             return 1
#         if "_" in split_name[1]:
#             index = int(eval(split_name[1].split("_")[-1], {}, {"inf": np.inf}))
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

class DataGenerator:
    """
    Class to handle data generation based on the model and parameters.

    Responsibilities:
    - Generate data using the specified model and parameters.
    - Handle different models and generate corresponding data.
    - Provide methods for generating data for specific models like Lin-KK.

    Integration:
    - Use this class within `DataManager` to generate data for plotting and analysis.
    - Instantiate with the model and call `generate` or `generate_linkk` with the required parameters.
    """

    def __init__(self, model):
        self.model = model

    def generate(self, params_values, model=None, freq=None, freq_start=-4.5, freq_stop=7, freq_num=200, area=25, thickness=450e-4, dx=0, **kwargs):
        """Create the fit data based on the current parameter values."""
        if model is None:
            model = self.model
        if model.lower() == "linkk":
            return self.generate_linkk(params_values, freq, **kwargs)
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
        return pd.DataFrame({"freq": freq, "real": Z[:, 0], "imag": Z[:, 1]})

    def generate_linkk(self, df, c=0.5, max_M=200, add_cap=False, f_min=-4.5, f_max=7, area=25, thickness=450e-4, dx=0, **kwargs):
        """Run the Lin-KK fit based on the selected model."""
        df = df[(df["freq"] >= f_min) & (df["freq"] <= f_max)]
        f = df["freq"].to_numpy()
        Z = ComplexSystem(df[["real", "imag"]]).Z
        M, mu, Z_linKK, _, _ = linKK(f, Z, c=c, max_M=max_M, fit_type="complex", add_cap=add_cap)
        return pd.DataFrame({"freq": f, "real": Z_linKK.real, "imag": Z_linKK.imag})

class DataUpdater(QObject):
    """
    Class to handle updating the system and retrieving data.

    Responsibilities:
    - Manage raw data and active dataset.
    - Update data for a given key and emit signals when data changes.
    - Provide access to the active dataset.

    Integration:
    - Use this class within `DataManager` to manage and update datasets.
    - Connect to the `data_changed` signal to handle data updates.
    """

    data_changed = pyqtSignal(object)  # Signal to notify data changes

    def __init__(self):
        super().__init__()
        self.raw = {}  # Holds all the simple dataframes
        self._active_key = None  # Holds the key for the active dataset

    @property
    def active(self):
        """Get the active dataset."""
        if self._active_key is not None:
            return self.raw.get(self._active_key)
        return None

    @active.setter
    def active(self, key):
        """Set the active dataset and emit data changed signal."""
        if key in self.raw:
            self._active_key = key
            self.data_changed.emit(self.raw[key])

    def update_data(self, key, data):
        """Update the data for the given key."""
        self.raw[key] = data
        if key == self._active_key:
            self.data_changed.emit(data)


class DataTransformer(QObject):
    """
    Class to handle data transformation.

    Responsibilities:
    - Transform data using specified form, thickness, area, and derivative level.
    - Calculate derivatives of transformed data.
    - Emit signals when the system is updated.

    Integration:
    - Use this class within `DataManager` to transform data before plotting or analysis.
    - Connect to the `system_updated` signal to handle system updates.
    """

    system_updated = pyqtSignal()  # Signal to notify system updates

    def __init__(self, form=None, thickness=None, area=None, dx=0):
        super().__init__()
        self._form = form
        self._thickness = thickness
        self._area = area
        self._dx = dx
        self.derivative_calculator = DerivativeCalculator(dx)

    @property
    def form(self):
        return self._form

    @form.setter
    def form(self, value):
        self._form = value
        self.system_updated.emit()

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, value):
        self._thickness = value
        self.system_updated.emit()

    @property
    def area(self):
        return self._area

    @area.setter
    def area(self, value):
        self._area = value
        self.system_updated.emit()

    @property
    def dx(self):
        return self._dx

    @dx.setter
    def dx(self, value):
        self._dx = value
        self.derivative_calculator.dx = value
        self.system_updated.emit()

    def custom_df(self, data, *args):
        """Get the base DataFrame for the selected key."""
        complex_system = ComplexSystem(data, form=self._form, thickness=self._thickness, area=self._area)
        return complex_system.get_df(*args)

    def transform(self, data, *args):
        """Transform data using ComplexSystem and calculate derivatives."""
        # complex_system = ComplexSystem(data, form=self._form, thickness=self._thickness, area=self._area)
        complex_system = self.custom_df(data, *args)
        
        transformed_data = self.derivative_calculator.calculate_derivative(complex_system.iloc[:,1:].to_numpy(), complex_system.iloc[:,0].to_numpy())
        return pd.DataFrame({"freq": complex_system.frequency, "real": transformed_data.real, "imag": transformed_data.imag})


class DataManager(QObject):
    """
    Class to manage data loading, generation, and transformation.

    Responsibilities:
    - Load and save data.
    - Generate data using `DataGenerator`.
    - Transform data using `DataTransformer`.
    - Update the system and manage datasets using `DataUpdater`.
    - Emit signals when data changes.

    Integration:
    - Use this class as the central manager for data-related operations in the application.
    - Connect to the `data_changed` signal to handle updates in the main program.
    - Call methods like `load_data`, `save_data`, `generate_data`, `update_system`, and `get_data` to manage data.
    """

    data_changed = pyqtSignal(object)  # Signal to notify data changes

    def __init__(self, model, transformer=None):
        super().__init__()
        self.model = model
        self.data_updater = DataUpdater()
        self.data_generator = DataGenerator(model)
        self.transformer = transformer if transformer else DataTransformer()

        # Connect signals
        self.data_updater.data_changed.connect(self.on_data_changed)
        self.transformer.system_updated.connect(self.on_system_updated)

    def load_data(self, key, data):
        """Load data into the manager."""
        self.data_updater.update_data(key, data)

    def save_data(self, key, filepath):
        """Save data to a file."""
        if key in self.data_updater.raw:
            self.data_updater.raw[key].to_csv(filepath)

    def generate_data(self, params_values, model=None, freq=None, *args, **kwargs):
        """Generate data using the DataGenerator."""
        generated_data = self.data_generator.generate(params_values, model, freq, **kwargs)
        transformed_data = self.transformer.transform(generated_data, *args)
        self.data_changed.emit(transformed_data)  # Emit signal when data is generated
        return transformed_data

    def update_system(self, key, data, **kwargs):
        """Update the system using the DataUpdater."""
        self.data_updater.update_data(key, data)
        self.data_changed.emit(self.data_updater.raw)  # Emit signal when system is updated

    def get_data(self, key, *args, **kwargs):
        """Get data using the DataUpdater."""
        data = self.data_updater.raw.get(key)
        if data is not None:
            return self.transformer.transform(data, *args)
        return None

    def on_data_changed(self, data):
        """Handle data changed signal."""
        self.data_changed.emit(data)

    def on_system_updated(self):
        """Handle system updated signal."""
        if self.data_updater.active is not None:
            transformed_data = self.transformer.transform(self.data_updater.active)
            self.data_changed.emit(transformed_data)


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
#         Z = ComplexSystem(df[["real", "imag"]]).Z
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
#         self.linkk = ComplexSystem(
#             Z_linKK,
#             frequency=f,
#             thickness=thickness,
#             area=area,
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




# class DataHandlerWidgets(DataHandler):
#     """A class to handle the data widgets for the EIS analysis."""
#     def __init__(self, parent=None, **kwargs):
#         self.raw = None
#         super().__init__()
#         self._raw = {}
#         self.root = parent
#         # self.raw_list = ListFrame(list(self.raw.keys()), parent)
#         self.raw_list = TableFrame([(k, str(len(v))) for k, v in self.raw.items()], parent)
#         self.window = None
#         self.var = None

#         self.callback = kwargs.get("callback", None)

#     def load_window(self):
#         """Show the list frame window."""
#         if not self.raw:
#             QMessageBox.warning(self.root, "Warning", "No data to display.")
#             return
#         for key, system in self.raw.items():
#             self._raw[key] = system.get_df("freq", "real", "imag")
#             self._raw[key].attrs["thickness"] = system.thickness
#             self._raw[key].attrs["area"] = system.area
        
#         self.raw_archive = {**self.raw_archive.copy(), **self.raw.copy()}

#         # self.raw_list.populate_list(list(self.raw.keys()))
#         self.raw_list.populate_table([(k, str(len(v))) for k, v in self.raw.items()])

#         self.window = QMainWindow(self.root)
#         self.window.setWindowTitle("Loaded Datasets")
#         self.window.setGeometry(100, 100, 400, 300)

#         self.layout = QVBoxLayout()

#         central_widget = QWidget()
#         # self.raw_list.setParent(central_widget)
#         self.raw_list.initialize(central_widget)

#         # layout = QVBoxLayout(central_widget)
#         self.layout.addWidget(self.raw_list)
#         central_widget.setLayout(self.layout)
        
#         self.raw_list.combine_button.clicked.disconnect()
#         self.raw_list.combine_button.clicked.connect(self.combine_items)
#         self.raw_list.rename_button.clicked.disconnect()
#         self.raw_list.rename_button.clicked.connect(self.rename_item)
#         self.raw_list.save_button.clicked.disconnect()
#         self.raw_list.save_button.clicked.connect(self.save_data)

#         self.window.setCentralWidget(central_widget)

#         self.window.closeEvent = self.closeEvent

#         self.window.show()
    
#     def show(self):
#         """Show the list frame window."""
#         self.load_window()

#     def closeEvent(self, event):
#         self.raw_list.setParent(None)
#         # self.window.hide()
        
#         if self.raw_list.changed:
#             reply = QMessageBox.question(
#                 self.raw_list,
#                 "Save Changes",
#                 "Do you want to save changes before closing?",
#                 QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
#                 QMessageBox.Cancel,
#             )
#             if reply == QMessageBox.Yes:
#                 self.save_data()
#                 event.accept()
#             elif reply == QMessageBox.Cancel:
#                 event.ignore()
#             else:
#                 event.accept()
#             if not sip.isdeleted(self.window):
#                 self.window.deleteLater()
#         else:
#             event.accept()
#             if not sip.isdeleted(self.window):
#                 self.window.deleteLater()

#     def save_data(self):
#         """Save the data to the raw dictionary. """
#         self.raw_archive = {**self.raw_archive.copy(), **self.raw.copy()}
#         self.raw = {}
#         for key in self.raw_list.get_info():
#             self.update_system(key, self._raw[key], "impedance", self._raw[key].attrs["thickness"], self._raw[key].attrs["area"])
#         if self.raw_list.changed:
#             self.update_var()
#         elif self.var is not None and not self.is_highlighted(self.primary()):
#             self.var.setCurrentText(self.raw_list.special_items[0])
#         self.raw_list.changed = False
#         if self.callback is not None:
#             self.callback()
    
#     def update_data(self, data):
#         """Update the raw data dictionary."""
#         self._raw = data
#         self.save_data()

#     def update_var(self):
#         """Update the variable in the list frame window."""
#         if self.var is not None and self.raw:
#             old_var = self.raw_list.special_items[0] if self.raw_list.special_items else self.var.currentText()
#             self.var.blockSignals(True)
#             self.var.clear()
#             self.var.setCurrentText("")
#             self.var.addItems(list(self.raw.keys()))
#             if old_var not in self.raw:
#                 old_var = next(iter(self.raw))
#             self.highlight(old_var)
#             self.var.setCurrentText(old_var)
#             self.var.blockSignals(False)

#     def set_var(self, var):
#         """Set the variable in the list frame window."""
#         if self.var is not None and var in self.raw:
#             self.var.blockSignals(True)
#             self.highlight(var)
#             self.var.setCurrentText(var)
#             self.var.blockSignals(False)

#     def primary(self):
#         if self.var is not None:
#             return self.var.currentText()
#         else:
#             return ""
        
#     def combine_items(self):
#         """Combine the selected entries in the list widget."""
#         new_key, keys = self.raw_list.combine_items()
#         if new_key:
#             data = {k: v for k, v in self._raw.items() if k in keys}
#             comb = pd.concat(data.values(), sort=False, keys=data.keys())
#             comb.sort_values("freq", ignore_index=True)
#             comb.attrs["thickness"] = np.mean([v.attrs["thickness"] for v in data.values()])
#             comb.attrs["area"] = np.mean([v.attrs["area"] for v in data.values()])
#             self._raw[new_key] = comb
#             self.raw_list.changed = True
#             self.raw_list.table_info[new_key]["Shape"] = str(len(comb))
#             self.raw_list.populate_table()
#             # self.update_var()

#     def rename_item(self):
#         """Rename the selected entry in the list widget."""
#         new_key, keys = self.raw_list.rename_item()
#         if new_key:
#             for key in keys:
#                 if key in self._raw:
#                     self._raw[new_key] = self._raw.pop(key)
#                     self.raw_list.changed = True
#                     # self.update_var()
#                     break
#             else:
#                 QMessageBox.warning(self.raw_list, "Warning", "Key not found.")
        
#     def highlight(self, strings, reset=False):
#         """Highlight the selected entry in the list widget."""
#         self.raw_list.highlight(strings, reset)

#     def is_highlighted(self, string):
#         """Check if the string is highlighted."""
#         return string in self.raw_list.special_items

#     def get_checked(self):
#         """Return the strings of the checked entries."""
#         return self.raw_list.get_checked()
    
#     def get_label(self, key):
#         """Return the label for the selected key."""
#         try:
#             return self.raw_list.table_info[key]["Label"]   
#         except KeyError:
#             return f"_{key}",

#     def get_mark(self, key):
#         """Return the mark for the selected key."""
#         checks = self.raw_list.get_checked(3)
#         if checks != self.raw_list.special_items:
#             return key in checks
#         try:
#             return self.raw_list.table_info[key]["Mark"]   
#         except KeyError:
#             return True

