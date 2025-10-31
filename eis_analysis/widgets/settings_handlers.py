# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""


import re
from pathlib import Path

import numpy as np
import psutil

from ..system_utilities.json_io import JSONSettings
from ..data_treatment.data_analysis import Statistics


def manage_settings_files(base_name="settings", base_path: str | Path = __file__) -> JSONSettings:
    """Manage settings files for the application."""
    count = 0
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        if "python" in proc.info["name"] and any("z_fit" in info for info in proc.info["cmdline"]):
            count += 1
        # return count
    if count == 0:
        count = 1
    # instance_count = count_instances()
    settings_file = Path(f"{base_name}_{count}.json")
    copy_from_file = Path(f"{base_name}_{count - 1}.json") if count > 1 else None

    # Initialize JSONSettings object
    json_settings = JSONSettings(settings_file, copy_from=copy_from_file, root_dir=base_path)

    # Get the true path from JSONSettings
    settings_dir = json_settings.settings_path.parent
    # print(settings_dir)
    # Remove any unwanted files
    for file in settings_dir.glob(f"{base_name}_*.json"):
        file_instance_number = int(file.stem.split("_")[-1])
        if file_instance_number > count:
            file.unlink()

    return json_settings


class SettingsManager(JSONSettings):
    """Class to store data for plotting graphs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = "p(R1,C1)"
        self.load_dir = Path().home()
        self.save_dir = Path().home()
        self.export_dir = Path().home()
        self.error_var = "Mean Abs Err"
        self.sel_error_var = "Z"
        self.afit_inputs = {}
        self.plot_var = {}
        self.is_log = {}
        self.var_units = {}
        self.option_inits = {}
        self.defaults = {}

        # Load settings from file and set attributes
        self.load_settings()

        self.error_methods = {
            key.replace("_", " ").title(): key for key in Statistics().single_method_list.copy()
        }
        self.error_methods_abbr = {
            k1: k2
            for k1, k2 in zip(self.error_methods.keys(), Statistics().single_method_abbr.keys())
        }
        if self.error_var not in self.error_methods:
            self.error_var = list(self.error_methods.keys())[0]

        if self.sel_error_var not in ["Nyquist", "Bode", "Both", "Z", "Y", "C", "M", "User"]:
            self.sel_error_var = "Z"

    def save_settings(self, **kwargs):
        """Save the local modified values to JSON file."""
        if not kwargs:
            return {}
        #     kwargs = self.option_inits
        settings = super().save_settings(**kwargs)
        for key, value in settings.items():
            if hasattr(self, key):
                setattr(self, key, value)

        return settings

    def load_settings(self, **kwargs):
        """Load settings from JSON files and set attributes."""
        settings = super().load_settings(**kwargs)  # Use JSONSettings method

        # Set attributes directly from the loaded settings
        for key, value in settings.items():
            setattr(self, key, value)

        return settings

    def restore_defaults(self, **kwargs):
        """Restore the default settings."""
        settings = super().restore_defaults(**kwargs)  # Use JSONSettings method

        # Set attributes directly from the restored settings
        for key, value in settings.items():
            setattr(self, key, value)

        return settings

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
            index = int(eval(split_name[1].split("_")[-1], {}, {"inf": np.inf}))
            return defaults[split_name[0]][index]
        return defaults[split_name[0]]

    def parse_label(self, form):
        """Parse the label based on the selected option."""
        if not isinstance(form, str):
            try:
                form = form.currentText()
            except AttributeError:
                form = str(form)

        # Handle special cases dynamically based on keywords
        if "θ" in form:
            units = "°"
        elif "slope" in form:
            units = "1"
        elif "freq" in form.lower():
            units = "Hz"
        elif "ω" in form:
            units = "1/s"
        elif "σ_dc" in form:
            units = "S/cm"
        else:
            # Search for the variable within the form using regex
            type_pattern = rf"({'|'.join(re.escape(t) for t in self.var_units.keys())})"
            type_match = re.search(type_pattern, form)
            if type_match is None:
                return f"{form} (1)"  # Default to unitless if no match is found

            # Extract the matched variable and its units
            type_ = type_match.group(1)
            units = self.var_units.get(type_, "1")  # Default to "1" if no unit is defined

        return f"{form} ({units})"

    # def parse_form(self, form: str, type_only: bool = False) -> str:
    #     """
    #     Parse the given form string to extract the type and mode.

    #     Args:
    #         form (str): The string representing the desired format (e.g., "Z'", "Z''", "|Z|", "Z θ", "Z tan(δ)").

    #     Returns:
    #         str: The parsed key in the format "<type>.<mode>" (e.g., "Z.real", "Z.imag").
    #     """
    #     if not isinstance(form, str):
    #         try:
    #             form = form.currentText()
    #         except AttributeError:
    #             form = str(form)
    #     # Define the mapping of suffixes to modes
    #     mode_mapping = {
    #         "'": "real",
    #         "''": "imag",
    #         "||": "mag",
    #         "-''": "neg_imag",
    #         "θ": "phase",
    #         "tan(δ)": "tan",
    #     }
    #     if form in ["f", "ω", "σ_dc"]:
    #         return form

    #     valid_keys = [re.escape(t) for t in self.var_units.keys() if t not in ["f", "ω", "σ_dc"]]
    #     type_pattern = rf"\W?({'|'.join(valid_keys)})(\s.+|\W+)$"
    #     t_match = re.match(type_pattern, form)
    #     key = t_match.group(1) if t_match else ""
    #     type_ = ComplexSystem.aliases[key.lower()]
    #     if type_only:
    #         return type_
    #     mode = mode_mapping[form.replace(key, "").strip()]
    #     return f"{type_}.{mode}"

    # def get_types(self, target: str = "complex") -> dict:
    #     """
    #     Generate a list of formatted strings by applying each mode to each type.

    #     Returns:
    #         dict: A dict of formatted strings (e.g., ["Z'", "Z''", "|Z|", "-Z''", "Z θ", "Z tan(δ)", ...]).
    #     """

    #     if not isinstance(target, str):
    #         target = "complex"

    #     arr_types = {
    #         k: v
    #         for k, v in zip(
    #             ["f", "ω", "σ_dc"], ["frequency", "angular_frequency", "dc_conductivity"]
    #         )
    #     }
    #     if "freq" in target or "omega" in target or "single" in target or "arr" in target:
    #         return arr_types

    #     formatted_types = []
    #     root_types = []
    #     for type_ in self.var_units.keys():
    #         if type_ in arr_types:
    #             continue
    #         root_type = ComplexSystem.aliases[type_.lower()]
    #         formatted_types += [
    #             f"{type_}'",
    #             f"{type_}''",
    #             f"|{type_}|",
    #             f"-{type_}''",
    #             f"{type_} θ",
    #             f"{type_} tan(δ)",
    #         ]
    #         root_types += [
    #             f"{root_type}.real",
    #             f"{root_type}.imag",
    #             f"{root_type}.mag",
    #             f"{root_type}.neg_imag",
    #             f"{root_type}.phase",
    #             f"{root_type}.tan",
    #         ]
    #     comp_types = {k: v for k, v in zip(formatted_types, root_types)}
    #     if "all" in target:
    #         return {**comp_types, **arr_types}
    #     if "append" in target:
    #         return {**comp_types, **arr_types}
    #     if "insert" in target:
    #         return {**arr_types, **comp_types}
    #     return comp_types


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
#     """Class to store data for plotting graphs."""

#     def __init__(self, settings_path=None, copy_from=None, root_dir: str | Path = __file__):
#         self.root_dir = Path(root_dir)
#         if self.root_dir.is_file():
#             self.root_dir = self.root_dir.parent
#         if not self.root_dir.exists():
#             raise NotADirectoryError(f"root_dir does not exist: {self.root_dir}")

#         self._defaults_path = self.root_dir / "defaults.json"
#         if isinstance(settings_path, JSONSettings):
#             settings_path = settings_path.settings_path
#         self.settings_path = self.get_path(settings_path, "settings.json")

#         # Copy default settings to local settings if local settings file doesn't exist
#         if not self.settings_path.exists():
#             self.replicate_settings_file(copy_from)

#     def get_path(self, file_path=None, default_name=None):
#         """Return the path of the file based on the given conditions."""

#         if isinstance(file_path, (str, Path)):
#             file_path = Path(file_path)
#             if len(file_path.parts) == 1:
#                 return self.root_dir / file_path.with_suffix(".json")
#             else:
#                 return file_path.with_suffix(".json")
#         else:
#             if isinstance(default_name, (str, Path)):
#                 return self.root_dir / Path(default_name).with_suffix(".json")
#             return self.root_dir / "defaults.json"

#         # self.settings = self.load_settings()

#     def from_json(self, file_path) -> dict:
#         """Load JSON file and return the data."""
#         try:
#             with open(file_path, "r", encoding="utf-8") as file:
#                 return json.load(file, object_hook=post_decoder)
#         except (json.JSONDecodeError, FileNotFoundError) as exc:
#             return self.json_loading_error(file_path, exc)

#     def json_loading_error(self, file_path, exc) -> dict:
#         """Handle JSON decode error."""
#         if file_path == self._defaults_path:
#             if isinstance(exc, FileNotFoundError):
#                 raise FileNotFoundError(f"Default file not found: {file_path}")
#             raise ValueError(
#                 f"Error decoding JSON from default path: {Path(self._defaults_path).name}"
#             )
#         else:
#             print(
#                 f"{type(exc).__name__} when decoding JSON from {file_path}. Attempting to load defaults."
#             )
#             res = self.from_json(self._defaults_path)
#             if res:
#                 print(
#                     f"Successfully loaded defaults from {Path(self._defaults_path).name}. Setting {Path(file_path).name} to defaults."
#                 )
#                 self.to_json(res, file_path)
#                 return res
#             return {}

#     def load_settings(self, **kwargs) -> dict:
#         """Load settings from JSON files and return the settings dictionary."""
#         settings = self.from_json(self._defaults_path)
#         local_settings = self.from_json(self.settings_path)

#         update_dict(settings, local_settings)

#         if kwargs:
#             kwargs = check_dict(kwargs, settings)
#             settings = filter_dict(settings, kwargs)

#         return settings

#     def to_json(self, settings, file_path):
#         """Save settings to a JSON file."""
#         if file_path != self._defaults_path:
#             with open(file_path, "w", encoding="utf-8") as file:
#                 json.dump(pre_encoder(settings), file, indent=4)

#     def save_settings(self, **kwargs) -> dict:
#         """Save the local modified values to JSON file."""
#         if not kwargs:
#             return {}
#         #     kwargs = self.option_inits
#         settings = self.from_json(self.settings_path)

#         # Update the local settings with the current settings
#         kwargs = check_dict(kwargs, settings)
#         update_dict(settings, kwargs)

#         self.to_json(settings, self.settings_path)
#         return settings

#     def restore_defaults(self, **kwargs) -> dict:
#         """Restore the default settings."""
#         return self.replicate_settings_file(self._defaults_path, **kwargs)

#     def replicate_settings_file(self, source_path=None, **kwargs) -> dict:
#         """Restore or copy from a settings file."""
#         source_path = (
#             self.get_path(source_path) if source_path != self._defaults_path else source_path
#         )
#         if source_path is not None and not source_path.exists():
#             source_path = self._defaults_path
#         settings = self.from_json(source_path)

#         if kwargs:
#             kwargs = check_dict(kwargs, settings)
#             kwargs = filter_dict(settings, kwargs)
#             settings = self.from_json(self.settings_path)
#             update_dict(settings, kwargs)

#         self.to_json(settings, self.settings_path)
#         return settings
