# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""


import json
import inspect
from pathlib import Path

import numpy as np


class WorkerError(Exception):
    """Custom exception class for handling unexpected errors."""


def update_dict(base_dict: dict, updater_dict: dict) -> None:
    """Recursively update the base dictionary with values from the update dictionary."""
    if not base_dict or not updater_dict:
        return

    for key, value in updater_dict.items():
        if isinstance(value, dict) and key in base_dict:
            update_dict(base_dict[key], value)
        else:
            base_dict[key] = value
        # return new_dict


def filter_dict(base_dict: dict, filtering_dict: dict) -> dict:
    """Recursively filter the base dictionary to keep only values present in the filter dictionary."""
    if not filtering_dict:
        return base_dict

    new_dict = {}
    for key, value in filtering_dict.items():
        if key in base_dict:
            if isinstance(value, dict) and isinstance(base_dict[key], dict):
                new_dict[key] = filter_dict(base_dict[key], value)
            else:
                new_dict[key] = base_dict[key]

    return new_dict


def check_dict(to_check_dict: dict, base_dict: dict) -> dict:
    """Recursively nest to_check_dict within base_dict if keys are not found at the top level."""
    if not to_check_dict:
        return to_check_dict

    # Check if any keys of to_check_dict are in base_dict
    keys_in_base = any(key in base_dict for key in to_check_dict)

    if keys_in_base:
        return to_check_dict
    else:
        # If no keys of to_check_dict are in base_dict, recurse through values of base_dict that are dicts
        for key, value in base_dict.items():
            if isinstance(value, dict):
                nested_dict = check_dict(to_check_dict, value)
                if nested_dict:
                    return {key: nested_dict}
        return {}


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


def get_caller_dir(f_name: str = "") -> Path:
    """Return the directory of the calling module."""
    stack = inspect.stack()
    # stack[0] = _get_caller_dir, stack[1] = __init__, stack[2] = caller of JSONSettings
    for frame_info in stack[2:]:
        module = inspect.getmodule(frame_info.frame)
        if module and hasattr(module, "__file__"):
            dir_path = Path(str(module.__file__)).parent
            if (dir_path / f_name).exists():
                return dir_path
    return Path.cwd()


class JSONSettings:
    """Class to store data for plotting graphs."""

    def __init__(self, settings_path=None, copy_from=None, root_dir: str | Path = __file__):
        self.root_dir = Path(root_dir)
        if self.root_dir.is_file():
            self.root_dir = self.root_dir.parent
        if not self.root_dir.exists():
            raise NotADirectoryError(f"root_dir does not exist: {self.root_dir}")

        self._defaults_path = self.root_dir / "defaults.json"
        if isinstance(settings_path, JSONSettings):
            settings_path = settings_path.settings_path
        self.settings_path = self.get_path(settings_path, "settings.json")

        # Copy default settings to local settings if local settings file doesn't exist
        if not self.settings_path.exists():
            self.replicate_settings_file(copy_from)

    def get_path(self, file_path=None, default_name=None):
        """Return the path of the file based on the given conditions."""

        if isinstance(file_path, (str, Path)):
            file_path = Path(file_path)
            if len(file_path.parts) == 1:
                return self.root_dir / file_path.with_suffix(".json")
            else:
                return file_path.with_suffix(".json")
        else:
            if isinstance(default_name, (str, Path)):
                return self.root_dir / Path(default_name).with_suffix(".json")
            return self.root_dir / "defaults.json"

        # self.settings = self.load_settings()

    def from_json(self, file_path) -> dict:
        """Load JSON file and return the data."""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return json.load(file, object_hook=post_decoder)
        except (json.JSONDecodeError, FileNotFoundError) as exc:
            return self.json_loading_error(file_path, exc)

    def json_loading_error(self, file_path, exc) -> dict:
        """Handle JSON decode error."""
        if file_path == self._defaults_path:
            if isinstance(exc, FileNotFoundError):
                raise FileNotFoundError(f"Default file not found: {file_path}")
            raise ValueError(
                f"Error decoding JSON from default path: {Path(self._defaults_path).name}"
            )
        else:
            print(
                f"{type(exc).__name__} when decoding JSON from {file_path}. Attempting to load defaults."
            )
            res = self.from_json(self._defaults_path)
            if res:
                print(
                    f"Successfully loaded defaults from {Path(self._defaults_path).name}. Setting {Path(file_path).name} to defaults."
                )
                self.to_json(res, file_path)
                return res
            return {}

    def load_settings(self, **kwargs) -> dict:
        """Load settings from JSON files and return the settings dictionary."""
        settings = self.from_json(self._defaults_path)
        local_settings = self.from_json(self.settings_path)

        update_dict(settings, local_settings)

        if kwargs:
            kwargs = check_dict(kwargs, settings)
            settings = filter_dict(settings, kwargs)

        return settings

    def to_json(self, settings, file_path):
        """Save settings to a JSON file."""
        if file_path != self._defaults_path:
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(pre_encoder(settings), file, indent=4)

    def save_settings(self, **kwargs) -> dict:
        """Save the local modified values to JSON file."""
        if not kwargs:
            return {}
        #     kwargs = self.option_inits
        settings = self.from_json(self.settings_path)

        # Update the local settings with the current settings
        kwargs = check_dict(kwargs, settings)
        update_dict(settings, kwargs)

        self.to_json(settings, self.settings_path)
        return settings

    def restore_defaults(self, **kwargs) -> dict:
        """Restore the default settings."""
        return self.replicate_settings_file(self._defaults_path, **kwargs)

    def replicate_settings_file(self, source_path=None, **kwargs) -> dict:
        """Restore or copy from a settings file."""
        source_path = (
            self.get_path(source_path) if source_path != self._defaults_path else source_path
        )
        if source_path is not None and not source_path.exists():
            source_path = self._defaults_path
        settings = self.from_json(source_path)

        if kwargs:
            kwargs = check_dict(kwargs, settings)
            kwargs = filter_dict(settings, kwargs)
            settings = self.from_json(self.settings_path)
            update_dict(settings, kwargs)

        self.to_json(settings, self.settings_path)
        return settings
