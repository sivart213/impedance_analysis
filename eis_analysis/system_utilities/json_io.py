# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""

import os
import json
import inspect
from typing import Self
from pathlib import Path
from itertools import islice

import numpy as np


class WorkerError(Exception):
    """Custom exception class for handling unexpected errors."""


NULL_PATH = "<:__NULL__PATH__:>" if os.name == "nt" else "/__NULL__PATH__"


def update_dict(subject: dict, source: dict, copy: bool = False) -> dict:
    """
    Recursively merge two dictionaries.

    Uses the union of keys/structure from both `subject` and `source`,
    with values taken from `source` where overlaps occur, and from
    `subject` otherwise.

    Parameters
    ----------
    subject : dict
        The dictionary to be updated. Modified in place unless `copy` is True.
    source : dict
        The dictionary providing new or overriding values.
    copy : bool, optional
        If True, a shallow copy of `subject` is created before updating.
        Default is False (modifies `subject` in place).

    Returns
    -------
    dict
        The updated subject dictionary.
    """
    if copy:
        subject = subject.copy()

    if not subject or not source:
        return subject

    for key, value in source.items():
        if isinstance(subject.get(key), dict) and isinstance(value, dict):
            # update_dict(subject[key], value, copy)
            subject[key] = update_dict(subject[key], value, copy)
        else:
            subject[key] = value
    return subject


def filter_dict(target: dict, reference: dict) -> dict:
    """
    Recursively extract a subset of `target` defined by `reference`.

    Uses the overlapping keys/structure of both dictionaries, with values
    taken from `target`. Keys not present in both are omitted.

    Parameters
    ----------
    target : dict
        The source dictionary to filter.
    reference : dict
        A dictionary whose keys specify which entries to keep.

    Returns
    -------
    dict
        A new dictionary containing only the filtered keys and values.
    """
    if not reference:
        return target

    new_dict = {}
    for key, value in reference.items():
        if key in target:
            if isinstance(value, dict) and isinstance(target[key], dict):
                new_dict[key] = filter_dict(target[key], value)
            else:
                new_dict[key] = target[key]

    return new_dict


def align_dict(target: dict, reference: dict) -> dict:
    """
    Recursively align the subset `target` with the structure of `reference`.

    Uses the keys/structure of `reference` to determine placement, with
    values taken from `target`. If keys in `target` match
    at the top level, it is returned directly; otherwise, it is nested
    under the appropriate sub-dictionary as seen in `reference`.

    Parameters
    ----------
    target : dict
        The dictionary to align with `reference`.
    reference : dict
        The reference dictionary structure.

    Returns
    -------
    dict
        Either `target` (if keys match at the top level),
        a nested version of it (if matched deeper), or `{}` if no alignment is found.
    """
    if not target:
        return target

    # Check if any keys of target are in reference
    keys_in_base = any(key in reference for key in target)

    if keys_in_base:
        return target
    else:
        # If no keys of target are in reference, recurse through values of reference that are dicts
        for key, value in reference.items():
            if isinstance(value, dict):
                nested_dict = align_dict(target, value)
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


def get_caller_dirs() -> list[Path]:
    frame = inspect.currentframe()
    files: dict[Path, None] = {}
    while frame:
        file = Path(frame.f_code.co_filename)
        if file.is_file() and file.parent.parts:
            files[file.parent] = None
            if frame.f_code.co_name == "<module>":
                mod = inspect.getmodule(frame)
                if mod and mod.__name__ == "__main__":
                    break
        frame = frame.f_back
    return list(files.keys())


def walk_caller_paths(
    f_name: str = "*.*",
    files: list[Path] | None = None,
    default_pth: str | Path = Path.home(),
    **_,
) -> Path:
    if not files:
        files = get_caller_dirs()

    # If f_name is provided, search downward from the last __main__ candidate
    if f_name and f_name != "*.*":
        for f in reversed(files):  # start from outermost
            if any(f.glob(f_name)):
                return f

    if files:  # Otherwise just return the outermost __main__ filename’s directory
        return list(files)[-1]
    default_pth = Path(default_pth).expanduser().resolve()
    return default_pth if default_pth.is_dir() else default_pth.parent


def _check_path_val(pth: str | Path | None) -> bool:
    if not pth:  # Catch None or ""
        return False
    if isinstance(pth, str):
        return pth != NULL_PATH  # must have value other wise would be caught above
    return bool(pth.parts) and pth != Path(NULL_PATH)


def _re_glob_path(file_dir: Path, file_name: str) -> list[Path]:
    files = (
        list(file_dir.glob(file_name))
        or list(file_dir.rglob(file_name))
        or list(file_dir.parent.rglob(file_name))
    )
    return files


class JSONSettings:
    """
    Manage a user-specific JSON settings file for plotting or configuration.

    The constructor attempts to locate an existing settings file based on
    the provided arguments. It does not reliably *create* new files; instead,
    it expects the file to already exist and only falls back to creating
    a directory or using a sentinel path when necessary. For reliably
    creating a new settings file at a desired location, use the static
    method :meth:`make_settings_file`.

    Parameters
    ----------
    settings_path : str or pathlib.Path or JSONSettings, optional
        Path to the active settings file. If another `JSONSettings` instance
        is provided, its `settings_path` is reused. If omitted or invalid,
        defaults to ``"settings.json"``. This argument is treated as the
        primary anchor for resolution.
    root_dir : str or pathlib.Path, optional
        Directory or file path used as a hint for locating the settings file.
        If a directory is provided, the settings file is assumed to be located
        inside it. If a file path is provided, its parent directory is used
        and the file name may override `settings_path`. If omitted or invalid,
        a sentinel path (``NULL_PATH/settings.json``) is used, which triggers
        discovery via `get_path` and `walk_caller_paths`.

    Notes
    -----
    Parsing behavior:
        * If `settings_path` is another `JSONSettings`, its path is cloned.
        * If `root_dir` points to a valid file, the file name is used to
          back up `settings_path`, and the parent directory becomes the
          anchor.
        * If `root_dir` points to a valid directory, the settings file is
          assumed to reside there.
        * If neither is valid, a sentinel path is assigned and resolution
          falls back to `get_path` and `walk_caller_paths`.

    Expectations:
        * The constructor expects the settings file to already exist.
        * If the file cannot be found, resolution escalates through caller
          paths and globbing. As a last resort, a directory is created to
          ensure a valid parent exists, but the file itself is not guaranteed
          to be created.
        * To reliably create a new settings file, use
          :meth:`JSONSettings.make_settings_file`.

    Examples
    --------
    >>> # Initialize from a direct path
    >>> s = JSONSettings("config/settings.json")

    >>> # Initialize from a directory hint
    >>> s = JSONSettings(root_dir="config")

    >>> # Clone from another instance
    >>> s2 = JSONSettings(s)

    >>> # Reliably create a new file
    >>> JSONSettings.make_settings_file("config/settings.json")
    """

    # __locations = get_caller_dirs()

    # def __init_subclass__(cls, **kwargs):
    #     super().__init_subclass__(**kwargs)
    #     cls.__locations = get_caller_dirs()

    def __init__(
        self,
        settings_path: str | Path | Self = "settings.json",
        root_dir: str | Path = "",
    ):
        if isinstance(settings_path, JSONSettings):
            self.settings_path = settings_path.settings_path
        else:
            # Check if root_dir can be used as the default path prior to calling get_path
            if _check_path_val(root_dir) and Path(root_dir).expanduser().exists():
                root_dir = Path(root_dir).expanduser()
                if root_dir.is_file():
                    if not _check_path_val(settings_path) and root_dir.suffix == ".json":
                        settings_path = root_dir
                    root_dir = root_dir.parent
                self.settings_path = root_dir / Path(settings_path).name
            else:
                self.settings_path = Path(NULL_PATH) / "settings.json"

            settings_path = settings_path if _check_path_val(settings_path) else "settings.json"
            try:
                self.settings_path = self.get_path(settings_path)
            except FileNotFoundError:
                settings_path = Path(settings_path)
                if (
                    any(p.exists() for p in islice(settings_path.parents, 2))
                    and settings_path.suffix == ".json"
                ):
                    self.settings_path = settings_path
                else:
                    name = (
                        settings_path.name if settings_path.suffix == ".json" else "settings.json"
                    )
                    self.settings_path = walk_caller_paths() / name

        if not self.settings_path.exists():
            if not self.settings_path.parent.exists():
                self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            self.settings_path.touch()

    def get_path(self, file_path: str | Path) -> Path:
        """
        Attempt to resolve a JSON file path:
        """
        if not _check_path_val(file_path):
            raise ValueError("file_path is invalid.")

        file_path = Path(file_path)
        base = self.settings_path.parent

        if len(file_path.parts) == 1:
            file_path = base / file_path

        file_dir = file_path.parent
        file_name = file_path.with_suffix(".json").name

        # Direct hit
        if (file_dir / file_name).exists():
            return file_dir / file_name

        file_dir = walk_caller_paths(file_name)
        if (file_dir / file_name).exists():
            return file_dir / file_name

        # Escalation: glob/rglob
        if base.exists() and base != file_dir:
            files = _re_glob_path(base, file_name)
            if files:
                return files[0]

        files = _re_glob_path(file_dir, file_name)

        if files:
            return files[0]

        raise FileNotFoundError(f"{file_name} not found")

    def from_json(self, file_path) -> dict:
        """Load JSON file and return the data."""
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file, object_hook=post_decoder)

    def to_json(self, settings, file_path):
        """Save settings to a JSON file."""
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(pre_encoder(settings), file, indent=4)

    def load_settings(self, **kwargs) -> dict:
        """Load settings from JSON files and return the settings dictionary."""
        settings = self.from_json(self.settings_path)

        if kwargs:
            kwargs = align_dict(kwargs, settings)
            settings = filter_dict(settings, kwargs)

        return settings

    def save_settings(self, **kwargs) -> dict:
        """Save the local modified values to JSON file."""
        if not kwargs:
            return {}

        settings = self.from_json(self.settings_path)

        # Update the local settings with the current settings
        kwargs = align_dict(kwargs, settings)
        update_dict(settings, kwargs)

        self.to_json(settings, self.settings_path)
        return settings

    def update_from_other(
        self, source_path: str | Path = "", overwrite: bool = True, sub_dict: dict | None = None
    ) -> dict:
        """
        Copy settings from another JSON file into this one.

        Behavior is controlled by two inputs:
        - `overwrite` (bool): whether to replace existing values or only fill in missing ones.
        - `sub_dict` (dict | None): optional structure to restrict the update to a subsection.

        Modes
        -----
        - overwrite=True, sub_dict=None : Clear settings and copy over values from the source
        - overwrite=True, sub_dict={} : Overwrite overlapping parts but retain entries not in source
        - overwrite=False, sub_dict=None : Fill in only missing keys from the source
        - overwrite=True, sub_dict={...} : Overwrite only the specified subsection
        - overwrite=False, sub_dict={...} : Fill in missing keys only within the specified subsection

        Parameters
        ----------
        source_path : str | Path
            Path to the source settings file.
        overwrite : bool, default True
            Whether to overwrite existing values (True) or only fill in missing ones (False).
        sub_dict : dict | None
            Optional subsection structure to restrict the update. An empty dict
            triggers mode 5 (retain extras while overwriting overlaps).

        Returns
        -------
        dict
            The updated settings dictionary.
        """
        try:
            source_path = self.get_path(source_path)
        except (ValueError, FileNotFoundError) as exc:
            raise FileNotFoundError(f"Source path invalid: {source_path}") from exc

        source = self.from_json(source_path)

        if sub_dict:  # Update only specified keys
            sub_dict = align_dict(sub_dict, source)  # Ensure structure matches
            source = filter_dict(source, sub_dict)  # limit source to keys in sub_dict

        if overwrite and sub_dict is None:
            settings = source
        else:
            settings = self.from_json(self.settings_path)
            if not overwrite:
                update_dict(source, settings)
            update_dict(settings, source)

        self.to_json(settings, self.settings_path)
        return settings

        # if not update_all:
        #     current_settings = self.from_json(self.settings_path)
        #     update_dict(settings, current_settings)

        # if sub_dict:  # Update only specified keys
        #     sub_dict = align_dict(sub_dict, settings)  # Ensure structure matches
        #     sub_settings = filter_dict(settings, sub_dict)  # update "kwargs" to values in settings
        #     settings = self.from_json(self.settings_path)  # Load current settings
        #     update_dict(settings, sub_settings)
        #     # Overwrite current settings subsection with values from source

    @staticmethod
    def make_settings_file(
        file_path: str | Path,
        default_settings: dict | None = None,
    ) -> None:
        """
        Create a JSON settings file at the specified location.

        This method validates the target path and ensures the parent directory exists.
        If the file does not already exist, it is created. Optionally, default
        settings can be written into the file. If no defaults are provided, an empty
        JSON file is created.

        Parameters
        ----------
        file_path : str or pathlib.Path
            Absolute path where the settings file should be created.
            The path must satisfy:
            - Have a ".json" suffix
            - Be absolute
            - Its first two components (drive/root and immediate folder) must already exist.
        default_settings : dict, optional
            Default settings to write into the file. If None, an empty
            file is created. If an empty dict is provided, "{}" is written.

        Raises
        ------
        ValueError
            If the path does not meet the criteria above.
        """
        file_path = Path(file_path).expanduser()
        if not file_path.exists():
            if (
                file_path.suffix != ".json"
                or not file_path.is_absolute()
                or (
                    file_path.parts[0] == Path.home().parts[0]
                    and not Path(*file_path.parts[:2]).exists()
                )
            ):
                raise ValueError(f"Invalid file path for JSON settings: {file_path}")
            if not file_path.parent.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.touch()
        if default_settings:
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(pre_encoder(default_settings), file, indent=4)


class DefaultJSONSettings(JSONSettings):
    """
    Settings class that manages both a defaults.json (package-specific)
    and a settings.json (active, user-specific).
    """

    def __init__(
        self,
        settings_path: str | Path = "",
        root_dir: str | Path = "",
        copy_from: str | Path = "",
        default_name: str = "defaults.json",
    ):
        self.settings_path = Path(NULL_PATH) / "settings.json"

        if isinstance(settings_path, DefaultJSONSettings):
            root_dir = settings_path._defaults_path.parent
            default_name = settings_path._defaults_path.name
            settings_path = settings_path.settings_path

        elif _check_path_val(root_dir):
            root_dir = Path(root_dir).expanduser()
            if root_dir.is_file():
                try:
                    root_dir = self.get_path(root_dir)
                except FileNotFoundError:
                    pass
                default_name = default_name or root_dir.name
                root_dir = root_dir.parent

        if not default_name:
            raise ValueError(
                "Cannot provide a null default_name without a root_dir that includes the file name."
            )

        default_name = str(Path(default_name).with_suffix(".json"))

        self._defaults_path = self.get_path(Path(root_dir) / default_name)

        super().__init__(settings_path, self._defaults_path.parent)

        self.update_from_other(copy_from, False)

    def from_json(self, file_path) -> dict:
        """Load JSON file and return the data."""
        try:
            return super().from_json(file_path)
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
            kwargs = align_dict(kwargs, settings)
            settings = filter_dict(settings, kwargs)

        return settings

    def to_json(self, settings, file_path):
        """Save settings to a JSON file."""
        if file_path != self._defaults_path:
            super().to_json(settings, file_path)

    def restore_defaults(self, **kwargs) -> dict:
        """Restore the default settings."""
        return self.update_from_other(self._defaults_path, sub_dict=kwargs)

    def update_from_other(
        self, source_path: str | Path = "", overwrite: bool = True, sub_dict: dict | None = None
    ) -> dict:
        """Restore or copy from a settings file."""
        try:
            source_path = self.get_path(source_path or self._defaults_path)
        except FileNotFoundError:
            source_path = self._defaults_path
        return super().update_from_other(source_path, overwrite, sub_dict)

        # if source_path and source_path != self._defaults_path:
        #     source_path = self.get_path(source_path)

        # if source_path is not None and not source_path.exists():
        #     source_path = self._defaults_path
        # settings = self.from_json(source_path)

        # if kwargs:
        #     kwargs = align_dict(kwargs, settings)
        #     kwargs = filter_dict(settings, kwargs)
        #     settings = self.from_json(self.settings_path)
        #     update_dict(settings, kwargs)

        # self.to_json(settings, self.settings_path)
        # return settings

        # self.settings_path = Path.home() / "settings.json"
        # self._defaults_path = Path(__file__).parent / "defaults.json"

        # if isinstance(settings_path, DefaultJSONSettings):
        #     root_dir = settings_path.root_dir
        #     settings_path = settings_path.settings_path

        # if not defaults_name:
        #     if not _check_path_val(root_dir) or not Path(root_dir).suffix:
        #         raise ValueError("Cannot provide a null defaults_name without a root_dir that includes the file name.")
        #     defaults_name = Path(root_dir).name

        # # Normalize settings_path input

        # # Discover package directory: caller override or walk_caller_paths
        # pkg_dir = Path(root_dir) if root_dir else walk_caller_paths(defaults_name)
        # if pkg_dir.is_file():
        #     pkg_dir = pkg_dir.parent

        # self._defaults_path = pkg_dir / defaults_name
        # if not self._defaults_path.exists() and root_dir:
        #     self._defaults_path = walk_caller_paths(defaults_name) / defaults_name

        # if not self._defaults_path.exists():
        #     files = list(pkg_dir.glob(defaults_name))
        #     if not files:
        #         files = list(pkg_dir.rglob(defaults_name))
        #     if not files:
        #         files = list(pkg_dir.parent.rglob(defaults_name))
        #     if files:
        #         self._defaults_path = files[0]
        #     else:
        #         raise FileNotFoundError(f"{defaults_name} not found in {pkg_dir}")

        # # Active settings file: either caller-supplied or default to settings.json
        # self.settings_path = self._resolve_path(settings_path, "settings.json")

        # super().__init__(self.settings_path, pkg_dir)

    # def get_path(self, file_path=None, default_name=None):
    #     """Return the path of the file based on the given conditions."""
    #     file_path = Path(file_path) if isinstance(file_path, (str, Path)) else self.root_dir
    #     default_name = (
    #         Path(default_name) if isinstance(default_name, (str, Path)) else "settings.json"
    #     )

    #     if len(file_path.parts) == 1:
    #         file_path = self.root_dir / file_path
    #     if file_path.is_file():
    #         return file_path.with_suffix(".json")
    #     if not file_path.exists() and file_path.suffix:
    #         file_path = (
    #             file_path.parent if file_path.parent.exists() else file_path.with_suffix("")
    #         )
    #     return (file_path / default_name).with_suffix(".json")

    # def _resolve_path(self, file_path: str | Path | None, default_name: str) -> Path:
    #     """
    #     Resolve a path to a JSON file:
    #     - If file_path is a bare name, place it under the package dir.
    #     - If file_path is a full path, normalize it.
    #     - If None, fall back to default_name under the package dir.
    #     """
    #     if file_path:
    #         file_path = Path(file_path)
    #         if len(file_path.parts) == 1:  # bare filename
    #             return self._defaults_path.parent / file_path.with_suffix(".json")
    #         return file_path.with_suffix(".json")
    #     return self._defaults_path.parent / Path(default_name).with_suffix(".json")


# class DefaultJSONSettings(JSONSettings):
#     """Class to store data for plotting graphs."""

#     def __init__(
#         self,
#         settings_path=None,
#         root_dir: str | Path = __file__,
#         copy_from=None,
#     ):
#         super().__init__(settings_path, root_dir)
#         self.root_dir = Path(root_dir)
#         if self.root_dir.is_file():
#             self.root_dir = self.root_dir.parent
#         if not self.root_dir.exists():
#             raise NotADirectoryError(f"root_dir does not exist: {self.root_dir}")

#         self._defaults_path = self.root_dir / "defaults.json"
#         if isinstance(settings_path, DefaultJSONSettings):
#             settings_path = settings_path.settings_path
#         self.settings_path = self.get_path(settings_path, "settings.json")

#         # Copy default settings to local settings if local settings file doesn't exist
#         if not self.settings_path.exists():
#             self.update_from_other(copy_from)

#     def get_path(self, file_path=None, default_name=None):
#         """Return the path of the file based on the given conditions."""

#         if isinstance(file_path, (str, Path)):
#             print("Yes path")
#             file_path = Path(file_path)
#             if len(file_path.parts) == 1:
#                 return self.root_dir / file_path.with_suffix(".json")
#             else:
#                 return file_path.with_suffix(".json")
#         else:
#             print("No path")
#             if isinstance(default_name, (str, Path)):
#                 return self.root_dir / Path(default_name).with_suffix(".json")
#             return self.root_dir / "defaults.json"


# def walk_pkg_paths(f_name: str = "*.*", default_pth: str | Path = Path.home(), **_) -> Path:
#     frame = inspect.currentframe()
#     files: dict[Path, None] = {}
#     while frame:
#         file = Path(frame.f_code.co_filename)
#         if file.is_file() and file.parent.parts:
#             files[file.parent] = None
#             if frame.f_code.co_name == "<module>":
#                 mod = inspect.getmodule(frame)
#                 if mod and mod.__name__ == "__main__":
#                     break
#         frame = frame.f_back
#     # print("\n".join([str(f) for f in main_files]))
#     # If f_name is provided, search downward from the last __main__ candidate
#     if f_name and f_name != "*.*":
#         for f in reversed(files):  # start from outermost
#             if any(f.glob(f_name)):
#                 return f

#     # Otherwise just return the outermost __main__ filename’s directory
#     return list(files.keys())[-1] if files else Path(default_pth).expanduser().resolve()

#     # if file.parent not in seen:
#     #     seen.add(file.parent)
#     #     main_files.append(file.parent)


# def walk_caller_paths(
#     f_name: str = "", ignore_till: str = "", default_pth: str | Path = Path.home(), **kwargs
# ) -> Path:
#     """
#     Attempt to return the directory of the calling module.  Walks the call stack
#     using inspect until it finds a directory containing f_name.
#     If none found, returns cwd or home based on cwd_as_default.


#     """
#     f_name = f_name or "*.*"
#     skip_past = Path(ignore_till)
#     skip_past = Path(__file__) if not skip_past.parts or not skip_past.exists() else skip_past

#     files = [Path(s.filename) for s in reversed(inspect.stack()) if Path(s.filename).is_file()]
#     files = files[: files.index(skip_past)][::-1] if skip_past in files else files[::-1]

#     for f in files:
#         if f.parent.parts and any(f.parent.glob(f_name)):
#             # print(f"Selected: {f}")
#             return f.parent

#     # print("No matching path found.")
#     return Path(default_pth)

# print("\n".join([str(f) for f in files]))
# if skip_past in files:
#     idx = files.index(skip_past) + 1
#     files = files[idx:]
# print("searching in files:")


# def get_source_dir_old(f_name: str = "", _callers: set = set(), cwd_as_default: bool = True) -> Path:
#     """Return the directory of the calling module."""
#     stack = inspect.stack()
#     _callers.add(__file__)
#     # stack[0] = _get_caller_dir, stack[1] = __init__, stack[2] = caller of DefaultJSONSettings
#     for frame_info in stack:
#         module = inspect.getmodule(frame_info.frame)
#         if module and hasattr(module, "__file__") and str(module.__file__) not in _callers:
#             dir_path = Path(str(module.__file__)).parent
#             if (dir_path / f_name).exists():
#                 return dir_path
#     return Path.cwd() if cwd_as_default else Path.home()


# pth = Path("C:/Users/j2cle/Documents/Python/impedance_analysis/eis_analysis/z_fit")

# class DefaultJSONSettings(JSONSettings):
#     """Class to store data for plotting graphs."""

#     def __init__(self, settings_path=None, copy_from=None, root_dir: str | Path = __file__):
#         self.root_dir = Path(root_dir)
#         if self.root_dir.is_file():
#             self.root_dir = self.root_dir.parent
#         if not self.root_dir.exists():
#             raise NotADirectoryError(f"root_dir does not exist: {self.root_dir}")

#         self._defaults_path = self.root_dir / "defaults.json"
#         if isinstance(settings_path, DefaultJSONSettings):
#             settings_path = settings_path.settings_path
#         self.settings_path = self.get_path(settings_path, "settings.json")

#         # Copy default settings to local settings if local settings file doesn't exist
#         if not self.settings_path.exists():
#             self.update_from_other(copy_from)

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
#             kwargs = align_dict(kwargs, settings)
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
#         kwargs = align_dict(kwargs, settings)
#         update_dict(settings, kwargs)

#         self.to_json(settings, self.settings_path)
#         return settings

#     def restore_defaults(self, **kwargs) -> dict:
#         """Restore the default settings."""
#         return self.update_from_other(self._defaults_path, **kwargs)

#     def update_from_other(self, source_path=None, **kwargs) -> dict:
#         """Restore or copy from a settings file."""
#         source_path = (
#             self.get_path(source_path) if source_path != self._defaults_path else source_path
#         )
#         if source_path is not None and not source_path.exists():
#             source_path = self._defaults_path
#         settings = self.from_json(source_path)

#         if kwargs:
#             kwargs = align_dict(kwargs, settings)
#             kwargs = filter_dict(settings, kwargs)
#             settings = self.from_json(self.settings_path)
#             update_dict(settings, kwargs)

#         self.to_json(settings, self.settings_path)
#         return settings
