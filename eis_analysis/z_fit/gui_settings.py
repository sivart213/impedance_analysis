import numpy as np
# import pandas as pd
# import sip
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import (
    # QApplication,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QWidget,
    QTabWidget,
    # QTableWidget,
    QLineEdit,
    QPushButton,
    # QFrame,
    QMessageBox,
    QFormLayout,
    QDialogButtonBox,
    # QTreeWidget,
    # QTreeWidgetItem,
    # QAbstractItemView,
    # QMainWindow,
    # QTableWidgetItem,
    # QTextEdit,
    # QProgressDialog,
    # QListWidget,
    # QListWidgetItem,
    # QInputDialog,
)
# from PyQt5.QtGui import QBrush, QColor, QFontMetrics

# import re
from pathlib import Path
# from typing import Optional
import json



# from copy import deepcopy



# from ..string_ops import format_number, common_substring
from ..string_ops import format_number, find_common_str, safe_eval

# from .gui_widgets import MultiEntryManager

# from .gui_workers import DataHandler, JSONSettings
from ..dict_ops import update_dict, filter_dict, check_dict, safe_deepcopy

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

class SudoDict:
    """Custom dictionary class that resembles a dictionary but has less conflicts with built-in methods."""
    def __init__(self, arg=None, **kwargs):
        self._items = arg if isinstance(arg, dict) else kwargs

    def __getitem__(self, key):
        return self._items[key]

    def __setitem__(self, key, value):
        self._items[key] = value

    def __delitem__(self, key):
        del self._items[key]

    def __contains__(self, key):
        return key in self._items
    
    def __copy__(self):
        return SudoDict(self._items.copy())

    def items(self):
        return self._items.items()

    def keys(self):
        return self._items.keys()

    def values(self):
        return self._items.values()

    def update(self, *args, **kwargs):
        self._items.update(*args, **kwargs)

    def clear(self):
        self._items.clear()

    def copy(self):
        return self.__copy__()
        


class CustomSimpleDialog(QDialog):
    """Class to create a simple dialog window."""

    def __init__(
        self, parent, title=None, prompt=None, initialvalue="", width=50
    ):
        super().__init__(parent)
        self.prompt = prompt
        self.initialvalue = initialvalue
        self.width = width
        self.setWindowTitle(title)
        self.init_ui()

    def init_ui(self):
        """Create the body of the dialog window."""
        layout = QVBoxLayout()

        label = QLabel(self.prompt)
        layout.addWidget(label)

        self.entry = QLineEdit(self)
        self.entry.setText(self.initialvalue)
        self.entry.setFixedWidth(self.width)
        layout.addWidget(self.entry)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def accept(self):
        """Apply the changes made in the dialog window."""
        self.result = self.entry.text()
        super().accept()
    
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
    """
    Class to store data for plotting graphs.

    Responsibilities:
    - Load and save settings from/to JSON files.
    - Restore default settings if local settings are not available.
    - Handle JSON loading errors and provide default settings.

    Integration:
    - Use this class to manage application settings.
    - Extend this class to create specific settings managers like `SettingsManager`.
    """

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


class SettingsGroup(SudoDict, QObject):
    """
    Class to manage a group of settings.
    """

    settings_changed = pyqtSignal()  # Signal to notify settings changes

    def __init__(self, settings, group_key):
        QObject.__init__(self)
        SudoDict.__init__(self)
        if not isinstance(settings, (dict, SettingsGroup)):
            raise ValueError("Settings must be a dictionary.")
        settings = settings.get(group_key, settings)
        self.update(settings)
        self._defaults = safe_deepcopy(settings)
        self.group_key = group_key
        self._emit_keys = []
    
    @property
    def defaults(self):
        """Get the default options."""
        return self._defaults

    @property
    def emit_keys(self):
        """Get the keys for the settings to emit signals."""
        return self._emit_keys
    
    @emit_keys.setter
    def emit_keys(self, keys):
        """Set the keys for the settings to emit signals."""
        if isinstance(keys, str) and keys in self and keys not in self._emit_keys:
            self._emit_keys.append(keys)
        elif isinstance(keys, (list, tuple)):
            for key in keys:
                self.emit_keys = key

    def set(self, key, value, suppress_signal=False):
        """Set a specific setting."""
        super().__setitem__(key, value)
        if key in self.emit_keys and not suppress_signal:
            self.settings_changed.emit()  # Emit signal when settings change

    def update(self, settings, suppress_signal=False):
        """Update the settings with new values."""
        send_emit = False
        for key, value in settings.items():
            self.set(key, value, True)
            if not send_emit and key in self.emit_keys:
                send_emit = True
        if send_emit and not suppress_signal:
            self.settings_changed.emit()  # Emit signal when done updating

class SettingsGroupManager(dict):
    """
    Class to manage multiple SettingsGroup objects.
    """

    def __init__(self, groups=None):
        super().__init__()
        if groups:
            self.update_groups(groups)

    def update_groups(self, groups):
        """Dynamically create settings groups based on keys under `option_inits`."""
        # option_inits = self.settings.get('option_inits', {})
        if not isinstance(groups, dict):
            return
        for key, gr in groups.items():
            if key not in self:
                self[key] = SettingsGroup(gr, key)
            else:
                self[key].update(gr)
            self[key] = SettingsGroup(gr, key)

    def as_dict(self):
        """Return a simple dictionary version of the settings groups."""
        return {key: dict(group) for key, group in self.items()}

    def get_defaults(self):
        """Return the default values of the settings groups."""
        return {key: group.defaults for key, group in self.items()}

class SettingsManager(JSONSettings):
    """
    Class to manage settings for the application.
    """

    settings_changed = pyqtSignal()  # Signal to notify settings changes

    def __init__(self):
        super().__init__()
        self.load_settings()
        self.options = SettingsGroupManager()

    # def _create_or_update_settings_groups(self, settings):
    #     """Dynamically create or update settings groups based on keys under `option_inits`."""
    #     option_inits = settings.get('option_inits', {})
    #     for key in option_inits:
    #         if key not in self.options:
    #             self.options[key] = SettingsGroup(settings, key)
    #             # self.options[key].settings_changed.connect(self.on_settings_changed)
    #         else:
    #             self.options[key].update(option_inits[key])


    def save_settings(self, **kwargs):
        """Save the local modified values to JSON file."""
        if not kwargs:
            return
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        super().save_settings(**kwargs)

    def load_settings(self, **kwargs):
        """Load settings from JSON files and set attributes."""
        settings = super().load_settings(**kwargs)
        for key, value in settings.items():
            setattr(self, key, value)
        self.options.update_groups(settings.get('option_inits', {}))
        # self._create_or_update_settings_groups(kwargs)  # Update settings groups after loading

    def restore_defaults(self, **kwargs):
        """Restore the default settings."""
        settings = super().restore_defaults(**kwargs)
        for key, value in settings.items():
            setattr(self, key, value)
        self.options.update_groups(settings.get('option_inits', {}))
        # self._create_or_update_settings_groups(kwargs)  # Update settings groups after restoring defaults
        self.settings_changed.emit()  # Emit signal when settings change


class DictWindow(SudoDict):
    """Class to create an options window."""

    def __init__(self, parent, options, name=None, title=None):
        SudoDict.__init__(self, options)
        self.parent = parent
        # self._items = deepcopy(options) if isinstance(options, dict) else {}
        # self._items = options if isinstance(options, dict) else {}
        self._defaults = safe_deepcopy(self._items)
        self.name = name
        self.title = self._format_name(title or name)
        self.entries = []
        self.qt_window = None

        
    @property
    def defaults(self):
        """Get the default options."""
        return self._defaults
    

    def window(self):
        """Create the options window."""
        self.entries = []
        self.destroy()

        self.qt_window = QDialog(self.parent)
        self.qt_window.setWindowTitle(self.title)
        self.qt_window.finished.connect(lambda: self.update_internal_dict(None))  # Update internal dict on close
        self.qt_window.finished.connect(self.destroy)

        layout = QVBoxLayout()

        def conv_items(key, value):
            label = QLabel(key)
            entry = QLineEdit()
            if isinstance(value, list):
                entry.setText(", ".join([format_number(v, 5) for v in value]))
            elif isinstance(value, (int, float)):
                entry.setText(format_number(value, 5))
            else:
                entry.setText(str(value))

            return label, entry

        if all(isinstance(value, dict) for value in self._items.values()):
            tabs = QTabWidget()
            layout.addWidget(tabs)

            self.previous_tab_index = 0  # Initialize previous tab index

            for index, (key, sub_dict) in enumerate(self._items.items()):
                tab = QWidget()
                tab_layout = QVBoxLayout()
                form_layout = QFormLayout()
                entries = {}

                for sub_key, sub_value in sub_dict.items():
                    label, entry = conv_items(sub_key, sub_value)
                    form_layout.addRow(label, entry)
                    entries[sub_key] = entry

                tab_layout.addLayout(form_layout)

                # Add tab-specific buttons to each tab
                tab_button_layout = QHBoxLayout()
                self.add_tab_buttons(tab_button_layout, index)
                tab_layout.addLayout(tab_button_layout)

                tab.setLayout(tab_layout)
                tabs.addTab(tab, key)

                self.entries.append([key, entries])

        else:
            form_layout = QFormLayout()
            entries = {}

            for key, value in self._items.items():
                label, entry = conv_items(key, value)
                form_layout.addRow(label, entry)
                entries[key] = entry

            layout.addLayout(form_layout)

            self.entries.append([None, entries]) 

        # Add global buttons
        global_button_layout = QHBoxLayout()
        self.add_global_buttons(global_button_layout)
        layout.addLayout(global_button_layout)

        self.qt_window.setLayout(layout)
        self.qt_window.resize(800, 400)

        self.qt_window.exec_()
    
    def add_tab_buttons(self, layout, index):
        """Add tab-specific buttons to the layout."""
        save_button = QPushButton("Save")
        save_button.clicked.connect(lambda: self.save(index))
        layout.addWidget(save_button)

        add_button = QPushButton("Add")
        add_button.clicked.connect(lambda: self.add_option(index))
        layout.addWidget(add_button)

        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(lambda: self.reset(index))
        layout.addWidget(reset_button)

    def add_global_buttons(self, layout):
        """Add global buttons to the layout."""
        save_all_button = QPushButton("Save All")
        save_all_button.clicked.connect(lambda: self.save(None))
        layout.addWidget(save_all_button)

        if len(self.entries) == 1:
            add_button = QPushButton("Add")
            add_button.clicked.connect(lambda: self.add_option(None))
            layout.addWidget(add_button)

        reset_all_button = QPushButton("Reset All")
        reset_all_button.clicked.connect(lambda: self.reset(None))
        layout.addWidget(reset_all_button)

    def update_internal_dict(self, tab_key=None):
        """Update the internal dictionary with the current values from the window."""
        if not self.entries:
            return
        if tab_key is None:
            if len(self.entries) == 1:
                # Update self._items as there are no sub-dicts
                entries = self.entries[0][1]
                self._items.update({key: safe_eval(val.text()) for key, val in entries.items()})
            else:
                # Update all sub-dicts
                for i in range(len(self.entries)):
                    self.update_internal_dict(i)
        else:
            # Update the specific sub-dict
            super_key, entries = self.entries[tab_key]
            self._items[super_key].update({key: safe_eval(val.text()) for key, val in entries.items()})
            
    def add_option(self, index=None):
        """Add a new option to the window."""
        dialog = CustomSimpleDialog(
            self.parent, "Add Option", "Enter the key:", width=50
        )
        if dialog.exec_() == QDialog.Accepted:
            sub_key = dialog.result
            if sub_key:
                if len(self.entries) == 1:
                    self._items[sub_key] = ""
                else:
                    if index is None:
                        index = self.qt_window.layout().itemAt(0).widget().currentIndex()
                    self._items[self.entries[index][0]][sub_key] = ""

                self.window()

    def save(self, index=None):
        """Save the options of the current tab to the local JSON and update _defaults."""
        if index is None:
            self.update_internal_dict()
            self._defaults = safe_deepcopy(self._items)
        else:
            key = self.entries[index][0]
            self.update_internal_dict(index)
            self._defaults[key] = safe_deepcopy(self._items[key])


    def reset(self, index=None):
        """Reset the options to the default values."""
        reply = QMessageBox.question(
            self.parent,
            "Confirm Reset",
            "Are you sure you want to reset the options to their initial values?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            if index is None:
                if len(self.entries) == 1:
                    # Update each sub-dict individually to preserve connections
                    for key in self._items:
                        self._items[key].clear()
                        self._items[key].update(safe_deepcopy(self._defaults[key]))
                else:
                    self._items.clear()
                    self._items.update(safe_deepcopy(self._defaults))
            else:
                key = self.entries[index][0]
                self._items[key].clear()
                self._items[key].update(safe_deepcopy(self._defaults[key]))
            self.window()  # Recreate the window to reflect the reset options

    def destroy(self):
        """Destroy the options window."""
        if self.qt_window is not None and self.qt_window.isVisible():
            self.qt_window.close()
            self.qt_window.deleteLater()

    def _format_name(self, name):
        """Format the name of the options window."""
        if isinstance(name, str):
            return (
                name if "option" in name.lower() else f"{name.title()} Options"
            )
        else:
            return "Options"

class JsonDictWindow(DictWindow):
    """Subclass of DictWindow with JSON handling."""

    def __init__(self, parent, options, name=None, title=None, json_settings=None):
        super().__init__(parent, options, name, title)
        self.json_settings = json_settings if json_settings else JSONSettings()

    def add_tab_buttons(self, layout, index):
        """Add tab-specific buttons to the layout."""
        super().add_tab_buttons(layout, index)

        restore_defaults_button = QPushButton("Restore Defaults")
        restore_defaults_button.clicked.connect(lambda: self.restore_defaults(index))
        layout.addWidget(restore_defaults_button)

    def add_global_buttons(self, layout):
        """Add global buttons to the layout."""
        super().add_global_buttons(layout)

        restore_all_defaults_button = QPushButton("Restore All Defaults")
        restore_all_defaults_button.clicked.connect(lambda: self.restore_defaults(None))
        layout.addWidget(restore_all_defaults_button)

    def save(self, index=False):
        """Save the options of the current tab to the local JSON and update _defaults."""
        super().save(index)
        self.json_settings.save_settings(**{self.name: self._defaults})

    def restore_defaults(self, index=None):
        """Restore the default settings from the local JSON and update _defaults and current values."""
        reply = QMessageBox.question(
            self.parent,
            "Confirm Reset",
            "Are you sure you want to reset the options to default?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            if index is None:
                # Restore all defaults
                defaults = self.json_settings.restore_defaults(**{self.name: {}})
                self._defaults = defaults[self.name].copy()
                self._items.clear()
                self._items.update(safe_deepcopy(self._defaults))
            else:
                # Restore defaults for the specified tab
                key = self.entries[index][0]
                restore_dict = {self.name: {key: {}}}
                defaults = self.json_settings.restore_defaults(**restore_dict)
                self._defaults[key] = defaults[self.name][key].copy()
                self._items[key].clear()
                self._items[key].update(safe_deepcopy(self._defaults[key]))
            self.window()  # Recreate the window to reflect the restored options
