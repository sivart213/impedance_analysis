# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
from copy import deepcopy
from typing import Any

from PyQt5 import sip  # type: ignore
from PyQt5.QtWidgets import (
    QLabel,
    QDialog,
    QWidget,
    QLineEdit,
    QTabWidget,
    QFormLayout,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QDesktopWidget,
)

from ..string_ops import safe_eval, format_number
from ..widgets.generic_widgets import SimpleDialog
from ..system_utilities.json_io import JSONSettings


class DictView:
    """
    A read-only view into a dict-like object.
    Supports flattening of nested dicts if all selected values are dicts
    and their subkeys do not overlap.
    """

    def __init__(self, source, keys, *, flatten=False, allow_backdoor=False, **_):
        self._source = source
        self._map = {}
        self._allow_backdoor = allow_backdoor

        if flatten:
            for k in keys:
                parts = k.split(".")
                if len(parts) == 1:
                    # top-level dict must itself be a dict
                    parent = parts[0]
                    if not isinstance(source.get(parent), dict):
                        raise TypeError(f"Flatten requested but {parent} is not a dict")
                    # expose all subkeys of this dict
                    for subk in source[parent]:
                        if subk in self._map:
                            raise ValueError(f"Flatten conflict: subkey '{subk}' already mapped")
                        self._map[subk] = (parent, subk)
                elif len(parts) == 2:
                    parent, subk = parts
                    if not isinstance(source.get(parent), dict):
                        raise TypeError(f"Flatten requested but {parent} is not a dict")
                    if subk not in source[parent]:
                        raise KeyError(f"Subkey '{subk}' not found in {parent}")
                    if subk in self._map:
                        raise ValueError(f"Flatten conflict: subkey '{subk}' already mapped")
                    self._map[subk] = (parent, subk)
                else:
                    raise ValueError(f"Only one level of '.' nesting supported: {k}")
        else:
            # Non-flattened: expose top-level keys directly
            self._map = {k: (k, None) for k in keys}

    def __getitem__(self, key):
        if key in self._map:
            parent, sub = self._map[key]
            return self._source[parent] if sub is None else self._source[parent][sub]
        if self._allow_backdoor and key in self._source:
            return self._source[key]
        raise KeyError(f"{key} not in view")

    def __iter__(self):
        return iter(self._map.keys())

    def __len__(self):
        return len(self._map)

    def __contains__(self, key):
        return key in self._map

    def keys(self):
        return list(self._map.keys())

    def values(self):
        return [self[k] for k in self._map]

    def items(self):
        return [(k, self[k]) for k in self._map]

    def as_dict(self):
        return {k: self[k] for k in self._map}

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __repr__(self):
        return f"<DictView keys={list(self._map.keys())} current={self.as_dict()}>"


class DictWindow:
    """Class to create an options window."""

    def __init__(
        self, parent: Any, options: dict, name: str = "window", title: str | None = None, **kwargs
    ):
        self.parent = parent
        self._items = deepcopy(options) if isinstance(options, dict) else {}
        self._defaults = deepcopy(self._items)
        self.name = name
        self.title = self._format_name(title or name)
        self.entries = []
        self.qt_window: Any = None
        self.canceled = False

        self._save_button = kwargs.get("save_button", True)
        self._add_button = kwargs.get("add_button", True)
        # self._rem_button = kwargs.get("rem_button", True)
        self._reset_button = kwargs.get("reset_button", True)
        self._cancel_button = kwargs.get("cancel_button", False)

        self._close_on_save = kwargs.get("close_on_save", False)
        self._close_on_save_all = kwargs.get("close_on_save_all", False)

        size = kwargs.get("size", (800, 400))
        if isinstance(size, str):
            if size.lower() == "large":
                size = (1200, 800)
            elif "large" in size.lower():
                size = (1400, 1000)
            elif size.lower() == "small":
                size = (600, 400)
            elif "small" in size.lower():
                size = (500, 300)
            else:
                size = (800, 400)
        if not isinstance(size, (tuple, list)):
            size = (800, 400)
        # self._size = size

        screen = QDesktopWidget().availableGeometry()
        max_width = int(screen.width() * 0.95)
        max_height = int(screen.height() * 0.95)
        self._size = (min(size[0], max_width), min(size[1], max_height))

    def __getitem__(self, key):
        return self._items[key]

    def __setitem__(self, key, value):
        self._items[key] = value

    def __delitem__(self, key):
        del self._items[key]

    def __contains__(self, key):
        return key in self._items

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

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def as_dict(self):
        """Return the internal dictionary."""
        return deepcopy(self._items)

    def get_view(self, keys, **kwargs):
        return DictView(self, keys, **kwargs)

    @property
    def defaults(self):
        """Get the default options."""
        return self._defaults

    def window(self):
        """Create the options window."""
        self.entries = []
        self.destroy()
        self.canceled = False

        self.qt_window = QDialog(self.parent)
        self.qt_window.setWindowTitle(self.title)
        self.qt_window.finished.connect(
            lambda: self.update_internal_dict(None)
        )  # Update internal dict on close
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

        # content_layout = QVBoxLayout()
        # content_layout.addLayout(layout)

        # Add global buttons
        global_button_layout = QHBoxLayout()
        self.add_global_buttons(global_button_layout)
        # layout.addLayout(global_button_layout)
        # content_layout.addLayout(global_button_layout)

        content_widget = QWidget()
        content_widget.setLayout(layout)
        # content_widget.setLayout(content_layout)

        scroll = QScrollArea(self.qt_window)
        scroll.setWidgetResizable(True)
        scroll.setWidget(content_widget)

        main_layout = QVBoxLayout(self.qt_window)
        main_layout.addWidget(scroll)
        main_layout.addLayout(global_button_layout)
        self.qt_window.setLayout(main_layout)

        # self.qt_window.setLayout(layout)
        self.qt_window.resize(*self._size)

        self.qt_window.exec_()

    def add_tab_buttons(self, layout, index):
        """Add tab-specific buttons to the layout."""
        if self._save_button:
            save_button = QPushButton("Save")
            save_button.clicked.connect(lambda: self.save(index))
            layout.addWidget(save_button)

        if self._add_button:
            add_button = QPushButton("Add")
            add_button.clicked.connect(lambda: self.add_option(index))
            layout.addWidget(add_button)

        # if self._rem_button:
        #     rem_button = QPushButton("Remove")
        #     rem_button.clicked.connect(lambda: self.rem_entry(index))
        #     layout.addWidget(rem_button)

        if self._reset_button:
            reset_button = QPushButton("Reset")
            reset_button.clicked.connect(lambda: self.reset(index))
            layout.addWidget(reset_button)

        if self._cancel_button:
            cancel_button = QPushButton("Cancel")
            cancel_button.clicked.connect(self.cancel)
            layout.addWidget(cancel_button)

    def add_global_buttons(self, layout):
        """Add global buttons to the layout."""
        if self._save_button:
            save_all_button = QPushButton("Save All")
            save_all_button.clicked.connect(lambda: self.save(None))
            layout.addWidget(save_all_button)

        if len(self.entries) == 1 and self._add_button:
            add_button = QPushButton("Add")
            add_button.clicked.connect(lambda: self.add_option(None))
            layout.addWidget(add_button)

        # if len(self.entries) == 1 and self._rem_button:
        #     rem_button = QPushButton("Remove")
        #     rem_button.clicked.connect(lambda: self.add_option(None))
        #     layout.addWidget(rem_button)

        if self._reset_button:
            reset_all_button = QPushButton("Reset All")
            reset_all_button.clicked.connect(lambda: self.reset(None))
            layout.addWidget(reset_all_button)

        if self._cancel_button:
            cancel_all_button = QPushButton("Cancel All")
            cancel_all_button.clicked.connect(self.cancel)
            layout.addWidget(cancel_all_button)

    def update_internal_dict(self, tab_key=None):
        """Update the internal dictionary with the current values from the window."""
        # breakpoint()
        if not self.entries or self.canceled:
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
            self._items[super_key] = {key: safe_eval(val.text()) for key, val in entries.items()}

    def add_option(self, index=None):
        """Add a new option to the window."""
        sub_key, ok = SimpleDialog.getResult(self.parent, "Add Option", "Enter the key:", width=50)
        if ok and sub_key:
            if sub_key:
                if len(self.entries) == 1:
                    self._items[sub_key] = ""
                else:
                    if index is None:
                        index = self.qt_window.layout().itemAt(0).widget().currentIndex()
                    self._items[self.entries[index][0]][sub_key] = ""

                self.window()

    def save(self, index: int | None = None):
        """Save the options of the current tab to the local JSON and update _defaults."""
        if index is None:
            self.update_internal_dict()
            self._defaults = deepcopy(self._items)
        else:
            key = self.entries[index][0]
            self.update_internal_dict(index)
            self._defaults[key] = deepcopy(self._items[key])

        if self._close_on_save or (self._close_on_save_all and index is None):
            self.destroy()

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
                self._items.clear()
                self._items.update(deepcopy(self._defaults))
            else:
                key = self.entries[index][0]
                self._items[key].clear()
                self._items[key].update(deepcopy(self._defaults[key]))
            self.window()  # Recreate the window to reflect the reset options

    def cancel(self):
        """Handle the cancel action."""
        self.canceled = True
        self.destroy()

    def destroy(self):
        """Destroy the options window."""
        if self.qt_window is not None and not sip.isdeleted(self.qt_window):
            # self.qt_window.isVisible()
            self.qt_window.close()
            self.qt_window.deleteLater()

    def _format_name(self, name):
        """Format the name of the options window."""
        if isinstance(name, str):
            return name if "option" in name.lower() else f"{name.title()} Options"
        else:
            return "Options"


class JsonDictWindow(DictWindow):
    """Subclass of DictWindow with JSON handling."""

    def __init__(
        self,
        parent: Any,
        options: dict,
        name: str = "window",
        title: str | None = None,
        json_settings: JSONSettings | None = None,
        **kwargs,
    ):
        super().__init__(parent, options, name, title, **kwargs)
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
                self._items.update(deepcopy(self._defaults))
            else:
                # Restore defaults for the specified tab
                key = self.entries[index][0]
                restore_dict = {self.name: {key: {}}}
                defaults = self.json_settings.restore_defaults(**restore_dict)
                self._defaults[key] = defaults[self.name][key].copy()
                self._items[key].clear()
                self._items[key].update(deepcopy(self._defaults[key]))
            self.window()  # Recreate the window to reflect the restored options
