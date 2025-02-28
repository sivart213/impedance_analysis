# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
import re
import sys

from copy import deepcopy

import numpy as np
import pandas as pd

import sip
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QWidget,
    QTabWidget,
    QTableWidget,
    QLineEdit,
    QPushButton,
    QFrame,
    QMessageBox,
    QFormLayout,
    QDialogButtonBox,
    QTreeWidget,
    QTreeWidgetItem,
    QAbstractItemView,
    QMainWindow,
    QTableWidgetItem,
    QTextEdit,
    QProgressDialog,
    QListWidget,
    QListWidgetItem,
    QInputDialog,
)
from PyQt5.QtGui import QBrush, QColor, QFontMetrics

# from ..string_ops import format_number, common_substring
from ..string_ops import format_number, find_common_str, safe_eval

from .gui_widgets import MultiEntryManager

from .gui_workers import DataHandler, JSONSettings

def create_progress_dialog(
    parent,
    title="Progress",
    label_text="Processing...",
    cancel=None,
    minimum=0,
    maximum=100,
):
    """Create and return a QProgressDialog."""
    progress_dialog = QProgressDialog(
        label_text, cancel, minimum, maximum, parent
    )
    progress_dialog.setWindowTitle(title)
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setMinimumDuration(0)
    return progress_dialog


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


class PrintWindow:
    def __init__(self, parent):
        self.parent = parent
        self.window = None
        self.text_area = None
        self.old_stdout = None
        self.history = ""

    def __enter__(self):
        """Enter the runtime context related to this object."""
        self.show()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context related to this object."""
        self.close()

    def show(self):
        """Show the print window and redirect stdout."""
        self.window = QDialog(self.parent)
        self.window.setWindowTitle("Print Output")
        self.window.finished.connect(self.close)

        self.text_area = QTextEdit(self.window)
        self.text_area.setReadOnly(True)

        # Create a layout for the buttons
        button_layout = QHBoxLayout()
        clear_button = QPushButton("Clear", self.window)
        clear_button.clicked.connect(self.clear_text)
        button_layout.addWidget(clear_button)

        close_button = QPushButton("Close", self.window)
        close_button.clicked.connect(self.close)
        button_layout.addWidget(close_button)

        # Create the main layout
        layout = QVBoxLayout(self.window)
        layout.addWidget(self.text_area)
        layout.addLayout(
            button_layout
        )  # Add the button layout to the main layout
        self.window.setLayout(layout)
        self.window.resize(800, 400)
        self.window.show()

        # Insert history into the text area
        self.text_area.setPlainText(self.history)

        # Redirect stdout to capture the output
        self.old_stdout = sys.stdout
        sys.stdout = self

    def close(self):
        """Restore stdout and close the print window."""
        self.text_area = None
        if self.old_stdout:
            sys.stdout = self.old_stdout
            self.old_stdout = None
        if self.window:
            self.window.close()
            self.window = None

    def clear_text(self):
        """Clear the text area and reset history."""
        self.history = ""
        if self.text_area:
            self.text_area.clear()

    def write(self, message, separate=False):
        """Write text to the text area and update history."""
        separator = "\n" + "=" * 150 + "\n"
        if separate:
            self.history += separator + message
        else:
            self.history += message
        if self.text_area:
            self.text_area.ensureCursorVisible()
            if separate:
                self.text_area.append(separator + message)
            else:
                self.text_area.append(message)
            # self.text_area.append(separator + message)
            self.text_area.ensureCursorVisible()
            QApplication.processEvents()  # Ensure the GUI updates in real-time

    def flush(self):
        """Flush the text area."""
        if self.text_area:
            self.text_area.ensureCursorVisible()

class DictWindow:
    """Class to create an options window."""

    def __init__(self, parent, options, name=None, title=None):
        self.parent = parent
        self._items = deepcopy(options) if isinstance(options, dict) else {}
        self._defaults = deepcopy(self._items)
        self.name = name
        self.title = self._format_name(title or name)
        self.entries = []
        self.qt_window = None

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
            self._items[super_key] = {key: safe_eval(val.text()) for key, val in entries.items()}
            
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
            self._defaults = deepcopy(self._items)
        else:
            key = self.entries[index][0]
            self.update_internal_dict(index)
            self._defaults[key] = deepcopy(self._items[key])


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

# class DictWindow(dict):
#     """Class to create an options window."""

#     def __init__(self, parent, options, name=None, title=None):
#         self.parent = parent
#         if isinstance(options, (dict, DictWindow)) and all(
#             isinstance(value, (dict, DictWindow)) for value in options.values()
#         ):
#             super().__init__({k: v.copy() for k, v in options.items()})
#             self._defaults = {k: v.copy() for k, v in options.items()}
#         elif isinstance(options, (dict, DictWindow)):
#             super().__init__(options.copy())
#             self._defaults = options.copy()
#         else:
#             super().__init__({})
#             self._defaults = {}

#         self.name = name
#         self.title = self._format_name(title or name)
#         self.entries = []  
#         self.qt_window = None

#     @property
#     def defaults(self):
#         """Get the default options."""
#         return self._defaults

#     @property
#     def current_index(self):
#         """Get the current index of the tab or default to 0."""
#         if self.qt_window is not None and isinstance(self.qt_window.layout().itemAt(0).widget(), QTabWidget):
#             return self.qt_window.layout().itemAt(0).widget().currentIndex()
#         return 0

#     def window(self):
#         """Create the options window."""
#         self.destroy()

#         self.qt_window = QDialog(self.parent)
#         self.qt_window.setWindowTitle(self.title)
#         self.qt_window.finished.connect(self.update_internal_dict)  # Update internal dict on close
#         self.qt_window.finished.connect(self.destroy)

#         layout = QVBoxLayout()

#         def conv_items(key, value):
#             label = QLabel(key)
#             entry = QLineEdit()
#             if isinstance(value, list):
#                 entry.setText(", ".join([format_number(v, 5) for v in value]))
#             elif isinstance(value, (int, float)):
#                 entry.setText(format_number(value, 5))
#             else:
#                 entry.setText(str(value))

#             return label, entry

#         if all(
#             isinstance(value, (dict, DictWindow)) for value in self.values()
#         ):
#             tabs = QTabWidget()
#             layout.addWidget(tabs)

#             # Create tab-specific buttons
#             save_button = QPushButton("Save")
#             save_button.clicked.connect(lambda: self.save(False))

#             reset_button = QPushButton("Reset")
#             reset_button.clicked.connect(lambda: self.reset(False))

#             restore_defaults_button = QPushButton("Restore Defaults")
#             restore_defaults_button.clicked.connect(lambda: self.restore_defaults(False))

#             for key, sub_dict in self.items():
#                 tab = QWidget()
#                 tab_layout = QVBoxLayout()
#                 form_layout = QFormLayout()
#                 entries = {}

#                 for sub_key, sub_value in sub_dict.items():
#                     label, entry = conv_items(sub_key, sub_value)
#                     form_layout.addRow(label, entry)
#                     entries[sub_key] = entry

#                 tab_layout.addLayout(form_layout)

#                 # Add tab-specific buttons to each tab
#                 tab_button_layout = QHBoxLayout()
#                 tab_button_layout.addWidget(save_button)
#                 tab_button_layout.addWidget(reset_button)
#                 tab_button_layout.addWidget(restore_defaults_button)
#                 tab_layout.addLayout(tab_button_layout)

#                 tab.setLayout(tab_layout)
#                 tabs.addTab(tab, key)

#                 self.entries.append([key, entries])  # CHANGE: Append to entries list
#         else:
#             form_layout = QFormLayout()
#             entries = {}

#             for key, value in self.items():
#                 label, entry = conv_items(key, value)
#                 form_layout.addRow(label, entry)
#                 entries[key] = entry

#             layout.addLayout(form_layout)



#             self.entries.append([None, entries])  # CHANGE: Append to entries list

#         # Add global buttons
#         global_button_layout = QHBoxLayout()

#         save_all_button = QPushButton("Save All")
#         save_all_button.clicked.connect(lambda: self.save(True))
#         global_button_layout.addWidget(save_all_button)

#         reset_all_button = QPushButton("Reset All")
#         reset_all_button.clicked.connect(lambda: self.reset(True))
#         global_button_layout.addWidget(reset_all_button)

#         restore_all_defaults_button = QPushButton("Restore All Defaults")
#         restore_all_defaults_button.clicked.connect(lambda: self.restore_defaults(True))
#         global_button_layout.addWidget(restore_all_defaults_button)

#         layout.addLayout(global_button_layout)

#         self.qt_window.setLayout(layout)
#         self.qt_window.resize(800, 400)

#         self.qt_window.exec_()

#     def update_internal_dict(self, tab_key=None):
#         """Update the internal dictionary with the current values from the window."""
#         index = self.current_index if tab_key is None else tab_key
#         entries = self.entries[index][1]
#         if super_key := self.entries[index][0] is not None:
#             to_update = self[super_key]
#         else:
#             to_update = self

#         for key, entry in entries.items():
#             value = entry.text()
#             to_update[key] = safe_eval(value)
        
#         return super_key
            

#     def add_option(self):
#         """Add a new option to the window."""
#         dialog = CustomSimpleDialog(
#             self.parent, "Add Option", "Enter the key:", width=50
#         )
#         if dialog.exec_() == QDialog.Accepted:
#             key = dialog.result
#             if key:
#                 index = self.current_index
#                 if self.entries[index][0] is not None:
#                     self[self.entries[index][0]][key] = ""
#                 else:
#                     self[key] = ""
#                 self.window()

#     def save(self, bulk=False):
#         """Save the options of the current tab to the local JSON and update _defaults."""
#         if len(self.entries) == 1:
#             self.update_internal_dict()
#             self._defaults = self.copy()
#         else:
#             index = range(len(self.entries)) if bulk else [self.current_index]
#             for i in index:
#                 key = self.update_internal_dict(i)
#                 self._defaults[key] = self[key].copy()
        
#         self.parent.data.save_local_settings(**{"option_inits": self._defaults})

#     def reset(self, bulk=False):
#         """Reset the options to the default values."""
#         reply = QMessageBox.question(
#             self.parent,
#             "Confirm Reset",
#             "Are you sure you want to reset the options to their initial values?",
#             QMessageBox.Yes | QMessageBox.No,
#             QMessageBox.No,
#         )
#         if reply == QMessageBox.Yes:
#             if len(self.entries) == 1:
#                 self.clear()
#                 self.update(self._defaults.copy())
#             else:
#                 index = range(len(self.entries)) if bulk else [self.current_index]
#                 for i in index:
#                     key = self.entries[i][0]
#                     self[key].clear()
#                     self[key].update(self._defaults[key].copy())
#             self.window()  # Recreate the window to reflect the reset options

#     def restore_defaults(self, bulk=False):
#         """Restore the default settings from the local JSON and update _defaults and current values."""
#         reply = QMessageBox.question(
#             self.parent,
#             "Confirm Reset",
#             "Are you sure you want to reset the options to default?",
#             QMessageBox.Yes | QMessageBox.No,
#             QMessageBox.No,
#         )
#         if reply == QMessageBox.Yes:
            
#             if len(self.entries) == 1:
#                 self._defaults = self.parent.option_inits.copy()
#                 self.update_internal_dict()
#             else:
#                 index = range(len(self.entries)) if bulk else [self.current_index]
#                 key_dict = {self.entries[i][0]: None for i in index}
#                 self.parent.data.restore_option_defaults(key_dict)
                
#                 for i in index:
#                     key = self.entries[i][0]
#                     self._defaults[key] = self.parent.option_inits[key].copy()
#                     self.update_internal_dict(i)

#             self.parent.restore_option_defaults()
#             self.window()  # Recreate the window to reflect the restored options


        
#     def destroy(self):
#         """Destroy the options window."""
#         if self.qt_window is not None and self.qt_window.isVisible():
#             self.qt_window.close()
#             self.qt_window.deleteLater()

#     def _format_name(self, name):
#         """Format the name of the options window."""
#         if isinstance(name, str):
#             return (
#                 name if "option" in name.lower() else f"{name.title()} Options"
#             )
#         else:
#             return "Options"

            # # Add tab-specific buttons
            # tab_button_layout = QHBoxLayout()
            # save_button = QPushButton("Save")
            # save_button.clicked.connect(lambda: self.save(False))
            # tab_button_layout.addWidget(save_button)

            # reset_button = QPushButton("Reset")
            # reset_button.clicked.connect(lambda: self.reset(False))
            # tab_button_layout.addWidget(reset_button)

            # restore_defaults_button = QPushButton("Restore Defaults")
            # restore_defaults_button.clicked.connect(lambda: self.restore_defaults(False))
            # tab_button_layout.addWidget(restore_defaults_button)

            # layout.addLayout(tab_button_layout)

    # def _get_dict(self, source): **{"option_inits": self._defaults}
    #     if self.tabs is not None:
    #         return source[self.tabs.tabText(self.tabs.currentIndex())]
    #     else:
    #         return source

        # self.update_internal_dict(index)
        # key = self.entries[index][0]
        # self.parent.save_local_settings(**{key: self[key]})
        # self._defaults[key] = self[key].copy()
        
        # current_tab = self.tabs.tabText(self.tabs.currentIndex())
        # self.update_internal_dict(current_tab)
        # self.parent.save_local_settings(**{current_tab: self[current_tab]})
        # self._defaults[current_tab] = self[current_tab].copy()

    # def save_all(self):
    #     """Save all options to the local JSON and update _defaults."""
    #     self.update_internal_dict()
    #     self.parent.save_local_settings(**self)
    #     self._defaults = {k: v.copy() for k, v in self.items()}
        

# class DictWindow(dict):
#     """Class to create an options window."""

#     def __init__(self, parent, options, name=None):
#         self.parent = parent
#         if isinstance(options, (dict, DictWindow)) and all(
#             isinstance(value, (dict, DictWindow)) for value in options.values()
#         ):
#             super().__init__({k: v.copy() for k, v in options.items()})
#             self._defaults = {k: v.copy() for k, v in options.items()}
#         elif isinstance(options, (dict, DictWindow)):
#             super().__init__(options.copy())
#             self._defaults = options.copy()
#         else:
#             super().__init__({})
#             self._defaults = {}

#         self.name = self._format_name(name)
#         self.entries = {}
#         self.qt_window = None
#         self.tabs = None

#     @property
#     def defaults(self):
#         """Get the default options."""
#         return self._defaults

#     def window(self):
#         """Create the options window."""
#         self.destroy()

#         self.qt_window = QDialog(self.parent)
#         self.qt_window.setWindowTitle(self.name)
#         self.qt_window.finished.connect(self.destroy)
        
#         layout = QVBoxLayout()

#         def conv_items(key, value):
#             label = QLabel(key)
#             entry = QLineEdit()
#             if isinstance(value, list):
#                 entry.setText(", ".join([format_number(v, 5) for v in value]))
#             elif isinstance(value, (int, float)):
#                 entry.setText(format_number(value, 5))
#             else:
#                 entry.setText(str(value))

#             return label, entry

#         if all(
#             isinstance(value, (dict, DictWindow)) for value in self.values()
#         ):
#             self.tabs = QTabWidget()
#             layout.addWidget(self.tabs)

#             for key, sub_dict in self.items():
#                 tab = QWidget()
#                 tab_layout = QVBoxLayout()
#                 form_layout = QFormLayout()
#                 self.entries[key] = {}

#                 for sub_key, sub_value in sub_dict.items():
#                     label, entry = conv_items(sub_key, sub_value)
#                     form_layout.addRow(label, entry)
#                     self.entries[key][sub_key] = entry

#                 tab_layout.addLayout(form_layout)
#                 tab.setLayout(tab_layout)
#                 self.tabs.addTab(tab, key)
#         else:
#             form_layout = QFormLayout()
#             self.entries = {}

#             for key, value in self.items():
#                 label, entry = conv_items(key, value)
#                 form_layout.addRow(label, entry)
#                 self.entries[key] = entry

#             layout.addLayout(form_layout)

#         button_layout = QHBoxLayout()

#         save_button = QPushButton("Save")
#         save_button.clicked.connect(self.save)
#         button_layout.addWidget(save_button)

#         add_button = QPushButton("Add")
#         add_button.clicked.connect(self.add_option)
#         button_layout.addWidget(add_button)

#         reset_button = QPushButton("Reset")
#         reset_button.clicked.connect(self.reset)
#         button_layout.addWidget(reset_button)

#         layout.addLayout(button_layout)
#         self.qt_window.setLayout(layout)
#         self.qt_window.resize(800, 400)

#         self.qt_window.exec_()

#     def add_option(self):
#         """Add a new option to the window."""
#         dialog = CustomSimpleDialog(
#             self.parent, "Add Option", "Enter the key:", width=50
#         )
#         if dialog.exec_() == QDialog.Accepted:
#             key = dialog.result
#             if key:
#                 if self.tabs is not None:
#                     current_tab = self.tabs.tabText(self.tabs.currentIndex())
#                     if current_tab:
#                         self[current_tab][key] = ""
#                 else:
#                     self[key] = ""
#                 self.window()

#     def save(self):
#         """Save the options to the dictionary."""

#         def eval_value(value):
#             if value.lower() == "none":
#                 return None
#             elif value.lower() == "true":
#                 return True
#             elif value.lower() == "false":
#                 return False
#             else:
#                 try:
#                     val = eval(value, {}, {"inf": np.inf})
#                     if isinstance(val, float) and val.is_integer():
#                         return int(val)
#                     return val
#                 except (NameError, SyntaxError):
#                     return value

#         for key, entry in self._get_dict(self.entries).items():
#             value = entry.text()
#             to_update = self._get_dict(self)
#             if "," in value and "{" in value:
#                 to_update[key] = eval_value(value)
#             elif "," in value:
#                 to_update[key] = [
#                     eval_value(v.strip()) for v in value.split(",")
#                 ]
#             else:
#                 to_update[key] = eval_value(value)

#         # self.qt_window.accept()

#     def reset(self):
#         """Reset the options to the default values."""
#         reply = QMessageBox.question(
#             self.parent,
#             "Confirm Reset",
#             "Are you sure you want to reset to default options?",
#             QMessageBox.Yes | QMessageBox.No,
#             QMessageBox.No,
#         )

#         if reply == QMessageBox.Yes:
#             to_reset = self._get_dict(self)
#             to_reset.clear()
#             to_reset.update(self._get_dict(self._defaults.copy()))
#             self.window()  # Recreate the window to reflect the reset options

#     def destroy(self):
#         """Destroy the options window."""
#         if self.qt_window is not None and self.qt_window.isVisible():
#             self.qt_window.close()
#             self.qt_window.deleteLater()

#     def _format_name(self, name):
#         """Format the name of the options window."""
#         if isinstance(name, str):
#             return (
#                 name if "option" in name.lower() else f"{name.title()} Options"
#             )
#         else:
#             return "Options"

#     def _get_dict(self, source):
#         if self.tabs is not None:
#             return source[self.tabs.tabText(self.tabs.currentIndex())]
#         else:
#             return source


class DataTreeWindow:
    def __init__(
        self,
        root,
        df,
        tree_cols,
        add_row_callback=None,
        use_row_callback=None,
        **kwargs,
    ):
        self.root = root
        self.window = None
        self.tabs = None
        self.tree1 = None  # Initialize tree attribute
        self.tree2 = None  # Initialize tree attribute
        self.name = kwargs.get("name", "Pinned Results")
        self.buttons = []

        # Initialize properties
        self.df = df
        self.tree_cols = tree_cols

        # Set the callback functions
        self.add_row_callback = add_row_callback
        self.use_row_callback = use_row_callback
        self.graphing_callback = kwargs.get("graphing_callback", None)

        self.df_base_cols = kwargs.get("df_base_cols", [])
        self.df_sort_cols = kwargs.get("df_sort_cols", [])
        self.tree_gr_cols = kwargs.get("tree_gr_cols", [])

        self.wide_cols = kwargs.get("wide_cols", [])
        self.narrow_cols = kwargs.get("narrow_cols", [])

    @property
    def df(self):
        if not hasattr(self, "_df"):
            self._df = pd.DataFrame()
        return self._df

    @df.setter
    def df(self, values):
        if isinstance(values, pd.DataFrame):
            self._df = values
        elif isinstance(values, list) and all(
            isinstance(col, str) for col in values
        ):
            self._df = pd.DataFrame(columns=values)

    @property
    def df_base_cols(self):
        if not hasattr(self, "_df_base_cols"):
            self._df_base_cols = []
        return self._df_base_cols

    @df_base_cols.setter
    def df_base_cols(self, values):
        if isinstance(values, list) and all(
            isinstance(col, str) for col in values
        ):
            self._df_base_cols = values

    @property
    def df_sort_cols(self):
        if not hasattr(self, "_df_sort_cols"):
            self._df_sort_cols = []
        return self._df_sort_cols

    @df_sort_cols.setter
    def df_sort_cols(self, values):
        if isinstance(values, list) and all(
            isinstance(col, str) for col in values
        ):
            self._df_sort_cols = values

    @property
    def tree_cols(self):
        if not hasattr(self, "_tree_cols"):
            return self.df.columns.tolist()
        return self._tree_cols

    @tree_cols.setter
    def tree_cols(self, columns):
        if isinstance(columns, list) and all(
            isinstance(col, str) for col in columns
        ):
            self._tree_cols = columns
        elif isinstance(columns, str):
            self._tree_cols.append(columns)

    @property
    def tree_gr_cols(self):
        if not hasattr(self, "_tree_gr_cols"):
            self._tree_gr_cols = []
        return self._tree_gr_cols

    @tree_gr_cols.setter
    def tree_gr_cols(self, columns):
        if isinstance(columns, list) and all(
            isinstance(col, str) for col in columns
        ):
            self._tree_gr_cols = [
                col for col in columns if col in self.tree_cols
            ]
        elif isinstance(columns, str) and columns in self.tree_cols:
            if not hasattr(self, "_tree_gr_cols"):
                self._tree_gr_cols = []
            self._tree_gr_cols.append(columns)

    @property
    def wide_cols(self):
        if not hasattr(self, "_wide_cols"):
            return []
        return self._wide_cols

    @wide_cols.setter
    def wide_cols(self, columns):
        if isinstance(columns, list) and all(
            isinstance(col, str) for col in columns
        ):
            self._wide_cols = [col for col in columns if col in self.tree_cols]
        elif isinstance(columns, str) and columns in self.tree_cols:
            if not hasattr(self, "_wide_cols"):
                self._wide_cols = []
            self._wide_cols.append(columns)

    @property
    def narrow_cols(self):
        if not hasattr(self, "_narrow_cols"):
            return []
        return self._narrow_cols

    @narrow_cols.setter
    def narrow_cols(self, columns):
        if isinstance(columns, list) and all(
            isinstance(col, str) for col in columns
        ):
            self._narrow_cols = [
                col for col in columns if col in self.tree_cols
            ]
        elif isinstance(columns, str) and columns in self.tree_cols:
            if not hasattr(self, "_narrow_cols"):
                self._narrow_cols = []
            self._narrow_cols.append(columns)

    def show(self):
        """View the pinned results in a new window."""
        self.window = QMainWindow(self.root)
        self.window.setWindowTitle(self.name)

        # Create a central widget and set it as the central widget of the QMainWindow
        central_widget = QWidget(self.window)
        self.window.setCentralWidget(central_widget)

        # Create a layout for the central widget
        layout = QVBoxLayout(central_widget)

        # Create a tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # ----------------- Create the TreeView -----------------
        tree_tab = QWidget()
        tree_layout = QVBoxLayout(tree_tab)

        # Create a TreeView to display the fit results
        self.tree1 = QTreeWidget()
        self.tree1.setColumnCount(len(self.tree_cols) + 1)
        self.tree1.setHeaderLabels([" "] + self.tree_cols)
        self.tree1.setSelectionMode(QAbstractItemView.ExtendedSelection)

        # Set column widths
        window_width = 0
        self.tree1.setColumnWidth(0, 5)
        for i, col in enumerate(self.tree_cols):
            if col in self.narrow_cols:
                self.tree1.setColumnWidth(i + 1, 50)
                window_width += 50
            elif col in self.wide_cols:
                self.tree1.setColumnWidth(i + 1, 500)
                window_width += 500
            else:
                if col in self.df.columns and not self.df[col].empty:
                    # Calculate the width of the longest string in the column
                    font_metrics = QFontMetrics(self.tree1.font())
                    max_width = (
                        self.df[col]
                        .astype(str)
                        .apply(font_metrics.width)
                        .max()
                    )
                    max_width *= 1.3
                    self.tree1.setColumnWidth(
                        i + 1, int(max_width)
                    )  # Add some padding
                    window_width += int(max_width)
                else:
                    self.tree1.setColumnWidth(
                        i + 1, 100
                    )  # Set default width to 100
                    window_width += 100

        self.window.resize(max(window_width, 1200), 600)

        tree_layout.addWidget(self.tree1)
        self.tabs.addTab(tree_tab, "Pinned Results")

        # ----------------- Create the DataFrame view -----------------
        data_tab = QWidget()
        data_layout = QVBoxLayout(data_tab)

        # Create a TreeView to display the DataFrame
        self.tree2 = QTreeWidget()
        self.tree2.setColumnCount(len(self.df.columns) + 1)
        self.tree2.setHeaderLabels([" "] + self.df.columns.tolist())

        self.tree2.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree2.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.tree2.setEditTriggers(QAbstractItemView.DoubleClicked)

        self.tree2.setColumnWidth(0, 5)

        data_layout.addWidget(self.tree2)
        self.tabs.addTab(data_tab, "DataFrame View")

        def calc_width(num, width, frac):
            if frac < 1:
                frac = int(1 / frac)
            return int(((frac + 1) * num + 1) * width / frac)

        # Create a frame for the buttons
        button_frame = QFrame(central_widget)
        button_layout = QHBoxLayout(button_frame)

        b_width = 100
        # Create frames for grouping buttons
        edit_frame = QFrame(button_frame)
        edit_frame.setFixedWidth(calc_width(2, b_width, 4))
        edit_layout = QHBoxLayout(edit_frame)
        button_layout.addWidget(edit_frame)

        action_frame = QFrame(button_frame)
        action_frame.setFixedWidth(calc_width(3, b_width, 4))
        action_layout = QHBoxLayout(action_frame)
        button_layout.addWidget(action_frame)

        navigation_frame = QFrame(button_frame)
        navigation_frame.setFixedWidth(calc_width(2, b_width, 4))
        navigation_layout = QHBoxLayout(navigation_frame)
        button_layout.addWidget(navigation_frame)

        # Initialize the buttons list
        self.buttons = []

        # Add buttons to the edit frame
        add_button = QPushButton("Add", edit_frame)
        add_button.setFixedWidth(b_width)
        add_button.clicked.connect(self.add_row)
        if self.add_row_callback is None:
            add_button.setDisabled(True)
            add_button.setStyleSheet("color: grey;")
        edit_layout.addWidget(add_button)
        self.buttons.append(add_button)

        remove_button = QPushButton("Remove", edit_frame)
        remove_button.setFixedWidth(b_width)
        remove_button.clicked.connect(self.remove_rows)
        edit_layout.addWidget(remove_button)
        self.buttons.append(remove_button)

        # Add buttons to the action frame
        plot_button = QPushButton("Plot", action_frame)
        plot_button.setFixedWidth(b_width)
        plot_button.clicked.connect(self.plot_rows)
        action_layout.addWidget(plot_button)
        self.buttons.append(plot_button)

        clear_button = QPushButton("Clear", action_frame)
        clear_button.setFixedWidth(b_width)
        clear_button.clicked.connect(self.unplot_rows)
        action_layout.addWidget(clear_button)
        self.buttons.append(clear_button)

        use_row_button = QPushButton("Use Values", action_frame)
        use_row_button.setFixedWidth(b_width)
        use_row_button.clicked.connect(self.use_row)
        if self.use_row_callback is None:
            use_row_button.setDisabled(True)
            use_row_button.setStyleSheet("color: grey;")
        action_layout.addWidget(use_row_button)
        self.buttons.append(use_row_button)

        # Add navigation buttons to the navigation frame
        up_button = QPushButton("Move Up", navigation_frame)
        up_button.setFixedWidth(b_width)
        up_button.clicked.connect(self.move_up)
        navigation_layout.addWidget(up_button)
        self.buttons.append(up_button)

        down_button = QPushButton("Move Down", navigation_frame)
        down_button.setFixedWidth(b_width)
        down_button.clicked.connect(self.move_down)
        navigation_layout.addWidget(down_button)
        self.buttons.append(down_button)

        layout.addWidget(button_frame)

        self.tree1.itemDoubleClicked.connect(self.edit_item)
        self.item_changed_connected = False

        self._refresh()
        # Connect the tab change signal to a slot
        self.tabs.currentChanged.connect(self.on_tab_changed)

        self.window.show()

    def on_tab_changed(self, index):
        """Enable or disable buttons based on the selected tab."""
        if self.tabs.tabText(index) == "DataFrame View":
            for button in self.buttons:
                button.setEnabled(False)
                button.setStyleSheet("color: grey;")
        else:
            for button in self.buttons:
                button.setEnabled(True)
                button.setStyleSheet("color: black;")
        self._refresh()

    def edit_item(self, item, column):
        """Make the item editable on double-click."""
        # self._refresh()

        if self.tree_cols[column - 1] in self.tree_gr_cols:
            return
        try:
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            self.tree1.editItem(item, column)

            self.item_changed_connected = True
            self.tree1.itemChanged.connect(self._update_dataframe)
            self.tree1.itemSelectionChanged.connect(self._refresh)
        except RuntimeError as exc:
            QMessageBox.critical(self.root, "Error", f"Error in editing of item: {exc}.")
            pass

    def _update_dataframe(self, item, column):
        """Update the DataFrame when an item is edited."""
        item_index = self.tree1.indexOfTopLevelItem(item)
        new_value = item.text(column)
        self.df.loc[item_index, self.tree_cols[column - 1]] = new_value

        self._refresh()

    def _refresh(self):
        """Refresh the pinned results in the treeview."""
        # Refresh the treeview col.replace(self.df_sort_cols[0], '')
        self.tree1.clear()
        self.tree2.clear()

        self.tree2.setColumnCount(len(self.df.columns) + 1)
        self.tree2.setHeaderLabels([" "] + self.df.columns.tolist())

        self._drop_nan_columns()

        for index, (_, row) in enumerate(self.df.iterrows()):
            display_row = [str(index)] + [
                str(row[col]) if col in row else "" for col in self.tree_cols
            ]
            for gr_col in self.tree_gr_cols:
                # Create a dict from the filtered columns using the prefix parts of the columns
                filtered_dict = {
                    col.replace(f"_{gr_col.lower()}", ""): format_number(
                        row[col], 2
                    )
                    for col in row.index
                    if col.endswith(f"_{gr_col.lower()}")
                    and not pd.isnull(row[col])
                    and row[col]
                }
                # Convert the dict to a string representation
                joined_values = str(filtered_dict)
                display_row[self.tree_cols.index(gr_col) + 1] = joined_values
            item1 = QTreeWidgetItem(display_row)
            item2 = QTreeWidgetItem(
                [str(index)] + [format_number(col, 5) for col in row]
            )
            if index % 2 == 0:
                for col in range(item1.columnCount()):
                    item1.setBackground(col, QBrush(QColor("#E2DDEF")))
                for col in range(item2.columnCount()):
                    item2.setBackground(col, QBrush(QColor("#E2DDEF")))
            self.tree1.addTopLevelItem(item1)
            self.tree2.addTopLevelItem(item2)

        if self.item_changed_connected:
            self.tree1.itemChanged.disconnect(self._update_dataframe)
            self.tree1.itemSelectionChanged.disconnect(self._refresh)
            self.item_changed_connected = False

    def _drop_nan_columns(self):
        """Helper function to drop all NaN columns of self.df."""
        # Get all columns that end with self.df_sort_cols[0]
        if self.df_sort_cols:
            base_to_drop = [
                col.replace("_" + self.df_sort_cols[0], "")
                for col in self.df.columns
                if col.endswith(self.df_sort_cols[0])
                and self.df[col].isna().all()
            ]
            cols_to_drop = [
                base_col + "_" + suffix
                for suffix in self.df_sort_cols
                for base_col in base_to_drop
                if base_col + "_" + suffix in self.df.columns
            ]
            self.df = self.df.drop(columns=cols_to_drop)

    def move_up(self):
        """Move the selected item up in the DataFrame and refresh the Treeview."""
        selected_items = self.tree1.selectedItems()
        if selected_items:
            indices = [
                self.tree1.indexOfTopLevelItem(item) for item in selected_items
            ]
            if min(indices) > 0:
                for index in sorted(indices):
                    # Swap rows in the DataFrame
                    self.df.iloc[[index, index - 1]] = self.df.iloc[
                        [index - 1, index]
                    ]
                self._refresh()
                # Reselect the moved items
                new_selection = [
                    self.tree1.topLevelItem(index - 1) for index in indices
                ]
                for item in new_selection:
                    item.setSelected(True)

    def move_down(self):
        """Move the selected item down in the DataFrame and refresh the Treeview."""
        selected_items = self.tree1.selectedItems()
        if selected_items:
            indices = [
                self.tree1.indexOfTopLevelItem(item) for item in selected_items
            ]
            if max(indices) < self.tree1.topLevelItemCount() - 1:
                for index in sorted(indices, reverse=True):
                    # Swap rows in the DataFrame
                    self.df.iloc[[index, index + 1]] = self.df.iloc[
                        [index + 1, index]
                    ]
                self._refresh()
                # Reselect the moved items
                new_selection = [
                    self.tree1.topLevelItem(index + 1) for index in indices
                ]
                for item in new_selection:
                    item.setSelected(True)

    def append_df(self, values):
        """Utility function to assist in adding data to the internal DataFrame."""

        def col_sort(column_name):
            """Custom sort key that prioritizes n1, n2, and then reverse alphabetical order."""
            match = re.match(r"(^[a-zA-Z]+)_?([0-9_]+)(_std)?$", column_name)
            if match:
                a1 = len(match.group(1))
                a2 = len(match.group(3)) if match.group(3) is not None else 0
                n1, n2 = (
                    map(int, match.group(2).split("_"))
                    if "_" in match.group(2)
                    else (int(match.group(2)), 0)
                )
                return (a2, n1, n2, a1)
            return (0, 0, 0, column_name)

        def safe_eval(x):
            try:
                return eval(x, {}, {"inf": np.inf})
            except (SyntaxError, TypeError, ValueError, NameError):
                return x

        # Ensure df_base_cols are in the keys
        if isinstance(values, pd.Series):
            values = values.to_frame().T
        elif isinstance(values, dict):
            values = pd.DataFrame([values])

        if isinstance(values, pd.DataFrame):
            if not all(col in values.columns for col in self.df_base_cols):
                print(values.columns)
                raise ValueError(
                    "All df_base_cols must be in the DataFrame columns"
                )
            value_cols = values.columns.difference(self.df_base_cols)
            values[value_cols] = values[value_cols].map(safe_eval)

            # Sort columns based on df_sort_cols
            sort_base = []
            ind = 0
            while sort_base == [] and ind < len(self.df_sort_cols):
                sort_base = [
                    col.replace("_" + self.df_sort_cols[ind], "")
                    for col in values.columns
                    if self.df_sort_cols[ind] in col
                ]
                ind += 1
            # sort_base = [col.replace("_"+self.df_sort_cols[0], '') for col in self.df.columns if self.df_sort_cols[0] in col]
            sort_base = list(sorted(reversed(sorted(sort_base)), key=col_sort))
            if all(col in values.columns for col in sort_base):
                for base in sort_base:
                    values[f"{base}_{self.df_sort_cols[max(0,ind-2)]}"] = (
                        values[base]
                    )
                values.drop(columns=sort_base, inplace=True)

            self._df = pd.concat([self._df, values], ignore_index=True)
        else:
            raise ValueError(
                "Unsupported data type for appending to DataFrame"
            )

        # # Sort columns based on df_sort_cols
        sort_base = [
            col.replace("_" + self.df_sort_cols[0], "")
            for col in self.df.columns
            if self.df_sort_cols[0] in col
        ]
        sort_base = list(sorted(reversed(sorted(sort_base)), key=col_sort))

        sorted_columns = []
        for identifier in self.df_sort_cols:
            sorted_columns += [f"{base}_{identifier}" for base in sort_base]

        other_cols = self.df.columns.difference(
            self.df_base_cols + sorted_columns
        ).tolist()

        self._df = self._df[self.df_base_cols + other_cols + sorted_columns]
        self._df[self.df_base_cols + other_cols] = self._df[
            self.df_base_cols + other_cols
        ].fillna("")

    def add_row(self):
        """Add a new pinned result to the treeview."""
        if self.add_row_callback:
            self.add_row_callback()
            self._refresh()
        else:
            QMessageBox.warning(
                self.root,
                "No Callback",
                "A callback is needed for this function.",
            )

    def add_row_popup(self, default_name):
        """Add a new pinned result to the treeview."""

        # Get the desired name
        dialog = CustomSimpleDialog(
            self.root,
            "Enter Name",
            "Enter the name for the pinned result:",
            initialvalue=default_name,
            width=max(50, min(500, len(default_name) * 10)),
        )
        if dialog.exec_() == QDialog.Rejected:
            return
        return dialog.result

    def use_row(self):
        """Use the values of a pinned result."""
        if self.use_row_callback:
            selected_items = self.tree1.selectedItems()
            if not selected_items:
                QMessageBox.warning(self.root, "Warning", "No row selected.")
                return

            # Get the first selected row
            selected_item = selected_items[0]
            item_index = self.tree1.indexOfTopLevelItem(selected_item)

            # Get the corresponding row from the DataFrame
            selected_row = self.df.iloc[item_index].to_dict()

            self.use_row_callback(selected_row)
        else:
            QMessageBox.warning(
                self.root,
                "No Callback",
                "A callback is needed for this function.",
            )

    def plot_rows(self):
        """Plot the selected pinned results."""
        selected_items = self.tree1.selectedItems()
        if not selected_items:
            # Clear the existing plots
            self._refresh()
            return

        format_strings = [
            style + color
            for style in ["--", "-.", ":"]
            for color in ["g", "c", "m", "k"]
        ]
        used_format_strings = self.df["Show"].tolist()
        format_usage_count = {
            fmt: used_format_strings.count(fmt) for fmt in format_strings
        }

        for item in selected_items:
            # Find the least used format string
            min_usage = min(format_usage_count.values())
            for fmt in format_strings:
                if format_usage_count[fmt] == min_usage:
                    format_string = fmt
                    break

            # Update the format usage count
            format_usage_count[format_string] += 1

            self.df.loc[self.df["Name"] == item.text(1), "Show"] = (
                format_string
            )
            used_format_strings.append(format_string)

        # Refresh the treeview
        self._refresh()

        # Update plots
        if self.graphing_callback:
            self.graphing_callback()

    def unplot_rows(self):
        """Remove the selected pinned results from the plots."""
        selected_items = self.tree1.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            # Update the "Show" column to None or ""
            self.df.loc[self.df["Name"] == item.text(1), "Show"] = ""

        # Refresh the treeview
        self._refresh()

        # Update plots
        if self.graphing_callback:
            self.graphing_callback()

    def remove_rows(self):
        """Remove the selected pinned results from the treeview."""
        selected_items = self.tree1.selectedItems()

        if not selected_items:
            return

        selected_item_info = [
            f"Index: {self.tree1.indexOfTopLevelItem(item)}, Name: {item.text(1)}"
            for item in selected_items
        ]
        message = (
            "Are you sure you want to remove the following rows?\n\n"
            + "\n".join(selected_item_info)
        )
        reply = QMessageBox.question(
            self.root,
            "Remove Rows",
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.No:
            return

        for item in selected_items:

            self.df = self.df.drop(
                self.df[
                    (self.df.index == int(item.text(0)))
                    & (self.df["Name"] == item.text(1))
                ].index
            )

            tree_index = self.tree1.indexOfTopLevelItem(item)
            self.tree1.takeTopLevelItem(tree_index)

        # Refresh the treeview
        self._refresh()

        # Update plots
        if self.graphing_callback:
            self.graphing_callback()


class CalcRCWindow:
    def __init__(self, main, root):
        self.main = main
        self.root = root
        self.window = None
        self.r_entry = None
        self.c_entry = None
        self.freq_entry = None

    def show(self):
        """Show the window to calculate R, C, or freq."""
        self.window = QDialog(self.root)
        self.window.setModal(False)
        self.window.setWindowTitle("Calculate RC")

        layout = QVBoxLayout(self.window)

        form_layout = QFormLayout()
        self.r_entry = QLineEdit()
        form_layout.addRow("R:", self.r_entry)

        self.c_entry = QLineEdit()
        form_layout.addRow("C:", self.c_entry)

        self.freq_entry = QLineEdit()
        form_layout.addRow("freq:", self.freq_entry)

        layout.addLayout(form_layout)

        calc_button = QPushButton("Calculate", self.window)
        calc_button.clicked.connect(self.calculate_rc_freq)
        layout.addWidget(calc_button)

        self.window.setLayout(layout)
        # self.window.exec_()
        self.window.show()

    def calculate_rc_freq(self):
        """Calculate R, C, or freq based on the other two values."""
        try:
            # Get the parameter names and values
            params = {p.name: p.values[0] for p in self.main.parameters}

            # Helper function to get the value from the entry or parameter
            def get_value(entry_value):
                try:
                    return (
                        float(eval(entry_value, params, {"inf": np.inf}))
                        if entry_value != ""
                        else None
                    )
                except (ValueError, SyntaxError):
                    return None

            # Get the values from the entries
            r_value = get_value(self.r_entry.text())
            c_value = get_value(self.c_entry.text())
            freq_value = get_value(self.freq_entry.text())

            if r_value is None:
                if c_value is not None and freq_value is not None:
                    res = 1 / (2 * np.pi * freq_value * c_value)
                    self.r_entry.setText(format_number(res, 6))
                else:
                    raise ValueError(
                        "Please provide values for C and freq to calculate R."
                    )
            elif c_value is None:
                if r_value is not None and freq_value is not None:
                    res = 1 / (2 * np.pi * freq_value * r_value)
                    self.c_entry.setText(format_number(res, 6))
                else:
                    raise ValueError(
                        "Please provide values for R and freq to calculate C."
                    )
            elif freq_value is None:
                if r_value is not None and c_value is not None:
                    res = 1 / (2 * np.pi * r_value * c_value)
                    self.freq_entry.setText(format_number(res, 6))
                else:
                    raise ValueError(
                        "Please provide values for R and C to calculate freq."
                    )
            else:
                raise ValueError(
                    "Please leave one field empty to calculate its value."
                )

        except ValueError as e:
            QMessageBox.critical(self.root, "Error", str(e))


class MultiEntryWindow(MultiEntryManager):
    def __init__(
        self,
        root,
        entries=None,
        num_entries=1,
        has_value=True,
        has_check=True,
        callbacks=None,
    ):
        super().__init__(None, entries, num_entries, has_value, has_check)
        self.root = root
        self.window = None
        self.callbacks = callbacks if isinstance(callbacks, dict) else {}

    def show(self):
        """Create and show the multi-entry window."""
        self.window = QDialog(self.root)
        self.window.setWindowTitle("Add Entries")
        self.window.finished.connect(self.save)
        self.window.setMinimumSize(200, 100)

        main_layout = QVBoxLayout(self.window)

        self.frame = QFrame(self.window)

        self.reset_frame(self.frame)

        # Create a frame for the buttons
        button_frame = QFrame(self.window)
        button_layout = QVBoxLayout(button_frame)

        # Create frames for grouping buttons
        base_frame = QFrame(button_frame)
        base_layout = QHBoxLayout(base_frame)
        button_layout.addWidget(base_frame)

        # Create buttons
        save_button = QPushButton("Save", base_frame)
        save_button.clicked.connect(self.save)
        base_layout.addWidget(save_button)

        clear_button = QPushButton("Clear", base_frame)
        clear_button.clicked.connect(self.clear)
        base_layout.addWidget(clear_button)

        if self.callbacks:
            callback_frame = QFrame(button_frame)
            callback_layout = QVBoxLayout(callback_frame)
            button_layout.addWidget(callback_frame)

            for key in self.callbacks.keys():
                if key.startswith("button_"):
                    name = key.replace("button_", "").replace("_", " ").title()
                    button = QPushButton(name, callback_frame)
                    button.clicked.connect(self.callbacks[key])
                    callback_layout.addWidget(button)

        main_layout.addWidget(self.frame)
        main_layout.addWidget(button_frame)

        self.window.setLayout(main_layout)
        self.window.exec_()

    def save(self):
        """Save the entries to the parameters and verify their validity."""

        if self.callbacks.get("save"):
            res = self.callbacks["save"](np.array(self.values))
            if res is not None:
                self.values = res

        self.destroy()
        # Destroy entry window if it exists
        if self.window:
            self.window.close()
            self.window = None
        if self.frame:
            self.frame.close()
            self.frame = None

    def clear(self):
        """Clear the entries from the parameters."""
        clear_callback = self.callbacks.get("clear", None)
        checked = self.callbacks.get("clear_flag", "unchecked")
        defaults = None
        if callable(clear_callback):
            defaults = clear_callback(self.values)

        if defaults is None:
            defaults = np.array(
                [entry.default_values for entry in self.entries]
            )
        if checked is True or checked == "checked":
            self.checked_values = defaults
        elif checked is False or checked == "unchecked":
            self.unchecked_values = defaults
        else:
            self.values = defaults


class DataViewer(QMainWindow):
    def __init__(self, data, parent=None, name=None):
        super().__init__(parent)
        self.data = data
        self.parent = parent
        self.name = name
        self.initUI()

    def initUI(self):
        self.setWindowTitle(
            f"Data Viewer - {self.name}" if self.name else "Data Viewer"
        )
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        self.table = QTableWidget()
        layout.addWidget(self.table)
        self.setCentralWidget(central_widget)

        self.setStyleSheet(
            "background-color: #f0f0f0;"
        )  # Set overall background color

        self.populate_table(self.data)
        self.table.cellDoubleClicked.connect(self.get_value)
        self.table.itemChanged.connect(self.set_value)
        self.table.installEventFilter(self)
        self.show()

    def populate_table(self, data):
        if isinstance(data, (np.ndarray, pd.DataFrame)):
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data)
            self.table.setRowCount(data.shape[0])
            self.table.setColumnCount(data.shape[1])
            self.table.setHorizontalHeaderLabels(
                [str(col) for col in data.columns]
            )
            for row in range(data.shape[0]):
                for col in range(data.shape[1]):
                    item = QTableWidgetItem(str(data.iloc[row, col]))
                    item.setBackground(QColor(240, 255, 255))
                    self.table.setItem(row, col, item)
            self.table.setVerticalHeaderLabels(
                [str(i) for i in range(data.shape[0])]
            )
            self.table.horizontalHeader().setStretchLastSection(
                False
            )  # Do not stretch the last column
        else:
            self.table.setRowCount(len(data))
            self.table.setColumnCount(2)
            if isinstance(data, (list, tuple, set)):
                data = {str(i): v for i, v in enumerate(data)}
                self.table.setHorizontalHeaderLabels(["Index", "Value"])
            else:
                self.table.setHorizontalHeaderLabels(["Key", "Value"])

            for row, (key, value) in enumerate(data.items()):
                key_item = QTableWidgetItem(str(key))
                value_item = QTableWidgetItem(str(value))
                key_item.setBackground(
                    QColor(0, 255, 255)
                )  # Set background color for key column
                value_item.setBackground(QColor(240, 255, 255))
                self.table.setItem(row, 0, key_item)
                self.table.setItem(row, 1, value_item)
            self.table.setVerticalHeaderLabels(
                [str(i) for i in range(len(data))]
            )
            self.table.horizontalHeader().setStretchLastSection(
                True
            )  # Stretch value column to fill width

    def get_value(self, row, column):
        if isinstance(self.data, np.ndarray):
            if len(self.data.shape) > 1:
                value = self.data[row, column]
            else:
                value = self.data[row]
        elif isinstance(self.data, pd.DataFrame):
            value = self.data.iloc[row, column]
        else:
            label = self.table.horizontalHeaderItem(0).text()
            key = self.table.item(row, 0).text()
            if label == "Key":
                value = self.data[key]
            else:
                value = list(self.data)[int(key)]
            if not isinstance(value, str) and hasattr(value, "__iter__"):
                self.viewer = DataViewer(
                    value, self, name=key
                )  # Use self.viewer and pass name
                self.viewer.show()

    def set_value(self, item):
        if isinstance(self.data, np.ndarray):
            if len(self.data.shape) > 1:
                dtype = type(self.data[item.row(), item.column()])
                self.data[item.row(), item.column()] = dtype(item.text())
            else:
                dtype = type(self.data[item.row()])
                self.data[item.row()] = dtype(item.text())
        elif isinstance(self.data, pd.DataFrame):
            dtype = type(self.data.iloc[item.row(), item.column()])
            self.data.iloc[item.row(), item.column()] = dtype(item.text())
        else:
            if item.column() == 0:
                return
            key = self.table.item(item.row(), 0).text()
            if self.table.horizontalHeaderItem(0).text() != "Key":
                key = int(key)
            dtype = (
                type(self.data[key])
                if not isinstance(self.data, set)
                else type(list(self.data)[key])
            )
            if dtype not in (int, float, str, np.number):
                return
            if isinstance(self.data, set):
                data_list = list(self.data)
                data_list[key] = dtype(item.text())
                self.data = set(data_list)
            else:
                self.data[key] = dtype(item.text())

        if self.parent and isinstance(self.parent, DataViewer) and self.name:
            key = self.name
            if self.parent.table.horizontalHeaderItem(0).text() != "Key":
                key = int(key)
            self.parent.data[key] = self.data  # Update parent data

    def closeEvent(self, event):
        # Handle the close event to ensure the application closes properly
        if self.parent and isinstance(self.parent, DataViewer):
            self.parent.refresh_table()  # Refresh parent table on close
        event.accept()
        self.deleteLater()  # Ensure the widget is properly deleted

    def refresh_table(self):
        self.populate_table(self.data)

    def eventFilter(self, source, event):
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Delete:
            if source is self.table:
                selected_rows = self.table.selectionModel().selectedRows()
                if selected_rows:
                    # for row in selected_rows:
                    #     self.delete_row(row.row())
                    # self.delete_row(selected_rows[0].row())
                    self.delete_row([row.row() for row in selected_rows])
                return True
        return super().eventFilter(source, event)

    def delete_row(self, rows):
        reply = QMessageBox.question(
            self,
            "Delete Row(s)",
            "Are you sure you want to delete data?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply == QMessageBox.Yes:
            for row in sorted(rows, reverse=True):
                if isinstance(self.data, np.ndarray):
                    self.data = np.delete(self.data, row, axis=0)
                elif isinstance(self.data, pd.DataFrame):
                    self.data = self.data.drop(self.data.index[row]).reset_index(drop=True)
                else:
                    key = self.table.item(row, 0).text()
                    if self.table.horizontalHeaderItem(0).text() != "Key":
                        key = int(key)
                    del self.data[key]
                self.refresh_table()

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class StrItem:
    string: str
    index: int = 0
    sub_strings: List[str] = field(default_factory=list)
    info: Dict = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.sub_strings, str):
            self.sub_strings = [self.sub_strings]
        elif not isinstance(self.sub_strings, (list, tuple)):
            self.sub_strings = []
        
        if not self.sub_strings:
            self.sub_strings = [self.string]

        if not isinstance(self.info, dict):
            self.info = {}

# needs update_entries, highlight, clear, checked_names, show
class ListFrame(QFrame):
    """
    A class to provide list interaction via a QListWidget.
    """

    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.strings = []
        self.special_items = []
        self.checked_items = []
        self.str_index = 0
        self.changed = False
        self.set_strings(data, False, False)

        self.initUI()

    def initUI(self):

        self.layout = QVBoxLayout()
        
        self.setLayout(self.layout)

        # List Widget for items
        self.items_list = QListWidget()
        self.items_list.setDragDropMode(QAbstractItemView.InternalMove)  # Enable drag-and-drop reordering
        self.items_list.installEventFilter(self)
        self.items_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.items_list.setEditTriggers(QAbstractItemView.DoubleClicked)
        self.items_list.itemChanged.connect(self.on_item_changed)
        self.items_list.itemDoubleClicked.connect(lambda item: self.highlight(item.text()))

        self.layout.addWidget(self.items_list)

        self.combine_button = QPushButton("Combine")
        self.combine_button.clicked.connect(self.combine_items)
        self.combine_button.setFixedWidth(75)

        self.rename_button = QPushButton("Rename")
        self.rename_button.clicked.connect(self.rename_item)
        self.rename_button.setFixedWidth(75)

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save)
        self.save_button.setFixedWidth(75)

        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.addWidget(self.combine_button)
        self.buttons_layout.addWidget(self.rename_button)
        self.buttons_layout.addWidget(self.save_button)

        self.layout.addLayout(self.buttons_layout)

        if self.items_list.count() == 0:
            self.populate_list()

    def eventFilter(self, source, event):
        """Handles key press events for deleting items."""
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Delete:
            if source is self.items_list:
                for item in self.items_list.selectedItems():
                    self.items_list.takeItem(self.items_list.row(item))
                return True
        return super().eventFilter(source, event)
    
    def initialize(self, parent=None):
        """Initialize the list widget."""
        if sip.isdeleted(self):
            super().__init__(parent)
            self.initUI()
        else:
            self.setParent(parent)

    def on_item_changed(self, item):
        self.changed = True

    def get_strings(self):
        """Return the strings of the entries in the list widget."""
        return [self.items_list.item(i).text() for i in range(self.items_list.count())]
        
    def set_strings(self, data=None, append=False, run_update_entries=True):
        """Update the entries in the list widget."""
        if data is None:
            return
        data = [data] if not isinstance(data, (list, tuple, set)) else data
        new_strings = []
        for ds in data:
            s = ds.string if isinstance(ds, StrItem) else str(ds)
            new_strings.append(StrItem(s, self.str_index))
            self.str_index += 1
        
        self.strings = self.strings + new_strings if append else new_strings

        if run_update_entries:
            self.populate_list()

    def get_checked(self):
        """Return the strings of the checked entries."""
        res = []
        for i in range(self.items_list.count()):
            if self.items_list.item(i).checkState() == Qt.Checked:
                res.append(self.items_list.item(i).text())
        if not res:
            return self.special_items
        return res

    def set_checked(self, strings):
        """Set the checked entries in the list widget."""
        strings = [str(string) for string in strings] if isinstance(strings, (tuple, list, set)) else [str(strings)]
        if self.items_list.count() == 0:
            self.populate_list()
        for i in range(self.items_list.count()):
            item = self.items_list.item(i)
            if item.text() in strings:
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)

    def set_highlight(self, item, strings=None):
        """Set the highlight for the selected item."""
        strings = strings if strings is not None else self.special_items
        font = item.font()
        bold = True if item.text() in strings else False
        font.setBold(bold)
        item.setFont(font)
        if bold:
            item.setCheckState(Qt.Checked)

    def highlight(self, strings, reset=False):
        """Highlight the selected entry in the list widget."""
        old_highlight = self.special_items.copy()
        self.special_items = [str(string) for string in strings] if isinstance(strings, (tuple, list, set)) else [str(strings)]
        if self.items_list.count() == 0:
            self.populate_list()
        self.items_list.blockSignals(True)
        for i in range(self.items_list.count()):
            item = self.items_list.item(i)
            self.set_highlight(item, None)
            if item.text() in self.special_items:
                item.setCheckState(Qt.Checked)
            elif item.text() in old_highlight:
                item.setCheckState(Qt.Unchecked)
        self.items_list.blockSignals(False)

    def populate_list(self, data=None, append=False):
        """Update the entries in the list widget."""
        if data is not None:
            self.set_strings(data, append, False)

        if sip.isdeleted(self):
            self.initialize()

        if self.items_list.count() != 0:
            checked_str = self.get_checked()
        else:
            checked_str = self.special_items
        self.items_list.blockSignals(True)
        self.items_list.clear()
        for col in self.strings:
            item = QListWidgetItem(str(col.string))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)  # Allow item to be checkable
            if str(col.string) in checked_str:
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)  # Default to unchecked
            self.set_highlight(item)
            self.items_list.addItem(item)
        self.items_list.blockSignals(False)
    
    def save(self):
        self.checked_items = self.get_checked()
        self.destroy()

    def clear(self):
        """Clear the list widget."""
        self.items_list.clear()
        self.checked_items = []
    
    def combine_items(self):
        """Combine the selected entries in the list widget."""
        selected_items = self.items_list.selectedItems()
        if len(selected_items) < 2:
            QMessageBox.warning(self, "Warning", "Select at least two items.")
            return None, None
        strings = [item.text() for item in selected_items]
        # autoname, _ = common_substring(strings)
        autoname, _ = find_common_str(*strings)
        new_name, ok = QInputDialog.getText(
            self, "Combine items", "Enter the new name:", text=autoname
        )
        if ok:
            new_list = []
            substrings = []
            inserted = False
            for item in self.strings:
                if item.string not in strings:
                    new_list.append(item)
                elif not inserted:
                    new_list.append(StrItem(new_name, self.str_index, substrings))
                    substrings.extend(item.sub_strings)
                    inserted = True
                else:
                    substrings.extend(item.sub_strings)
            self.strings = new_list
            self.str_index += 1
            self.populate_list()
            return new_name, strings
        return None, None

    def rename_item(self):
        """Rename the selected entry in the list widget."""
        selected_items = self.items_list.selectedItems()
        if len(selected_items) != 1:
            QMessageBox.warning(self, "Warning", "Select one item to rename.")
            return
        item = selected_items[0]
        name = item.text()
        new_name, ok = QInputDialog.getText(
            self, "Rename item", "Enter the new name:", text=name
        )
        if ok:
            loc = self.items_list.row(item)
            item.setText(new_name)
            if name in self.strings[loc].sub_strings:
                substrings = self.strings[loc].sub_strings
            else:
                substrings = self.strings[loc].sub_strings + [name]
            
            self.strings[loc] = StrItem(new_name, self.str_index, substrings)
            self.str_index += 1
            return new_name, substrings
        return None, None

class TableFrame(QFrame):
    """
    A class to provide table interaction via a QTableWidget.
    """

    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.table_info = {}  
        self.special_items = []
        self.checked_items = []
        self.str_index = 0
        self.changed = False
        self.items_table = None
        self.set_info(data, False, False)

        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Table Widget for items
        self.items_table = QTableWidget()
        self.items_table.setColumnCount(5)
        self.items_table.setHorizontalHeaderLabels(["Plot", "Dataset", "Shape", "Mark", "Label"])
        self.items_table.installEventFilter(self)
        self.items_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.items_table.setEditTriggers(QAbstractItemView.DoubleClicked)
        self.items_table.itemChanged.connect(self.on_item_changed)
        self.items_table.itemDoubleClicked.connect(lambda item: self.highlight(self.items_table.item(item.row(), 1).text()))
        self.items_table.setColumnWidth(0, 20)
        self.items_table.setColumnWidth(1, 100)
        self.items_table.setColumnWidth(2, 75)
        self.items_table.setColumnWidth(3, 20)
        self.items_table.setColumnWidth(4, 100)


        self.layout.addWidget(self.items_table)

        self.combine_button = QPushButton("Combine")
        self.combine_button.clicked.connect(self.combine_items)
        self.combine_button.setFixedWidth(75)

        self.rename_button = QPushButton("Rename")
        self.rename_button.clicked.connect(self.rename_item)
        self.rename_button.setFixedWidth(75)

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save)
        self.save_button.setFixedWidth(75)

        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.addWidget(self.combine_button)
        self.buttons_layout.addWidget(self.rename_button)
        self.buttons_layout.addWidget(self.save_button)

        self.layout.addLayout(self.buttons_layout)

        if self.items_table.rowCount() == 0:
            self.populate_table()

    def eventFilter(self, source, event):
        """Handles key press events for deleting items."""
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Delete:
            if source is self.items_table:
                rows_to_delete = set()
                for item in self.items_table.selectedItems():
                    rows_to_delete.add(item.row())
                for row in sorted(rows_to_delete, reverse=True):
                    self.items_table.removeRow(row)
                return True
        return super().eventFilter(source, event)
    
    def initialize(self, parent=None):
        """Initialize the table widget."""
        if sip.isdeleted(self):
            super().__init__(parent)
            self.initUI()
        else:
            self.setParent(parent)

    def on_item_changed(self, item):
        self.changed = True

    def get_info(self):
        """Return the strings of the entries in the table widget."""
        return [self.items_table.item(row, 1).text() for row in range(self.items_table.rowCount())]
        
    def set_info(self, data=None, append=False, run_update_entries=True):
        """Update the entries in the table widget."""
        if data is None:
            return
        data = [data] if not isinstance(data, (list, tuple, set)) else data
        data = [(ds, "") if not isinstance(ds, (list, tuple)) else ds for ds in data]
        checked_str = self.get_checked() if self.items_table is not None and self.items_table.rowCount() != 0 else self.special_items
        new_table_info = {}
        for ds, shp in data:
            s = ds.string if isinstance(ds, StrItem) else str(ds)
            new_table_info[s] = {
                "Plot": True if s in checked_str else False,
                "Dataset": StrItem(s, self.str_index),
                "Shape": str(shp),
                "Mark": True if s in self.special_items else False,
                "Label": f"_{s}" if s in self.special_items else s,
            }
            self.str_index += 1
        
        if append:
            self.table_info.update(new_table_info) 
        else:
            self.table_info = new_table_info

        if run_update_entries:
            self.populate_table()

    def get_checked(self, column=0):
        """Return the strings of the checked entries."""
        column = column if column in (0, 3) else 0
        res = []
        for row in range(self.items_table.rowCount()):
            if self.items_table.item(row, column).checkState() == Qt.Checked:
                res.append(self.items_table.item(row, 1).text())
        if not res:
            return self.special_items
        return res

    def set_checked(self, strings):
        """Set the checked entries in the table widget."""
        strings = [str(string) for string in strings] if isinstance(strings, (tuple, list, set)) else [str(strings)]
        if self.items_table.rowCount() == 0:
            self.populate_table()
        for row in range(self.items_table.rowCount()):
            item = self.items_table.item(row, 1)
            if item.text() in strings:
                self.items_table.item(row, 0).setCheckState(Qt.Checked)
            else:
                self.items_table.item(row, 0).setCheckState(Qt.Unchecked)

    def set_highlight(self, row, strings=None):
        """Set the highlight for the selected row."""
        strings = strings if strings is not None else self.special_items
        font = self.items_table.item(row, 1).font()
        bold = True if self.items_table.item(row, 1).text() in strings else False
        font.setBold(bold)
        self.items_table.item(row, 1).setFont(font)
        if bold:
            self.items_table.item(row, 0).setCheckState(Qt.Checked)
            self.items_table.item(row, 3).setCheckState(Qt.Checked)

    def highlight(self, strings, reset=False):
        """Highlight the selected entry in the table widget."""
        if isinstance(strings, int):
            strings = [self.items_table.item(strings, 1).text()]
        old_highlight = self.special_items.copy()
        self.special_items = [str(string) for string in strings] if isinstance(strings, (tuple, list, set)) else [str(strings)]
        if self.items_table.rowCount() == 0:
            self.populate_table()
        self.items_table.blockSignals(True)
        for row in range(self.items_table.rowCount()):
            self.set_highlight(row, None)
            if self.items_table.item(row, 1).text() in self.special_items:
                self.items_table.item(row, 0).setCheckState(Qt.Checked)
                self.items_table.item(row, 3).setCheckState(Qt.Checked)
            elif self.items_table.item(row, 1).text() in old_highlight:
                self.items_table.item(row, 0).setCheckState(Qt.Unchecked)
                self.items_table.item(row, 3).setCheckState(Qt.Unchecked)
        self.items_table.blockSignals(False)

    def populate_table(self, data=None, append=False):
        """Update the entries in the table widget."""
        # if data is not None:
        self.set_info(data, append, False)

        if sip.isdeleted(self):
            self.initialize()

        checked_str = self.get_checked() if self.items_table.rowCount() != 0 else self.special_items
        self.items_table.blockSignals(True)
        self.items_table.setRowCount(0)
        for key, value in self.table_info.items():
            row_position = self.items_table.rowCount()
            self.items_table.insertRow(row_position)
            check_item = QTableWidgetItem()
            check_item.setFlags(check_item.flags() | Qt.ItemIsUserCheckable)
            check_item.setCheckState(Qt.Checked if key in checked_str else Qt.Unchecked)
            self.items_table.setItem(row_position, 0, check_item)
            dataset_item = QTableWidgetItem(value["Dataset"].string)
            dataset_item.setFlags(dataset_item.flags() & ~Qt.ItemIsEditable)  # Make dataset_item un-editable
            self.items_table.setItem(row_position, 1, dataset_item)
            shape_item = QTableWidgetItem(value["Shape"])
            self.items_table.setItem(row_position, 2, shape_item)
            mark_item = QTableWidgetItem()
            mark_item.setFlags(mark_item.flags() | Qt.ItemIsUserCheckable)
            mark_item.setCheckState(Qt.Checked if value["Mark"] else Qt.Unchecked)
            self.items_table.setItem(row_position, 3, mark_item)
            label_item = QTableWidgetItem(value["Label"])
            self.items_table.setItem(row_position, 4, label_item)
            self.set_highlight(row_position)
        self.items_table.blockSignals(False)
    
    def save(self):
        self.checked_items = self.get_checked()
        self.destroy()

    def clear(self):
        """Clear the table widget."""
        self.items_table.setRowCount(0)
        self.checked_items = []
    
    def combine_items(self):
        """Combine the selected entries in the table widget."""
        selected_items = self.items_table.selectedItems()
        if len(selected_items) < 2:
            QMessageBox.warning(self, "Warning", "Select at least two items.")
            return None, None
        # strings = [self.items_table.item(item.row(), 1).text() for item in selected_items]
        selected_rows = list(set(item.row() for item in selected_items))
        strings = [self.items_table.item(row, 1).text() for row in selected_rows]
        # autoname, _ = common_substring(strings)
        autoname, _ = find_common_str(*strings)
        new_name, ok = QInputDialog.getText(
            self, "Combine items", "Enter the new name:", text=autoname
        )
        if ok:
            new_table_info = {}
            substrings = []
            inserted = False
            for key, value in self.table_info.items():
                if key not in strings:
                    new_table_info[key] = value
                elif not inserted:
                    new_table_info[new_name] = {
                        "Plot": Qt.Unchecked,
                        "Dataset": StrItem(new_name, self.str_index, substrings),
                        "Shape": "",
                        "Mark": Qt.Unchecked,
                        "Label": f"_{new_name}"
                    }
                    substrings.extend(value["Dataset"].sub_strings)
                    inserted = True
                else:
                    substrings.extend(value["Dataset"].sub_strings)
            new_table_info[new_name]["Dataset"].sub_strings = substrings
            self.table_info = new_table_info
            self.str_index += 1
            self.populate_table()
            return new_name, strings
        return None, None

    def rename_item(self):
        """Rename the selected entry in the table widget."""
        selected_items = self.items_table.selectedItems()
        if len(selected_items)  != 5:
            QMessageBox.warning(self, "Warning", "Select one item to rename.")
            return None, None
        # Get the selected row
        selected_row = selected_items[0].row()
        name = self.items_table.item(selected_row, 1).text()
        new_name, ok = QInputDialog.getText(
            self, "Rename item", "Enter the new name:", text=name
        )
        if ok:
            self.items_table.item(selected_row, 1).setText(new_name)
            self.items_table.item(selected_row, 4).setText(f"_{new_name}")

            if name in self.table_info[name]["Dataset"].sub_strings:
                substrings = self.table_info[name]["Dataset"].sub_strings
            else:
                substrings = self.table_info[name]["Dataset"].sub_strings + [name]
            self.table_info[new_name] = self.table_info.pop(name)
            self.table_info[new_name]["Dataset"] = StrItem(new_name, self.str_index, substrings)
            self.table_info[new_name]["Label"] = f"_{new_name}"
            self.str_index += 1
            return new_name, substrings
        return None, None
    

class DataHandlerWidgets(DataHandler):
    """A class to handle the data widgets for the EIS analysis."""
    def __init__(self, parent=None, **kwargs):
        self.raw = None
        super().__init__()
        self._raw = {}
        self.root = parent
        # self.raw_list = ListFrame(list(self.raw.keys()), parent)
        self.raw_list = TableFrame([(k, str(len(v))) for k, v in self.raw.items()], parent)
        self.window = None
        self.var = None

        self.callback = kwargs.get("callback", None)

    def load_window(self):
        """Show the list frame window."""
        if not self.raw:
            QMessageBox.warning(self.root, "Warning", "No data to display.")
            return
        for key, system in self.raw.items():
            self._raw[key] = system.get_df("freq", "real", "imag")
            self._raw[key].attrs["thickness"] = system.thickness
            self._raw[key].attrs["area"] = system.area
        
        self.raw_archive = {**self.raw_archive.copy(), **self.raw.copy()}

        # self.raw_list.populate_list(list(self.raw.keys()))
        self.raw_list.populate_table([(k, str(len(v))) for k, v in self.raw.items()])

        self.window = QMainWindow(self.root)
        self.window.setWindowTitle("Loaded Datasets")
        self.window.setGeometry(100, 100, 400, 300)

        self.layout = QVBoxLayout()

        central_widget = QWidget()
        # self.raw_list.setParent(central_widget)
        self.raw_list.initialize(central_widget)

        # layout = QVBoxLayout(central_widget)
        self.layout.addWidget(self.raw_list)
        central_widget.setLayout(self.layout)
        
        self.raw_list.combine_button.clicked.disconnect()
        self.raw_list.combine_button.clicked.connect(self.combine_items)
        self.raw_list.rename_button.clicked.disconnect()
        self.raw_list.rename_button.clicked.connect(self.rename_item)
        self.raw_list.save_button.clicked.disconnect()
        self.raw_list.save_button.clicked.connect(self.save_data)

        self.window.setCentralWidget(central_widget)

        self.window.closeEvent = self.closeEvent

        self.window.show()
    
    def show(self):
        """Show the list frame window."""
        self.load_window()

    def closeEvent(self, event):
        self.raw_list.setParent(None)
        # self.window.hide()
        
        if self.raw_list.changed:
            reply = QMessageBox.question(
                self.raw_list,
                "Save Changes",
                "Do you want to save changes before closing?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if reply == QMessageBox.Yes:
                self.save_data()
                event.accept()
            elif reply == QMessageBox.Cancel:
                event.ignore()
            else:
                event.accept()
            if not sip.isdeleted(self.window):
                self.window.deleteLater()
        else:
            event.accept()
            if not sip.isdeleted(self.window):
                self.window.deleteLater()

    def save_data(self):
        """Save the data to the raw dictionary. """
        self.raw_archive = {**self.raw_archive.copy(), **self.raw.copy()}
        self.raw = {}
        for key in self.raw_list.get_info():
            self.update_system(key, self._raw[key], "impedance", self._raw[key].attrs["thickness"], self._raw[key].attrs["area"])
        if self.raw_list.changed:
            self.update_var()
        elif self.var is not None and not self.is_highlighted(self.primary()):
            self.var.setCurrentText(self.raw_list.special_items[0])
        self.raw_list.changed = False
        if self.callback is not None:
            self.callback()
    
    def update_data(self, data):
        """Update the raw data dictionary."""
        self._raw = data
        self.save_data()

    def update_var(self):
        """Update the variable in the list frame window."""
        if self.var is not None and self.raw:
            old_var = self.raw_list.special_items[0] if self.raw_list.special_items else self.var.currentText()
            self.var.blockSignals(True)
            self.var.clear()
            self.var.setCurrentText("")
            self.var.addItems(list(self.raw.keys()))
            if old_var not in self.raw:
                old_var = next(iter(self.raw))
            self.highlight(old_var)
            self.var.setCurrentText(old_var)
            self.var.blockSignals(False)

    def set_var(self, var):
        """Set the variable in the list frame window."""
        if self.var is not None and var in self.raw:
            self.var.blockSignals(True)
            self.highlight(var)
            self.var.setCurrentText(var)
            self.var.blockSignals(False)

    def primary(self):
        if self.var is not None:
            return self.var.currentText()
        else:
            return ""
        
    def combine_items(self):
        """Combine the selected entries in the list widget."""
        new_key, keys = self.raw_list.combine_items()
        if new_key:
            data = {k: v for k, v in self._raw.items() if k in keys}
            comb = pd.concat(data.values(), sort=False, keys=data.keys())
            comb.sort_values("freq", ignore_index=True)
            comb.attrs["thickness"] = np.mean([v.attrs["thickness"] for v in data.values()])
            comb.attrs["area"] = np.mean([v.attrs["area"] for v in data.values()])
            self._raw[new_key] = comb
            self.raw_list.changed = True
            self.raw_list.table_info[new_key]["Shape"] = str(len(comb))
            self.raw_list.populate_table()
            # self.update_var()

    def rename_item(self):
        """Rename the selected entry in the list widget."""
        new_key, keys = self.raw_list.rename_item()
        if new_key:
            for key in keys:
                if key in self._raw:
                    self._raw[new_key] = self._raw.pop(key)
                    self.raw_list.changed = True
                    # self.update_var()
                    break
            else:
                QMessageBox.warning(self.raw_list, "Warning", "Key not found.")
        
    def highlight(self, strings, reset=False):
        """Highlight the selected entry in the list widget."""
        self.raw_list.highlight(strings, reset)

    def is_highlighted(self, string):
        """Check if the string is highlighted."""
        return string in self.raw_list.special_items

    def get_checked(self):
        """Return the strings of the checked entries."""
        return self.raw_list.get_checked()
    
    def get_label(self, key):
        """Return the label for the selected key."""
        try:
            return self.raw_list.table_info[key]["Label"]   
        except KeyError:
            return f"_{key}",

    def get_mark(self, key):
        """Return the mark for the selected key."""
        checks = self.raw_list.get_checked(3)
        if checks != self.raw_list.special_items:
            return key in checks
        try:
            return self.raw_list.table_info[key]["Mark"]   
        except KeyError:
            return True

