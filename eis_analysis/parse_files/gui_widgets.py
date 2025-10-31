import re
from datetime import datetime
from collections import namedtuple, defaultdict

import pandas as pd
from PyQt5.QtCore import Qt, QEvent, pyqtSignal
from PyQt5.QtWidgets import (
    QMenu,
    QAction,
    QDialog,
    # QErrorMessage,
    QHeaderView,
    QListWidget,
    QMessageBox,
    QTreeWidget,
    QVBoxLayout,
    QInputDialog,
    QRadioButton,
    QTableWidget,
    QListWidgetItem,
    QTreeWidgetItem,
    QDialogButtonBox,
    QTableWidgetItem,
    QAbstractItemView,
    QTreeWidgetItemIterator,
)

from ..dict_ops import flatten_dict
from .gui_plots import PopupGraph
from ..string_ops import sci_note, compile_search_patterns

# from PyQt5.QtWidgets import QListWidget, QMenu, QInputDialog, QListWidgetItem
# from PyQt5.QtCore import Qt
from .gui_windows import DataViewer
from ..data_treatment import find_nearest, impedance_concat

DatasetEntry = namedtuple("DatasetEntry", ["df", "uid", "name", "cr_time", "row_0_hash"])
# DatasetEntryIndex = {"df": 0, "uid": 1, "name": 2, "cr_time": 3}


def parse_df_to_entry(df, uid=None, name=None, cr_time=None, first_row_na=0):
    """
    Parses a DataFrame into a DatasetEntry.

    Parameters:
    - df (pd.DataFrame): The DataFrame to parse.
    - uid (int, optional): The unique identifier for the entry. Defaults to id(df).
    - name (str, optional): The name for the entry. Defaults to df.attrs["name"].
    - cr_time (int, optional): The created timestamp for the entry. Defaults to df.attrs["createdtime"].
    - first_row_na (any, optional): Value to replace NaN in the first row. Defaults to 0.

    Returns:
    - DatasetEntry: The parsed DatasetEntry.
    """
    if uid is None:
        uid = id(df)
    if name is None:
        name = df.attrs.get("name", "Unnamed")
    if cr_time is None:
        cr_time = df.attrs.get("createdtime", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Compute the hash of the first row, with a fallback to column names
    try:
        # Attempt to hash the first row
        row_0_hash = hash(tuple(df.iloc[0].fillna(first_row_na).to_list()))
    except (IndexError, AttributeError, TypeError):
        # Fallback: Hash the column names if the first row is unavailable or invalid
        row_0_hash = hash(tuple(df.columns.to_list()))

    return DatasetEntry(df=df, uid=uid, name=name, cr_time=cr_time, row_0_hash=row_0_hash)


class DataGroup:
    """
    A class to manage a group of DatasetEntry objects.
    """

    def __init__(self, entries: list | tuple):
        """
        Initializes the DataGroup with a list of DatasetEntry objects.
        """
        if not isinstance(entries, (tuple, list)):
            entries = [entries]
        entries = list(entries)
        # Ensure each entry in the list is a DatasetEntry
        self.entries: list[DatasetEntry] = []
        for i, entry in enumerate(entries):
            if isinstance(entry, pd.DataFrame):
                self.entries.append(parse_df_to_entry(entry))
            elif isinstance(entry, DatasetEntry):
                self.entries.append(entry)
            else:
                raise TypeError("each entry must be a DatasetEntry")

        self.active_index = 0

    @property
    def active(self):
        """
        Returns the active DatasetEntry.
        """
        return self.entries[self.active_index]

    @active.setter
    def active(self, value):
        """
        Sets the active DatasetEntry.
        """
        if isinstance(value, int) and -1 <= value < len(self.entries):
            self.active_index = value
        elif isinstance(value, DatasetEntry):
            try:
                self.active_index = self.uids.index(value.uid)
            except ValueError:
                self.entries.append(value)
                self.active_index = len(self.entries) - 1
        elif isinstance(value, pd.DataFrame):
            py_id = value.attrs.get("py_id")
            try:
                self.active_index = self.uids.index(py_id)
            except ValueError:
                new_entry = parse_df_to_entry(value)
                self.entries.append(new_entry)
                self.active_index = len(self.entries) - 1
        else:
            raise ValueError(
                "Invalid type for active. Must be valid int index, DatasetEntry, or DataFrame."
            )

    @property
    def dfs(self):
        """
        Returns a list of DataFrames in the group.
        """
        return [entry.df for entry in self.entries]

    @property
    def uids(self):
        """
        Returns a list of unique identifiers for the entries.
        """
        return [entry.uid for entry in self.entries]

    @property
    def names(self):
        """
        Returns a list of names for the entries.
        """
        return [entry.name for entry in self.entries]

    @property
    def cr_time(self):
        """
        Returns the creation time for the active entry.
        """
        return self.active.cr_time

    @property
    def row_0_hash(self):
        """
        Returns the hash of the first row for the active entry.
        """
        return self.active.row_0_hash

    def append(self, entry):
        """
        Appends a new entry to the group.
        """
        if isinstance(entry, pd.DataFrame):
            entry = parse_df_to_entry(entry)
        elif not isinstance(entry, DatasetEntry):
            print("entry:", entry)
            raise TypeError("entry must be a DataFrame or DatasetEntry")
        self.entries.append(entry)


class ListWidgetWithMenu(QListWidget):
    def __init__(self, pattern_check_str="not ", callbacks=None, has_checks=True, parent=None):
        """
        A QListWidget subclass with a context menu and customizable actions.

        Args:
            pattern_check_str (str): String to check for patterns (default: "not ").
            callbacks (dict): A dictionary of additional menu options where keys are menu text and values are callables.
            has_checks (bool): Whether to make the items checkable (default: True).
            parent (QWidget): Parent widget.
        """
        self._context_menu_actions = {}
        self._default_actions = ["add", "sort", "reset", "delete"]

        super().__init__(parent)

        self.pattern_check_str = pattern_check_str
        self.has_checks = has_checks

        # Initialize _context_menu_actions with callbacks first, then add missing defaults
        self.context_menu_actions = dict(
            zip(
                self._default_actions,
                [self.add_item, self.sort_items, self.reset_items, self.delete_selected_items],
            )
        )
        if callbacks:
            self.context_menu_actions = callbacks

        # Default actions
        # default_actions = {
        #     "add": self.add_item,
        #     "delete": self.delete_selected_items,
        #     "sort": self.sort_items,
        #     "reset": self.reset_items,
        # }
        # if callbacks:
        #     for key, value in callbacks.items():
        #         self._context_menu_actions[key] = value
        # for key, value in default_actions.items():
        #     if key not in self._context_menu_actions:
        #         self._context_menu_actions[key] = value

        # Install the event filter
        self.installEventFilter(self)

    def eventFilter(self, source, event):
        """
        Event filter to handle key press events for deleting items.
        """
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Delete:
            self.delete_selected_items()
            return True
        return super().eventFilter(source, event)

    @property
    def context_menu_actions(self):
        """Property to get the context menu actions."""
        return self._context_menu_actions

    @context_menu_actions.setter
    def context_menu_actions(self, actions):
        """
        Setter for context menu actions. Validates the input and updates the internal dictionary.

        Args:
            actions (dict): A dictionary where keys are strings (menu text) and values are callables.
        """
        if isinstance(actions, dict):
            # Ensure all keys are strings and all values are callables
            actions_dict = {
                k.lower(): v for k, v in actions.items() if isinstance(k, str) and callable(v)
            }
            if not actions_dict:
                raise ValueError(
                    "Actions dictionary must contain at least one valid key-value pair."
                )
            self._context_menu_actions.update(actions_dict)
            # if all(isinstance(k, str) and callable(v) for k, v in actions.items()):

            #     self._context_menu_actions.update({k.lower(): v for k, v in actions.items() if isinstance(k, str) and callable(v)})
            # else:
            #     raise ValueError("All keys must be strings and all values must be callables.")
        elif isinstance(actions, (list, tuple)) and len(actions) == 2:
            # Ensure the first element is a string and the second is a callable
            if isinstance(actions[0], str) and callable(actions[1]):
                self._context_menu_actions[actions[0].lower()] = actions[1]
            else:
                raise ValueError(
                    "The first element must be a string and the second must be a callable."
                )
        else:
            raise ValueError(
                "Actions must be a dict or a list/tuple of length 2 with a string and a callable."
            )

    def contextMenuEvent(self, event):
        """Creates and displays the context menu."""
        menu = QMenu(self)

        def add_action(name, function):
            action = QAction(name.title(), self)
            action.triggered.connect(function)
            menu.addAction(action)

        # Add actions from the context_menu_actions dictionary
        for action_name in [
            key for key in self.context_menu_actions if key not in self._default_actions
        ]:
            add_action(action_name, self.context_menu_actions[action_name])

        menu.addSeparator()

        for action_name in self._default_actions[:-1]:
            add_action(action_name, self.context_menu_actions[action_name])

        menu.addSeparator()

        add_action(self._default_actions[-1], self.context_menu_actions[self._default_actions[-1]])

        # Show the menu at the cursor position
        menu.exec_(event.globalPos())

        # Add actions from the context_menu_actions dictionary
        # for action_name, action_function in self.context_menu_actions.items():
        #     if action_name == "delete":
        #         continue
        #     custom_action = QAction(action_name.title(), self)
        #     custom_action.triggered.connect(action_function)
        #     menu.addAction(custom_action)

        # menu.addSeparator()

        # custom_action = QAction("Delete", self)
        # custom_action.triggered.connect(self.context_menu_actions["delete"])
        # menu.addAction(custom_action)

        # # Show the menu at the cursor position
        # menu.exec_(event.globalPos())

    def add_item(self, *_, items=None):
        """
        Adds item(s) to the list. If `items` is None, uses a QInputDialog to get input.

        Args:
            items (None, str, or list of str): The item(s) to add. If None, prompts the user for input.
        """
        if items is None:
            # Prompt the user for input
            text, ok = QInputDialog.getText(
                self,
                "Add Item",
                "Enter the item text:",
            )
            if ok and text.strip():
                if "," in text or ";" in text:
                    items = re.split(r"\s*[,;]\s+", text.strip())
                else:
                    items = [text.strip()]
            else:
                return

        # Ensure items is a list
        if isinstance(items, str):
            items = [items]

        # Add each item to the list
        for text in items:
            check_state = Qt.Unchecked
            if self.has_checks and text.startswith(self.pattern_check_str):
                text = text[len(self.pattern_check_str) :].strip()
                check_state = Qt.Checked

            item = QListWidgetItem(text)
            if self.has_checks:
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(check_state)
            self.addItem(item)

    def delete_selected_items(self):
        """Deletes the selected items from the list."""
        for item in self.selectedItems():
            self.takeItem(self.row(item))

    def sort_items(self):
        """Sorts the items in the list alphabetically."""
        self.sortItems(Qt.AscendingOrder)

    def reset_items(self):
        """Resets the list by clearing all items."""
        self.clear()

    def get_list(self, transform=None, transform_checked=None):
        """
        Returns a list of item texts, optionally applying a transformation to checked or unchecked items.

        Args:
            transform (callable or None, optional): A callable or string to transform the text of unchecked items.
                If None, no transformation is applied.
            transform_checked (callable or None, optional): A callable or string to transform the text of checked items.
                If None, no transformation is applied.

        Returns:
            list: A list of item texts, with transformations applied to checked or unchecked items if specified.
        """
        items = []
        for i in range(self.count()):
            item = self.item(i)
            text = item.text()

            # Apply transformation for checked items
            if (
                self.has_checks
                and item.checkState() == Qt.Checked
                and transform_checked is not None
            ):
                if callable(transform_checked):
                    text = transform_checked(text)
                elif isinstance(transform_checked, (str, int, float)) and not bool(
                    transform_checked
                ):
                    text = ""

            # Apply transformation for unchecked items
            elif transform is not None:
                if callable(transform):
                    text = transform(text)
                elif isinstance(transform, (str, int, float)) and not bool(transform):
                    text = ""

            # Add the transformed or original text to the list
            if text:
                items.append(text)

        return items

    # def get_list(self, get_checked=None, ):
    #     """
    #     Returns a list of items in the list widget.

    #     Args:
    #         get_checked (bool): If True, returns only checked items. Defaults to False.

    #     Returns:
    #         list: A list of item texts.
    #     """
    #     if not self.has_checks:
    #         # return all
    #         return [self.item(i).text() for i in range(self.count())]
    #     if get_checked is None:
    #         # return all
    #         return [self.item(i).text() for i in range(self.count())]
    #     if get_checked:
    #         # return checked
    #         return [
    #             self.item(i).text()
    #             for i in range(self.count())
    #             if self.item(i).checkState() == Qt.Checked
    #         ]
    #     # return unchecked
    #     return [
    #             self.item(i).text()
    #             for i in range(self.count())
    #             if self.item(i).checkState() != Qt.Checked
    #         ]


class ColumnListWidget(ListWidgetWithMenu):
    def __init__(
        self,
        preferred_keys=None,
        pattern_check_str="use ",
        callbacks=None,
        has_checks=True,
        parent=None,
    ):
        """
        A specialized ListWidgetWithMenu for managing column names.

        Args:
            pattern_check_str (str): String to check for patterns (default: "use ").
            callbacks (dict): A dictionary of additional menu options where keys are menu text and values are callables.
            has_checks (bool): Whether to make the items checkable (default: True).
            parent (QWidget): Parent widget.
        """
        base_callbacks = {
            "rename": self.merge_columns,
            "reset": self.reset_columns_list,
            "sort": self.sort_items,
        }
        if not isinstance(callbacks, dict):
            callbacks = {}

        super().__init__(
            pattern_check_str=pattern_check_str,
            callbacks={**base_callbacks, **callbacks},
            has_checks=has_checks,
            parent=parent,
        )

        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setDragDropMode(QAbstractItemView.InternalMove)

        # Dictionary to store column name mappings
        # Note: keys -> original column names, values -> new merged column name (display name)
        self.column_map = pd.DataFrame(columns=["original", "display"])
        self.checked_columns = []
        self.preferred_keys = (
            [] if not isinstance(preferred_keys, (list, tuple)) else list(preferred_keys)
        )

    def _get_unique_columns(self, data_org=None):
        """
        Updates column_map with unique columns from data_org and returns the display names.
        Ensures no null/empty values are returned.
        """
        if data_org is not None:
            unique_columns = set()
            for data in data_org.values():
                if isinstance(data.active.df, pd.DataFrame):
                    unique_columns.update(data.active.df.columns)

            # Build DataFrame: original = original column, display = current display name (default to original)
            self.column_map = pd.DataFrame(
                {
                    "original": list(unique_columns),
                    "display": [
                        re.sub(r"([\'\"])(.+?)\1", r"\2", str(col)) for col in unique_columns
                    ],
                }
            )

        if not self.column_map.empty:
            # Remove rows where display is null/empty
            # filtered = self.column_map[
            #     self.column_map["display"].notnull() & (self.column_map["display"] != "")
            # ]
            # return set(filtered["display"])
            # return set(filter(None, self.column_map["display"]))
            return set(self.column_map["display"][self.column_map["display"] != ""].dropna())
        return set()

    def _get_checked_columns(self, current_only=False):
        """Helper function to clean and validate checked columns."""

        current_checked = list(
            dict.fromkeys(
                [
                    self.item(i).text()
                    for i in range(self.count())
                    if self.item(i).checkState() == Qt.Checked
                ]
            )
        )
        if current_only:
            # Only keep checked columns that are currently in the list
            return current_checked
        else:
            # Keep all checked columns
            # self.checked_columns = list(set(self.checked_columns) & set(current_checked))
            return current_checked + list(
                set(map(str, self.checked_columns)).difference(current_checked)
            )

    def _parse_column_keys(self, str_list=None):
        """
        Converts a list of strings into valid column keys (tuples or strings).
        Handles tuple-string conversions, e.g., '(a, b)' -> ('a', 'b').
        """
        if str_list is None:
            str_list = [self.item(i).text() for i in range(self.count())]

        keys = []
        for col in str_list:
            if isinstance(col, str):
                parts = re.sub(r"([\'\"])(.+?)\1", r"\2", col)
                if parts.startswith("(") and parts.endswith(")"):
                    # Split by comma and strip whitespace
                    parts = tuple(re.split(r"\s*,\s*", parts[1:-1]))

                keys.append(parts)
            else:
                keys.append(col)
        return keys

    def reset_columns_list(self, data_org=None):
        """Populate the columns list with unique columns from the dataframes ignoring any checked list."""
        if not data_org:
            QMessageBox.warning(self, "No Data", "Please load and group data first.")
            return

        if self.checked_columns:
            # Show a confirmation dialog
            reply = QMessageBox.question(
                self,
                "Reset Columns",
                "Do you also want to clear the checked columns?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )

            if reply == QMessageBox.Yes:
                self.checked_columns.clear()
            else:
                self.checked_columns = self._get_checked_columns()

        if data_org is None:
            # # If data_org is None, use current "original" column names
            # unique_columns = set(self.column_map["original"])
            # update "display" to be str of the original column names
            self.column_map["display"] = (
                self.column_map["original"]
                .astype(str)
                .str.replace(r"([\'\"])(.+?)\1", r"\2", regex=True)
            )
            # If data_org is None, use current "original" column names
            unique_columns = set(self.column_map["display"])

        else:
            # Collect all unique columns from the dataframes
            unique_columns = self._get_unique_columns(data_org)

        # Populate the list widget with unique columns
        self.clear()
        for col in unique_columns:
            item = QListWidgetItem(str(col))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)  # Default to checked
            self.addItem(item)

    def refresh_columns_list(self, data_org=None):
        """Refresh the columns list to reflect unique columns in data_org while preserving checked states."""
        if not data_org:
            QMessageBox.warning(self, "No Data", "Please load and group data first.")
            return

        # Collect all unique columns from data_org
        unique_columns = self._get_unique_columns(data_org)

        self.checked_columns = self._get_checked_columns()

        # Clear the list widget and repopulate it
        self.clear()
        for col in unique_columns:
            item = QListWidgetItem(str(col))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            # Restore the checked state if it exists, otherwise default to unchecked
            item.setCheckState(Qt.Checked if str(col) in self.checked_columns else Qt.Unchecked)
            self.addItem(item)

    def sort_columns_list(self):
        """Sort the columns list by check state first, then alphabetically."""
        if self.count() == 0:
            return

        c_items = []
        c_i = 0
        while c_i < self.count():
            item = self.item(c_i)
            if item.checkState() == Qt.Checked:
                # Remove the item from the list
                c_items.append((item.text(), item.checkState()))
                c_i += 1
            else:
                break

        # Sort the remaining items (checked items here will be sorted to the top of this section)
        items = [
            (self.item(i).text(), self.item(i).checkState()) for i in range(c_i, self.count())
        ]

        items.sort(
            key=lambda x: (
                x[1] == Qt.Unchecked,  # Checked first
                -sum(
                    pk.lower() in x[0].lower() for pk in self.preferred_keys
                ),  # More preferred keys = higher rank
                x[0].lower(),  # Alphabetical
            )
        )

        items = c_items + items

        self.clear()
        for text, checkState in items:
            item = QListWidgetItem(text)
            item.setCheckState(checkState)
            self.addItem(item)

    def delete_selected_items(self):
        selected_items = self.selectedItems()
        for item in selected_items:
            # Replace all occurrences of item_text in the "display" column with ""
            self.column_map["display"] = self.column_map["display"].where(
                self.column_map["display"] != item.text(), ""
            )
            # Optionally, remove the item from the widget
            self.takeItem(self.row(item))

    def merge_columns(self):
        """
        Merges selected columns into a single name.
        Updates the column_map to reflect the merged name.
        """
        selected_items = self.selectedItems()
        if not selected_items:  # or len(selected_items) < 2:
            QMessageBox.warning(
                self, "Merge Columns", "Please select at least two columns to merge."
            )
            return

        if len(selected_items) < 2:
            new_name, ok = QInputDialog.getText(
                self,
                "Rename Column",
                "Enter the new name for the column:",
                text=selected_items[0].text(),
            )
        else:
            new_name, ok = QInputDialog.getText(
                self,
                "Merge Columns",
                "Enter the new name for the merged columns:",
                text=selected_items[0].text(),
            )
        if not ok or not new_name.strip():
            return

        new_name = new_name.strip()
        selected_names = [item.text() for item in selected_items]

        # Update the display name for all selected columns
        self.column_map.loc[self.column_map["display"].isin(selected_names), "display"] = new_name

        selected_items += [
            self.item(i) for i in range(self.count()) if self.item(i).text() == new_name
        ]

        # Remove merged items from the widget
        rows = []
        for item in selected_items:
            rows.append(self.row(item))
            self.takeItem(self.row(item))

        first_row = min(rows)

        # Add the new merged column to the list
        merged_item = QListWidgetItem(new_name)
        merged_item.setFlags(merged_item.flags() | Qt.ItemIsUserCheckable)
        merged_item.setCheckState(Qt.Checked)
        self.insertItem(first_row, merged_item)

    def get_translated_columns(self, checked_only=False, df=None):
        """
        Returns a mapping of original column names to display names.
        If checked_only is True, only includes checked columns.
        If df is provided, returns a DataFrame with columns renamed to display names.
        Otherwise, returns the mapping dict.
        """
        if checked_only:
            display_names = self._get_checked_columns(current_only=True)
            # mask = self.column_map["display"].isin(display_names)
        else:
            display_names = [self.item(i).text() for i in range(self.count())]
            # All non-null display names
            # mask = self.column_map["display"].notnull() & (self.column_map["display"] != "")

        if self.column_map["display"].is_unique:
            sorted_map = self.column_map.set_index("display").reindex(display_names).reset_index()

        else:
            # Make a copy to pop rows from
            df_pop = self.column_map.copy().T
            res_rows = []

            for val in list(dict.fromkeys(display_names)):
                idx = df_pop.columns[df_pop.loc["display"] == val]
                for i in idx:
                    res_rows.append(df_pop.pop(i))

            sorted_map = pd.DataFrame(res_rows).reset_index(drop=True)

        mapping = dict(zip(sorted_map["original"], self._parse_column_keys(sorted_map["display"])))

        # mapping = dict(zip(
        #     self.column_map.loc[mask, "original"],
        #     self._parse_column_keys(self.column_map.loc[mask, "display"])
        # ))

        if df is not None and isinstance(df, pd.DataFrame):
            # Only rename columns that exist in the DataFrame
            rename_dict = {k: v for k, v in mapping.items() if k in df.columns}
            return df.rename(columns=rename_dict)
        return mapping


class RadioButtonDialog(QDialog):
    """
    Simple dialog with radio buttons for selecting options.
    """

    def __init__(self, title, options, default_index=0, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)

        layout = QVBoxLayout(self)

        self.radio_buttons = []
        for i, option in enumerate(options):
            radio_button = QRadioButton(option)
            if i == default_index:
                radio_button.setChecked(True)
            self.radio_buttons.append(radio_button)
            layout.addWidget(radio_button)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        layout.addWidget(button_box)

    def selected_option(self):
        """Returns the text of the selected radio button."""
        for radio_button in self.radio_buttons:
            if radio_button.isChecked():
                return radio_button.text()
        return None


class DraggableTableWidget(QTableWidget):
    """
    A QTableWidget subclass that supports drag-and-drop reordering of rows.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setDragDropOverwriteMode(False)
        self.setDropIndicatorShown(True)

        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setDragDropMode(QAbstractItemView.InternalMove)

    def dropEvent(self, event):
        """
        Handles the drop event for reordering rows within the table.
        """
        if not event.isAccepted() and event.source() == self:
            if self.isSortingEnabled():
                self.sortItems(-1)
            drop_row = self.drop_on(event)

            rows = sorted(set(item.row() for item in self.selectedItems()))
            rows_to_move = [
                [
                    QTableWidgetItem(self.item(row_index, column_index))
                    for column_index in range(self.columnCount())
                ]
                for row_index in rows
            ]
            for row_index in reversed(rows):
                self.removeRow(row_index)
                if row_index < drop_row:
                    drop_row -= 1

            for row_index, data in enumerate(rows_to_move):
                row_index += drop_row
                self.insertRow(row_index)
                for column_index, column_data in enumerate(data):
                    self.setItem(row_index, column_index, column_data)
            event.accept()
            for row_index in range(len(rows_to_move)):
                self.item(drop_row + row_index, 0).setSelected(True)
                self.item(drop_row + row_index, 1).setSelected(True)

        super().dropEvent(event)

    def drop_on(self, event):
        """
        Determines the row index where the item is dropped.
        """
        index = self.indexAt(event.pos())
        if not index.isValid():
            return self.rowCount()

        return index.row() + 1 if self.is_below(event.pos(), index) else index.row()

    def is_below(self, pos, index):
        """
        Checks if the drop position is below the given index.
        """
        rect = self.visualRect(index)
        margin = 2
        if pos.y() - rect.top() < margin:
            return False
        elif rect.bottom() - pos.y() < margin:
            return True
        # noinspection PyTypeChecker
        return (
            rect.contains(pos, True)
            and not (int(self.model().flags(index)) & Qt.ItemIsDropEnabled)
            and pos.y() >= rect.center().y()
        )


class DataTreeWidget(QTreeWidget):
    """
    A QTreeWidget subclass for managing and displaying hierarchical data.
    """

    saveDataRequested = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.all_data = {}
        self.data_org = {}
        self.data_parents = {}
        self.data_viewer = None  # DataViewer object for viewing data
        self.setColumnCount(4)
        self.setHeaderLabels(["Name", "Duplicates", "Created time", "Details"])

        # Set initial column widths
        self.setColumnWidth(0, 200)  # Name
        self.setColumnWidth(1, 100)  # Duplicates
        self.setColumnWidth(2, 150)  # Created time
        self.setColumnWidth(3, 200)  # Details

        # Set resize modes
        header = self.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Name column expands
        header.setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )  # Duplicates column packs left
        header.setSectionResizeMode(
            2, QHeaderView.ResizeToContents
        )  # Created time column packs left
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Details column packs left

        # header.setStyleSheet("QTreeWidget::item { padding-right: 40px; }")  # Adjust the value as needed

        self.popup_graph = PopupGraph()

    def refresh_tree(self):
        """
        Updates the tree view with the organized data.
        """
        self.clear()
        self.data_parents.clear()

        def add_items(parent, structure, parent_names):
            for key, value in structure.items():
                if isinstance(value, DataGroup):
                    a_df = value.active.df.copy()
                    details_str = f"{a_df.shape[0]}x{a_df.shape[1]} DataFrame"
                    # if ("imps", "frequency") in a_df.columns:
                    f_data = a_df.get(
                        ("imps", "frequency"),
                        a_df.get(find_nearest(a_df.columns, "freq", False, ignore_case=True)),
                    )
                    if f_data is not None:
                        # f_data = a_df[("imps", "frequency")]
                        details_str += f"; Freq. range: {sci_note(f_data.min(),2)} - {sci_note(f_data.max(),2)} Hz"
                        if (repeats := len(f_data) // f_data.nunique()) > 1:
                            details_str += f"; has {repeats} combined sets"
                    tree_item = QTreeWidgetItem(
                        [
                            key,
                            str(len(value.dfs) - 1),
                            str(value.active.cr_time) + " " * 2,
                            details_str,
                        ]
                    )
                    parent.addChild(tree_item)
                    tree_item.setData(0, Qt.UserRole, parent_names + (key,))
                else:
                    tree_item = QTreeWidgetItem([key])
                    parent.addChild(tree_item)
                    current_parent_names = parent_names + (key,)
                    tree_item.setData(0, Qt.UserRole, current_parent_names)

                    self.data_parents[current_parent_names] = tree_item

                    add_items(tree_item, value, current_parent_names)

                    # Fold or expand logic
                    if key == "residuals":
                        parent.treeWidget().collapseItem(tree_item)
                    else:
                        parent.treeWidget().expandItem(tree_item)

        # Create a temporary dictionary to hold the tree structure
        tree_structure = defaultdict(dict)
        for path, data in self.data_org.items():
            current_level = tree_structure
            for part in path[:-1]:
                current_level = current_level.setdefault(part, {})
            current_level[path[-1]] = data

        # Add items to the tree view
        add_items(self.invisibleRootItem(), tree_structure, ())

    def get_leaf_children(self, arg_item=None):
        """
        Retrieves all bottom-level (leaf) children of the given tree item.
        The input can be None, a tuple key, or a QTreeWidgetItem.
        """
        if isinstance(arg_item, list):
            for item in arg_item:
                yield from self.get_leaf_children(item)
            return

        # Determine the item based on the input type
        item = self.data_parents.get(arg_item) if isinstance(arg_item, tuple) else arg_item

        if item is None:
            # If item is None, use the invisible root item
            item = self.invisibleRootItem()
            leaf_path = arg_item if arg_item in self.data_org else ()
        elif not isinstance(item, QTreeWidgetItem):
            raise TypeError("arg_item must be None, a tuple, or a QTreeWidgetItem")
        else:
            leaf_path = item.data(0, Qt.UserRole)

        path_len = len(leaf_path)  # type: ignore

        # Initialize the iterator with the item and the NoChildren flag
        iterator = QTreeWidgetItemIterator(item, QTreeWidgetItemIterator.NoChildren)

        while current_item := iterator.value():
            if current_item.data(0, Qt.UserRole)[:path_len] != leaf_path:
                break
            yield current_item
            iterator += 1

    def show_tree_context_menu(self, position, items):
        """
        Shows a context menu for the tree item.
        """
        context_menu = QMenu()

        view_action = QAction("View Data", self)
        view_action.triggered.connect(lambda: self.view_tree_item_data(items[0]))
        context_menu.addAction(view_action)

        plot_action = QAction("Plot Data", self)
        plot_action.triggered.connect(lambda: self.view_tree_item_graph(items))
        context_menu.addAction(plot_action)

        # Add a separator
        context_menu.addSeparator()

        # Create a submenu for "Select Version"
        select_version_menu = QMenu("Select Version", self)

        key_path = items[0].data(0, Qt.UserRole)
        data_group = self.data_org.get(key_path)

        if data_group and len(data_group.entries) > 1:
            for index, entry in enumerate(data_group.entries):
                version_action = QAction(entry.name, self)
                version_action.setCheckable(True)
                version_action.setChecked(data_group.active.uid == entry.uid)
                version_action.triggered.connect(
                    lambda checked, idx=index: self.swap_tree_item(key_path, idx)
                )
                select_version_menu.addAction(version_action)
        else:
            select_version_menu.setEnabled(False)

        select_version_action = QAction("Select Version", self)
        select_version_action.setMenu(select_version_menu)
        context_menu.addAction(select_version_action)

        # Add another separator
        context_menu.addSeparator()

        rename_action = QAction("Rename", self)
        rename_action.triggered.connect(lambda: self.rename_tree_item(items=items))
        context_menu.addAction(rename_action)

        regex_rename_action = QAction("Rename w/ RegEx", self)
        regex_rename_action.triggered.connect(lambda: self.rename_tree_item_simply(items=items))
        context_menu.addAction(regex_rename_action)

        concat_action = QAction("Combine", self)
        concat_action.triggered.connect(lambda: self.concat_items(items=items))
        context_menu.addAction(concat_action)

        # Add "Split Group" action
        split_action = QAction("Split Group", self)
        split_action.triggered.connect(lambda: self.split_group(items))
        context_menu.addAction(split_action)

        # Add another separator
        context_menu.addSeparator()

        save_action = QAction("Save", self)
        save_action.triggered.connect(lambda: self.save_tree_items(items))
        context_menu.addAction(save_action)

        delete_action = QAction("Remove", self)
        delete_action.triggered.connect(lambda: self.hide_tree_items(items))
        context_menu.addAction(delete_action)

        context_menu.exec_(position)

    def split_group(self, items):
        """
        Splits grouped data into individual datasets.
        """
        for item in self.get_leaf_children(items):
            key_path = item.data(0, Qt.UserRole)
            data_group = self.data_org.get(key_path)

            if data_group and len(data_group.entries) > 1:
                # Remove the grouped entry
                self.data_org.pop(key_path, None)

                # Add each entry as a separate dataset
                for entry in data_group.entries:
                    new_key = tuple(list(key_path[:-1]) + [entry.name])
                    self.data_org[new_key] = DataGroup([entry])

        self.refresh_tree()

    def view_tree_item_data(self, item):
        """
        View the data when a dataset is double-clicked.
        """
        if isinstance(item, (tuple, list)):
            item = item[0]
        if not isinstance(item, QTreeWidgetItem) or item.childCount() > 0:
            return
        data = self.data_org.get(item.data(0, Qt.UserRole))
        if data is not None and isinstance(data.active.df, pd.DataFrame):
            self.data_viewer = DataViewer(
                data.active.df, self, name=" > ".join(item.data(0, Qt.UserRole))
            )

    def view_tree_item_graph(self, items):
        """Trigger the popup graph window or add data."""
        for item in self.get_leaf_children(items):
            data_group = self.data_org.get(item.data(0, Qt.UserRole))
            if data_group:
                df = data_group.active.df

                # Create a new dataframe with the updated column names
                new_df = df.copy(deep=True)
                new_df.columns = [".".join(map(str, col)) for col in df.columns]

                self.popup_graph.handle_trigger(name=data_group.active.name, data=new_df)

    def swap_tree_item(self, key_path, index):
        """
        Replaces the active dataset with the selected version.
        """
        data_group = self.data_org.get(key_path)
        if data_group:
            data_group.active = index  # Set the new active index
            self.update_data_pathkeys(
                {key_path: tuple(key_path[:-1] + (data_group.active.name,))}
            )  # Update the tree view
            self.refresh_tree()

    def rename_tree_item(self, *_, items=None, names=None):
        """Rename the selected dataset."""
        # for item in items:
        if items is None:
            return
        item = items
        if isinstance(items, (tuple, list)):
            item = items[0]

        key_path = item.data(0, Qt.UserRole)
        if key_path in self.data_org:
            new_name = names[0] if isinstance(names, (tuple, list)) else names
            ok = True
            if names is None:
                # Show input dialog to get the new name
                new_name, ok = QInputDialog.getText(
                    self,
                    "Rename Dataset",
                    "Enter new name:",
                    text=key_path[-1],
                )

            if ok:
                if new_name:
                    self.update_data_pathkeys({key_path: tuple(key_path[:-1] + (new_name,))})

                else:
                    self.update_data_pathkeys(
                        {key_path: tuple(key_path[:-1] + (self.data_org[key_path].active.name,))}
                    )
                self.refresh_tree()

    def rename_all_tree_items(self, *_, items=None, names=None):
        """
        Renames the dataset using some of the renaming logic from simplify_data_keys.
        """

        def get_next_number(string_list):
            """
            Determines the next available number for the renaming convention.
            """
            string_list = string_list or []
            next_number = (
                max(
                    [
                        int(re.search(r"_r(\d+)$", name).group(1))  # type: ignore
                        for name in string_list
                        if re.search(r"_r(\d+)$", name)
                    ],
                    default=0,
                )
                + 1
            )
            return next_number

        reset_names = False
        gr_names = {}
        name_dict = {}
        init_name_dict = {}
        for item in self.get_leaf_children(items):
            touple_key = item.data(0, Qt.UserRole)
            if callable(names):
                name = names(touple_key[:-1])
            elif not isinstance(names, str):
                name = ""
            else:
                name = names

            new_key = tuple(touple_key[:-1] + (self.data_org[touple_key].active.name,))
            if items is None or not reset_names:
                if touple_key[:-1] not in gr_names:
                    gr_items = list(self.get_leaf_children(item.parent()))
                    gr_names[touple_key[:-1]] = [it.data(0, Qt.UserRole)[-1] for it in gr_items]

                next_number = get_next_number(gr_names[touple_key[:-1]])

                if next_number != 1 and next_number == len(gr_names[touple_key[:-1]]):
                    reset_names = True
                else:
                    if new_key != touple_key:
                        init_name_dict[touple_key] = new_key
                        if next_number == 1:
                            reset_names = True

                    if re.search(r"_r(\d+)$", touple_key[-1]):
                        new_key = touple_key
                    else:
                        new_key = tuple(touple_key[:-1] + (f"{name}_r{next_number}",))
                        gr_names[touple_key[:-1]].append(f"{name}_r{next_number}")

            if new_key != touple_key:
                name_dict[touple_key] = new_key

        if (items is not None and reset_names) or not name_dict:
            name_dict.update(init_name_dict)
        self.update_data_pathkeys(name_dict)
        self.refresh_tree()

    def rename_tree_item_simply(self, *_, items=None):
        """
        Simplifies renaming of tree items by applying a regex pattern to extract new names.

        Parameters:
        - items (list or QTreeWidgetItem): The tree items to rename.

        Returns:
        - list: A list of new names derived from the regex pattern.
        """
        if not items:
            return []

        # Ensure items is a list
        if not isinstance(items, (list, tuple)):
            items = [items]

        # Prompt the user for a regex pattern with a brief "cheatsheet"
        regex_pattern, ok = QInputDialog.getText(
            self,
            "Rename Items - Regex Pattern",
            (
                "Enter regex pattern to extract new names (use parentheses for capture groups):\n"
                "Cheatsheet:\n"
                "  ( )  - Capturing group\n"
                "  (?: ) - Non-capturing group\n"
                "  (?= ) - Lookahead assertion\n"
                "  (?<= ) - Lookbehind assertion\n"
                "Example: .*/(.*) extracts the last part after '/'."
            ),
            text=".*/(.*)",
        )

        if not ok:
            return []

        regex = re.compile(".*")
        use_groups = False
        if regex_pattern == ".*" or not regex_pattern.strip():
            use_groups = True

        # Compile the regex pattern
        if not use_groups:
            try:
                regex = re.compile(regex_pattern)
            except re.error as e:
                # Handle invalid regex
                QInputDialog.getText(
                    self,
                    "Invalid Regex",
                    f"Error in regex pattern: {e}. Please try again.",
                )
                return []

        join_string = None  # To store the string for joining groups if needed
        new_names = {}
        for item in self.get_leaf_children(items):
            # Get the current name of the item
            key_path = item.data(0, Qt.UserRole)
            if key_path not in self.data_org:
                continue
            current_name = key_path[-1] if key_path else ""

            new_name = ""
            if use_groups and len(key_path) > 1:
                # Construct a regex pattern dynamically from the grouping strings
                group_strings = [re.escape(group) for group in key_path[:-1]]
                group_pattern = r"|".join(group_strings)  # Match any of the group strings
                regex = re.compile(rf"(?:{group_pattern})", flags=re.IGNORECASE)

                # Remove the group strings from the current name
                new_name = regex.sub("", current_name).strip()
            else:
                # Apply the regex to extract the new name
                match = regex.search(current_name)
                if match:
                    # Handle multiple capture groups
                    if len(match.groups()) > 1:
                        if join_string is None:
                            # Prompt the user for a string to join groups
                            join_string, ok = QInputDialog.getText(
                                self,
                                "Join Multiple Groups",
                                "Multiple groups detected. Enter a string to join them:",
                                text="_",
                            )
                            if not ok:
                                join_string = "_"

                        # Join the groups using the specified string
                        new_name = join_string.join(match.groups())
                    else:
                        # Use the first capture group if available, otherwise use the full match
                        new_name = match.group(1) if match.groups() else match.group(0)
            if new_name:
                # Perform substitutions to remove duplicate separators
                new_name = re.sub(
                    r"[\s_/]*([\s_/])", r"\1", new_name
                )  # Replace sequences of separators with the last one
                new_name = new_name.strip("_/ ")  # Strip leading/trailing separators

                new_names[key_path] = tuple(key_path[:-1] + (new_name,))

            #     new_names.append(new_name)
            # else:
            #     # If no match, keep the original name
            #     new_names.append(current_name)
            # self.update_data_pathkeys({key_path: tuple(key_path[:-1] + (new_name,))})
        self.update_data_pathkeys(new_names)
        self.refresh_tree()
        # return new_names

    def concat_items(self, *_, items=None, new_name=None):
        """
        Concatenates the DataFrames and replaces the last string in the path with 'data' or a user-provided name.
        If items are provided, it processes selected items; otherwise, it processes all items.
        """
        if items is not None:
            key_path = items[0].data(0, Qt.UserRole)
            if not new_name:
                new_name, ok = QInputDialog.getText(
                    self,
                    "Combine Datasets",
                    "Enter new name:",
                    text=" ".join(key_path[:-1]),
                )
                if not ok or not new_name:
                    return
        else:
            new_name = new_name or "data"

        groups = defaultdict(dict)
        for item in self.get_leaf_children(items):
            key_path = item.data(0, Qt.UserRole)
            data_group = self.data_org.get(key_path)
            if items is not None:
                self.data_org.pop(key_path, None)
            sub_key = key_path[-1]
            if data_group is None:
                raise ValueError(f"Data group for key path {key_path} not found in data_org.")
            groups[key_path[:-1]][sub_key] = data_group.active.df

        if items is None:
            self.data_org = {}

        self.data_org.update(
            {
                tuple(key + (new_name,)): DataGroup(
                    [parse_df_to_entry(impedance_concat(dfs), name=new_name)]
                )
                for key, dfs in groups.items()
            }
        )

        self.refresh_tree()

    def save_tree_items(self, items):
        """
        Saves the selected datasets to a file.
        """
        data_to_save = {}
        for item in self.get_leaf_children(items):
            key_path = item.data(0, Qt.UserRole)
            if key_path in self.data_org:
                data_to_save["/".join(key_path)] = self.data_org[key_path].active.df

        if data_to_save:
            self.saveDataRequested.emit(data_to_save)

    def get_flat_data_org(self, sep="/", data=None):
        """
        Sanitizes data_org for use elsewhere.
        """
        if not isinstance(data, dict):
            data = self.data_org

        result = {}
        for key, val in data.items():
            result[sep.join(key)] = val.active.df
        return result

    def hide_tree_items(self, items):
        """
        Removes a dataset from the organized data.
        """
        for item in self.get_leaf_children(items):
            key_path = item.data(0, Qt.UserRole)
            self.data_org.pop(key_path, None)

        self.refresh_tree()

    def set_all_data(self, df_dict, deep_copy=False):
        """
        Sets the all_data dictionary using the provided dictionary of DataFrames.
        """
        df_dict = flatten_dict(df_dict)
        if deep_copy:
            df_dict = {key: df.copy() for key, df in df_dict.items()}

        self.all_data = {}
        for key, df in df_dict.items():
            entry = parse_df_to_entry(df, name=key)
            self.all_data[entry.uid] = entry

    # def set_data_org(self):
    #     """
    #     Sets the data_org dictionary using the all_data dictionary.
    #     """

    #     def check_duplicates(old, new):
    #         if old is None:
    #             return DataGroup([new])

    #         try:
    #             time_delta = int(
    #                 new.df.attrs.get("changedtimestamp", 1)
    #                 - new.df.attrs.get("createdtimestamp", 1)
    #             )
    #         except TypeError:
    #             time_delta = 0

    #         try:
    #             new_name = new.df.attrs.get("name", new.name)
    #             old_name = old.active.df.attrs.get("name", old.active.name)
    #             pattern = re.compile("loaded")
    #             if "/" in new_name and "/" in old_name:
    #                 pattern = re.compile(r"/[a-zA-Z]")
    #             if (
    #                 # "loaded" not in new_name
    #                 pattern.search(new_name) is None
    #                 and time_delta > 1
    #                 and (
    #                     new_name >= old_name
    #                     or pattern.search(old_name) is not None
    #                     # or "loaded" in old_name
    #                 )
    #             ):
    #                 old.append(new)
    #                 old.active = len(old.entries) - 1
    #             else:
    #                 old.append(new)
    #         except (TypeError, IndexError, AttributeError):
    #             old.append(new)
    #         return old

    #     duplicates = {}
    #     self.data_org = {}
    #     for val in self.all_data.values():
    #         key = (val.cr_time, val.row_0_hash)
    #         duplicates[key] = check_duplicates(
    #             duplicates.get(key),
    #             val,
    #         )
    #     for val in duplicates.values():
    #         self.data_org[(val.active.name,)] = val

    #     self.refresh_tree()

    def set_data_org(self):
        """
        Organizes datasets into the data_org dictionary using the all_data dictionary.
        Groups datasets by (cr_time, row_0_hash) and evaluates duplicates.
        """

        def evaluate_active_dataset(old, new):
            """
            Evaluates whether the new dataset should be added to the DataGroup and/or set as active.

            Parameters:
            - old (DataGroup): The existing DataGroup for the dataset.
            - new (DatasetEntry): The new dataset to evaluate.

            Returns:
            - DataGroup: The updated DataGroup.
            """
            try:
                # Calculate the time delta between changed and created timestamps
                time_delta = int(
                    new.df.attrs.get("changedtimestamp", 1)
                    - new.df.attrs.get("createdtimestamp", 1)
                )
            except TypeError:
                time_delta = 0

            try:
                # Extract names for comparison
                new_name = new.df.attrs.get("name", new.name)
                old_name = old.active.df.attrs.get("name", old.active.name)

                # Define a pattern for name comparison
                pattern = re.compile("loaded")
                if "/" in new_name and "/" in old_name:
                    pattern = re.compile(r"/[a-zA-Z]")

                # Determine if the new dataset should replace the active one
                if (
                    pattern.search(new_name) is None
                    and time_delta > 1
                    and (new_name >= old_name or pattern.search(old_name) is not None)
                ):
                    old.append(new)
                    old.active = len(old.entries) - 1  # Set the new dataset as active
                else:
                    old.append(new)  # Add the new dataset without changing the active one
            except (TypeError, IndexError, AttributeError):
                old.append(new)  # Add the new dataset in case of an error

            return old

        duplicates = {}
        self.data_org = {}

        # Group datasets by (cr_time, row_0_hash)
        for val in self.all_data.values():
            key = (val.cr_time, val.row_0_hash)
            if key not in duplicates:
                # Create a new DataGroup if no group exists for the key
                duplicates[key] = DataGroup([val])
            else:
                # Evaluate the new dataset against the existing group
                duplicates[key] = evaluate_active_dataset(duplicates[key], val)

        # Populate data_org with the grouped datasets
        for val in duplicates.values():
            self.data_org[(val.active.name,)] = val

        self.refresh_tree()

    def update_data_pathkeys(self, path_updates):
        """
        Updates the paths in data_org based on the provided dictionary or mapping function.
        """
        if callable(path_updates):
            # If path_updates is a function, apply it to each key in data_org
            for old_path in list(self.data_org.keys()):
                if old_path in self.data_org:
                    new_path = path_updates(old_path)
                    self.data_org[new_path] = self.data_org.pop(old_path)
        elif isinstance(path_updates, dict):
            # If path_updates is a dictionary, use it to update the paths
            for old_path, new_path in path_updates.items():
                if old_path in self.data_org:
                    self.data_org[new_path] = self.data_org.pop(old_path)
        else:
            raise TypeError("path_updates must be a dict or a callable")

        self.refresh_tree()

    def eval_grouping(self, search_terms, reject_terms, keys, suppress_refresh=False):
        """
        Separates a dictionary into multiple dictionaries based on search terms.
        """
        # Determine keys for the result dictionary
        if keys is None:
            keys = [str(term) for term in search_terms]
        keys.append("residuals")

        patterns = compile_search_patterns(search_terms, reject_terms)
        for key in list(self.data_org.keys()):
            for pattern, key_name in zip(patterns, keys):
                if key in self.data_org and pattern.search(key[-1]):
                    new_key = tuple(list(key[:-1]) + [key_name] + [key[-1]])
                    self.data_org[new_key] = self.data_org.pop(key)
                    break
            else:
                new_key = tuple(list(key[:-1]) + ["residuals"] + [key[-1]])
                self.data_org[new_key] = self.data_org.pop(key)
        if not suppress_refresh:
            self.refresh_tree()
        # self.refresh_tree()

    # def show_tree_context_menu(self, position, items):
    #     """
    #     Shows a context menu for the tree item.
    #     """
    #     # items = list(self.get_leaf_children(items))

    #     context_menu = QMenu()

    #     view_action = QAction("View Data", self)
    #     view_action.triggered.connect(
    #         lambda: self.view_tree_item_data(items[0])
    #     )
    #     context_menu.addAction(view_action)

    #     plot_action = QAction("Plot Data", self)
    #     plot_action.triggered.connect(lambda: self.view_tree_item_graph(items))
    #     context_menu.addAction(plot_action)

    #     # Add a separator
    #     context_menu.addSeparator()

    #     # Create a submenu for "Select Version"
    #     select_version_menu = QMenu("Select Version", self)

    #     # key_path = items[0].data(0, Qt.UserRole)
    #     # if items[0].childCount() > 0:
    #     #     key_path = list(self.get_leaf_children(items[0]))[0].data(0, Qt.UserRole)
    #     key_path = items[0].data(0, Qt.UserRole)
    #     data_group = self.data_org.get(key_path)
    #     # Get the dataset key path
    #     # data_group, key_path = self.get_tree_item_data(items[0])

    #     if data_group and len(data_group.entries) > 1:
    #         for index, entry in enumerate(data_group.entries):
    #             version_action = QAction(entry.name, self)
    #             version_action.setCheckable(True)
    #             version_action.setChecked(data_group.active.uid == entry.uid)
    #             version_action.triggered.connect(
    #                 lambda checked, idx=index: self.swap_tree_item(
    #                     key_path, idx
    #                 )
    #             )
    #             select_version_menu.addAction(version_action)
    #     else:
    #         select_version_menu.setEnabled(False)

    #     select_version_action = QAction("Select Version", self)
    #     select_version_action.setMenu(select_version_menu)
    #     context_menu.addAction(select_version_action)

    #     # Add another separator
    #     context_menu.addSeparator()

    #     rename_action = QAction("Rename", self)
    #     rename_action.triggered.connect(lambda: self.rename_tree_item(items))
    #     context_menu.addAction(rename_action)

    #     rename_action = QAction("Simple Name", self)
    #     rename_action.triggered.connect(
    #         lambda: self.rename_all_tree_items(items=items, names=" ".join)
    #     )
    #     context_menu.addAction(rename_action)

    #     concat_action = QAction("Combine", self)
    #     concat_action.triggered.connect(lambda: self.concat_items(items=items))
    #     context_menu.addAction(concat_action)

    #     # Add another separator
    #     context_menu.addSeparator()

    #     save_action = QAction("Save", self)
    #     save_action.triggered.connect(lambda: self.save_tree_items(items))
    #     context_menu.addAction(save_action)

    #     delete_action = QAction("Remove", self)
    #     delete_action.triggered.connect(lambda: self.hide_tree_items(items))
    #     context_menu.addAction(delete_action)

    #     context_menu.exec_(position)

    # def get_data_groupings(self, items=None, rem_from_tree=False):
    #     """
    #     Groups the data by path[:-1] with subdicts keyed by path[-1].
    #     The input can be None, a tuple key, or a QTreeWidgetItem.
    #     """
    #     groups = defaultdict(dict)
    #     for item in self.get_leaf_children(items):
    #         key_path = item.data(0, Qt.UserRole)
    #         data_group = self.data_org.get(key_path)
    #         if rem_from_tree:
    #             self.data_org.pop(key_path, None)
    #         sub_key = key_path[-1]
    #         groups[key_path[:-1]][sub_key] = data_group.active.df

    #     return groups

    # def concat_tree_items(self, items):
    #     """
    #     Concatenates the DataFrames and replaces the last string in the path with 'data'.
    #     """
    #     # _, key_path = self.get_tree_item_data(items[0])
    #     key_path = items[0].data(0, Qt.UserRole)

    #     new_name, ok = QInputDialog.getText(
    #         self,
    #         "Combine Datasets",
    #         "Enter new name:",
    #         text= " ".join(key_path[:-1]),
    #     )

    #     if ok and new_name:
    #         groups = self.get_data_groupings(items, "df", True)
    #         self.data_org.update(
    #             {
    #                 tuple(key + (new_name,)): DataGroup(
    #                     [
    #                         parse_df_to_entry(
    #                             impedance_concat(dfs), name=new_name
    #                         )
    #                     ]
    #                 )
    #                 for key, dfs in groups.items()
    #             }
    #         )

    #         self.refresh_tree()

    # def concat_all_tree_items(self):
    #     """
    #     Concatenates the DataFrames and replaces the last string in the path with 'data'.
    #     """
    #     groups = self.get_data_groupings()
    #     self.data_org = {
    #         tuple(key + ("data",)): DataGroup(
    #             [parse_df_to_entry(impedance_concat(dfs), name="data")]
    #         )
    #         for key, dfs in groups.items()
    #     }

    #     self.refresh_tree()


# def construct_data_items(self, items):
#     """
#     Safely constructs data_items from the given items.
#     """
#     if isinstance(items, QTreeWidgetItem):
#         items = [items]

#     data_items = []
#     for item in items:
#         if not isinstance(item, QTreeWidgetItem):
#             return []
#         if item.childCount() == 0:
#             data_items.extend(list(self.get_leaf_children(item.parent())))
#         else:
#             data_items.extend(list(self.get_leaf_children(item)))
#     return data_items

# def construct_flat_dict(self, data_items):
#     """
#     Constructs a flat dictionary with tuple keys similar to self.data_org.
#     """
#     flat_dict = {}
#     for data_item in data_items:
#         key_path = data_item.data(0, Qt.UserRole)
#         data_group = self.data_org[key_path]

#         if data_group:
#             flat_dict[key_path] = data_group
#     return flat_dict

# def construct_nested_dict(self, flat_dict, target="df", grp_lvl=-1, rem_from_tree=False):
#     """
#     Constructs a nested dictionary from the flat dictionary.
#     """
#     groups = defaultdict(dict)
#     for key_path, data_group in flat_dict.items():
#         if rem_from_tree:
#             self.data_org.pop(key_path, None)
#         sub_key = key_path[grp_lvl] if grp_lvl == -1 or grp_lvl == len(key_path) - 1 else key_path[grp_lvl:]
#         if isinstance(target, str) and target in DatasetEntryIndex:
#             groups[key_path[:grp_lvl]][sub_key] = data_group.active[DatasetEntryIndex[target]]
#         else:
#             groups[key_path[:grp_lvl]][sub_key] = data_group
#     return groups

# def get_data_groupings(self, items=None, target="df", rem_from_tree=False, grp_lvl=-1):
#     """
#     Groups the data by path[:-1] with subdicts keyed by path[-1].
#     The input can be None, a tuple key, or a QTreeWidgetItem.
#     """
#     if isinstance(grp_lvl, (tuple, list)):
#         grp_lvl = len(grp_lvl)

#     if isinstance(items, QTreeWidgetItem):
#         items = [items]

#     if not isinstance(items, (list, tuple)):
#         # If no items are provided, group all data in self.data_org
#         flat_dict = self.data_org
#     else:
#         data_items = self.construct_data_items(items)
#         flat_dict = self.construct_flat_dict(data_items)

#     return self.construct_nested_dict(flat_dict, target, grp_lvl, rem_from_tree)

# def find_data_by_tuple(self, key_tuple):
#     """
#     Finds the data associated with the given tuple key.
#     """
#     # Check if the key is in self.data_org
#     if key_tuple in self.data_org:
#         return {key_tuple: self.data_org[key_tuple]}
#     else:
#         items = list(self.get_leaf_children(key_tuple))
#         return {item.data(0, Qt.UserRole): self.data_org[item.data(0, Qt.UserRole)] for item in items}


# def get_leaf_children(self, arg_item=None):
#     """
#     Retrieves all bottom-level (leaf) children of the given tree item.
#     The input can be None, a tuple key, or a QTreeWidgetItem.
#     """
#     if isinstance(arg_item, list):
#         res = []
#         for item in arg_item:
#             res.extend(self.get_leaf_children(item))
#         return res

#     # Determine the item based on the input type
#     item = self.data_parents.get(arg_item) if isinstance(arg_item, tuple) else arg_item

#     if item is None:
#         # If item is None, use the invisible root item
#         item = self.invisibleRootItem()
#         leaf_path = arg_item if arg_item in self.data_org else ()
#     elif not isinstance(item, QTreeWidgetItem):
#         raise TypeError("arg_item must be None, a tuple, or a QTreeWidgetItem")
#     else:
#         leaf_path = item.data(0, Qt.UserRole)

#     leaf_children = []
#     path_len = len(leaf_path)

#     # Initialize the iterator with the item and the NoChildren flag
#     iterator = QTreeWidgetItemIterator(item, QTreeWidgetItemIterator.NoChildren)

#     while (current_item := iterator.value()):
#         if current_item.data(0, Qt.UserRole)[:path_len] != leaf_path:
#             break
#         leaf_children.append(current_item)
#         iterator += 1

#     return leaf_children

# def get_parent_items(self, items, as_list=False):
#     """
#     Safely constructs data_items from the given items.
#     """
#     if isinstance(items, QTreeWidgetItem):
#         items = [items]

#     parent_items = {}
#     for item in items:
#         if not isinstance(item, QTreeWidgetItem):
#             return {}
#         if item.childCount() == 0 and item.parent():
#             item = item.parent()


#         parent_items[item.data(0, Qt.UserRole)] = item

#             if as_tuple:
#                 parent_items.append(item.parent().data(0, Qt.UserRole))
#             else:
#                 parent_items.append(item.parent())
#         elif item.childCount() > 0 and item not in parent_items:
#             if as_tuple:
#                 parent_items.append(item.data(0, Qt.UserRole))
#             else:
#                 parent_items.append(item)
#     return parent_items


# def get_data_groupings(self, items=None, target="df", rem_from_tree=False, grp_lvl=-1):
#     """
#     Groups the data by path[:-1] with subdicts keyed by path[-1].
#     The input can be None, a tuple key, or a QTreeWidgetItem.
#     """
#     if isinstance(grp_lvl, (tuple, list)):
#         grp_lvl = len(grp_lvl)

#     if isinstance(items, QTreeWidgetItem):
#         items = [items]

#     groups = defaultdict(dict)

#     if not isinstance(items, (list, tuple)):
#         # If no items are provided, group all data in self.data_org
#         for key_path, data_group in self.data_org.items():
#             sub_key = key_path[grp_lvl] if grp_lvl == -1 or grp_lvl == len(key_path) - 1 else key_path[grp_lvl:]
#             if isinstance(target, str) and target in DatasetEntryIndex:
#                 groups[key_path[:grp_lvl]][sub_key] = data_group.active[DatasetEntryIndex[target]]
#             else:
#                 groups[key_path[:grp_lvl]][sub_key] = data_group
#     else:
#         data_items = []
#         for item in items:
#             if not isinstance(item, QTreeWidgetItem):
#                 return
#             if item.childCount() == 0:
#                 data_items.extend(self.get_leaf_children(item.parent()))
#             else:
#                 data_items.extend(self.get_leaf_children(item))

#         for data_item in data_items:
#             key_path = data_item.data(0, Qt.UserRole)
#             data_group = self.data_org[key_path]
#             if rem_from_tree:
#                 self.data_org.pop(key_path, None)
#             if data_group:
#                 sub_key = key_path[grp_lvl] if grp_lvl == -1 or grp_lvl == len(key_path) - 1 else key_path[grp_lvl:]
#                 if isinstance(target, str) and target in DatasetEntryIndex:
#                     groups[key_path[:grp_lvl]][sub_key] = data_group.active[DatasetEntryIndex[target]]
#                 else:
#                     groups[key_path[:grp_lvl]][sub_key] = data_group

#     return groups

# def get_data_groupings(self, items=None, target="df", rem_from_tree=False, grp_lvl=-1):
#     """
#     Groups the data by path[:-1] with subdicts keyed by path[-1].
#     """
#     if isinstance(grp_lvl, (tuple, list)):
#         grp_lvl = len(grp_lvl)

#     if isinstance(items, QTreeWidgetItem):
#         items = [items]

#     groups = defaultdict(dict)
#     if not isinstance(items, (list, tuple)):
#         for key_path, data_group in self.data_org.items():
#             sub_key = key_path[grp_lvl] if grp_lvl == -1 or grp_lvl == len(key_path) - 1 else key_path[grp_lvl:]
#             if isinstance(target, str) and target in DatasetEntryIndex:
#                 groups[key_path[:grp_lvl]][sub_key] = data_group.active[DatasetEntryIndex[target]]  # if as_df else data_group
#             else:
#                 groups[key_path[:grp_lvl]][sub_key] = data_group
#     else:
#         for item in items:
#             data_group, key_path = self.get_tree_item_data(item)
#             if rem_from_tree:
#                 self.data_org.pop(key_path, None)
#             if data_group:
#                 sub_key = key_path[grp_lvl] if grp_lvl == -1 or grp_lvl == len(key_path) - 1 else key_path[grp_lvl:]
#                 if isinstance(target, str) and target in DatasetEntryIndex:
#                     groups[key_path[:grp_lvl]][sub_key] = data_group.active[
#                         DatasetEntryIndex[target]
#                     ]  # if as_df else data_group
#                 else:
#                     groups[key_path[:grp_lvl]][sub_key] = data_group
#     return groups

# def get_tree_item_data(self, item):
#     """
#     Retrieve the data and key path from the tree item.
#     """
#     return self.data_org.get(item.data(0, Qt.UserRole)), item.data(0, Qt.UserRole)

# def get_tree_item_data(self, item):
#     """
#     Retrieve the data and key path from the tree item.
#     """
#     key_path_list = []
#     while item:
#         key_path_list.insert(0, item.text(0))
#         item = item.parent()

#     key_path = tuple(key_path_list)
#     return self.data_org.get(key_path), key_path

# def get_leaf_children(self, items):
#     """
#     Returns a flat list of children that have no further children.
#     """
#     leaf_children = []

#     def find_leaf_children(check_item):
#         if check_item.childCount() == 0:
#             leaf_children.append(check_item)
#         else:
#             for i in range(check_item.childCount()):
#                 find_leaf_children(check_item.child(i))

#     if not isinstance(items, list):
#         items = [items]
#     for item in items:
#         find_leaf_children(item)
#     return leaf_children
# t_groups = self.get_parent_items(items, True)
# groups = self.get_data_groupings(items, None)
# name_dict = {}
# alternate_name_dict = {}
# all_keys_same = True

# for group_key, group in groups.items():
#     if callable(names):
#         name = names(group_key)
#     elif not isinstance(names, str):
#         name = ""
#     else:
#         name = names

#     next_number = max(
#         [
#             int(re.search(r'_r(\d+)$', name).group(1))
#             for name in existing_names
#             if re.search(r'_r(\d+)$', name)
#         ],
#         default=0,
#     ) + 1

#     for n, k in enumerate(group.keys(), next_number):
#         new_key = tuple(group_key + (f"{name}_r{n}",))
#         old_key = tuple(group_key + (k,))
#         name_dict[old_key] = new_key

#         # Update alternate_name_dict if all_keys_same is True
#         if all_keys_same:
#             if new_key != old_key:
#                 all_keys_same = False
#             else:
#                 alternate_name_dict[old_key] = tuple(
#                     group_key + (group[k].active.name,)
#                 )

# if all_keys_same:
#     self.update_data_pathkeys(alternate_name_dict)
#     self.refresh_tree()
# else:
#     self.update_data_pathkeys(name_dict)
#     self.refresh_tree()
# # If all keys were the same, update the tree view with alternate names
# def get_data_groupings(self, items=None, target="df", rem_from_tree=False, grp_lvl=-1):
#     """
#     Groups the data by path[:-1] with subdicts keyed by path[-1].
#     """
#     if isinstance(grp_lvl, (tuple, list)):
#         grp_lvl = len(grp_lvl)

#     if isinstance(items, QTreeWidgetItem):
#         items = [items]

#     groups = defaultdict(dict)
#     if not isinstance(items, (list, tuple)):
#         for key_path, data_group in self.data_org.items():
#             sub_key = key_path[grp_lvl] if grp_lvl == -1 or grp_lvl == len(key_path) - 1 else key_path[grp_lvl:]
#             if isinstance(target, str) and target in DatasetEntryIndex:
#                 groups[key_path[:grp_lvl]][sub_key] = data_group.active[DatasetEntryIndex[target]]  # if as_df else data_group
#             else:
#                 groups[key_path[:grp_lvl]][sub_key] = data_group
#     else:
#         for item in items:
#             data_group, key_path = self.get_tree_item_data(item)
#             if rem_from_tree:
#                 self.data_org.pop(key_path, None)
#             if data_group:
#                 sub_key = key_path[grp_lvl] if grp_lvl == -1 or grp_lvl == len(key_path) - 1 else key_path[grp_lvl:]
#                 if isinstance(target, str) and target in DatasetEntryIndex:
#                     groups[key_path[:grp_lvl]][sub_key] = data_group.active[
#                         DatasetEntryIndex[target]
#                     ]  # if as_df else data_group
#                 else:
#                     groups[key_path[:grp_lvl]][sub_key] = data_group
#     return groups


#     def concat_tree_items(self, items):
#         """
#         Concatenates the DataFrames and replaces the last string in the path with 'data'.
#         """
#         _, key_path = self.get_tree_item_data(items[0])

#         new_name, ok = QInputDialog.getText(self, "Combine Datasets", "Enter new name:", text=" ".join(key_path[:-1]))

#         if ok and new_name:

#             groups = self.get_data_groupings(items, "df", True)

#             self.data_org.update({tuple(key + (new_name,)): DataGroup([parse_df_to_entry(impedance_concat(dfs), name=new_name)]) for key, dfs in groups.items()})

#             self.refresh_tree()

#             # groups = defaultdict(dict)
#             # for item in items:
#             #     data_group, key_path = self.get_tree_item_data(item)
#             #     self.data_org.pop(key_path, None)
#             #     if data_group:
#             #         group_key = key_path[:-1]
#             #         groups[group_key][key_path[-1]] = data_group.active.df
#             #         # key = tuple(key_path[:-1])
#             #         # if key not in groups:
#             #         #     groups[key] = []
#             #         # groups[key].append(data_group.active.df)
