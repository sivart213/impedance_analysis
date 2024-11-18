# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
import sys
import json
import re
from pathlib import Path
from datetime import datetime

import pandas as pd

from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtWidgets import (
    QDialog,
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QFileDialog,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QFormLayout,
    QLineEdit,
    QTabWidget,
    QFrame,
    QHBoxLayout,
    QMessageBox,
    QTableWidget,
    QListWidget,
    QGridLayout,
    QInputDialog,
    QListWidgetItem,
    QTableWidgetItem,
    QCheckBox,
    QAbstractItemView,
    QComboBox,
    QDialogButtonBox,
)
from PyQt5.QtGui import QBrush, QColor

from .gui_windows import DataViewer
from ..data_treatment import (
    remove_duplicate_datasets,
    impedance_concat,
    simplify_multi_index,
)
from ..dict_ops import separate_dict, flatten_dict, dict_key_sep
from ..string_ops import re_not
from ..equipment.mfia_ops import parse_mfia_file, convert_mfia_data, convert_mfia_df_for_fit

from ..system_utilities import find_files, load_file, load_hdf, save
from ..system_utilities.special_io import parse_files

class MFIAFileConverter(QMainWindow):
    """
    A class to convert MFIA files using a GUI.
    Attributes:
        is_xlsx (bool): Flag to indicate if the files are in xlsx format.
        files (list): List of files to be processed.
        in_path (Path): Input directory path.
        t_files (DataFrame): DataFrame containing target files information.
        data_in (dict): Dictionary to store input data.
        data_org (dict): Dictionary to store organized data.
        keywords_list (QLineEdit): Line edit for keywords.
        checked_columns (list): List of checked columns.
        tab_widget (QTabWidget): Tab widget for grouping iterations.
        patterns_list (QListWidget): List widget for file patterns.
        columns_list (QListWidget): List widget for columns.
        files_table (QTableWidget): Table widget for files.
        datasets_tree (QTreeWidget): Tree widget for datasets.
        t_files_tree (QTreeWidget): Tree widget for target files.
        files_datasets_tab_widget (QTabWidget): Tab widget for files and datasets.
        save_format_combo (QComboBox): Combo box for save format.
        get_all_checkbox (QCheckBox): Checkbox to get all files.
    Methods:
        initUI(): Initializes the user interface.
        add_grouping_tab(search_terms="", reject_terms="", keys=""): Adds a grouping tab.
        eventFilter(source, event): Handles key press events for deleting items.
        browse_in_path(): Opens a dialog to browse for input directory.
        add_pattern(): Adds a pattern to the patterns list.
        update_files(): Updates the files table with the selected patterns.
        set_data_org(): Sets the organized data.
        apply_group(tab=None): Applies grouping logic to the data.
        apply_all_groups(): Applies all grouping logic to the data.
        update_tree_view(): Updates the tree view with the organized data.
        update_t_files(): Updates the target files tree view.
        convert_files(): Converts the files based on the selected patterns.
        remove_dataset(item): Removes a dataset from the organized data.
        reset_columns_list(): Resets the columns list with unique columns from the dataframes.
        refresh_columns_list(): Refreshes the columns list keeping any checked list.
        sort_columns_list(): Sorts the columns list.
        concat_data(): Concatenates the data.
        simplify_data(): Simplifies the data for fitting.
        save(): Saves the converted data.
        save_settings(): Saves the current settings.
        load_empty_settings(): Loads empty settings.
        load_all_settings(): Loads all settings.
        load_settings(full=True): Loads the settings from a file.
    """

    def __init__(self):
        super().__init__()
        self.is_xlsx = True
        self.files = None
        self.in_path = None
        self.t_files = None
        self.data_in = None
        self.data_org = None
        self.keywords_list = None
        self.checked_columns = []

        self.setWindowTitle("MFIA File Conversion")

        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)

        def create_separator(frame=None):
            separator = QFrame(frame)
            separator.setFrameShape(QFrame.HLine)
            separator.setFrameShadow(QFrame.Sunken)
            return separator

        max_width = 300

        # Input Path
        self.in_path_label = QLabel("<b>Input Path:</b>")
        in_path_button = QPushButton("Browse")
        in_path_button.setFixedWidth(100)
        in_path_button.clicked.connect(self.browse_in_path)
        in_path_layout = QHBoxLayout()
        in_path_layout.addWidget(self.in_path_label)
        in_path_layout.addWidget(in_path_button)
        in_path_frame = QFrame()
        in_path_frame.setLayout(in_path_layout)
        in_path_frame.setFrameShape(QFrame.StyledPanel)
        layout.addWidget(in_path_frame)

        layout.addWidget(create_separator())

        # File Patterns
        self.patterns_list = QListWidget()
        self.patterns_list.setMaximumWidth(max_width)
        self.patterns_list.installEventFilter(self)
        self.patterns_edit = QLineEdit()
        self.patterns_edit.setPlaceholderText("pattern")
        self.patterns_edit.returnPressed.connect(self.add_pattern)

        patterns_layout = QVBoxLayout()
        patterns_layout.addWidget(QLabel("<b>File Patterns:</b>"))
        patterns_layout.addWidget(self.patterns_list)

        patterns_edit_layout = QHBoxLayout()
        patterns_edit_layout.addWidget(QLabel("Add Item:"))
        patterns_edit_layout.addWidget(self.patterns_edit)

        patterns_layout.addLayout(patterns_edit_layout)

        patterns_frame = QFrame()
        patterns_frame.setLayout(patterns_layout)
        patterns_frame.setMaximumWidth(max_width)

        # Tab Widget for Grouping Iterations
        self.tab_widget = QTabWidget()
        self.add_grouping_tab()

        # List Widget for Columns
        self.columns_list = QListWidget()
        self.columns_list.setDragDropMode(
            QAbstractItemView.InternalMove
        )  # Enable drag-and-drop reordering
        reset_columns_button = QPushButton("Reset")
        reset_columns_button.clicked.connect(self.reset_columns_list)
        reset_columns_button.setFixedWidth(75)
        refresh_columns_button = QPushButton("Refresh")
        refresh_columns_button.clicked.connect(self.refresh_columns_list)
        refresh_columns_button.setFixedWidth(75)
        sort_columns_button = QPushButton("Sort")
        sort_columns_button.clicked.connect(self.sort_columns_list)
        sort_columns_button.setFixedWidth(75)

        columns_layout = QVBoxLayout()
        columns_layout.addWidget(QLabel("<b>Columns:</b>"))
        columns_layout.addWidget(self.columns_list)
        columns_buttons_layout = QHBoxLayout()
        columns_buttons_layout.addWidget(reset_columns_button)
        columns_buttons_layout.addWidget(refresh_columns_button)
        columns_buttons_layout.addWidget(sort_columns_button)
        columns_layout.addLayout(columns_buttons_layout)

        columns_frame = QFrame()
        columns_frame.setLayout(columns_layout)
        columns_frame.setMaximumWidth(max_width)

        grid1 = QGridLayout()
        grid1.addWidget(patterns_frame, 0, 0)
        grid1.addWidget(self.tab_widget, 0, 1)
        grid1.addWidget(columns_frame, 0, 2)

        layout.addLayout(grid1)

        layout.addWidget(create_separator())

        self.files_datasets_tab_widget = QTabWidget()

        # Files Table
        self.files_table = QTableWidget()
        self.files_table.setColumnCount(3)
        self.files_table.setHorizontalHeaderLabels(
            ["File Name", "Relative Path", "Path"]
        )
        self.files_table.setColumnWidth(0, 150)
        self.files_table.setColumnWidth(1, 250)
        self.files_table.setColumnWidth(2, 250)
        files_update_button = QPushButton("Update Files")
        files_update_button.clicked.connect(self.update_files)
        files_update_button.setFixedWidth(150)
        files_layout = QVBoxLayout()
        files_layout.addWidget(self.files_table)
        files_layout.addWidget(files_update_button)
        files_frame = QFrame()
        files_frame.setLayout(files_layout)
        files_frame.setMinimumWidth(int(max_width * 1.5))

        self.files_datasets_tab_widget.addTab(files_frame, "Loaded Files")

        # Tree View for Datasets
        datasets_layout = QVBoxLayout()

        self.datasets_tree = QTreeWidget()
        self.datasets_tree.setHeaderLabels(["Name", "Details"])
        self.datasets_tree.setColumnWidth(0, 400)
        self.datasets_tree.installEventFilter(self)
        self.datasets_tree.itemDoubleClicked.connect(self.view_data)
        datasets_layout.addWidget(self.datasets_tree)

        datasets_buttons = QHBoxLayout()

        reset_gr_button = QPushButton("Reset")
        reset_gr_button.clicked.connect(self.set_data_org)
        reset_gr_button.setFixedWidth(150)
        datasets_buttons.addWidget(reset_gr_button)

        apply_gr_button = QPushButton("Apply Groups")
        apply_gr_button.clicked.connect(self.apply_all_groups)
        apply_gr_button.setFixedWidth(150)
        datasets_buttons.addWidget(apply_gr_button)
        
        datasets_rename_button = QPushButton("Rename Datasets")
        datasets_rename_button.clicked.connect(self.simplify_data_keys)
        datasets_rename_button.setFixedWidth(150)
        datasets_buttons.addWidget(datasets_rename_button)

        datasets_concat_button = QPushButton("Combine Datasets")
        datasets_concat_button.clicked.connect(self.concat_data)
        datasets_concat_button.setFixedWidth(150)
        datasets_buttons.addWidget(datasets_concat_button)

        datasets_layout.addLayout(datasets_buttons)

        self.datasets_frame = QFrame()
        self.datasets_frame.setLayout(datasets_layout)
        self.datasets_frame.setMinimumWidth(int(max_width * 1.5))

        self.files_datasets_tab_widget.addTab(self.datasets_frame, "Datasets")

        # Tree Widget for t_files
        t_files_layout = QVBoxLayout()

        self.t_files_tree = QTreeWidget()
        self.t_files_tree.setHeaderLabels(["Name", "ID"])
        t_files_layout.addWidget(self.t_files_tree)

        t_files_update_button = QPushButton("Update Target Files")
        t_files_update_button.clicked.connect(self.update_t_files)
        t_files_update_button.setFixedWidth(150)
        self.get_all_checkbox = QCheckBox("Get All")
        convert_button = QPushButton("Convert")
        convert_button.setFixedWidth(100)
        convert_button.clicked.connect(self.convert_files)

        check_update_layout = QHBoxLayout()
        check_update_layout.addWidget(t_files_update_button)
        check_update_layout.addWidget(self.get_all_checkbox)
        check_update_layout.addWidget(convert_button)
        t_files_layout.addLayout(check_update_layout)

        self.t_files_frame = QFrame()
        self.t_files_frame.setLayout(t_files_layout)
        self.t_files_frame.setMinimumWidth(int(max_width * 1.5))

        self.files_datasets_tab_widget.addTab(
            self.t_files_frame, "Target Files"
        )
        self.files_datasets_tab_widget.setTabVisible(2, False)

        # layout.addLayout(grid2)
        layout.addWidget(self.files_datasets_tab_widget)
        layout.addWidget(create_separator())

        bottom_layout = QHBoxLayout()
        # Save Format
        self.save_format_combo = QComboBox()
        self.save_format_combo.addItems(
            ["Use Columns", "Use Freq, Real, Imag"]
        )
        self.save_format_combo.setFixedWidth(150)
        save_button = QPushButton("Save")
        save_button.setFixedWidth(100)
        save_button.clicked.connect(self.save)

        save_format_layout = QHBoxLayout()
        save_format_layout.addWidget(QLabel("Save Format: "))
        save_format_layout.addWidget(self.save_format_combo)
        save_format_layout.addWidget(save_button)
        save_format_layout.setAlignment(Qt.AlignLeft)

        # Add Buttons for Save and Load Settings
        save_settings_button = QPushButton("Save Settings")
        save_settings_button.clicked.connect(self.save_settings)
        load_settings_button = QPushButton("Load Settings")
        load_settings_button.clicked.connect(self.load_settings)
        # load_all_settings_button = QPushButton("Load All Settings")
        # load_all_settings_button.clicked.connect(self.load_all_settings)

        # Add buttons to the layout
        settings_buttons_layout = QHBoxLayout()
        settings_buttons_layout.addWidget(save_settings_button)
        settings_buttons_layout.addWidget(load_settings_button)
        # settings_buttons_layout.addWidget(load_all_settings_button)
        settings_buttons_layout.setAlignment(Qt.AlignRight)

        bottom_layout.addLayout(save_format_layout)
        bottom_layout.addLayout(settings_buttons_layout)

        layout.addLayout(bottom_layout)

        self.setCentralWidget(central_widget)

    def add_grouping_tab(self, search_terms="", reject_terms="", keys=""):
        """
        Adds a new grouping tab to the tab widget. The tab contains fields for search terms, reject terms, and keys.
        If the tab widget already contains tabs, it clears the existing tabs before adding the new one.
        Args:
            search_terms (str or list, optional): Terms to search for. Defaults to an empty string.
            reject_terms (str or list, optional): Terms to reject. Defaults to an empty string.
            keys (str or list, optional): Keys to use. Defaults to an empty string.
        """
        search_terms = "" if isinstance(search_terms, bool) else search_terms
        search_terms = (
            ", ".join(search_terms)
            if isinstance(search_terms, list)
            else str(search_terms)
        )
        if self.keywords_list is not None:
            search_terms = (
                search_terms if search_terms else self.keywords_list.text()
            )
        if self.is_xlsx:
            if (
                self.tab_widget.count() > 0
                and self.tab_widget.tabText(0) == "Parse Files Keywords"
            ):
                self.tab_widget.clear()
            title = f"Group {self.tab_widget.count() + 1}"
            reject_terms = (
                ", ".join(reject_terms)
                if isinstance(reject_terms, list)
                else str(reject_terms)
            )
            keys = ", ".join(keys) if isinstance(keys, list) else str(keys)

            tab = QWidget()
            layout = QVBoxLayout(tab)
            form_layout = QFormLayout()
            form_layout.addRow(QLabel("Search Terms"), QLineEdit(search_terms))
            form_layout.addRow(QLabel("Reject Terms"), QLineEdit(reject_terms))
            form_layout.addRow(QLabel("Keys"), QLineEdit(keys))
            use_group_button = QPushButton("Apply Group")
            use_group_button.clicked.connect(self.apply_group)
            use_group_button.setFixedWidth(100)
            # Add Group Button
            add_group_button = QPushButton("Add Group")
            add_group_button.setFixedWidth(100)
            add_group_button.clicked.connect(self.add_grouping_tab)
            button_layout = QHBoxLayout()
            button_layout.addWidget(use_group_button)
            button_layout.addWidget(add_group_button)
            layout.addLayout(form_layout)
            layout.addLayout(button_layout)

            tab.setLayout(layout)
            self.tab_widget.addTab(tab, title)

            if self.tab_widget.count() > 1:
                previous_tab = self.tab_widget.widget(
                    self.tab_widget.count() - 2
                )
                previous_layout = previous_tab.layout().itemAt(
                    previous_tab.layout().count() - 1
                )
                previous_layout.removeWidget(
                    previous_layout.itemAt(
                        previous_layout.count() - 1
                    ).widget()
                )
            else:
                self.keywords_list = QLineEdit(search_terms)
        else:
            if self.tab_widget.count() > 0:
                self.tab_widget.clear()
            tab = QWidget()
            # layout = QVBoxLayout(tab)
            form_layout = QFormLayout()
            self.keywords_list = QLineEdit(search_terms)
            form_layout.addRow(QLabel("Keywords"), self.keywords_list)
            tab.setLayout(form_layout)
            self.tab_widget.addTab(tab, "Parse Files Keywords")

    def eventFilter(self, source, event):
        """Handles key press events for deleting items."""
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Delete:
            if source is self.patterns_list:
                for item in self.patterns_list.selectedItems():
                    self.patterns_list.takeItem(self.patterns_list.row(item))
                return True
            elif source is self.datasets_tree:
                selected_items = self.datasets_tree.selectedItems()
                for item in selected_items:
                    self.remove_dataset(item)
                return True
        return super().eventFilter(source, event)
    
    def view_data(self, item, column):
        """View the data when a dataset is double-clicked."""
        key_path = []
        while item:
            key_path.insert(0, item.text(0))
            item = item.parent()

        data = self.data_org
        for key in key_path:
            data = data[key]

        if isinstance(data, pd.DataFrame):
            self.data_viewer = DataViewer(data, self, name=" > ".join(key_path))


    def browse_in_path(self):
        """Opens a dialog to browse for the input directory"""
        path = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if path:
            self.in_path_label.setText(f"<b>Input Path:</b> {path}")
            self.in_path = Path(path)

    def add_pattern(self):
        """Adds a pattern to the patterns list."""

        def add_worker(pattern):
            check = Qt.Unchecked
            if pattern.startswith("not "):
                pattern = pattern.replace("not ", "")
                check = Qt.Checked
            item = QListWidgetItem(pattern.strip())
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(check)
            self.patterns_list.addItem(item)
            self.patterns_edit.clear()

        patterns = self.patterns_edit.text()
        if patterns:
            for p in re.split(r"\s*[,;]\s+", patterns):
                add_worker(p)

    def update_files(self):
        """Updates the files table with the selected patterns."""
        is_xlsx = False
        patterns = []
        for i in range(self.patterns_list.count()):
            item = self.patterns_list.item(i)
            pattern = item.text()
            use_re_not = item.checkState() == Qt.Checked
            if use_re_not:
                pattern = re_not(pattern)
            patterns.append(pattern)
            if "xls" in pattern:
                is_xlsx = True

        self.files = find_files(self.in_path, patterns=patterns)
        self.files_table.setRowCount(len(self.files))
        for row, file in enumerate(self.files):
            relative_path = str(file.relative_to(self.in_path))
            self.files_table.setItem(row, 0, QTableWidgetItem(file.name))
            self.files_table.setItem(row, 1, QTableWidgetItem(relative_path))
            self.files_table.setItem(row, 2, QTableWidgetItem(str(file)))
            if not is_xlsx and file.suffix == ".xlsx":
                is_xlsx = True

        if is_xlsx:
            self.is_xlsx = True
            # self.data_in = {
            #     f.stem: load_file(f, header=[0, 1], index_col=0)[0]
            #     for f in self.files
            #     if f.suffix == ".xlsx"
            # }
            self.data_in = {
                f.stem: load_file(f, index_col=0)[0]
                for f in self.files
                if f.suffix == ".xlsx"
            }
            self.set_data_org()
        else:
            self.files_datasets_tab_widget.setTabVisible(1, False)
            self.files_datasets_tab_widget.setTabVisible(2, True)
            self.is_xlsx = False
            self.add_grouping_tab()

    def set_data_org(self):
        """Sets the organized data."""
        if self.is_xlsx:
            if self.data_in is None and self.files is not None:
                self.data_in = {
                    f.stem: load_file(f, header=[0, 1], index_col=0)[0]
                    for f in self.files
                    if f.suffix == ".xlsx"
                }
            if self.data_in is not None:
                self.data_org = flatten_dict(
                    remove_duplicate_datasets(self.data_in.copy())
                )
                self.update_tree_view()
                self.datasets_tree.setColumnWidth(0, 400)

    def apply_group(self, tab=None):
        """Applies grouping logic to the data."""
        if self.data_org is None:
            QMessageBox.warning(self, "No Files", "Please load files first.")
            return
        # Iterate through the tabs and apply the grouping logic
        if tab is None or not tab:
            tab = self.tab_widget.widget(self.tab_widget.currentIndex())
        form_layout = tab.children()
        if form_layout[2].text():
            search_terms = form_layout[2].text().split(", ")
            reject_terms = (
                form_layout[4].text().split(", ")
                if form_layout[4].text()
                else None
            )
            keys = (
                form_layout[6].text().split(", ")
                if form_layout[6].text()
                else None
            )
            self.data_org = separate_dict(
                self.data_org, search_terms, reject_terms, keys
            )

            self.update_tree_view()

    def apply_all_groups(self):
        """Applies all grouping logic to the data."""
        if self.data_org is None:
            QMessageBox.warning(self, "No Files", "Please load files first.")
            return

        # Iterate through the tabs and apply the grouping logic
        for tab_index in range(self.tab_widget.count()):
            tab = self.tab_widget.widget(tab_index)
            self.apply_group(tab)

    def update_tree_view(self):
        """Updates the tree view with the organized data."""
        if self.is_xlsx:
            self.datasets_tree.clear()

            def add_items(parent, data):
                for data_k, data_v in data.items():
                    if isinstance(data_v, dict):
                        if data_v:
                            tree_item = QTreeWidgetItem([data_k])
                            parent.addChild(tree_item)
                            add_items(tree_item, data_v)
                    elif isinstance(data_v, pd.DataFrame):
                        tree_item = QTreeWidgetItem(
                            [
                                data_k,
                                f"{data_v.shape[0]}x{data_v.shape[1]} DataFrame",
                            ]
                        )
                        parent.addChild(tree_item)

            for key in self.data_org.keys():
                item = QTreeWidgetItem([key])
                self.datasets_tree.addTopLevelItem(item)
                add_items(item, self.data_org[key])

            self.datasets_tree.expandAll()
        else:
            self.t_files_tree.clear()
            self.t_files_tree.setHeaderLabels(self.t_files.columns)
            for index, row in self.t_files.iterrows():
                item = QTreeWidgetItem([str(r) for r in row])
                if index % 2 == 0:
                    for col in range(item.columnCount()):
                        item.setBackground(col, QBrush(QColor("#E2DDEF")))
                self.t_files_tree.addTopLevelItem(item)

    def update_t_files(self):
        """Updates the target files tree view."""
        if self.files is None:
            self.update_files()

        keywords = (
            [k.strip() for k in self.keywords_list.text().split(", ")]
            if self.keywords_list.text()
            else None
        )
        get_all = self.get_all_checkbox.isChecked()
        self.t_files = parse_files(
            self.files, parse_mfia_file, keywords, get_all=get_all
        )
        self.update_tree_view()

    def convert_files(self):
        """Converts the files based on the selected patterns."""
        if self.t_files is None:
            self.update_t_files()
        get_all = self.get_all_checkbox.isChecked()
        data = {}
        for ind in self.t_files.index:
            raw_data = load_hdf(self.t_files["id"][ind], key_sep=True)
            data[self.t_files["name"][ind]] = convert_mfia_data(
                raw_data, flip=False, flatten=2
            )

        self.data_in = {}
        for ind in self.t_files.index:
            data_attrs = pd.DataFrame(
                {
                    k2: v2.attrs
                    for k2, v2 in data[self.t_files["name"][ind]].items()
                }
            ).T.infer_objects()
            tmp = {**data[self.t_files["name"][ind]], **{"attrs": data_attrs}}
            f_name = self.t_files["name"][ind]
            if get_all:
                f_name = self.t_files["id"][ind].parent.stem
            self.data_in[f_name] = tmp

        self.is_xlsx = True
        self.files_datasets_tab_widget.setCurrentIndex(0)
        self.files_datasets_tab_widget.setTabVisible(1, True)
        self.files_datasets_tab_widget.setTabVisible(2, False)
        self.add_grouping_tab()
        self.set_data_org()

    def remove_dataset(self, item):
        """Removes a dataset from the organized data."""
        key_path = []
        while item:
            key_path.insert(0, item.text(0))
            item = item.parent()

        # Remove the dataset from self.data_org
        current_dict = self.data_org
        for key in key_path[:-1]:
            current_dict = current_dict.get(key, {})
        current_dict.pop(key_path[-1], None)

        self.update_tree_view()

    def reset_columns_list(self):
        """Populate the columns list with unique columns from the dataframes ignoring any checked list."""
        if self.data_org is None:
            QMessageBox.warning(
                self, "No Data", "Please load and group data first."
            )
            return

        # Flatten the data organization dictionary
        flattened_data = flatten_dict(self.data_org)

        # Collect all unique columns from the dataframes
        unique_columns = set()
        for df in flattened_data.values():
            if isinstance(df, pd.DataFrame):
                unique_columns.update(df.columns)

        check = Qt.Checked if len(unique_columns) < 10 else Qt.Unchecked
        # Populate the list widget with unique columns
        self.columns_list.clear()
        for col in unique_columns:
            item = QListWidgetItem(str(col))
            item.setFlags(
                item.flags() | Qt.ItemIsUserCheckable
            )  # Allow item to be checkable
            item.setCheckState(check)  # Default to checked
            self.columns_list.addItem(item)

    def refresh_columns_list(self):
        """Populate the columns list with unique columns from the dataframes keeping any checked list."""
        if self.data_org is None and self.checked_columns == []:
            QMessageBox.warning(
                self, "No Data", "Please load and group data first."
            )
            return

        if self.data_org is None:
            for col in self.checked_columns:
                item = QListWidgetItem(str(col))
                item.setFlags(
                    item.flags() | Qt.ItemIsUserCheckable
                )  # Allow item to be checkable
                item.setCheckState(Qt.Checked)  # Default to checked
                self.columns_list.addItem(item)
            return

        if self.columns_list.count() == 0:
            self.reset_columns_list()

        # Flatten the data organization dictionary
        flattened_data = flatten_dict(self.data_org)

        # Collect all unique columns from the dataframes
        unique_columns = set()
        for df in flattened_data.values():
            if isinstance(df, pd.DataFrame):
                unique_columns.update(df.columns)

        # Make sure checked columns match the unique columns
        clean_check = []
        for col in self.checked_columns:
            if col in unique_columns:
                clean_check.append(col)
            elif any(str(col) in str(u_col) for u_col in unique_columns):
                clean_check.append(str(col))
            else:
                for u_col in unique_columns:
                    if str(u_col) in str(col):
                        clean_check.append(u_col)
                        break
        self.checked_columns = (
            clean_check  # if clean_check != [] else self.checked_columns
        )

        items = {}
        c_items = {}
        for i in range(self.columns_list.count()):
            item = self.columns_list.item(i)
            if item.checkState() == Qt.Checked:
                c_items[item.text()] = item
            else:
                items[item.text()] = item

        if self.checked_columns == [] and c_items:
            self.checked_columns = list(c_items.keys())
        elif self.checked_columns != [] and c_items:
            self.checked_columns = [
                c for c in self.checked_columns if c in c_items
            ]
            if items:
                self.checked_columns.extend(
                    [
                        c
                        for c in c_items.keys()
                        if c not in self.checked_columns
                    ]
                )
            else:
                for key, item in c_items.items():
                    if key not in self.checked_columns:
                        item.setCheckState(Qt.Unchecked)
        elif self.checked_columns != [] and not c_items:
            for key, item in items.items():
                if key in self.checked_columns:
                    item.setCheckState(Qt.Checked)
        self.sort_columns_list()

    def sort_columns_list(self):
        """Sort the columns list by check state first, then alphabetically."""
        if self.columns_list.count() == 0:
            self.reset_columns_list()

        items = [(c, Qt.Checked) for c in self.checked_columns]
        unc_items = []
        for i in range(self.columns_list.count()):
            if self.columns_list.item(i).text() in self.checked_columns:
                continue
            item = self.columns_list.item(i)
            unc_items.append((item.text(), item.checkState()))

        # Sort by check state first, then alphabetically
        unc_items.sort(key=lambda x: (x[1] == Qt.Unchecked, x[0].lower()))

        items.extend(unc_items)

        self.columns_list.clear()
        for text, checkState in items:
            item = QListWidgetItem(text)
            item.setCheckState(checkState)
            self.columns_list.addItem(item)

    def concat_data(self):
        """Concatenates the data."""
        self.data_org = impedance_concat(self.data_org)
        self.update_tree_view()
        self.datasets_tree.setColumnWidth(0, 150)

    def simplify_data_keys(self):
        """Simplifies the data for fitting."""
        def simplify(data):
            if isinstance(data, dict):
                new_data = {}
                run = 1
                for k, v in data.items():
                    if isinstance(v, dict):
                        new_data[k] = simplify(v)
                    else:
                        new_data[f"_r{run}"] = v
                        run += 1
                return new_data
            return data

        self.data_org = simplify(self.data_org)

        self.update_tree_view()

    def save(self):
        """Saves the converted data."""
        save_format = self.save_format_combo.currentText()
        columns = []
        if save_format == "Use Columns":
            num_cols = self.columns_list.count()
            if num_cols == 0:
                self.refresh_columns_list()
                self.sort_columns_list()
            for i in range(num_cols):
                if self.columns_list.item(i).checkState() == Qt.Checked:
                    columns.append(self.columns_list.item(i).text())
            if not columns:
                columns = [
                    self.columns_list.item(i).text() for i in range(num_cols)
                ]
            columns = [
                eval(tc) if tc[0] == "(" and tc[-1] == ")" else tc
                for tc in columns
            ]

        if self.t_files is not None and len(self.t_files["name"]) == len(
            self.data_in
        ):
            out_path = QFileDialog.getExistingDirectory(
                self,
                "Select Output Directory",
            )
            out_path = Path(out_path)
            get_all = self.get_all_checkbox.isChecked()

            for ind in self.t_files.index:
                f_name = self.t_files["name"][ind]
                if get_all:
                    f_name = self.t_files["id"][ind].parent.stem

                tmp = flatten_dict(self.data_in[f_name])
                if save_format == "Use Columns":
                    for k, v in tmp.items():
                        tmp[k] = simplify_multi_index(
                            v[[c for c in columns if c in v.columns]]
                        )
                    save(
                        tmp,
                        out_path / self.t_files["id"][ind].parent.parent.stem,
                        f_name + "_reduced",
                        merge_cells=True,
                    )
                else:
                    save(
                        tmp,
                        out_path / self.t_files["id"][ind].parent.parent.stem,
                        f_name,
                        merge_cells=True,
                    )
        else:
            out_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save File",
                "Converted_Data",
                "Excel files (*.xlsx);;CSV files (*.csv);;All files (*.*)",
            )
            if not out_path:
                return
            flat_data = separate_dict(flatten_dict(self.data_org), ["resid"])[
                "residuals"
            ]
            if save_format == "Use Columns":
                for k, v in flat_data.items():
                    flat_data[k] = simplify_multi_index(
                        v[[c for c in columns if c in v.columns]]
                    )
            else:
                flat_data = convert_mfia_df_for_fit(flat_data)

            save(
                flat_data,
                Path(out_path).parent,
                name=Path(out_path).stem,
                file_type=Path(out_path).suffix,
            )

    def save_settings(self):
        """Saves the current settings."""
        patterns = [
            self.patterns_list.item(i)
            for i in range(self.patterns_list.count())
        ]
        columns = [
            self.columns_list.item(i) for i in range(self.columns_list.count())
        ]
        settings = {
            "directory": str(self.in_path),
            "patterns": [[p.text(), p.checkState()] for p in patterns],
            "groups": [],
            "columns": [
                c.text() for c in columns if c.checkState() == Qt.Checked
            ],
        }
        if self.is_xlsx:
            for tab_index in range(self.tab_widget.count()):
                tab = self.tab_widget.widget(tab_index)
                form_layout = tab.layout().itemAt(0).layout()
                group = {
                    "search_terms": form_layout.itemAt(1).widget().text(),
                    "reject_terms": form_layout.itemAt(3).widget().text(),
                    "keys": form_layout.itemAt(5).widget().text(),
                }
                settings["groups"].append(group)
        else:
            settings["groups"].append({"keywords": self.keywords_list.text()})

        settings_file = Path(__file__).parent / "settings.json"
        if settings_file.exists():
            with open(settings_file, "r", encoding="utf-8") as f:
                all_settings = json.load(f)
        else:
            all_settings = {}

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        name, ok = QInputDialog.getText(
            self,
            "Save Settings",
            "Enter name for the settings:",
            text=timestamp,
        )
        if not ok or not name:
            return
        all_settings[name] = settings

        with open(settings_file, "w", encoding="utf-8") as f:
            json.dump(all_settings, f, indent=4)

    # def load_empty_settings(self):
    #     """Loads empty settings."""
    #     self.load_settings(full=False)

    # def load_all_settings(self):
    #     """Loads all settings."""
    #     self.load_settings(full=True)

    def load_settings(self):
        """Loads the settings from a file."""
        settings_file = Path(__file__).parent / "settings.json"
        if not settings_file.exists():
            QMessageBox.warning(
                self, "No Settings", "Settings file not found."
            )
            return

        with open(settings_file, "r", encoding="utf-8") as f:
            all_settings = json.load(f)

        names = list(all_settings.keys())
        name, ok = QInputDialog.getItem(
            self,
            "Select Settings",
            "Select the settings to load:",
            names,
            0,
            False,
        )
        if not ok or not name:
            return

        settings = all_settings[name]

        # Create a dialog to select which parts to load
        parts_dialog = QDialog(self)
        parts_dialog.setWindowTitle("Select Parts to Load")
        parts_layout = QVBoxLayout(parts_dialog)

        directory_checkbox = QCheckBox("Directory")
        directory_checkbox.setChecked(True)
        patterns_checkbox = QCheckBox("Patterns")
        patterns_checkbox.setChecked(True)
        groups_checkbox = QCheckBox("Groups")
        groups_checkbox.setChecked(True)
        columns_checkbox = QCheckBox("Columns")
        columns_checkbox.setChecked(True)

        parts_layout.addWidget(directory_checkbox)
        parts_layout.addWidget(patterns_checkbox)
        parts_layout.addWidget(groups_checkbox)
        parts_layout.addWidget(columns_checkbox)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(parts_dialog.accept)
        buttons.rejected.connect(parts_dialog.reject)
        parts_layout.addWidget(buttons)

        if parts_dialog.exec_() != QDialog.Accepted:
            return

        if directory_checkbox.isChecked():
            try:
                path = Path(settings["directory"])
                if settings["directory"] and path.exists():
                    self.in_path_label.setText(
                        f"<b>Input Path:</b> {str(path)}"
                    )
                    self.in_path = path
            except KeyError:
                pass

        if patterns_checkbox.isChecked():
            try:
                if settings["patterns"]:
                    self.patterns_list.clear()
                for pattern in settings["patterns"]:
                    pattern, check = (
                        (pattern, False)
                        if isinstance(pattern, str)
                        else pattern
                    )
                    item = QListWidgetItem(pattern)
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    state = Qt.Checked if check else Qt.Unchecked
                    item.setCheckState(state)
                    self.patterns_list.addItem(item)
            except (KeyError, ValueError):
                pass

        if groups_checkbox.isChecked():
            try:
                if settings["groups"]:
                    self.tab_widget.clear()
                for group in settings["groups"]:
                    if "keywords" in group:
                        self.keywords_list.setText(group["keywords"])
                        self.add_grouping_tab(group["keywords"])
                    else:
                        self.add_grouping_tab(
                            group["search_terms"].split(", "),
                            group["reject_terms"].split(", "),
                            group["keys"].split(", "),
                        )
            except (KeyError, ValueError):
                pass

        if columns_checkbox.isChecked():
            try:
                if settings["columns"]:
                    self.checked_columns = settings["columns"]
                    self.refresh_columns_list()
            except (KeyError, ValueError):
                pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MFIAFileConverter()
    main_window.show()
    sys.exit(app.exec_())
