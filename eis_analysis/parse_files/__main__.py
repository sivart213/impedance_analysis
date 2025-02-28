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
# from collections import namedtuple, defaultdict
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
    QFormLayout,
    QLineEdit,
    QTabWidget,
    QFrame,
    QHBoxLayout,
    QMessageBox,
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

from .gui_widgets import DataTreeWidget, DraggableTableWidget, RadioButtonDialog
from .gui_workers import WorkerFunctions, LoadDatasetsWorker, SaveDatasetsWorker
from .gui_plots import PopupGraph

from ..data_treatment import (
    get_valid_keys,
    simplify_multi_index,
)
from ..dict_ops import separate_dict, flatten_dict
from ..string_ops import re_not
from ..equipment.mfia_ops import (
    parse_mfia_files,
    convert_mfia_data,
    convert_mfia_df_for_fit,
)

from ..system_utilities import find_files, load_file, load_hdf, save
from ..system_utilities.special_io import parse_file_info

# Define the named tuple for dataset entries



class MFIAFileConverter(QMainWindow, WorkerFunctions):
    """
    A class to convert MFIA files using a GUI.
    Attributes:
        is_xlsx (bool): Flag to indicate if the files are in xlsx format.
        files (list): List of files to be processed.
        in_path (Path): Input directory path.
        t_files (DataFrame): DataFrame containing target files information.
        all_datasets (dict): Dictionary to store input data.
        data_org (dict): Dictionary to store organized data.
        keywords_list (QLineEdit): Line edit for keywords.
        checked_columns (list): List of checked columns.
        tab_widget (QTabWidget): Tab widget for grouping iterations.
        patterns_list (QListWidget): List widget for file patterns.
        columns_list (QListWidget): List widget for columns.
        files_table (QTableWidget): Table widget for files.
        tree (QTreeWidget): Tree widget for datasets.
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
        group_by_tab(tab=None): Applies grouping logic to the data.
        apply_all_groups(): Applies all grouping logic to the data.
        refresh_tree(): Updates the tree view with the organized data.
        update_t_files(): Updates the target files tree view.
        load_files(): Converts the files based on the selected patterns.
        hide_tree_items(item): Removes a dataset from the organized data.
        reset_columns_list(): Resets the columns list with unique columns from the dataframes.
        refresh_columns_list(): Refreshes the columns list keeping any checked list.
        sort_columns_list(): Sorts the columns list.
        concat_tree_items(): Concatenates the data.
        simplify_data(): Simplifies the data for fitting.
        save(): Saves the converted data.
        save_settings(): Saves the current settings.
        load_empty_settings(): Loads empty settings.
        load_all_settings(): Loads all settings.
        load_settings(full=True): Loads the settings from a file.
    """

    def __init__(self):
        super().__init__()
        # self.is_xlsx = True
        self.files = None
        self.in_path = None
        self.t_files = None
        self.loaded_data = None # flat dict of df containing loaded data (retained for resets)
        self.data_viewer = None

        self.worker = None  # Initialize worker
        self.thread = None  # Initialize thread
        self.progress_dialog = None  # Initialize progress dialog
        self.kill_operation = False  # Initialize kill operation flag
        
        self.grouping_history = []
        self.checked_columns = []

        self.popup_graph = PopupGraph()

        self.setWindowTitle("MFIA File Conversion")

        # Widget Creation

        # Labels
        self.in_path_label = QLabel("<b>Input Path:</b>")

        # List Widgets
        self.file_kw_and_list = QListWidget()
        self.file_kw_and_list.setMaximumWidth(300)
        self.file_kw_and_list.installEventFilter(self)

        self.file_kw_or_list = QListWidget()
        self.file_kw_or_list.setMaximumWidth(300)
        self.file_kw_or_list.installEventFilter(self)

        self.columns_list = QListWidget()
        self.columns_list.setDragDropMode(QAbstractItemView.InternalMove)

        # Tree Widgets
        self.tree = DataTreeWidget()
        self.tree.installEventFilter(self)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.tree.saveDataRequested.connect(self.save_data)

        # Replace t_files_tree with t_files_table
        self.files_table = DraggableTableWidget()
        self.files_table.setColumnCount(2)
        self.files_table.setHorizontalHeaderLabels(["Name", "Path"])
        self.files_table.setSortingEnabled(True)  # Enable column sorting
        self.files_table.installEventFilter(self)

        # Buttons
        in_path_button = QPushButton("Browse")
        in_path_button.setFixedWidth(100)
        in_path_button.clicked.connect(self.browse_in_path)

        reset_columns_button = QPushButton("Reset")
        reset_columns_button.clicked.connect(self.reset_columns_list)
        reset_columns_button.setFixedWidth(75)

        refresh_columns_button = QPushButton("Refresh")
        refresh_columns_button.clicked.connect(self.refresh_columns_list)
        refresh_columns_button.setFixedWidth(75)

        sort_columns_button = QPushButton("Sort")
        sort_columns_button.clicked.connect(self.sort_columns_list)
        sort_columns_button.setFixedWidth(75)

        reset_tree_button = QPushButton("Reset")
        reset_tree_button.clicked.connect(self.tree.set_data_org)
        reset_tree_button.setFixedWidth(150)

        reset_gr_button = QPushButton("Reset Grouping")
        reset_gr_button.clicked.connect(self.remove_grouping)
        reset_gr_button.setFixedWidth(150)

        apply_fn_button = QPushButton("Group by File")
        apply_fn_button.clicked.connect(self.group_by_subname)
        apply_fn_button.setFixedWidth(150)

        apply_gr_button = QPushButton("Apply Groups")
        apply_gr_button.clicked.connect(self.apply_all_groups)
        apply_gr_button.setFixedWidth(150)

        datasets_rename_button = QPushButton("Rename Datasets")
        datasets_rename_button.clicked.connect(self.tree.rename_all_tree_items)
        datasets_rename_button.setFixedWidth(150)

        datasets_concat_button = QPushButton("Combine Datasets")
        datasets_concat_button.clicked.connect(self.tree.concat_items)
        datasets_concat_button.setFixedWidth(150)

        t_files_update_button = QPushButton("(Re)Load File List")
        t_files_update_button.clicked.connect(self.update_files)
        t_files_update_button.setFixedWidth(150)

        convert_button = QPushButton("Convert")
        convert_button.setFixedWidth(100)
        convert_button.clicked.connect(self.load_files)

        save_button = QPushButton("Save to File")
        save_button.setFixedWidth(100)
        save_button.clicked.connect(lambda: self.save_data(None, path=self.get_save_path(False)))

        save_gr_button = QPushButton("Save to Folder")
        save_gr_button.setFixedWidth(100)
        save_gr_button.clicked.connect(lambda: self.save_data(None, path=self.get_save_path(True)))

        save_settings_button = QPushButton("Save Settings")
        save_settings_button.clicked.connect(self.save_settings)

        load_settings_button = QPushButton("Load Settings")
        load_settings_button.clicked.connect(self.load_settings)

        # Line Edits
        self.patterns_edit = QLineEdit()
        self.patterns_edit.setPlaceholderText("pattern")
        self.patterns_edit.returnPressed.connect(self.add_pattern)

        # Combo Boxes
        self.save_format_combo = QComboBox()
        self.save_format_combo.addItems(
            ["All Columns", "Selected Columns", "Freq, Real, & Imag"]
        )
        self.save_format_combo.setFixedWidth(150)

        # Check Boxes
        self.get_all_checkbox = QCheckBox("Get All")

        # Tab Widgets
        self.file_kw_tabs = QTabWidget()
        self.group_tabs = QTabWidget()
        self.add_grouping_tab()
        self.data_tabs = QTabWidget()

        # Frame and Layout Insertion
        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)

        def create_separator(frame=None):
            separator = QFrame(frame)
            separator.setFrameShape(QFrame.HLine)
            separator.setFrameShadow(QFrame.Sunken)
            return separator

        max_width = 300

        # Input Path
        in_path_layout = QHBoxLayout()
        in_path_layout.addWidget(self.in_path_label)
        in_path_layout.addWidget(in_path_button)
        in_path_frame = QFrame()
        in_path_frame.setLayout(in_path_layout)
        in_path_frame.setFrameShape(QFrame.StyledPanel)
        layout.addWidget(in_path_frame)

        layout.addWidget(create_separator())

        # Patterns and Keywords Tab Widget
        file_kw_layout = QVBoxLayout()
        file_kw_layout.addWidget(QLabel("<b>File Patterns and Keywords:</b>"))
        file_kw_layout.addWidget(self.file_kw_tabs)
        file_kw_frame = QFrame()
        file_kw_frame.setLayout(file_kw_layout)
        file_kw_frame.setMaximumWidth(max_width)

        self.file_kw_tabs.addTab(self.file_kw_and_list, "File Patterns")
        self.file_kw_tabs.addTab(self.file_kw_or_list, "File Keywords")

        # Entry Widget for Adding Patterns/Keywords
        patterns_edit_layout = QHBoxLayout()
        patterns_edit_layout.addWidget(QLabel("Add Item:"))
        patterns_edit_layout.addWidget(self.patterns_edit)
        file_kw_layout.addLayout(patterns_edit_layout)

        # Grouping Iterations Tab Widget
        # layout.addWidget(self.group_tabs)

        # Columns List
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
        grid1.addWidget(file_kw_frame, 0, 0)
        grid1.addWidget(self.group_tabs, 0, 1)
        grid1.addWidget(columns_frame, 0, 2)

        layout.addLayout(grid1)

        layout.addWidget(create_separator())


        # Tree View for Datasets
        datasets_layout = QVBoxLayout()
        datasets_layout.addWidget(self.tree)

        datasets_buttons = QHBoxLayout()
        datasets_buttons.addWidget(reset_tree_button)
        datasets_buttons.addWidget(reset_gr_button)
        datasets_buttons.addWidget(apply_fn_button)
        datasets_buttons.addWidget(apply_gr_button)
        datasets_buttons.addWidget(datasets_rename_button)
        datasets_buttons.addWidget(datasets_concat_button)
        datasets_layout.addLayout(datasets_buttons)

        datasets_frame = QFrame()
        datasets_frame.setLayout(datasets_layout)
        datasets_frame.setMinimumWidth(int(max_width * 1.5))

        # Tree Widget for t_files
        t_files_layout = QVBoxLayout()
        # t_files_layout.addWidget(self.t_files_tree)
        t_files_layout.addWidget(self.files_table)

        check_update_layout = QHBoxLayout()
        check_update_layout.addWidget(t_files_update_button)
        check_update_layout.addWidget(self.get_all_checkbox)
        check_update_layout.addWidget(convert_button)
        t_files_layout.addLayout(check_update_layout)

        t_files_frame = QFrame()
        t_files_frame.setLayout(t_files_layout)
        t_files_frame.setMinimumWidth(int(max_width * 1.5))

        # self.data_tabs.addTab(files_frame, "Loaded Files")
        self.data_tabs.addTab(t_files_frame, "Files")
        self.data_tabs.addTab(datasets_frame, "Datasets")

        layout.addWidget(self.data_tabs)
        layout.addWidget(create_separator())

        bottom_layout = QHBoxLayout()
        save_format_layout = QHBoxLayout()
        save_format_layout.addWidget(QLabel("Save Format: "))
        save_format_layout.addWidget(self.save_format_combo)
        save_format_layout.addWidget(save_button)
        save_format_layout.addWidget(save_gr_button)
        save_format_layout.setAlignment(Qt.AlignLeft)

        settings_buttons_layout = QHBoxLayout()
        settings_buttons_layout.addWidget(save_settings_button)
        settings_buttons_layout.addWidget(load_settings_button)
        settings_buttons_layout.setAlignment(Qt.AlignRight)

        bottom_layout.addLayout(save_format_layout)
        bottom_layout.addLayout(settings_buttons_layout)

        layout.addLayout(bottom_layout)

        self.setCentralWidget(central_widget)
    
    def eventFilter(self, source, event):
        """Handles key press events for deleting items."""
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Delete:
            if source is self.file_kw_and_list:
                for item in self.file_kw_and_list.selectedItems():
                    self.file_kw_and_list.takeItem(
                        self.file_kw_and_list.row(item)
                    )
                return True
            elif source is self.file_kw_or_list:
                for item in self.file_kw_or_list.selectedItems():
                    self.file_kw_or_list.takeItem(
                        self.file_kw_or_list.row(item)
                    )
                return True
            elif source is self.files_table:
                rows_to_delete = set()
                for item in self.files_table.selectedItems():
                    rows_to_delete.add(item.row())
                for row in sorted(rows_to_delete, reverse=True):
                    self.files_table.removeRow(row)
                return True
        if event.type() == QEvent.ContextMenu and source is self.tree:
            item = self.tree.selectedItems()
            if item:
                self.tree.show_tree_context_menu(event.globalPos(), item)
                return True
        return super().eventFilter(source, event)

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

        title = f"Group {self.group_tabs.count() + 1}"
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
        # Group buttons
        use_group_button = QPushButton("Apply Group")
        use_group_button.clicked.connect(self.group_by_tab)
        use_group_button.setFixedWidth(100)
        drop_group_button = QPushButton("Drop Group")
        drop_group_button.setFixedWidth(100)
        drop_group_button.clicked.connect(self.drop_grouping_tab)
        add_group_button = QPushButton("Add Group")
        add_group_button.setFixedWidth(100)
        add_group_button.clicked.connect(self.add_grouping_tab)
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(use_group_button)
        button_layout.addWidget(drop_group_button)
        button_layout.addWidget(add_group_button)
        
        layout.addLayout(form_layout)
        layout.addLayout(button_layout)

        tab.setLayout(layout)
        self.group_tabs.addTab(tab, title)

        if self.group_tabs.count() > 1:
            previous_tab = self.group_tabs.widget(
                self.group_tabs.count() - 2
            )
            previous_layout = previous_tab.layout().itemAt(
                previous_tab.layout().count() - 1
            )
            previous_layout.removeWidget(
                previous_layout.itemAt(
                    previous_layout.count() - 1
                ).widget()
            )

    def drop_grouping_tab(self):
        """Drops the current tab from the group_tabs widget."""
        current_index = self.group_tabs.currentIndex()
        
        if n_tabs := self.group_tabs.count() > 1:
            is_last_tab = current_index == n_tabs - 1
            self.group_tabs.removeTab(current_index)

            if is_last_tab:
                last_tab = self.group_tabs.widget(n_tabs - 1)
                last_layout = last_tab.layout().itemAt(
                    last_tab.layout().count() - 1
                )

                add_group_button = QPushButton("Add Group")
                add_group_button.setFixedWidth(100)
                add_group_button.clicked.connect(self.add_grouping_tab)

                last_layout.addWidget(add_group_button)

    def browse_in_path(self):
        """Opens a dialog to browse for the input directory"""
        path = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if path:
            self.in_path_label.setText(f"<b>Input Path:</b> {path}")
            self.in_path = Path(path)

    def add_pattern(self):
        """Adds a pattern to the patterns list."""
        def add_worker(pattern):
            if self.file_kw_tabs.currentIndex() == 0:
                check = Qt.Unchecked
                if pattern.startswith("not "):
                    pattern = pattern.replace("not ", "")
                    check = Qt.Checked
                item = QListWidgetItem(pattern.strip())
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(check)
            else:
                item = QListWidgetItem(pattern.strip())

            self.file_kw_tabs.currentWidget().addItem(item)
            self.patterns_edit.clear()

        patterns = self.patterns_edit.text()
        if patterns:
            for p in re.split(r"\s*[,;]\s+", patterns):
                add_worker(p)

    def update_files(self):
        """Updates the files table with the selected patterns and linked to button."""
        patterns = []
        for i in range(self.file_kw_and_list.count()):
            item = self.file_kw_and_list.item(i)
            pattern = item.text()
            use_re_not = item.checkState() == Qt.Checked
            if use_re_not:
                pattern = re_not(pattern)
            patterns.append(pattern)
        try:
            files = find_files(self.in_path, patterns=patterns) # loading of filenames

            keywords = ([
                self.file_kw_or_list.item(i).text()
                for i in range(self.file_kw_or_list.count())
            ] if self.file_kw_or_list.count() else None)

            get_all = self.get_all_checkbox.isChecked()

            files = parse_file_info(
                files, parse_mfia_files, keywords, get_all=get_all
            )
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error: {e}")
            return
        
        self.files = files

        # Insert the "relative path" column
        self.files.insert(
            self.files.columns.get_loc("path"),
            "relative path",
            self.files["path"].apply(lambda p: p.relative_to(self.in_path))
        )

        # # TODO: Test simpler Clear Table
        # self.files_table.sortItems(-1)
        # self.files_table.clear()

        # Clear the table
        self.files_table.clearContents()
        # Reset the row and column counts
        self.files_table.setRowCount(0)
        self.files_table.setColumnCount(0)

        # Clear the horizontal header labels
        self.files_table.setHorizontalHeaderLabels([])

        # Set the number of columns and their headers
        self.files_table.setColumnCount(len(self.files.columns))
        self.files_table.setHorizontalHeaderLabels(self.files.columns)

        # Set the number of rows
        self.files_table.setRowCount(len(self.files))

        # Populate the table with data from the DataFrame
        for row_idx, row in self.files.iterrows():
            for col_idx, value in enumerate(row):
                if "date" in self.files.columns[col_idx].lower():
                    self.files_table.setItem(row_idx, col_idx, QTableWidgetItem(value.strftime("%Y-%m-%d")))
                elif "time" in self.files.columns[col_idx].lower():
                    self.files_table.setItem(row_idx, col_idx, QTableWidgetItem(value.strftime("%H:%M:%S")))
                else:
                    self.files_table.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))
    
    def update_t_files(self):
        """Updates the target files tree view."""
        # Create a list to store the updated file indices
        updated_files = []

        self.t_files=None
        name_idx = None
        path_idx = None
        name_str = ""
        path_str = ""
        for col in range(self.files_table.columnCount()):
            header = self.files_table.horizontalHeaderItem(col).text()
            if header.lower() == "name":
                name_idx = col
                name_str = self.files.columns[col]
            elif header.lower() == "path":
                path_idx = col
                path_str = self.files.columns[col]

        # Ensure the id column is found
        if name_idx is None or path_idx is None:
            QMessageBox.warning(self, "Name Error", "Table must have a 'name' and 'path' column")
            return

        # Iterate through the rows of the table to get the current state
        for row in range(self.files_table.rowCount()):
            # Extract the name and path from the table
            name_item = self.files_table.item(row, name_idx)
            path_item = self.files_table.item(row, path_idx)

            # Ensure the items are not None
            if name_item and path_item:
                name_val = name_item.text()
                path_val = Path(path_item.text())

                # Append the extracted data to the updated_files list
                if self.files.apply(lambda x: x[name_str] == name_val and x[path_str] == path_val, axis=1).any():
                    updated_files.append({name_str: name_val, path_str: path_val})
        
        # Check if updated_files is empty
        if not updated_files:
            QMessageBox.warning(self, "No Matching Files", "No matching files found in the table.")
            return
        
        # Generate self.t_files based on the updated data
        self.t_files = pd.DataFrame(updated_files)
        
    def load_files(self):
        """Converts the files based on the selected patterns."""
        # if self.t_files is None:
        self.update_t_files()

        self.worker = LoadDatasetsWorker(self.t_files, self.tree, self.get_all_checkbox.isChecked())
        self.create_progress_dialog(
            self, title="Load", label_text="Loading Files...", cancel="Cancel", maximum=0
        )
        self.run_in_thread(
            self.io_finished,
            self.on_worker_error,
            progress_dialog=self.progress_dialog,
        )

        # # load the data
        # name_str = self.t_files.columns[0]
        # path_str = self.t_files.columns[1]
        # self.loaded_data = {}
        # for ind in self.t_files.index:
        #     name = self.t_files[name_str][ind]
        #     pth = self.t_files[path_str][ind]
            
        #     if ".h" in pth.suffix:
        #         if self.get_all_checkbox.isChecked():
        #             name = pth.parent.stem
        #         raw_data = load_hdf(pth, key_sep=True)
        #         self.loaded_data[name] = convert_mfia_data(
        #             raw_data, flip=False, flatten=2
        #         )
        #     elif ".xls" in pth.suffix:
        #         self.loaded_data[name] = load_file(pth, index_col=0)[0]

        # self.loaded_data = flatten_dict(self.loaded_data)

        # self.tree.set_all_data(self.loaded_data, True)
        # self.tree.set_data_org()

        self.data_tabs.setCurrentIndex(1)

    def io_finished(self, loaded_data=None):
        """Handle the completion of data I/O operations."""
        if isinstance(loaded_data, dict):
            self.tree.set_all_data(loaded_data, True)
            self.tree.set_data_org()

        self.progress_dialog.close()
        self.progress_dialog.deleteLater()
        self.kill_operation = False

    def group_by_subname(self):
        if not self.tree.data_org:
            QMessageBox.warning(self, "No Files", "Please load files first.")
            return
        
        self.tree.update_data_pathkeys(lambda k: tuple("/".join(k).split("/")))
        self.grouping_history.append("grouped_by_subname")


    def group_by_tab(self, tab=None, suppress_refresh=False):
        """Applies grouping logic to the data."""
        if not self.tree.data_org:
            QMessageBox.warning(self, "No Files", "Please load files first.")
            return

        # Iterate through the tabs and apply the grouping logic
        if tab is None or not tab:
            tab = self.group_tabs.widget(self.group_tabs.currentIndex())
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
            # self.data_org = separate_dict(
            #     self.data_org, search_terms, reject_terms, keys
            # )
            self.tree.eval_grouping(search_terms, reject_terms, keys, suppress_refresh)

            # Log the grouping operation
            self.grouping_history.append({
                "search_terms": search_terms,
                "reject_terms": reject_terms,
                "keys": keys
            })

            # self.refresh_tree()

    def apply_all_groups(self):
        """Applies all grouping logic to the data."""
        if not self.tree.data_org:
            QMessageBox.warning(self, "No Files", "Please load files first.")
            return

        # Iterate through the tabs and apply the grouping logic
        for tab_index in range(self.group_tabs.count()):
            tab = self.group_tabs.widget(tab_index)
            self.group_by_tab(tab, True)
        
        self.tree.refresh_tree()

    def reapply_groups(self):
        """Reapplies all grouping logic from the history to the data."""
        if not self.tree.data_org:
            QMessageBox.warning(self, "No Files", "Please load files first.")
            return

        # Flatten the data_org
        # self.data_org = flatten_dict(self.data_org)
        self.tree.set_data_org()

        # Reapply the grouping history
        for group in self.grouping_history:
            if group == "grouped_by_subname":
                self.tree.update_data_pathkeys(lambda k: tuple("/".join(k).split("/")))
            else:
                self.tree.eval_grouping(group["search_terms"], group["reject_terms"], group["keys"])

        # self.refresh_tree()
    
    def remove_grouping(self):
        """Removes all grouping while retaining other modifications."""
        if not self.tree.data_org:
            QMessageBox.warning(self, "No Files", "Please load files first.")
            return

        self.grouping_history = []
        name_dict = {}
        for key in self.tree.data_org.keys():
            name_dict[key] = (key[-1],)
        self.tree.update_data_pathkeys(name_dict)
        # self.data_org = flatten_dict(self.data_org)
        # self.tree.set_data_org()


    def reset_columns_list(self):
        """Populate the columns list with unique columns from the dataframes ignoring any checked list."""
        if not self.tree.data_org:
            QMessageBox.warning(
                self, "No Data", "Please load and group data first."
            )
            return

        # Collect all unique columns from the dataframes
        unique_columns = set()
        for data in self.tree.data_org.values():
            if isinstance(data.active.df, pd.DataFrame):
                unique_columns.update(data.active.df.columns)

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
        if not self.tree.data_org and self.checked_columns == []:
            QMessageBox.warning(
                self, "No Data", "Please load and group data first."
            )
            return

        if not self.tree.data_org:
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

        # Collect all unique columns from the dataframes
        unique_columns = set()
        for data in self.tree.data_org.values():
            if isinstance(data.active.df, pd.DataFrame):
                unique_columns.update(data.active.df.columns)

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

    def parse_columns_for_save(self, save_format):
        """Parses the columns list for saving."""
        columns = []
        num_cols = self.columns_list.count()
        if num_cols == 0:
            self.refresh_columns_list()
            self.sort_columns_list()
        for i in range(num_cols):
            col = self.columns_list.item(i).text()
            if col.startswith("(") and col.endswith(")"):
                col = re.split(r'\s*,\s*', col[1:-1])
                # Extract words that may or may not have quotes
                col = [re.findall(r'\"(.*?)\"|\'(.*?)\'|(\S+)', word) for word in col]
                # Flatten the list of tuples and filter out empty strings
                col = tuple([item for sublist in col for group in sublist for item in group if item])
            columns.append(col)

        if save_format == "Selected Columns":
            sel_col = []
            for i, col in enumerate(columns):
                if self.columns_list.item(i).checkState() == Qt.Checked:
                    sel_col.append(col)
            if not sel_col:
                return columns
            return sel_col
        elif save_format == "Freq, Real, and Imag":
            sel_col = get_valid_keys(["freq", "real", "imag"], columns)
            if not sel_col:
                return columns
            return sel_col
        return columns
    
    def get_save_path(self, to_dir=None):
        """
        Get the save path for a file or directory.
        
        Parameters:
        to_dir (bool): If True, select a directory. If False, select a file.
        
        Returns:
        Path: The selected file or directory path.
        """
        if not isinstance(to_dir, bool):
            dialog = RadioButtonDialog(
                title="Select Save Option",
                options=["Save to File", "Save to Directory"],
                default_index=0
            )
            if dialog.exec_() == QDialog.Accepted:
                selected_option = dialog.selected_option()
                to_dir = selected_option == "Save to Directory"
            else:
                return None

        if to_dir:
            # Select a directory
            dir_path = QFileDialog.getExistingDirectory(
                None,
                "Select Output Directory",
            )
            if not dir_path:
                return None
            return Path(dir_path)
        else:
            # Select a file
            file_path, _ = QFileDialog.getSaveFileName(
                None,
                "Save File",
                "Converted_Data",
                "Excel files (*.xlsx);;CSV files (*.csv);;All files (*.*)",
            )
            if not file_path:
                return None
            return Path(file_path)

    def save_data(self, data=None, path=None):
        """Saves the converted data."""
        if path is None:
            path = self.get_save_path()
            if not path:
                return
        
        save_format = self.save_format_combo.currentText()

        columns = self.parse_columns_for_save(save_format)

        if data is None:
            data = self.tree.get_flat_data_org()

        flat_data = separate_dict(data, ["resid"])[
            "residuals"
        ]

        self.worker = SaveDatasetsWorker(
            path,
            flat_data,
            columns,
            save_format,
            self.t_files,
            self.get_all_checkbox.isChecked(),
        )
        self.create_progress_dialog(
            self,
            title="Save Data",
            label_text="Saving datasets...",
            cancel="Cancel",
            maximum=0,
        )
        self.run_in_thread(
            self.io_finished,
            self.on_worker_error,
            progress_dialog=self.progress_dialog,
        )

    def save_settings(self):
        """Saves the current settings."""
        patterns = [
            self.file_kw_and_list.item(i)
            for i in range(self.file_kw_and_list.count())
        ]
        keywords = [
            self.file_kw_or_list.item(i).text()
            for i in range(self.file_kw_or_list.count())
        ]
        columns = [
            self.columns_list.item(i) for i in range(self.columns_list.count())
        ]
        settings = {
            "directory": str(self.in_path),
            "patterns": [[p.text(), p.checkState()] for p in patterns],
            "keywords": keywords,
            "groups": [],
            "columns": [
                c.text() for c in columns if c.checkState() == Qt.Checked
            ],
        }
        # if self.is_xlsx:
        for tab_index in range(self.group_tabs.count()):
            tab = self.group_tabs.widget(tab_index)
            form_layout = tab.layout().itemAt(0).layout()
            group = {
                "search_terms": form_layout.itemAt(1).widget().text(),
                "reject_terms": form_layout.itemAt(3).widget().text(),
                "keys": form_layout.itemAt(5).widget().text(),
            }
            settings["groups"].append(group)
        # else:
        #     settings["groups"].append({"keywords": self.keywords_list.text()})

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
        patterns_checkbox = QCheckBox("Patterns/KWs")
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
                    self.file_kw_and_list.clear()
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
                        self.file_kw_and_list.addItem(item)
                if settings["keywords"]:
                    self.file_kw_or_list.clear()
                    for keywords in settings["keywords"]:
                        item = QListWidgetItem(keywords)
                        self.file_kw_or_list.addItem(item)
            except (KeyError, ValueError):
                pass

        if groups_checkbox.isChecked():
            try:
                if settings["groups"]:
                    self.group_tabs.clear()
                    for group in settings["groups"]:
                        if "keywords" not in group:
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

