# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney
pound !usr/bin/env python3
General function file
"""

import sys
from IPython import get_ipython
import numpy as np
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)

from PyQt5.QtCore import Qt, QThread, QEventLoop

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QMainWindow,
    QFileDialog,
    QMessageBox,
    QMenu,
    QMenuBar,
    QLabel,
    QCheckBox,
    QLineEdit,
    QPushButton,
    QFrame,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QSizePolicy,
    QInputDialog,
)

from impedance.models.circuits.fitting import wrapCircuit

from ..data_treatment import ConfidenceAnalysis, Statistics
from ..utils.plot_factory import GeneratePlot

from .gui_widgets import (
    AlignComboBox,
    MultiEntryManager,
)
from .gui_windows import (
    create_progress_dialog,
    DictWindow,
    DataTreeWindow,
    CalcRCWindow,
    PrintWindow,
    MultiEntryWindow,
    DataViewer,
)
from .gui_workers import (
    DataHandler,
    FittingWorker,
    LoadDataWorker,
    SaveResultsWorker,
    SaveFiguresWorker,
)

if get_ipython() is not None:
    get_ipython().run_line_magic("matplotlib", "inline")


class GraphGUI:
    """Class to create a GUI for plotting graphs."""

    def __init__(self, root):
        self.root = root
        self.root.closeEvent = self.closeEvent
        self.root.setWindowTitle("EIS Fitting")

        # Create a central widget and set it as the central widget of the main window
        central_widget = QWidget(self.root)
        self.root.setCentralWidget(central_widget)

        # Initialize parameters
        self.data = DataHandler()

        self.plotted_data = None
        self.line = None
        self.line_count = 0
        self.ci_inputs = None
        self.kill_operation = False
        self.thread = None
        self.worker = None
        self.progress_dialog = None
        self.fit_results = None
        self.data_viewer = None

        self.options = DictWindow(self.root, self.data.option_inits)
        self.quick_bound_vals = DictWindow(
            self.root, {"low": 0.1, "high": 10}, "Quick Bounds"
        )

        # Windows and special classes
        self.calculation_window = CalcRCWindow(self, None)
        self.print_window = PrintWindow(None)
        self.pinned = DataTreeWindow(
            None,
            ["Name", "Dataset", "Model", "Show", "Comments"],
            ["Name", "Dataset", "Model", "Values", "Show", "Comments"],
            self.add_pin,
            self.pull_pin,
            graphing_callback=self.update_graphs,
            df_base_cols=["Name", "Dataset", "Model", "Comments"],
            df_sort_cols=["values", "std"],
            tree_gr_cols=["Values"],
            narrow_cols=["Show"],
            wide_cols=["Values", "Comments"],
        )

        self.parameters = MultiEntryManager(
            None,
            callback=self.update_line_data,
            interval=self.options["simulation"]["interval"],
        )
        self.bounds = MultiEntryWindow(
            self.root,
            num_entries=2,
            callbacks=dict(
                save=self.validate_bounds,
                button_Quick_Bounds=self.quick_bounds,
                button_Boundary_Options=self.quick_bound_vals.window,
            ),
        )
        self.datasets = MultiEntryWindow(
            None,
            has_value=False,
            callbacks=dict(
                save=lambda x: self.update_graphs(),
                clear=self.clear_datasets,
            ),
        )

        self.parameters_std = MultiEntryWindow(
            None,
            has_check=False,
        )

        # Create a file menu
        self.create_menu()

        # Create a frame for the graphs
        self.graph_frame = QFrame(central_widget)
        self.graph_frame_layout = QHBoxLayout()
        self.graph_frame.setLayout(self.graph_frame_layout)

        # Create a frame for the parameter inputs
        self.control_frame = QFrame(central_widget)
        self.control_frame_layout = QVBoxLayout()
        self.control_frame_layout.setAlignment(Qt.AlignTop)
        self.control_frame.setLayout(self.control_frame_layout)
        self.control_frame.setMaximumSize(300, 16777215)

        # Add frames to the main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.addWidget(self.control_frame)
        main_layout.addWidget(self.graph_frame)
        central_widget.setLayout(main_layout)

        # Create a frame for the parameter inputs
        self.param_frame = QFrame(self.control_frame)
        self.param_layout = QVBoxLayout(self.param_frame)
        self.param_frame.setLayout(self.param_layout)

        cols_layout = QGridLayout()
        error_layout = QGridLayout()
        # freq_layout = QGridLayout()
        freq_layout = QVBoxLayout()
        freq_sub_layout = QHBoxLayout()
        freq_layout.addLayout(freq_sub_layout)

        button_layout = QHBoxLayout()

        def interlock_comboboxes(sender):
            top_type = self.top_type_var.currentText()
            top_mode = self.top_mode_var.currentIndex()
            bot_type = self.bot_type_var.currentText()
            bot_mode = self.bot_mode_var.currentIndex()

            if top_type == bot_type and top_mode == bot_mode:
                if sender == self.top_type_var or sender == self.top_mode_var:
                    ind = 1 if bot_mode in [0, 1] else 5
                    self.bot_mode_var.setCurrentIndex(ind - bot_mode)
                else:
                    ind = 1 if top_mode in [0, 1] else 5
                    self.top_mode_var.setCurrentIndex(ind - top_mode)

        plot_types = [
            "Z",
            "Y",
            "M",
            "ε",
            "εᵣ",
            "σ",
            "ρ",
        ]

        plot_modes = [
            "Real",
            "Imag",
            "Mag",
            "Θ",
            "+Imag",
            "tan(δ)",
        ]

        # Group 1: Creation
        # Create combo boxes
        self.ny_type_var = AlignComboBox(self.control_frame)
        self.ny_type_var.setTextAlignment(Qt.AlignCenter)
        self.ny_type_var.addItems(plot_types)
        self.ny_type_var.setCurrentIndex(
            0
        )  # Set the default value to "Z' & Z''"

        self.top_type_var = AlignComboBox(self.control_frame)
        self.top_type_var.setTextAlignment(Qt.AlignCenter)
        self.top_type_var.addItems(plot_types)
        self.top_type_var.setCurrentIndex(0)
        self.top_type_var.currentIndexChanged.connect(
            lambda: interlock_comboboxes(self.top_type_var)
        )

        self.top_mode_var = AlignComboBox(self.control_frame)
        self.top_mode_var.setTextAlignment(Qt.AlignCenter)
        self.top_mode_var.addItems(plot_modes)
        self.top_mode_var.setCurrentIndex(0)
        self.top_mode_var.currentIndexChanged.connect(
            lambda: interlock_comboboxes(self.top_mode_var)
        )

        self.bot_type_var = AlignComboBox(self.control_frame)
        self.bot_type_var.setTextAlignment(Qt.AlignCenter)
        self.bot_type_var.addItems(plot_types)
        self.bot_type_var.setCurrentIndex(0)
        self.bot_type_var.currentIndexChanged.connect(
            lambda: interlock_comboboxes(self.bot_type_var)
        )

        self.bot_mode_var = AlignComboBox(self.control_frame)
        self.bot_mode_var.setTextAlignment(Qt.AlignCenter)
        self.bot_mode_var.addItems(plot_modes)
        self.bot_mode_var.setCurrentIndex(1)
        self.bot_mode_var.currentIndexChanged.connect(
            lambda: interlock_comboboxes(self.bot_mode_var)
        )

        self.dataset_var = AlignComboBox(self.control_frame)
        self.dataset_var.setTextAlignment(Qt.AlignCenter)
        self.dataset_var.setFixedWidth(150)
        self.dataset_var.addItems(["None"])
        self.dataset_var.setCurrentText("None")
        self.dataset_var.currentIndexChanged.connect(self.update_datasets)

        self.cursor_var = AlignComboBox(self.control_frame)
        self.cursor_var.setTextAlignment(Qt.AlignCenter)
        self.cursor_var.addItems(["Model"])
        self.cursor_var.setCurrentText("Model")

        self.error_var = AlignComboBox(self.control_frame)
        self.error_var.addItems(self.data.error_methods.keys())
        self.error_var.setCurrentText(list(self.data.error_methods.keys())[0])
        self.error_var.currentIndexChanged.connect(self.update_graphs)

        # Create line edits
        self.model_entry = QLineEdit(self.control_frame)
        self.model_entry.setText(self.data.model)
        self.model_entry.setAlignment(Qt.AlignCenter)

        # Create buttons
        update_model_button = QPushButton("Use Model", self.control_frame)
        update_model_button.setFixedWidth(150)
        update_model_button.clicked.connect(self.update_model_frame)

        update_graphs_button = QPushButton("Update Graphs", self.control_frame)
        update_graphs_button.setFixedWidth(100)
        update_graphs_button.clicked.connect(self.update_graphs)

        update_format_button = QPushButton("Update Format", self.control_frame)
        update_format_button.setFixedWidth(100)
        update_format_button.clicked.connect(self.update_format)

        # Create checkboxes
        self.nyquist_checkbox = QCheckBox("Nyquist", self.control_frame)
        self.bode_checkbox = QCheckBox("Bode", self.control_frame)

        self.top_log_checkbox = QCheckBox("Top", self.control_frame)
        self.bot_log_checkbox = QCheckBox("Bottom", self.control_frame)

        # Create labels
        dataset_label = QLabel("Measured Data", self.control_frame)
        font = dataset_label.font()
        font.setPointSize(10)  # Set font size
        font.setBold(True)  # Set font to bold
        dataset_label.setFont(font)

        model_label = QLabel("Model", self.control_frame)
        font = model_label.font()
        font.setPointSize(10)  # Set font size
        font.setBold(True)  # Set font to bold
        model_label.setFont(font)

        error_label = QLabel("Error using: ", self.control_frame)
        font = error_label.font()
        font.setPointSize(10)  # Set font size
        font.setBold(True)  # Set font to bold
        error_label.setFont(font)

        cursor_label = QLabel("Point at cursor of ", self.control_frame)
        font = cursor_label.font()
        font.setPointSize(10)  # Set font size
        font.setBold(True)  # Set font to bold
        cursor_label.setFont(font)

        nyq_label = QLabel("Nyquist")
        font = nyq_label.font()
        font.setBold(True)  # Set font to bold
        nyq_label.setFont(font)

        bode1_label = QLabel("BodeTop")
        font = bode1_label.font()
        font.setBold(True)  # Set font to bold
        bode1_label.setFont(font)

        bode2_label = QLabel("BodeBot")
        font = bode2_label.font()
        font.setBold(True)  # Set font to bold
        bode2_label.setFont(font)

        bands_label = QLabel("Bands")
        font = bands_label.font()
        font.setBold(True)  # Set font to bold
        bands_label.setFont(font)

        bode_log_label = QLabel("Y Log Scale")
        font = bode_log_label.font()
        font.setBold(True)  # Set font to bold
        bode_log_label.setFont(font)

        # Create labels for displaying values
        self.error_printout = QLabel("Error: N/A", self.control_frame)
        self.cursor_printout = QLabel("Points: N/A")

        # Create separators
        def create_separator(frame=None):
            separator = QFrame(frame)
            separator.setFrameShape(QFrame.HLine)
            separator.setFrameShadow(QFrame.Sunken)
            return separator

        # Group 2: Insertion into layout
        self.control_frame_layout.addWidget(
            dataset_label, alignment=Qt.AlignCenter
        )
        self.control_frame_layout.addWidget(
            self.dataset_var, alignment=Qt.AlignCenter
        )
        self.control_frame_layout.addWidget(
            create_separator(self.control_frame)
        )

        self.control_frame_layout.addWidget(
            model_label, alignment=Qt.AlignCenter
        )
        self.control_frame_layout.addWidget(
            self.model_entry
        )  # , alignment=Qt.AlignCenter)
        self.control_frame_layout.addWidget(
            update_model_button, alignment=Qt.AlignCenter
        )
        self.control_frame_layout.addWidget(
            create_separator(self.control_frame)
        )

        self.control_frame_layout.addWidget(self.param_frame)

        self.control_frame_layout.addWidget(
            create_separator(self.control_frame)
        )

        # self.control_frame_layout.addWidget(cols_frame)
        self.control_frame_layout.addLayout(cols_layout)
        # Add widgets to the grid layout
        cols_layout.addWidget(
            bands_label, 0, 0
        )  # Add to grid at position (0, 0)
        cols_layout.addWidget(
            self.nyquist_checkbox, 0, 1
        )  # Add to grid at position (0, 1)
        cols_layout.addWidget(
            self.bode_checkbox, 0, 2
        )  # Add to grid at position (0, 2)

        cols_layout.addWidget(
            nyq_label, 1, 0
        )  # Add to grid at position (1, 0)
        cols_layout.addWidget(
            self.ny_type_var, 1, 1
        )  # Add to grid at position (1, 1)

        cols_layout.addWidget(
            bode_log_label, 2, 0
        )  # Add to grid at position (2, 0)
        cols_layout.addWidget(
            self.top_log_checkbox, 2, 1
        )  # Add to grid at position (2, 1)
        cols_layout.addWidget(
            self.bot_log_checkbox, 2, 2
        )  # Add to grid at position (2, 2)

        cols_layout.addWidget(
            bode1_label, 3, 0
        )  # Add to grid at position (3, 0)
        cols_layout.addWidget(
            self.top_type_var, 3, 1
        )  # Add to grid at position (3, 1)
        cols_layout.addWidget(
            self.top_mode_var, 3, 2
        )  # Add to grid at position (3, 2)

        cols_layout.addWidget(
            bode2_label, 4, 0
        )  # Add to grid at position (4, 0)
        cols_layout.addWidget(
            self.bot_type_var, 4, 1
        )  # Add to grid at position (4, 1)
        cols_layout.addWidget(self.bot_mode_var, 4, 2)

        self.control_frame_layout.addLayout(button_layout)
        button_layout.addWidget(update_graphs_button)
        button_layout.addWidget(update_format_button)

        self.control_frame_layout.addWidget(
            create_separator(self.control_frame)
        )

        # self.control_frame_layout.addWidget(error_frame)
        self.control_frame_layout.addLayout(error_layout)
        error_layout.addWidget(error_label, 0, 0)
        error_layout.addWidget(self.error_var, 0, 1)
        error_layout.addWidget(self.error_printout, 1, 0)

        self.control_frame_layout.addWidget(
            create_separator(self.control_frame)
        )

        # # self.control_frame_layout.addWidget(freq_frame)
        # self.control_frame_layout.addLayout(freq_layout)
        # freq_layout.addWidget(cursor_label, 0, 0)
        # freq_layout.addWidget(self.cursor_var, 0, 1)
        # freq_layout.addWidget(self.cursor_printout, 1, 0)

        # self.control_frame_layout.addWidget(freq_frame)
        self.control_frame_layout.addLayout(freq_layout)
        freq_sub_layout.addWidget(cursor_label)
        freq_sub_layout.addWidget(self.cursor_var)
        freq_layout.addWidget(self.cursor_printout)

        # Add parts for the graph frame
        # Create matplotlib figures
        self.fig1, self.ax1 = GeneratePlot.subplots(
            figsize=(6, 6)
        )  # Nyquist plot
        self.fig1.subplots_adjust(left=0.175)
        self.fig2, (self.ax2, self.ax3) = GeneratePlot.subplots(
            2, 1, figsize=(10, 10), sharex=True
        )  # Bode plots
        self.ax2.set_label("Bode Top Figure")
        self.ax3.set_label("Bode Bottom Figure")

        # Create a frame for the Nyquist plot
        self.nyquist_frame = QFrame(self.graph_frame)
        self.nyquist_frame_layout = QVBoxLayout()
        self.nyquist_frame.setLayout(self.nyquist_frame_layout)
        self.graph_frame_layout.addWidget(self.nyquist_frame)

        self.nyquist_frame.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.nyquist_frame.setMaximumSize(600, 16777215)

        # Create a frame for the Bode plot
        self.bode_frame = QFrame(self.graph_frame)
        self.bode_frame_layout = QVBoxLayout()
        self.bode_frame.setLayout(self.bode_frame_layout)
        self.graph_frame_layout.addWidget(self.bode_frame)

        # Create canvases to display the figures
        self.canvas1 = FigureCanvasQTAgg(self.fig1)
        self.canvas2 = FigureCanvasQTAgg(self.fig2)

        self.nyquist_frame_layout.addWidget(self.canvas1)
        self.bode_frame_layout.addWidget(self.canvas2)

        # Add Matplotlib toolbars CustomNavigationToolbar
        self.toolbar1 = NavigationToolbar2QT(self.canvas1, self.nyquist_frame)
        self.toolbar1.update()
        self.nyquist_frame_layout.addWidget(self.toolbar1)

        # Connect the cursor event to update the cursor position 
        self.canvas1.mpl_connect(
            "motion_notify_event", self.nyquist_cursor_position
        )
        self.canvas2.mpl_connect(
            "motion_notify_event", self.bode_cursor_position
        )

        self.toolbar2 = NavigationToolbar2QT(self.canvas2, self.bode_frame)
        self.toolbar2.update()
        self.bode_frame_layout.addWidget(self.toolbar2)

        self.nyquist_plot = GeneratePlot(
            self.ax1,
            title=f"{self.ny_type_var.currentText()} Nyquist of Model",
            scales="LinFrom0Scaler",
            init_formats=["scale", "format", "square"],
            fkwargs=dict(power_lim=2),
        )

        self.bode_plot = GeneratePlot(
            [self.ax2, self.ax3],
            labels=[
                "Frequency [Hz]",
                self.top_type_var.currentText(),
                self.top_type_var.currentText(),
            ],
            title=f"{self.top_type_var.currentText()} & {self.top_type_var.currentText()} of Model",
            scales=[
                "log",
                self.data.var_scale[self.top_mode_var.currentText()],
                self.data.var_scale[self.bot_mode_var.currentText()],
            ],
            init_formats=["scale", "format"],
        )

        # Initial parameter setup
        self.update_model_frame()

    def create_menu(self):
        """Create the menu bar for the GUI."""
        menu_bar = QMenuBar(self.root)
        self.root.setMenuBar(menu_bar)

        file_menu = QMenu("File", self.root)
        menu_bar.addMenu(file_menu)

        file_menu.addAction("Load Data", self.load_data)
        file_menu.addAction("Save Results", self.save_results)
        file_menu.addAction("Save Figures", self.save_figures)
        file_menu.addSeparator()
        file_menu.addAction("Close", self.root.close)

        data_menu = QMenu("Data", self.root)
        menu_bar.addMenu(data_menu)

        data_menu.addAction("Undo", self.undo)
        data_menu.addAction("View Std Values", self.parameters_std.show)
        
        data_menu.addSeparator()
        data_menu.addAction("Dataset Data", self.edit_data)
        data_menu.addAction("Dataset Visibility", self.datasets.show)

        fitting_menu = QMenu("Fitting", self.root)
        menu_bar.addMenu(fitting_menu)
        
        fitting_menu.addSeparator()
        fitting_menu.addAction("Quick Bounds", self.quick_bounds)
        fitting_menu.addAction("Modify Bounds", self.bounds.show)
        fitting_menu.addSeparator()
        fitting_menu.addAction("Run Fit", self.run_fit)
        fitting_menu.addAction("Run Bootstrap", self.run_bootstrap)
        fitting_menu.addAction("Apply Results", self.apply_fit_results)

        tools_menu = QMenu("Tools", self.root)
        menu_bar.addMenu(tools_menu)

        tools_menu.addAction("Pinned Results", self.pinned.show)
        tools_menu.addAction("Print Window", self.print_window.show)
        tools_menu.addSeparator()
        tools_menu.addAction("Calculate RC", self.calculation_window.show)
        tools_menu.addSeparator()
        tools_menu.addAction("Options", self.options.window)

        # options_menu = QMenu("Options", self.root)
        # menu_bar.addMenu(options_menu)
        
    def close_all_windows(self):
        """Close all windows associated with the GUI."""
        if self.calculation_window.window:
            self.calculation_window.window.close()
        if self.print_window.window:
            self.print_window.window.close()
        if self.pinned.window:
            self.pinned.window.close()
        if self.datasets.window:
            self.datasets.window.close()

    def closeEvent(self, event):
        """Catch the close event and close all windows."""
        self.close_all_windows()
        event.accept()

    def update_model_frame(self, update_plots=True):
        """Update the model frame with the new model."""
        self.data.model = self.model_entry.text().strip()
        self.parameters.interval = self.options["simulation"]["interval"]

        if self.data.model.lower() == "linkk":
            param_values = self.data.generate_linkk(
                self.data.raw[self.dataset_var.currentText()].base_df(
                    None, "frequency"
                ),
                area=self.options["simulation"]["area (cm^2)"],
                thickness=self.options["simulation"]["thickness (cm)"],
                **self.options["linkk"],
            )

            self.parameters.update_entries(
                ["M", "mu"], param_values, self.param_frame
            )
            self.parameters_std.update_entries(["M", "mu"], [0.1, 0.1], None)
            self.bounds.update_entries(["M", "mu"], [[0, 200], [0, 1]], None)

        else:
            param_names = self.data.parse_parameters()

            self.parameters.update_entries(
                param_names,
                [
                    self.data.parse_default(name, self.options["element"])
                    for name in param_names
                ],
                self.param_frame,
            )

            self.parameters_std.update_entries(
                param_names,
                [
                    self.data.parse_default(name, self.options["element"])
                    * 0.1
                    for name in param_names
                ],
                None,
            )

            self.bounds.update_entries(
                param_names,
                [
                    self.data.parse_default(
                        name, self.options["element_range"]
                    )
                    for name in param_names
                ],
                None,
            )

        self.ci_inputs = None
        self.print_window.write("Model updated")
        if update_plots:
            self.update_graphs()

    def undo(self):
        """Undo the last parameter change."""
        self.parameters.undo_recent()
        self.parameters_std.undo_recent()

    def run_in_thread(
            self,
            finished_slot,
            error_slot,
            progress_slot=None,
            progress_dialog=None,
    ):
        """Helper function to run a worker in a separate thread with optional progress dialog."""
        self.thread = QThread()
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(finished_slot)
        self.worker.error.connect(error_slot)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Connect progress signal if provided
        if progress_slot:
            self.worker.progress.connect(progress_slot)

        self.thread.start()
        return progress_dialog

    def on_worker_error(self, error_message):
        self.progress_dialog.close()
        self.kill_operation = False
        QMessageBox.critical(
            self.root, "Error", f"Operation failed: {error_message}"
        )

    def update_progress(self, value):
        """Update the progress bar."""
        self.progress_dialog.setValue(value)

    def cancel_operation(self):
        """Cancel the bootstrap fit."""
        self.kill_operation = True  # Set cancellation flag

    def load_data(self):
        """Load the data from a file."""
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Open File",
            "",
            "Excel files (*.xlsx);;CSV files (*.csv);;All files (*.*)",
        )

        if file_path:
            self.worker = LoadDataWorker(file_path, self.options)
            self.progress_dialog = create_progress_dialog(
                self.root, title="Loading Data", label_text="Loading data..."
            )
            self.progress_dialog.show()
            self.progress_dialog.setMaximum(0)
            self.run_in_thread(
                self.io_data_finished,
                self.on_worker_error,
                progress_dialog=self.progress_dialog,
            )

    def edit_data(self):
        """Edit the data in a separate window."""
        if self.data.raw:
            form = self.data.var_val[self.ny_type_var.currentText()]
            dataset = self.data.raw[self.dataset_var.currentText()].base_df(form, "frequency")
            self.data_viewer = DataViewer(dataset, self.root, "Raw Data")

            # Create an event loop to block execution until the DataViewer is closed
            loop = QEventLoop()
            self.data_viewer.destroyed.connect(loop.quit)
            loop.exec_()

            if len(dataset) != len(self.data_viewer.data) or (dataset.to_numpy() != self.data_viewer.data.to_numpy()).any():
                reply = QMessageBox.question(
                    self.root,
                    "Apply Changes",
                    "The data has been modified. Do you want to apply?\n(Note: This will overwrite the original data)",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if reply == QMessageBox.Yes:
                    self.data.raw[self.dataset_var.currentText()].update(self.data_viewer.data, form=form)
                    self.data_viewer = None
                    self.update_graphs()

    def save_results(self):
        """Save the fit results to a file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self.root,
            "Save File",
            self.dataset_var.currentText(),
            "Excel files (*.xlsx);;CSV files (*.csv);;All files (*.*)",
        )

        if file_path:
            self.worker = SaveResultsWorker(
                file_path,
                self.data,
                self.options,
                self.pinned,
                self.parameters,
                self.dataset_var,
            )
            self.progress_dialog = create_progress_dialog(
                self.root,
                title="Saving Results",
                label_text="Saving results...",
            )
            self.progress_dialog.show()
            self.progress_dialog.setMaximum(0)
            self.run_in_thread(
                self.io_data_finished,
                self.on_worker_error,
                progress_dialog=self.progress_dialog,
            )

    def save_figures(self):
        """Save the figures to a file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self.root,
            "Save Figures",
            self.dataset_var.currentText(),
            "PNG files (*.png);;All files (*.*)",
        )

        if file_path:
            self.worker = SaveFiguresWorker(file_path, self.fig1, self.fig2)
            self.progress_dialog = create_progress_dialog(
                self.root,
                title="Saving Figures",
                label_text="Saving figures...",
            )
            self.progress_dialog.show()
            self.progress_dialog.setMaximum(0)
            self.run_in_thread(
                self.io_data_finished,
                self.on_worker_error,
                progress_dialog=self.progress_dialog,
            )

    def io_data_finished(self, valid_sheets=None, df_in=None):
        """Handle the completion of data I/O operations."""
        if valid_sheets is not None:
            self.data.raw.update(valid_sheets)
            self.datasets.update_entries(list(self.data.raw.keys()))
            self.dataset_var.clear()
            self.dataset_var.setCurrentText("")
            self.dataset_var.addItems(list(self.data.raw.keys()) + ["None"])

            if self.data.raw:
                first_key = next(iter(self.data.raw))
                self.dataset_var.setCurrentText(first_key)
                self.datasets.highlight(
                    self.dataset_var.currentText(), check=True
                )
                self.update_graphs()

        if df_in is not None:
            self.pinned.append_df(df_in)

        self.progress_dialog.close()
        self.kill_operation = False

    def clear_datasets(self):
        """Clear the current datasets."""
        self.data.raw = {}
        self.datasets.clear()
        self.dataset_var.clear()
        self.dataset_var.setCurrentText("None")
        self.update_graphs()

    def update_datasets(self, update_plots=True):  # , *args
        """Update the scatter data based on the selected dataset."""
        if self.data.raw and self.dataset_var.currentText() in self.data.raw:
            self.datasets.highlight(self.dataset_var.currentText(), check=True)
            if update_plots:
                self.update_graphs()

    def add_pin(self):
        """Add a new pinned result to the treeview."""

        # Get the current dataset name
        dataset_name = self.dataset_var.currentText()

        # Determine the default name
        if not self.pinned.df.empty:
            n = self.pinned.df["Dataset"].str.contains(dataset_name).sum()
        else:
            n = 0
        default_name = f"{dataset_name}_fit{n + 1}"

        name = self.pinned.add_row_popup(default_name)

        # Prepare the new row data
        new_row = {
            "Name": name,
            "Dataset": dataset_name,
            "Model": self.data.model,
            "Show": "",
            "Comments": self.error_printout.text().split("\n")[0],
            **{
                f"{name}_values": entry[0]
                for name, entry in zip(
                    self.parameters.names, self.parameters.values
                )
            },
            **{
                f"{name}_std": entry[0]
                for name, entry in zip(
                    self.parameters_std.names, self.parameters_std.values
                )
            },
        }

        # Add to the DataFrame using append_df
        self.pinned.append_df(new_row)

    def pull_pin(self, selected_row):
        """Use the values of a pinned result."""
        # Extract the model, dataset, and comments
        model = selected_row["Model"]
        dataset = selected_row["Dataset"]

        params = self.data.parse_parameters(model)
        # Extract parameter values and stds
        param_values = {
            name: selected_row[f"{name}_values"] for name in params
        }
        param_stds = {
            name: (
                selected_row[f"{name}_std"]
                if not np.isnan(selected_row[f"{name}_std"])
                   and isinstance(selected_row[f"{name}_std"], (int, float))
                else selected_row[f"{name}_values"] * 0.1
            )
            for name in params
        }

        # Update the dataset if available
        if self.data.raw != {} and dataset in self.data.raw.keys():
            self.dataset_var.setCurrentText(dataset)
            self.update_datasets(False)

        # Update the model
        if hasattr(self, "model_entry"):
            self.model_entry.setText(model)
        self.data.model = model

        self.parameters.update_entries(
            params,
            list(param_values.values()),
            # self.control_frame,
        )

        self.parameters_std.update_entries(
            params,
            list(param_stds.values()),
            # self.control_frame,
        )

        # Update the control_frame with the new model
        self.update_model_frame(False)

        # # Initialize an index for the fit_results
        self.parameters.values = list(param_values.values())
        self.parameters_std.values = list(param_stds.values())

        self.ci_inputs = None
        self.root.activateWindow()
        self.update_datasets()

    def validate_bounds(self, bounds):
        """Validate the bounds for the parameter values."""
        params = np.array(self.parameters.values)[:, 0]

        # Update all bounds
        res = np.column_stack(
            (
                np.minimum(bounds[:, 0], params),
                np.maximum(bounds[:, 1], params),
            )
        ).tolist()
        return res

    def quick_bounds(self, *args):
        """Set the bounds to ±10% of the current parameter values."""
        bounds = [
            sorted(
                [
                    p.values[0] * self.quick_bound_vals["low"],
                    p.values[0] * self.quick_bound_vals["high"],
                ]
            )
            for p in self.parameters
        ]
        # set unchecked values
        self.bounds.unchecked_values = bounds

    def update_error(self, scatter_df=None, generated_df=None):
        """Update the graphs based on the selected option."""
        bode_y1 = f"{self.data.var_val[self.top_type_var.currentText()]}.{self.data.var_val[self.top_mode_var.currentText()]}"
        bode_y2 = f"{self.data.var_val[self.bot_type_var.currentText()]}.{self.data.var_val[self.bot_mode_var.currentText()]}"

        if scatter_df is None:
            # Parse data
            if (
                    self.data.raw != {}
                    and self.dataset_var.currentText() in self.data.raw.keys()
            ):
                system_data = self.data.raw[self.dataset_var.currentText()]
                scatter_df = system_data.get_custom_df(
                    "freq", bode_y1, bode_y2
                )
            else:
                self.error_printout.setText("Error: N/A")
                return

        if generated_df is None:
            generated_df = self.data.generate(
                [entry.values[0] for entry in self.parameters],
                freq=scatter_df["freq"].to_numpy(),
                **self.options["simulation"],
            ).get_custom_df("freq", bode_y1, bode_y2)

        if self.options["simulation"]["limit_error"]:
            # Retrieve f_min and f_max from options
            f_min = self.options["simulation"]["freq_start"]
            f_max = self.options["simulation"]["freq_stop"]

            # Filter the scatter_data based on f_min and f_max
            scatter_df = scatter_df[
                (scatter_df["freq"] >= f_min) & (scatter_df["freq"] <= f_max)
                ]
            generated_df = generated_df[
                (generated_df["freq"] >= f_min)
                & (generated_df["freq"] <= f_max)
                ]

            if scatter_df.empty:
                QMessageBox.warning(
                    self.root,
                    "Warning",
                    "No data points within the specified frequency range.",
                )
                return

            if len(scatter_df) != len(generated_df):
                generated_df = self.data.generate(
                    [entry.values[0] for entry in self.parameters],
                    freq=scatter_df["freq"].to_numpy(),
                    **self.options["simulation"],
                ).get_custom_df("freq", bode_y1, bode_y2)

        if len(scatter_df) != len(generated_df):
            self.error_printout.setText("Error: N/A")
            return

        # Error calculation needs generated data of same length as scatter data
        error_type = self.error_var.currentText()
        if error_type in self.data.error_methods.keys():
            # if error_type in self.data.error_methods:
            data = [
                scatter_df[bode_y1],
                scatter_df[bode_y2],
                generated_df[bode_y1],
                generated_df[bode_y2],
            ]

            if "log" in error_type.lower():
                data = [abs(d) for d in data]

            error_all = Statistics()[self.data.error_methods[error_type]](
                np.hstack(data[:2]), np.hstack(data[2:])
            )

            if abs(np.log10(abs(error_all))) > 2:
                self.error_printout.setText(f"Total Error: {error_all:.4e}")
            else:
                self.error_printout.setText(f"Total Error: {error_all:.4f}")
        else:
            self.error_printout.setText("Error: N/A")
        return

    def update_graphs(self, *args, clear_axis=True):
        """Update the graphs based on the selected option."""
        self.plotted_data = {}
        self.parameters.update_interval(self.options["simulation"]["interval"])
        params_values = [entry.values[0] for entry in self.parameters]
        generated_df = self.data.generate(
            params_values, **self.options["simulation"]
        )

        ny_x = f"{self.data.var_val[self.ny_type_var.currentText()]}.real"
        ny_y = f"{self.data.var_val[self.ny_type_var.currentText()]}.pos_imag"

        bode_y1 = f"{self.data.var_val[self.top_type_var.currentText()]}.{self.data.var_val[self.top_mode_var.currentText()]}"
        bode_y2 = f"{self.data.var_val[self.bot_type_var.currentText()]}.{self.data.var_val[self.bot_mode_var.currentText()]}"

        cmaps = [
            [
                scatter.get_cmap().name
                for scatter in GeneratePlot.get_scatter(ax)
            ]
            for ax in [self.ax1, self.ax2, self.ax3]
        ]

        base_cmaps = [
            "Spectral_r",
            "coolwarm",
            "Blues_r",
            "Greens_r",
            "Oranges_r",
            "Reds_r",
            "YlOrBr_r",
            "YlOrRd_r",
            "OrRd_r",
            "PuRd_r",
            "RdPu_r",
            "BuPu_r",
            "GnBu_r",
            "PuBu_r",
            "YlGnBu_r",
            "PuBuGn_r",
            "BuGn_r",
            "YlGn_r",
            "Greys_r",
            "Purples_r",
        ]

        for i, ax_cmaps in enumerate(cmaps):
            for cmap in base_cmaps:
                if cmap not in ax_cmaps:
                    ax_cmaps.append(cmap)
            cmaps[i] = ax_cmaps

        markers = (
            "o",
            "v",
            "^",
            "<",
            ">",
            "8",
            "s",
            "p",
            "*",
            ".",
            "h",
            "H",
            "D",
            "d",
            "P",
            "X",
        )

        # Parse data
        main_dataset = (
            "Model" if self.data.raw == {} else self.dataset_var.currentText()
        )
        checked_names = []
        if self.data.raw != {}:
            main_dataset = self.dataset_var.currentText()
            checked_names = self.datasets.checked_names.copy()

            if main_dataset in checked_names:
                checked_names.remove(main_dataset)
                checked_names.insert(0, main_dataset)

        self.plotted_data["Model"] = generated_df

        # Clear the axes in preparation
        if clear_axis:
            self.nyquist_plot.clear()
            self.bode_plot.clear()

        charge = (
                generated_df[
                    self.data.var_val[self.ny_type_var.currentText()]
                ].sign
                * -1
        )
        charge = "-" if charge < 0 else ""

        self.nyquist_plot.title = (
            f"{self.ny_type_var.currentText()} Nyquist of {main_dataset}"
        )
        self.nyquist_plot.xlabel = self.data.parse_label(
            self.ny_type_var.currentText(), "real"
        )
        self.nyquist_plot.ylabels = charge + self.data.parse_label(
            self.ny_type_var.currentText(), "imag"
        )

        bode_label1 = self.data.parse_label(
            self.top_type_var.currentText(),
            self.top_mode_var.currentText(),
        )

        if "imag" in self.top_mode_var.currentText():
            charge = (
                    generated_df[
                        self.data.var_val[self.top_type_var.currentText()]
                    ].sign
                    * -1
            )
            charge = "-" if charge < 0 else ""
            bode_label1 = charge + bode_label1

        bode_label2 = self.data.parse_label(
            self.bot_type_var.currentText(),
            self.bot_mode_var.currentText(),
        )

        if "imag" in self.bot_mode_var.currentText():
            charge = (
                    generated_df[
                        self.data.var_val[self.bot_type_var.currentText()]
                    ].sign
                    * -1
            )
            charge = "-" if charge < 0 else ""
            bode_label2 = charge + bode_label2

        space = " "

        def scale_check(mode_var, check):
            mode_text = mode_var.currentText().lower()
            if check.isChecked():
                if mode_text == "imag":
                    mode_var.setCurrentText("+Imag")
                    return "log"
                elif mode_text == "θ":
                    check.setChecked(False)
                    return self.data.var_scale[mode_var.currentText()]
                else:
                    return "log"
            else:
                if mode_text == "mag" or mode_text == "+imag":
                    check.setChecked(True)
                return self.data.var_scale[mode_var.currentText()]

        self.bode_plot.title = f"{bode_label1.split(space)[0]} & {bode_label2.split(space)[0]} of {main_dataset}"
        self.bode_plot.ylabels = [bode_label1, bode_label2]
        self.bode_plot.yscales = [
            scale_check(self.top_mode_var, self.top_log_checkbox),
            scale_check(self.bot_mode_var, self.bot_log_checkbox),
        ]

        self.line = []

        ### Begin Plots
        ## Begin Scatter plots
        if self.data.raw != {}:
            self.update_error()
            n_cmap = 0
            for key in checked_names:
                system_data = self.data.raw[key]
                # Plot scatter
                self.nyquist_plot.plot(
                    "scatter",
                    system_data.get_custom_df(ny_x, ny_y, "freq"),
                    label=f"_{key}",
                    styling="freq",
                    marker=markers[n_cmap],
                    **GeneratePlot.DecadeCmapNorm(
                        system_data["freq"], cmap=cmaps[0][n_cmap]
                    ),
                )
                self.nyquist_plot.annotate(
                    system_data.get_custom_df(ny_x, ny_y, "freq")
                )

                self.bode_plot.plot(
                    "scatter",
                    system_data.get_custom_df("freq", bode_y1, bode_y2),
                    label=f"_{key}",
                    styling="freq",
                    marker=markers[n_cmap],
                    **GeneratePlot.DecadeCmapNorm(
                        system_data["freq"], cmap=cmaps[1][n_cmap]
                    ),
                )

                for i, scatter in enumerate(
                        GeneratePlot.get_scatter(self.ax3)
                ):
                    cmap_norm = GeneratePlot.DecadeCmapNorm(
                        scatter.get_array(), cmaps[2][i]
                    )
                    scatter.set_cmap(cmap_norm["cmap"])
                    scatter.set_norm(cmap_norm["norm"])

                self.plotted_data[key] = system_data
                n_cmap += 1
                if n_cmap >= min(len(cmap) for cmap in cmaps):
                    n_cmap = 0

        # plot generated
        self.nyquist_plot.plot(
            "line",
            generated_df.get_custom_df(ny_x, ny_y, "freq"),
            label="_model",
            styling="r",
        )

        self.nyquist_plot.annotate(
            generated_df.get_custom_df(ny_x, ny_y, "freq"), color="r"
        )

        self.bode_plot.plot(
            "line",
            generated_df.get_custom_df("freq", bode_y1, bode_y2),
            label="_model",
            styling="r",
        )

        self.line.append(self.ax1.lines[-1])
        self.line.append(self.ax2.lines[-1])
        self.line.append(self.ax3.lines[-1])
        self.line_count = self.nyquist_plot.count

        ## Begin Band plots
        if self.nyquist_checkbox.isChecked() or self.bode_checkbox.isChecked():
            if isinstance(self.options["bands"]["std_devs"], (int, float)):
                std_devs = [
                    self.options["bands"]["std_devs"] * param
                    for param in params_values
                ]
            else:
                std_devs = [v[0] for v in self.parameters_std.values]

            ci_inputs = {
                "percentile": self.options["bands"]["percentile"],
                "params_values": params_values,
                "std_devs": std_devs,
                "model": self.data.model,
                "freq": generated_df["freq"],
                "num_freq_points": self.options["bands"]["band_freq_num"],
                "target_form": ["freq", bode_y1, bode_y2],
                "thickness": self.options["simulation"]["thickness (cm)"],
                "area": self.options["simulation"]["area (cm^2)"],
            }
            if self.ci_inputs is None:
                self.ci_inputs = ci_inputs
                remake_ci = True
            else:
                remake_ci = False
                for key, value in ci_inputs.items():
                    if self.ci_inputs[key] != value:
                        self.ci_inputs = ci_inputs
                        remake_ci = True
                        break

            if remake_ci:
                # Generate confidence interval data self.parameters_std
                ci_analysis = ConfidenceAnalysis(
                    self.options["bands"][
                        "percentile"
                    ],  # 5 converts to 97.5 and 2.5
                    params_values,  # popt
                    std=std_devs,  # standard
                    func=wrapCircuit(self.data.model, {}),
                )

                ci_df = ci_analysis.gen_conf_band(
                    generated_df["freq"],
                    num_freq_points=self.options["bands"]["band_freq_num"],
                    main_col="real",
                    target_form=["freq", bode_y1, bode_y2],
                    thickness=self.options["simulation"]["thickness (cm)"],
                    area=self.options["simulation"]["area (cm^2)"],
                )
                self.ci_inputs["ci_df"] = ci_df
            else:
                ci_df = self.ci_inputs["ci_df"]

            perc_cols = [
                col for col in list(ci_df.values())[0].columns if "%" in col
            ]
            min_col = perc_cols[0] if perc_cols else "min"
            max_col = perc_cols[-1] if perc_cols else "max"

            if self.nyquist_checkbox.isChecked():
                self.nyquist_plot.plot(
                    "band",
                    ci_df,
                    ["nyquist"],
                    ["real", min_col, max_col],
                    styling=self.options["bands"]["band_color"],
                    alpha=self.options["bands"]["band_alpha"],
                )
            if self.bode_checkbox.isChecked():
                self.bode_plot.plot(
                    "band",
                    ci_df,
                    [bode_y1, bode_y2],
                    ["freq", min_col, max_col],
                    styling=self.options["bands"]["band_color"],
                    alpha=self.options["bands"]["band_alpha"],
                )

        ## Begin pinned line plots
        for _, row in self.pinned.df.iterrows():
            if row["Show"] != "":
                name = row["Name"]
                model = row["Model"]

                # Extract parameter values and stds
                params_values = [
                    row[f"{name}_values"]
                    for name in self.data.parse_parameters(model)
                ]

                pinned_df = self.data.generate(
                    params_values, model=model, **self.options["simulation"]
                )

                self.plotted_data[name] = pinned_df
                # Plot pinned results using GeneratePlot objects
                self.nyquist_plot.plot(
                    "line",
                    pinned_df.get_custom_df(ny_x, ny_y, "freq"),
                    styling=row["Show"],
                    label=name,
                )
                self.bode_plot.plot(
                    "line",
                    pinned_df.get_custom_df("freq", bode_y1, bode_y2),
                    styling=row["Show"],
                    label=name,
                )

        if any(self.pinned.df["Show"] != ""):
            self.ax1.legend()
            self.ax2.legend()

        self.ax1 = self.nyquist_plot.ax[0]
        self.ax2, self.ax3 = self.bode_plot.ax

        self.nyquist_plot.apply_formats()
        self.bode_plot.apply_formats()

        self.canvas1.draw()
        self.canvas2.draw()

        self.cursor_var.clear()
        self.cursor_var.addItems(list(self.plotted_data.keys()))
        self.print_window.write("Graphs updated")

    def update_line_data(self):
        """Update the line data in real-time."""
        # Get the selected option from the combobox
        if len(self.parameters.names) != len(
                self.data.parse_parameters(self.data.model)
        ):
            return

        generated_df = self.data.generate(
            [entry.values[0] for entry in self.parameters],
            **self.options["simulation"],
        )
        self.parameters.update_history()
        self.parameters_std.update_history()
        self.parameters.update_interval(self.options["simulation"]["interval"])

        ny_x = f"{self.data.var_val[self.ny_type_var.currentText()]}.real"
        ny_y = f"{self.data.var_val[self.ny_type_var.currentText()]}.pos_imag"

        bode_y1 = f"{self.data.var_val[self.top_type_var.currentText()]}.{self.data.var_val[self.top_mode_var.currentText()]}"
        bode_y2 = f"{self.data.var_val[self.bot_type_var.currentText()]}.{self.data.var_val[self.bot_mode_var.currentText()]}"

        keys = [[ny_x, ny_y], ["freq", bode_y1], ["freq", bode_y2]]
        if self.line is not None:
            if self.data.raw != {}:
                self.update_error()
            self.plotted_data["Model"] = generated_df

            for line, key in zip(self.line, keys):
                line.set_xdata(generated_df[key[0]])
                line.set_ydata(generated_df[key[1]])

            if self.nyquist_plot is not None:
                self.nyquist_plot.formatting()
                self.nyquist_plot.scale()
                self.nyquist_plot.square()
                self.nyquist_plot.update_annotation(
                    data=generated_df.get_custom_df(*keys[0], "freq"),
                    index=self.line_count - 1,
                    color="r",
                )

            self.canvas1.draw()
            self.canvas2.draw()
        self.print_window.write("Line data updated")

    def update_format(self):
        if self.nyquist_plot is not None:
            self.nyquist_plot.formatting()
            self.nyquist_plot.scale()
            self.nyquist_plot.square()

        if self.bode_plot is not None:
            self.bode_plot.formatting()
            self.bode_plot.scale()

        self.canvas1.draw()
        self.canvas2.draw()

    def nyquist_cursor_position(self, event):
        """Update the cursor position display with x, y, and z values."""
        x_in, y_in = event.xdata, event.ydata
        if x_in is not None and y_in is not None and self.plotted_data is not None:
            df = self.plotted_data[self.cursor_var.currentText()]
            x_col = f"{self.data.var_val[self.ny_type_var.currentText()]}.real"
            y_col = (
                f"{self.data.var_val[self.ny_type_var.currentText()]}.pos_imag"
            )
            z_col = "freq"

            distances = np.sqrt((df[x_col] - x_in) ** 2 + (df[y_col] - y_in) ** 2)
            nearest_index = np.argmin(distances)
            if distances[nearest_index] < 0.05 * self.ax1.get_xlim()[1]:
                x = df[x_col][nearest_index]
                y = df[y_col][nearest_index]
                z = df[z_col][nearest_index]
                self.cursor_printout.setText(f"Point {int(nearest_index)} at:\n({x:.3e}, {y:.3e}, {z:.3e})")
            else:
                self.cursor_printout.setText("Point: N/A")
        else:
            self.cursor_printout.setText("Point: N/A")

    def bode_cursor_position(self, event):
        """Update the cursor position display with x, y, and z values."""
        x_in, y_in = event.xdata, event.ydata
        if x_in is not None and y_in is not None and self.plotted_data is not None:
            ax = event.inaxes
            x_in = np.log10(x_in) if ax.get_xscale() == "log" else x_in
            y_in = np.log10(y_in) if ax.get_yscale() == "log" else y_in

            df = self.plotted_data[self.cursor_var.currentText()]
            x_col = "freq"
            y_col = None
            if ax == self.ax2: 
                y_col = f"{self.data.var_val[self.top_type_var.currentText()]}.{self.data.var_val[self.top_mode_var.currentText()]}"
            elif ax == self.ax3:
                y_col = f"{self.data.var_val[self.bot_type_var.currentText()]}.{self.data.var_val[self.bot_mode_var.currentText()]}"
            if y_col is not None:
                x_arr = np.log10(df[x_col]) if ax.get_xscale() == "log" else df[x_col]
                y_arr = np.log10(df[y_col]) if ax.get_yscale() == "log" else df[y_col]
                xlim = np.diff(np.log10(ax.get_xlim()))[0] if ax.get_xscale() == "log" else np.diff(ax.get_xlim())[0]
                ylim = np.diff(np.log10(ax.get_ylim()))[0] if ax.get_yscale() == "log" else np.diff(ax.get_ylim())[0]
                
                distances = np.sqrt((x_arr/xlim - x_in/xlim) ** 2 + (y_arr/ylim - y_in/ylim) ** 2)
                nearest_index = np.argmin(distances)
                if distances[nearest_index] < 0.05:
                    x = df[x_col][nearest_index]
                    y = df[y_col][nearest_index]
                    self.cursor_printout.setText(f"Point {int(nearest_index)} at:\n({x:.3e}, {y:.3e})")
                else:
                    self.cursor_printout.setText("Point: N/A")
            else:
                self.cursor_printout.setText("Point: N/A")
        else:
            self.cursor_printout.setText("Point: N/A")

    def run_bootstrap(self):
        """Run the bootstrap fit based on the selected model."""
        # Create a popup to get the number of desired iterations
        if self.data.model.lower() == "linkk":
            self.run_fit()
        else:
            iterations, ok = QInputDialog.getInt(
                self.root,
                "Bootstrap Fit",
                "Enter the number of iterations:",
                500,  # Default value
                1,  # Minimum value
                100000,  # Maximum value
                1,  # Step
            )

            if ok:
                self.run_fit(iterations=iterations)

    # Method to handle fitting and status updates
    def run_fit(self, iterations=1):
        """Run the circuit fit based on the selected model."""
        if self.data.raw == {}:
            QMessageBox.critical(self.root, "Error", "No data to fit.")
            return

        if self.data.model.lower() == "linkk":
            param_values = self.data.generate_linkk(
                self.data.raw[self.dataset_var.currentText()].base_df(
                    None, "frequency"
                ),
                area=self.options["simulation"]["area (cm^2)"],
                thickness=self.options["simulation"]["thickness (cm)"],
                **self.options["linkk"],
            )
            self.parameters.values = param_values

            self.print_window.write("\n")
            self.print_window.write(
                "Completed Lin-KK Fit\nM = {:d}\nmu = {:.2f}".format(
                    param_values[0], param_values[1]
                )
            )
            self.update_graphs()
            self.progress_dialog.close()
            self.kill_operation = False
            return

        if iterations <= 1:
            self.progress_dialog = create_progress_dialog(
                self.root,
                title="Running Fit",
                label_text="Fitting data...",
                maximum=0,
            )
        else:
            self.progress_dialog = create_progress_dialog(
                None,
                title="Running Fit",
                label_text="Fitting data...",
                cancel="Cancel Bootstrap",
            )
        self.progress_dialog.canceled.connect(self.cancel_operation)
        self.progress_dialog.show()

        self.worker = FittingWorker(self, iterations)

        self.run_in_thread(
            self.on_fitting_finished,
            self.on_worker_error,
            self.update_progress,
            self.progress_dialog,
        )

    def on_fitting_finished(self, fit_results):
        """Handle the completion of the fitting operation."""

        if "mean" not in fit_results.index or "std" not in fit_results.index:
            fit_attrs = fit_results.attrs.copy()
            fit_results = fit_results.describe()
            fit_results.attrs = fit_attrs

        self.fit_results = fit_results
        self.progress_dialog.close()
        self.kill_operation = False
        # Normal fit or parsed bootstrap fit
        reply = QMessageBox.question(
            self.root,
            "Update Parameters",
            "Do you want to update the parameters with the fit results?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.apply_fit_results()

    def apply_fit_results(self):
        """Apply the fit results to the parameters."""
        if self.fit_results is None:
            return

        df_attrs = self.fit_results.attrs
        if (
                self.data.model != df_attrs["model"]
                or self.dataset_var.currentText() != df_attrs["dataset_var"]
        ):
            self.data.model = df_attrs["model"]
            self.model_entry.setText(self.data.model)
            self.dataset_var.setCurrentText(df_attrs["dataset_var"])
            self.update_model_frame(False)

        self.options["simulation"]["thickness (cm)"] = df_attrs["thickness"]
        self.options["simulation"]["area (cm^2)"] = df_attrs["area"]

        for name in self.fit_results.columns:
            self.parameters[name] = [self.fit_results.loc["mean", name]]
            self.parameters_std[name] = [self.fit_results.loc["std", name]]

        self.update_graphs()

# Main entry point for the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = QMainWindow()
    ui = GraphGUI(main_window)
    main_window.show()
    sys.exit(app.exec_())
