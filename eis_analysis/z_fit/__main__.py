# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney
pound !usr/bin/env python3
General function file
"""
# cSpell: ignore whos, linkk, stds, interp, savgol
# cSpell: ignoreRegExp /cmap[\w]?/gi
# cSpell:includeRegExp #.*
# cSpell:includeRegExp /(["]{3}|[']{3})[^\1]*?\1/g
# only comments and block strings will be checked for spelling.
import re
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QEventLoop, pyqtSignal
from PyQt5.QtWidgets import (
    QMenu,
    QFrame,
    QLabel,
    QAction,
    QWidget,
    QMenuBar,
    QCheckBox,
    QSplitter,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QApplication,
    QInputDialog,
)
from IPython.core.getipython import get_ipython
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from .options import DictWindow, JsonDictWindow
from .pin_widget import DataTreeWindow
from ..string_ops import ContainerEvaluator
from .gui_helpers import tools
from .gui_windows import (
    MultiEntryWindow,
    DataHandlerWidgets,
)
from .gui_workers import (
    FittingWorker,
    LoadDataWorker,
    WorkerFunctions,
    SaveFiguresWorker,
    SaveResultsWorker,
)
from .data_handlers import DataGenerator
from .model_widgets import ModelLineEdit
from ..data_treatment import Statistics
from .pkg_vault_funcs import (
    GraphGUIError,
    logger,
    set_style,
    log_warning,
    show_error_message,
    graceful_error_handler,
    construct_error_message,
)
from ..widgets.ipython import MainConsole
from .parameter_widgets import MultiEntryManager, ParameterStatPanel
from ..widgets.data_view import DataViewer
from ..utils.plot_factory import StylizedPlot
from .plot_control_widgets import PlotControlPanel
from ..impedance_supplement import (
    ImpedanceFunc,
    parse_parameters,
    extract_ckt_elements,
)
from ..widgets.widget_helpers import create_separator
from ..widgets.generic_widgets import (
    FormDialog,
    AlignComboBox,
    SetIncLineEdit,
    CollapsibleSection,
    tLabel,
)
from ..z_system.impedance_band import ImpedanceConfidence
from ..widgets.settings_handlers import SettingsManager, manage_settings_files

warnings.showwarning = log_warning

np.seterr(all="raise")


class GraphGUI(QMainWindow, WorkerFunctions):
    """Class to create a GUI for plotting graphs."""

    # connect the other to the signal which will be called on "emit"
    close_children = pyqtSignal()

    def __init__(self, **kwargs):
        super().__init__()
        logger.info("Starting EIS Fitting GUI")

        self.setWindowTitle("EIS Fitting")

        # Create a central widget and set it as the central widget of the main window
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        settings_manager = manage_settings_files(base_path=__file__)
        # ---- Initialize parameters ----
        self._is_debugging = kwargs.get("debug", False)
        self.plotted_data = []
        self.line = []
        self.line_count = 0
        self.ci_inputs = {}
        self.cancel_event = WorkerFunctions.cancel_event
        self.fit_results = None
        self.data_viewer = None
        self.last_save_name = ""
        self.backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.auto_fit_params = {}
        # self.fig1: Any = None
        # self.fig2: Any = None
        # self.ax1: Any = None
        # self.ax2: Any = None
        # self.ax3: Any = None
        self._block_update = False

        # ---- Specialized Widgets and internal components ----
        # self.data = DataHandlerWidgets(
        #     callback=self.update_graphs, settings_path=settings_manager.settings_path
        # )
        self.data = DataHandlerWidgets(callback=self.update_graphs)
        self.close_children.connect(self.data.destroy)
        self.settings = SettingsManager(
            settings_path=settings_manager.settings_path, root_dir=__file__
        )

        self.options = JsonDictWindow(
            self,
            self.settings.option_inits,
            "option_inits",
            "Options",
            json_settings=settings_manager,
            close_on_save_all=True,
            # root_dir=__file__,
        )
        self.close_children.connect(self.options.destroy)

        self.quick_bound_vals = DictWindow(
            self, {"low": 0.1, "high": 10}, "Quick Bounds", add_button=False, close_on_save=True
        )
        self.close_children.connect(self.quick_bound_vals.destroy)

        # Windows and special classes full_shutdown
        self.ipython = MainConsole(self)
        self.ipython.console.push({"root": self, "np": np, **tools, "DataViewer": DataViewer})
        self.close_children.connect(self.ipython.full_shutdown)

        self.pinned = DataTreeWindow(
            None,  # parent/root
            ["Name", "Dataset", "Model", "Show", "Comments"],  # columns for dataframe
            ["Name", "Dataset", "Model", "Values", "Show", "Comments"],  # columns for tree_cols
            self.add_pin,  # Add row callback
            self.pull_pin,  # Use row callback
            graphing_callback=self.update_graphs,
            df_base_cols=["Name", "Dataset", "Model", "Comments"],
            df_sort_cols=["values", "std"],
            tree_gr_cols=["Values"],
            narrow_cols=["Show"],
            wide_cols=["Values", "Comments"],
        )
        self.close_children.connect(self.pinned.close)

        self.bounds = MultiEntryWindow(
            self,
            num_entries=2,
            callbacks=dict(
                save=self.validate_bounds,
                button_Quick_Bounds=self.quick_bounds,
                button_Boundary_Options=self.quick_bound_vals.window,
            ),
        )
        self.close_children.connect(self.bounds.close)

        self.parameters_std = MultiEntryWindow(
            None,
            has_check=False,
        )
        self.close_children.connect(self.parameters_std.close)

        self.parameters = MultiEntryManager(
            None,
            callback=self.update_line_data,
            interval=0.1,
        )

        self.generator = DataGenerator(options=self.options.get_view(["simulation"], flatten=True))

        # Create a file menu
        self.create_menu()

        # ----------------- Frame Creation and Splitting -----------------
        # ---- Control Section ----
        # -- Primary Frame for control --
        self.control_frame = QFrame(central_widget)
        control_frame_layout = QVBoxLayout()
        control_frame_layout.setAlignment(Qt.AlignTop)
        self.control_frame.setLayout(control_frame_layout)
        self.control_frame.setMaximumSize(500, 16777215)
        self.control_frame.setFrameStyle(QFrame.WinPanel | QFrame.Sunken)
        self.control_frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        # -- Parameter Frame --
        self.param_frame = QFrame()
        self.param_frame.setMaximumHeight(250)
        self.param_frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        # -- Plot control Frame --
        plot_controls_section = CollapsibleSection("Plot Controls", self.control_frame)

        # ---- Graph Section ----
        # -- Primary Frame for graphs --
        self.graph_frame = QFrame(central_widget)
        graph_frame_layout = QHBoxLayout()
        self.graph_frame.setLayout(graph_frame_layout)

        # -- Subplot Frames --
        self.nyquist_frame = QFrame(self.graph_frame)
        self.nyquist_frame.setMinimumSize(300, 300)

        self.bode_frame = QFrame(self.graph_frame)
        self.bode_frame.setMinimumSize(400, 300)

        # ---- Splitters ----
        # -- Splitter to divide control and graph frames --
        splitter = QSplitter(Qt.Horizontal, central_widget)
        splitter.addWidget(self.control_frame)
        splitter.addWidget(self.graph_frame)
        splitter.setSizes([300, 800])

        # -- Splitter to divide the graph frames --
        graph_splitter = QSplitter(Qt.Horizontal, self.graph_frame)
        graph_splitter.setStyleSheet("QSplitter::handle { background-color: #f0f0f0; }")  # #bdbdbd
        graph_splitter.addWidget(self.nyquist_frame)
        graph_splitter.addWidget(self.bode_frame)
        graph_splitter.setSizes([400, 600])

        # ---- Finalyze Frames and splitters ----
        main_layout = QHBoxLayout(central_widget)
        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)

        graph_frame_layout.addWidget(graph_splitter)

        # ----------------- Widget Creation -----------------
        # ---- Line Edits ----
        # -- Model entry --
        self.model_entry = ModelLineEdit(
            self.control_frame, ignore_error=True, default=self.settings.model
        )
        self.model_entry.setAlignment(Qt.AlignCenter)
        self.model_entry.onError.connect(
            lambda e: show_error_message(
                e, "Model Validation", "invalid model string", parent=self
            )
        )

        # -- Incriment widgets --
        self.coarse_var = SetIncLineEdit(parent=self.control_frame)
        self.coarse_var.setText("0.1")
        self.coarse_var.setToolTip("up/dwn arrows or mouse wheel")
        self.coarse_var.editingFinished.connect(
            lambda: self.parameters.update_entry_args(interval=self.coarse_var.incr())
        )

        self.fine_var = SetIncLineEdit(parent=self.control_frame)
        self.fine_var.setText("0.01")
        self.fine_var.setToolTip("hold shift for fine control")
        self.fine_var.editingFinished.connect(
            lambda: self.parameters.update_entry_args(fine_interval=self.fine_var.incr())
        )

        # Link the two incriment widgets
        self.coarse_var.gt_wigets.append(self.fine_var)
        self.fine_var.lt_wigets.append(self.coarse_var)

        # -- Axis def widgets --
        self.cont = PlotControlPanel(
            self.control_frame, self.settings.plot_var, self.settings.is_log
        )
        self.param_stats = ParameterStatPanel(
            self.control_frame, parameters=self.parameters, model_entry=self.model_entry
        )
        self.param_stats.hide()

        # ---- Dropdown menus ----
        # -- Dataset sel --
        self.data.var = AlignComboBox(self.control_frame)
        self.data.var.setTextAlignment(Qt.AlignCenter)
        self.data.var.setFixedWidth(150)
        self.data.var.addItems(["None"])
        self.data.var.setCurrentText("None")
        self.data.var.currentIndexChanged.connect(self.update_graphs)
        self.data.var.currentIndexChanged.connect(
            lambda _: self.param_stats.update(data=self.data.primary_df())
        )

        # -- Point cursor sel --
        self.cursor_var = AlignComboBox(self.control_frame)
        self.cursor_var.setTextAlignment(Qt.AlignCenter)
        self.cursor_var.addItems(["Model"])
        self.cursor_var.setCurrentText("Model")
        self.cursor_var.currentIndexChanged.connect(self.update_format)

        # -- Error type sel --
        self.error_var = AlignComboBox(self.control_frame)
        self.error_var.setTextAlignment(Qt.AlignRight)
        self.error_var.addItems(self.settings.error_methods.keys())
        self.error_var.setCurrentText(self.settings.error_var)
        self.error_var.currentIndexChanged.connect(lambda _: self.update_error())
        self.error_var.currentIndexChanged.connect(
            lambda _: self.settings.save_settings(error_var=self.error_var.currentText())
        )

        # -- Error data sel --
        self.sel_error_var = AlignComboBox(self.control_frame)
        self.sel_error_var.addItems(["Nyquist", "Bode", "Both", "Z", "Y", "C", "M", "User"])
        self.sel_error_var.setCurrentText(self.settings.sel_error_var)
        self.sel_error_var.currentIndexChanged.connect(lambda _: self.update_error())
        self.sel_error_var.currentIndexChanged.connect(
            lambda _: self.settings.save_settings(sel_error_var=self.sel_error_var.currentText())
        )

        # ---- Buttons ----
        # -- "Use Model" aka impliment model --
        update_model_button = QPushButton("Use Model", self.control_frame)
        update_model_button.setFixedWidth(150)
        update_model_button.clicked.connect(lambda *_: self.update_param_section(False, True))

        self.model_entry.validityChanged.connect(update_model_button.setEnabled)
        self.model_entry.replaceElement.connect(self.replace_element)
        self.model_entry.convertSection.connect(self.convert_section)

        # -- "Update Graphs" aka Replot data --
        graphing_button = QPushButton("Update Graphs", self.control_frame)
        graphing_button.setFixedWidth(150)
        graphing_button.clicked.connect(self.update_graphs)
        graphing_button.clicked.connect(self.cont.parse_axes)

        # ---- Checkboxes ----
        self.model_check = QCheckBox("Model", self.control_frame)
        self.sub_model_check = QCheckBox("Sub Models", self.control_frame)
        self.legend_check = QCheckBox("Legend", self.control_frame)
        self.model_check.setChecked(True)
        self.legend_check.setChecked(True)

        # ---- labels ----
        data_header = tLabel("Measured Data", 12, self.control_frame)
        model_header = tLabel("Model", 12, self.control_frame)
        info_header = tLabel("Information", 12, self.control_frame)
        error_label = tLabel("Error of -> ", 10, self.control_frame)
        cursor_label = tLabel("Point at cursor of ", 10, self.control_frame)
        coarse_label = tLabel("Coarse", parent=self.control_frame)
        fine_label = tLabel("Fine", parent=self.control_frame)

        # Create labels for displaying values
        self.error_printout = QLabel("N/A", self.control_frame)
        self.error_printout.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
        )
        self.cursor_printout = QLabel("Point:\nN/A")
        self.cursor_printout.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
        )

        # ----------------- Compile Layout/Widget Insertion -----------------
        # ---- Construct addtional layouts ----
        # -- Layouts for the Control Frame (lvl 1) --
        incr_layout = QHBoxLayout()
        cols_layout = QGridLayout()
        error_layout = QGridLayout()
        freq_layout = QVBoxLayout()

        # -- Sub-Layouts for Frequency and Cursor (lvl 2) --
        freq_sub_layout = QHBoxLayout()

        # ---- Add widgets to the control frame layouts ----
        # -- Dataset section --
        control_frame_layout.addWidget(data_header, alignment=Qt.AlignCenter)
        control_frame_layout.addWidget(self.data.var, alignment=Qt.AlignCenter)
        control_frame_layout.addWidget(create_separator(self.control_frame))

        # -- Model section --
        control_frame_layout.addWidget(model_header, alignment=Qt.AlignCenter)
        control_frame_layout.addWidget(self.model_entry)  # , alignment=Qt.AlignCenter)
        control_frame_layout.addWidget(update_model_button, alignment=Qt.AlignCenter)
        control_frame_layout.addWidget(create_separator(self.control_frame))

        # -- Parameter section (partial) --
        incr_layout.addWidget(coarse_label)
        incr_layout.addWidget(self.coarse_var)
        incr_layout.addWidget(fine_label)
        incr_layout.addWidget(self.fine_var)
        control_frame_layout.addLayout(incr_layout)
        control_frame_layout.addWidget(self.param_frame)
        control_frame_layout.addWidget(create_separator(self.control_frame))

        plot_controls_section.frame.setLayout(self.cont.layout())
        control_frame_layout.addWidget(plot_controls_section)

        # -- Checkbox grid below plot controls --
        cols_layout.addWidget(self.legend_check, 0, 0)
        cols_layout.addWidget(self.model_check, 0, 1)
        cols_layout.addWidget(self.sub_model_check, 0, 2)
        control_frame_layout.addLayout(cols_layout)

        # -- Button/checks below checkbox grid --
        control_frame_layout.addWidget(graphing_button, alignment=Qt.AlignCenter)
        control_frame_layout.addWidget(create_separator(self.control_frame))

        # -- Info (Error & Freq) section --
        control_frame_layout.addWidget(info_header, alignment=Qt.AlignCenter)
        # -- Error section --
        error_layout.addWidget(error_label, 0, 0, alignment=Qt.AlignRight)
        error_layout.addWidget(self.sel_error_var, 0, 1)
        error_layout.addWidget(self.error_var, 1, 0)
        error_layout.addWidget(self.error_printout, 1, 1)
        control_frame_layout.addLayout(error_layout)
        control_frame_layout.addWidget(create_separator(self.control_frame))

        # -- Frequency and cursor section --
        freq_sub_layout.addWidget(cursor_label)
        freq_sub_layout.addWidget(self.cursor_var)
        freq_layout.addLayout(freq_sub_layout)
        freq_layout.addWidget(self.cursor_printout)
        control_frame_layout.addLayout(freq_layout)

        # ----------------- Graph/Canvas Creation -----------------
        # ---- Styling and variables ----
        set_style(self.options["plotting"]["style"])
        self.style_history = {
            "style": self.options["plotting"]["style"],
            "background": self.options["plotting"]["background"],
        }
        self._show_crosshairs = False
        self._show_linked_crosshairs = False

        # ---- Get matplotlib figures and axis ----
        self.fig1, self.ax1 = StylizedPlot.subplots(
            figsize=(6, 6), layout="constrained"
        )  # Nyquist plot

        self.fig2, (self.ax2, self.ax3) = StylizedPlot.subplots(
            2, 1, figsize=(10, 10), sharex=True, layout="constrained"
        )  # Bode plots
        self.ax2.set_label("Bode Top Figure")
        self.ax3.set_label("Bode Bottom Figure")

        if self.options["plotting"]["background"]:
            try:
                self.ax1.set_facecolor(self.options["plotting"]["background"])
                self.ax2.set_facecolor(self.options["plotting"]["background"])
                self.ax3.set_facecolor(self.options["plotting"]["background"])
            except ValueError as exc:
                show_error_message(
                    exc, title="Initialization", message="setting background color", popup=False
                )

        # -- Create StylizedPlot objects to manage the plots --
        self.nyquist_plot = StylizedPlot(
            self.ax1,
            title="Nyquist of Model",
            scales="LinFrom0Scaler",
            init_formats=["scale", "format", "square"],
            f_kwargs=dict(power_lim=2),
        )

        self.bode_plot = StylizedPlot(
            [self.ax2, self.ax3],
            labels=[
                "Frequency (Hz)",
                "Z'",
                "Z''",
            ],
            title="Z' & Z'' of Model",
            scales=[
                "log",
                "log" if self.cont.bd.ax["y1"].is_log() else "linear",
                "log" if self.cont.bd.ax["y2"].is_log() else "linear",
            ],
            init_formats=["scale", "format"],
        )

        # ---- Canvases to display the figures ----
        self.canvas1 = FigureCanvasQTAgg(self.fig1)
        self.canvas2 = FigureCanvasQTAgg(self.fig2)

        # Connect the cursor event to update the cursor position
        self.canvas1.mpl_connect("motion_notify_event", self.nyquist_cursor_position)
        self.canvas2.mpl_connect("motion_notify_event", self.bode_cursor_position)

        self.canvas1.mpl_connect("button_press_event", self.show_canvas_context_menu)
        self.canvas2.mpl_connect("button_press_event", self.show_canvas_context_menu)

        # ---- Create the matplotlib navigation toolbars ----
        self.toolbar1 = NavigationToolbar2QT(self.canvas1, self.nyquist_frame)
        self.toolbar2 = NavigationToolbar2QT(self.canvas2, self.bode_frame)
        self.toolbar1.update()
        self.toolbar2.update()

        # ----------------- Graph Section Construction/Insertion -----------------
        # ---- Plot 1: Nyquist plot ----
        nyquist_frame_layout = QVBoxLayout()
        nyquist_frame_layout.addWidget(self.canvas1)
        nyquist_frame_layout.addWidget(self.toolbar1)
        self.nyquist_frame.setLayout(nyquist_frame_layout)

        # ---- Plot 2: Bode plot ----
        bode_frame_layout = QVBoxLayout()
        bode_frame_layout.addWidget(self.canvas2)
        bode_frame_layout.addWidget(self.toolbar2)
        self.bode_frame.setLayout(bode_frame_layout)

        # ----------------- Finalize initialization -----------------
        self.update_param_section(True, True)

        if self._is_debugging:
            QMessageBox.information(
                None,
                "Debug Warning",
                "Detected Debugging.\nAll threaded operations will will now run in the main thread",
            )

    def create_menu(self):
        """Create the menu bar for the GUI."""
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)

        file_menu = QMenu("File", self)
        menu_bar.addMenu(file_menu)

        file_menu.addAction("Load Data", self.load_data)
        file_menu.addAction("Save", lambda: self.save_results(direct_save=True, export=False))
        file_menu.addAction("Save As", lambda: self.save_results(direct_save=False, export=False))
        file_menu.addAction(
            "Export Data", lambda: self.save_results(direct_save=False, export=True)
        )
        file_menu.addAction("Export Figures", self.save_figures)
        file_menu.addSeparator()
        file_menu.addAction("Close", self.close)

        data_menu = QMenu("Data", self)
        menu_bar.addMenu(data_menu)

        data_menu.addAction("Undo", self.undo)
        data_menu.addAction("View Std Values", self.parameters_std.show)
        data_menu.addAction("View Parameter Stats", self.param_stat_show)

        data_menu.addSeparator()
        data_menu.addAction("Data List", self.data.show)
        data_menu.addAction("Dataset Points", self.edit_data)
        data_menu.addSeparator()
        data_menu.addAction("Clear Data", self.clear_datasets)

        fitting_menu = QMenu("Fitting", self)
        menu_bar.addMenu(fitting_menu)

        fitting_menu.addSeparator()
        fitting_menu.addAction("Quick Bounds", self.quick_bounds)
        fitting_menu.addAction("Modify Bounds", self.bounds.show)
        fitting_menu.addSeparator()
        fitting_menu.addAction("Run Normal Fit", self.run_fit)
        fitting_menu.addAction("Run Iterative Fit", self.run_iterative)
        fitting_menu.addAction("Run Bootstrap Fit", self.run_bootstrap)
        fitting_menu.addAction("Run Automated Seq", self.auto_fit_datasets)
        fitting_menu.addSeparator()
        fitting_menu.addAction("View Last Results", self.apply_fit_results)

        tools_menu = QMenu("Tools", self)
        menu_bar.addMenu(tools_menu)

        tools_menu.addAction("Pinned Results", self.pinned.show)
        tools_menu.addAction("Attempt Recovery", self.recover_fit_results)

        tools_menu.addSeparator()
        tools_menu.addAction("IPython Terminal", self.open_ipython_terminal)
        tools_menu.addAction("Push Variables", self.push_variables_to_terminal)
        tools_menu.addAction("Reset Terminal", self.reset_terminal)
        tools_menu.addSeparator()
        tools_menu.addAction("Options", self.options.window)

    def param_stat_show(self):
        self.param_stats.refresh()
        self.param_stats.window("Parameter Statistics")

    def closeEvent(self, event):
        """Catch the close event and close all windows."""
        self.close_children.emit()
        event.accept()

    def open_ipython_terminal(self):
        """Open the IPython terminal or redirect to Spyder's terminal."""
        # self.ipython.show()
        i_shell = get_ipython()
        if i_shell and "SPYDER" in i_shell.__class__.__name__.upper():
            # Push variables to Spyder's IPython terminal
            i_shell.push({"root": self, "np": np, **tools, "DataViewer": DataViewer})
            QMessageBox.information(
                self,
                "Spyder IPython Terminal",
                "Detected active Spyder IPython terminal. Use the console to interact dynamically."
                "The primary class is available as 'root', use '%whos' to view other supplied functions.",
            )
            # pdb.set_trace()
        else:
            QMessageBox.information(
                self,
                "IPython Terminal",
                "The IPython terminal will open in a separate window."
                "The primary class is available as 'root', use '%whos' to view other supplied functions.",
            )
            # Open the MainConsole window
            self.ipython.show()

    def reset_terminal(self):
        """Push important variables to the IPython terminal."""
        self.ipython.full_shutdown()
        self.ipython = MainConsole(self)
        self.ipython.console.push({"root": self, "np": np, **tools, "DataViewer": DataViewer})
        self.close_children.connect(self.ipython.full_shutdown)

    def push_variables_to_terminal(self):
        """Push important variables to the IPython terminal."""
        i_shell = get_ipython()
        if i_shell:
            i_shell.push({"root": self, "np": np, **tools, "DataViewer": DataViewer})

    def show_canvas_context_menu(self, event):
        """Show a context menu when right-clicking on a canvas."""
        if event.button == 3:  # Right click
            # Create a context menu
            context_menu = QMenu(self)

            # Add a point option
            add_point_action = QAction("Add Point", self)
            add_point_action.triggered.connect(lambda: self.save_current_point(event))
            context_menu.addAction(add_point_action)

            # Toggle crosshairs option with checkmark for current state
            crosshairs_action = QAction("Crosshairs", self)
            crosshairs_action.setCheckable(True)
            crosshairs_action.setChecked(self._show_crosshairs)
            crosshairs_action.triggered.connect(self.toggle_crosshairs)
            context_menu.addAction(crosshairs_action)

            # Show the menu at the cursor position
            context_menu.exec_(event.guiEvent.globalPos())

    def toggle_crosshairs(self):
        """Toggle crosshairs visibility on all plots."""
        # Invert the state of _show_crosshairs
        self._show_crosshairs = not self._show_crosshairs

        # Apply the state to both plots
        self.nyquist_plot.toggle_crosshair(self._show_crosshairs)
        self.bode_plot.toggle_crosshair(self._show_crosshairs)

        # # Redraw the canvases to ensure crosshairs appear/disappear immediately
        self.canvas1.draw_idle()
        self.canvas2.draw_idle()

    def update_param_section(self, refresh_values=False, refresh_graphs=True):
        """Update the model frame with the new model."""
        self.settings.model = self.model_entry.text()
        self.generator.model = self.model_entry.text()
        # self.parameters.interval = self.coarse_var.incr()
        # self.parameters.fine_interval = self.fine_var.incr()
        self.settings.save_settings(model=self.settings.model)

        block_status = self._block_update
        self._block_update = True

        if self.settings.model.lower() == "linkk":
            param_values = self.generator.get_linkk(
                self.data.get(self.data.primary()),
                **self.options["linkk"],
            )[1]
            self.param_stats.model = self.generator.linkk_info[0]
            self.param_stats.values = self.generator.linkk_info[1]

            self.parameters.update_entries(["M", "mu"], param_values, self.param_frame)
            self.parameters_std.update_entries(["M", "mu"], [0.1, 0.1], None)
            self.bounds.update_entries(["M", "mu"], [[0, 200], [0, 1]], None)

        else:
            # param_names = self.data.parse_parameters()
            self.param_stats.model = self.model_entry.text()
            param_names = self.generator.parse_parameters(self.model_entry.text())

            if refresh_values or param_names != self.parameters.names:
                self.parameters.update_entries(
                    param_names,
                    [
                        self.settings.parse_default(name, self.options["element"])
                        for name in param_names
                    ],
                    self.param_frame,
                    [
                        self.settings.parse_default(name, self.options["element_range"])
                        for name in param_names
                    ],
                )

                self.parameters_std.update_entries(
                    param_names,
                    [
                        self.settings.parse_default(name, self.options["element"]) * 0.1
                        for name in param_names
                    ],
                    None,
                )

                self.bounds.update_entries(
                    param_names,
                    [
                        (
                            self.bounds[name].values
                            if name in self.bounds
                            else self.settings.parse_default(name, self.options["element_range"])
                        )
                        for name in param_names
                    ],
                    None,
                )

        self.param_stats.update(data=self.data.primary_df())
        self.param_stats.refresh()

        self.ci_inputs = {}

        # Set the tab order
        self.parameters.set_tab_order(previous_widget=self.fine_var)

        self._block_update = block_status

        if refresh_graphs:
            self.update_graphs()

    def replace_element(self, old, new):
        """
        Replace an element in the model and transfer its parameter values to the new element.

        Args:
            old (str): The name of the element to replace (e.g. 'R1')
            new (str): The name of the new element (e.g. 'CPE1')
        """
        if old not in self.parameters.names and f"{old}_0" not in self.parameters.names:
            return

        try:
            self._block_update = True

            self._replace_element(old, new)

            # Update the graphs to reflect the changes
            self.update_graphs()

        except Exception as exc:
            self._block_update = False
            show_error_message(
                exc,
                "Element Replacement",
                f"Failed to replace {old} with {new}",
                parent=self,
                popup=True,
            )

    def _replace_element(self, old, new):
        """
        Replace an element in the model and transfer its parameter values to the new element.

        Args:
            old (str): The name of the element to replace (e.g. 'R1')
            new (str): The name of the new element (e.g. 'CPE1')
        """
        # Determine which form of the element exists in parameters
        param_old = old if old in self.parameters.names else f"{old}_0"

        # Save the current value and checked state for the old element
        old_values = self.parameters[param_old].values.copy()
        old_checked = self.parameters[param_old].is_checked.copy()

        # Update the model string by replacing the old element with the new element
        current_model = self.model_entry.text()
        updated_model = current_model.replace(old, new)

        # Set the new model
        self.model_entry.setText(updated_model)

        # Update the model frame to refresh the parameters
        self.update_param_section(False, False)

        # Determine target parameter name (with _0 suffix if needed)
        param_new = new if new in self.parameters.names else f"{new}_0"

        # Restore the saved value to the new element if it exists
        if param_new in self.parameters.names:
            self.parameters[param_new].values = old_values
            self.parameters[param_new].is_checked = old_checked

    def convert_section(self, current_section, old_section, new_section, converter_func, kwargs):
        """
        Convert a section of the model to a different equivalent circuit.

        Args:
            old_section (str): The original circuit section
            new_section (str): The replacement circuit section
            converter_func (callable): Function to convert parameter values
        """
        try:
            self._block_update = True
            # Extract parameter names for both old and new sections
            elements = extract_ckt_elements(current_section, lambda x: (x[0], f"{x[1]}{x[2]}"))
            if any(e[0] in ["CPE", "ICPE"] for e in elements):
                convert_cpe = QMessageBox.question(
                    self,
                    "CPE Elements Found",
                    "The selection contains CPE/ICPE elements"
                    "which need to be converted first. Convert CPEs to C and ICPEs to R?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if not convert_cpe:
                    return
                for i, (elem, suffix) in enumerate(elements):
                    if elem not in ["CPE", "ICPE"]:
                        continue
                    new_elem = f"C{suffix}" if elem == "CPE" else f"R{suffix}"
                    self._replace_element(f"{elem}{suffix}", new_elem)
                    current_section = current_section.replace(f"{elem}{suffix}", new_elem)

            old_params = parse_parameters(old_section)
            new_params = parse_parameters(new_section)

            # Check if required parameters exist
            if not all(param in self.parameters.names for param in old_params):
                show_error_message(
                    ValueError(f"Not all parameters from {old_section} exist in current model"),
                    "Section Conversion",
                    "Missing parameters for conversion",
                    parent=self,
                    popup=True,
                )
                return

            # Get current parameter values from the old section
            param_values = []
            for param in old_params:
                if param in self.parameters.names:
                    param_values.append(self.parameters[param].values[0])

            # Convert the parameter values using the provided function
            try:
                converted_values = converter_func(*param_values, **kwargs)
            except Exception:
                converted_values = converter_func(*param_values)

            # Verify that we got the correct number of converted values
            if len(converted_values) != len(new_params):
                show_error_message(
                    ValueError(
                        f"Converter function returned {len(converted_values)} values, "
                        f"but {len(new_params)} were expected"
                    ),
                    "Section Conversion",
                    "Parameter count mismatch",
                    parent=self,
                    popup=True,
                )
                return

            # Update the model string
            current_model = self.model_entry.text()
            updated_model = current_model.replace(current_section, new_section)

            # Set the new model
            self.model_entry.setText(updated_model)

            # Update the model frame to refresh the parameters
            self.update_param_section(False, False)

            # Update the parameter values with the converted values
            for i, param in enumerate(new_params):
                if param in self.parameters.names:
                    self.parameters[param].values = [converted_values[i]]

            self.update_graphs()

        except Exception as exc:
            self._block_update = False
            show_error_message(
                exc,
                "Section Conversion",
                f"Failed to convert {old_section} to {new_section}",
                parent=self,
                popup=True,
            )

    def undo(self):
        """Undo the last parameter change."""
        self.parameters.undo_recent()
        self.parameters_std.undo_recent()

    def edit_data(self):
        """Edit the data in a separate window."""
        # if self.data.primary() in self.data.raw:
        try:
            if self.data.primary() not in self.data.raw:
                raise ValueError("No active dataset")

            forms = self.cont.ny.get_text() + self.cont.bd.get_text()
            forms = [f.lower() for f in forms if f]  # remove empty strings

            if "freq" in forms:
                forms[forms.index("freq")] = "frequency"
            if "frequency" not in forms:
                forms.insert(0, "frequency")
            elif forms[0] != "frequency":
                # move frequency to the front
                forms.insert(0, forms.pop(forms.index("frequency")))

            if "impedance.real" not in forms:
                forms.insert(1, "impedance.real")
            elif forms[1] != "impedance.real":
                # move impedance.real to the front
                forms.insert(1, forms.pop(forms.index("impedance.real")))

            if "impedance.imag" not in forms:
                forms.insert(2, "impedance.imag")
            elif forms[2] != "impedance.imag":
                # move impedance.imag to the front
                forms.insert(2, forms.pop(forms.index("impedance.imag")))

            forms[0] = "freq"
            dataset = self.data.raw[self.data.primary()].get_df(*forms)
            self.data_viewer = DataViewer(dataset, self, "Raw Data")
            self.close_children.connect(self.data_viewer.close)

            loop = QEventLoop()
            self.data_viewer.destroyed.connect(loop.quit)
            loop.exec_()

            if (
                len(dataset) != len(self.data_viewer.data)
                or (dataset.to_numpy(copy=True) != self.data_viewer.data.to_numpy(copy=True)).any()
            ):
                reply = QMessageBox.question(
                    self,
                    "Apply Changes",
                    "The data has been modified. Do you want to apply?\n(Note: This will overwrite the original data)",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if reply == QMessageBox.Yes:
                    self.data.update_system(
                        self.data.primary(),
                        self.data_viewer.data,
                        form=None,
                        thickness=self.options["simulation"]["thickness"],
                        area=self.options["simulation"]["area"],
                    )
                    self.data_viewer = None
                    self.update_graphs()

        except Exception as exc:
            show_error_message(exc, title="Edit Data", message="opening data editor", parent=self)

    def load_data(self):
        """Load the data from a file."""
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Open File",
            str(self.settings.load_dir),
            "Excel files (*.xlsx);;CSV files (*.csv);;All files (*.*)",
        )

        if file_path:
            self.settings.save_settings(load_dir=Path(file_path).parent)
            self.last_save_name = ""
            self._load(file_path)

    def _load(self, file_path):
        """Load the data from a file."""

        self.worker = LoadDataWorker(file_path, self.options)
        self.create_progress_dialog(
            self, title="Loading Data", label_text="Loading data...", maximum=0
        )
        self.run_in_thread(
            self.io_data_finished,
            self.on_worker_error,
        )

    def save_results(self, *_, direct_save=False, export=False):
        """Save the fit results to a file."""
        file_path = str(self.settings.save_dir / (self.last_save_name or self.data.primary()))

        # Prompt if: no last file name, direct save is False, or file does not exist
        if not self.last_save_name or not direct_save or not Path(file_path).exists():
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save File",
                str(self.settings.save_dir),
                "Excel files (*.xlsx);;CSV files (*.csv);;All files (*.*)",
            )

        if file_path:
            self.settings.save_settings(save_dir=Path(file_path).parent)
            self.last_save_name = Path(file_path).name

            self.create_progress_dialog(
                self,
                title="Save Results",
                label_text="Saving results...",
                maximum=0,
            )

            self._save(file_path, export=export)

    def _save(self, file_path, export=False):
        """Save the fit results to a file."""

        self.worker = SaveResultsWorker(
            file_path,
            self,
            export=export,
        )

        self.run_in_thread(
            self.io_data_finished,
            self.on_worker_error,
        )

    def save_figures(self):
        """Save the figures to a file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Figures",
            str(self.settings.export_dir / self.data.primary()),
            "PNG files (*.png);;All files (*.*)",
        )

        if file_path:
            self.settings.save_settings(export_dir=Path(file_path).parent)
            self.worker = SaveFiguresWorker(file_path, self.fig1, self.fig2)
            self.create_progress_dialog(
                self,
                title="Save Figures",
                label_text="Saving figures...",
                maximum=0,
            )

            self.run_in_thread(
                self.io_data_finished,
                self.on_worker_error,
                progress_dialog=self.progress_dialog,
            )

    def io_data_finished(self, valid_sheets=None, df_in=None):
        """Handle the completion of data I/O operations."""
        if valid_sheets is not None:
            for key, value in valid_sheets.items():
                self.data.update_system(key, value)
                # self.data*5

            self.data.update_var()
            self.update_graphs()
        if df_in is not None:
            self.pinned.append_df(df_in)

        try:
            self.progress_dialog.close()
            self.progress_dialog.deleteLater()
        except (RuntimeError, AttributeError) as exc:
            show_error_message(
                exc, title="Input/Output", message="during progress dialog close", popup=False
            )

        # self.kill_operation = False
        self.cancel_event.clear()

    def clear_datasets(self):
        """Clear the current datasets."""
        self.data.raw = {}

        self.data.var.clear()
        self.data.var.addItems(["None"])
        self.data.set_var("None")

        if self.pinned.window:
            self.pinned.window.close()
        self.pinned.df = ["Name", "Dataset", "Model", "Show", "Comments"]
        self.backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.update_graphs()

    def update_datasets(self, update_plots=True):  # , *args
        """Update the scatter data based on the selected dataset."""

        if self.data.primary() in self.data.raw:
            self.data.highlight(self.data.primary())
            if update_plots:
                self.update_graphs()

    @graceful_error_handler(popup=True, title="Adding Pin")
    def add_pin(self, *, new_name=""):
        """Add a new pinned result to the treeview."""
        try:
            # Get the current dataset name
            dataset_name = self.data.primary()

            if not dataset_name or dataset_name not in self.data.raw:
                raise ValueError("No valid dataset selected to pin.")

            n = 1
            # Determine the default name
            if not self.pinned.df.empty:
                # Increment `n` until no matching name exists
                while any(
                    row["Name"].endswith(f"fit{n}") and dataset_name in row["Dataset"]
                    for _, row in self.pinned.df.iterrows()
                ):
                    n += 1

            default_name = f"{dataset_name}_fit{n}"

            name = new_name or self.pinned.add_row_popup(default_name)

            if not name:
                return

            error_printout = "Error: N/A"

            if self.data.raw != {}:
                try:
                    error_method = self.settings.error_methods[self.error_var.currentText()]
                    error_short = self.settings.error_methods_abbr[self.error_var.currentText()]
                    error_all = self.calculate_error(
                        ["Z.real", "Z.imag"],
                        error_method,
                    )

                    if abs(np.log10(abs(error_all))) > 2:
                        error_printout = f"{error_short}: {error_all:.4e}"
                    else:
                        error_printout = f"{error_short}: {error_all:.4f}"
                except (ValueError, TypeError) as exc:
                    show_error_message(
                        exc,
                        title="Add Pin",
                        message="while calculating error for pinning",
                        popup=False,
                    )

            # Prepare the new row data
            new_row = {
                "Name": name,
                "Dataset": dataset_name,
                "Model": self.settings.model,
                "Show": "",
                "Comments": error_printout,
                **{
                    f"{name}_values": entry[0]
                    for name, entry in zip(self.parameters.names, self.parameters.values)
                },
                **{
                    f"{name}_std": entry[0]
                    for name, entry in zip(self.parameters_std.names, self.parameters_std.values)
                },
            }

            # Add to the DataFrame using append_df
            self.pinned.append_df(new_row)

            self.backup_fit_results()

        except Exception as exc:
            message = construct_error_message(exc, "while adding to pinned data")
            raise GraphGUIError(message) from exc

    @graceful_error_handler(popup=True, title="Applying Pin Settings")
    def pull_pin(self, selected_row):
        """Use the values of a pinned result."""
        # Extract the model, dataset, and comments
        try:
            model = selected_row["Model"]
            dataset = selected_row["Dataset"]
            if model == np.nan or dataset == np.nan:
                return

            # params = self.data.parse_parameters(model)
            params = self.generator.parse_parameters(model)
            # Extract parameter values and stds
            param_values = {name: selected_row[f"{name}_values"] for name in params}
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
            if dataset in self.data.raw:
                self.data.set_var(dataset)

            # Update the model
            if hasattr(self, "model_entry"):
                self.model_entry.setText(model)
            self.settings.model = model

            self._block_update = True

            self.parameters.update_entries(
                params,
                list(param_values.values()),
                None,
                [self.settings.parse_default(n, self.options["element_range"]) for n in params],
            )

            self.parameters_std.update_entries(
                params,
                list(param_stds.values()),
                # self.control_frame,
            )

            # Update the control_frame with the new model
            self.update_param_section(True, False)

            # # Initialize an index for the fit_results
            self.parameters.values = list(param_values.values())
            self.parameters_std.values = list(param_stds.values())

            self.ci_inputs = {}
            self.activateWindow()
            if self.data.primary() in self.data.raw:
                self.data.highlight(self.data.primary())

            self.update_graphs()

        except Exception as exc:
            self._block_update = False
            message = construct_error_message(exc, "while extracting/applying the parameters")
            raise GraphGUIError(message) from exc

    def recover_fit_results(self):
        """Recover the fit results from a backup file."""
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Open File",
            str(Path(self.options["general"]["backup path ('~' == home dir)"]).expanduser()),
            "Excel files (*.xlsx);;CSV files (*.csv);;All files (*.*)",
        )

        if file_path:
            self._load(file_path)

    def backup_fit_results(self):
        """Backup the fit results to a file."""
        if self.options["general"]["backup pins"]:
            f_name = Path(self.options["general"]["backup path ('~' == home dir)"]).expanduser()
            f_name = f_name / (self.backup_timestamp + ".xlsx")
            # print(f_name)
            self._save(f_name)

            # Check the number of backup files and delete the oldest if necessary
            backup_files = sorted(f_name.parent.glob("*.xlsx"), key=lambda p: p.stat().st_mtime)
            max_backups = self.options["general"]["number of backups"]
            if len(backup_files) > max_backups:
                for file_to_delete in backup_files[:-max_backups]:
                    file_to_delete.unlink()

    def validate_bounds(self, bounds):
        """Validate the bounds for the parameter values."""
        if self.options["fit"]["prioritize_bounds"]:
            return
        params = np.array(self.parameters.values)[:, 0]

        # Update all bounds
        res = np.column_stack(
            (
                np.minimum(bounds[:, 0], params),
                np.maximum(bounds[:, 1], params),
            )
        ).tolist()
        return res

    def quick_bounds(self, *_):
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

    @graceful_error_handler(title="Error Calculation")
    def calculate_error(
        self,
        forms: tuple | list,
        error_method: str,
        scatter_df: pd.DataFrame | None = None,
        generated_df: pd.DataFrame | None = None,
    ) -> float:
        """
        Calculate the error between scatter data and generated data.

        Args:
            scatter_df (pd.DataFrame): DataFrame containing the scatter data.
            generated_df (pd.DataFrame): DataFrame containing the generated data.
            error_methods (str): A dictionary mapping error types to calculation methods.

        Returns:
            float: The calculated error value.
        """
        try:
            if scatter_df is None:
                # Parse data
                if self.data.primary() in self.data.raw:
                    scatter_df = self.data.get_df(
                        self.data.primary(),
                        "freq",
                        *forms,
                        area=self.options["simulation"]["area"],
                        thickness=self.options["simulation"]["thickness"],
                    )
                else:
                    self.error_printout.setText("N/A")
                    return np.inf

            if generated_df is None:
                generated_df = self.generator.get(
                    [entry.values[0] for entry in self.parameters],
                    freq=scatter_df["freq"].to_numpy(copy=True),
                    interp=False,
                ).get_df("freq", *forms)

            if len(scatter_df) != len(generated_df):
                raise ValueError("Data and Fit must have the same length.")

            # Prepare data for error calculation
            data = [scatter_df[form] for form in forms] + [generated_df[form] for form in forms]

            # Take absolute values if "log" is in the error type
            if "log" in error_method.lower():
                data = [abs(d) for d in data]

            # Calculate the error using the specified method
            error_all = Statistics()[error_method](np.hstack(data[:2]), np.hstack(data[2:]))
        except Exception as exc:
            message = construct_error_message(exc, "while calculating the error")
            raise GraphGUIError(message) from exc

        return error_all

    @graceful_error_handler(title="Error Update")
    def update_error(self, scatter_df=None, generated_df=None):
        """Update the graphs based on the selected option."""
        try:
            if self.sel_error_var.currentText() == "Nyquist":
                forms = self.cont.ny.get_text()
            elif self.sel_error_var.currentText() == "Bode":
                forms = self.cont.bd.get_text()[1:]
            elif self.sel_error_var.currentText() == "Both":
                forms = self.cont.ny.get_text() + self.cont.bd.get_text()[1:]

            elif len(self.sel_error_var.currentText()) == 1:
                forms = [
                    f"{self.sel_error_var.currentText()}.real",
                    f"{self.sel_error_var.currentText()}.imag",
                ]
            elif (
                self.sel_error_var.currentText() == "User"
                and self.options["general"]["error_forms"]
            ):
                forms = self.options["general"]["error_forms"]
            else:
                forms = ["Z.real", "Z.imag"]

            if scatter_df is None:
                # Parse data
                if self.data.primary() in self.data.raw:
                    scatter_df = self.data.get_df(
                        self.data.primary(),
                        "freq",
                        # bode_y1,
                        # bode_y2,
                        *forms,
                        area=self.options["simulation"]["area"],
                        thickness=self.options["simulation"]["thickness"],
                    )
                else:
                    self.error_printout.setText("N/A")
                    return

            if generated_df is None:
                generated_df = self.generator.get(
                    [entry.values[0] for entry in self.parameters],
                    freq=scatter_df["freq"].to_numpy(copy=True),
                    interp=False,
                ).get_df("freq", *forms)

            if not isinstance(scatter_df, pd.DataFrame) or not isinstance(
                generated_df, pd.DataFrame
            ):
                self.error_printout.setText("N/A")
                return

            if self.options["general"]["limit_error"]:
                f_min = 10 ** self.options["simulation"]["freq_start"]
                f_max = 10 ** self.options["simulation"]["freq_stop"]

                scatter_df = scatter_df[
                    (scatter_df["freq"] >= f_min) & (scatter_df["freq"] <= f_max)
                ]
                generated_df = generated_df[
                    (generated_df["freq"] >= f_min) & (generated_df["freq"] <= f_max)
                ]

                if scatter_df.empty:
                    logger.warning(
                        "Attempted error update: The specified frequency range resulted in an empty dataset."
                    )
                    QMessageBox.warning(
                        self,
                        "Warning",
                        "No data points within the specified frequency range.",
                    )
                    return

                if len(scatter_df) != len(generated_df):
                    generated_df = self.generator.get(
                        [entry.values[0] for entry in self.parameters],
                        freq=scatter_df["freq"].to_numpy(copy=True),
                    ).get_df("freq", *forms)

            if len(scatter_df) != len(generated_df):
                self.error_printout.setText("N/A")
                return
        except Exception as exc:
            self.error_printout.setText(f"Error: {str(exc)}")
            message = construct_error_message(exc, "while calculating the error")
            raise GraphGUIError(message) from exc

        try:
            error_method = self.settings.error_methods[self.error_var.currentText()]
            error_all = self.calculate_error(
                forms,
                error_method,
                scatter_df,
                generated_df,
            )

            if abs(np.log10(abs(error_all))) > 2:
                self.error_printout.setText(f"{error_all:.4e}")
            else:
                self.error_printout.setText(f"{error_all:.4f}")
        except ValueError as exc:
            self.error_printout.setText(f"Error: {str(exc)}")

    @graceful_error_handler(popup=True, title="Graph Update")
    def update_graphs(self, *_, clear_axis=True):
        """Update the graphs based on the selected option."""
        try:
            ny_lin_type = "LinScaler"
            if (
                self.options["plotting"]["plot1 as nyquist"]
                and self.nyquist_plot.x_scale.__class__.__name__ != "LinFrom0Scaler"
            ):
                self.nyquist_plot.scales = "LinFrom0Scaler"
                ny_lin_type = "LinFrom0Scaler"
            elif (
                not self.options["plotting"]["plot1 as nyquist"]
                and self.nyquist_plot.x_scale.__class__.__name__ == "LinFrom0Scaler"
            ):
                self.nyquist_plot.scales = "LinScaler"
                ny_lin_type = "LinScaler"

            if self.style_history["style"] != self.options["plotting"]["style"]:
                set_style(self.options["plotting"]["style"])
                self.style_history["style"] = self.options["plotting"]["style"]

            if (
                self.options["plotting"]["background"]
                and self.style_history["background"] != self.options["plotting"]["background"]
            ):

                # Set a darker background color for the Nyquist plot
                self.ax1.set_facecolor(self.options["plotting"]["background"])

                # Set a darker background color for the Bode plots
                self.ax2.set_facecolor(self.options["plotting"]["background"])
                self.ax3.set_facecolor(self.options["plotting"]["background"])

                self.style_history["background"] = self.options["plotting"]["background"]

        except Exception as exc:
            message = construct_error_message(exc, "while modifying the scales")
            raise GraphGUIError(message) from exc

        try:
            self.plotted_data = ["All"]

            params_values = [entry.values[0] for entry in self.parameters]

            sim_freq = None
            if (
                self.data.primary() in self.data.raw
                and not self.options["simulation"]["sim_param_freq"]
            ):
                sim_freq = self.data.raw[self.data.primary()]["freq"]

            generated_df = self.generator.get(params_values, freq=sim_freq)

            generated_sub_dfs = None
            if self.sub_model_check.isChecked():
                generated_sub_dfs = self.generator.get_sub_groups(
                    params_values,
                    freq=sim_freq,
                    shift=self.options["plotting"]["suplots: shift by real"],
                    re_order_shift=self.options["plotting"]["suplots: shift using max"],
                    use_numbers=self.options["plotting"]["suplots: use numbers"],
                    sub_groups=self.model_entry.sub_models,
                )
        except Exception as exc:
            message = construct_error_message(exc, "while generating/getting data")
            raise GraphGUIError(message) from exc

        try:
            ny_x, ny_y = self.cont.ny.get_text()
            bode_x, bode_y1, bode_y2 = self.cont.bd.get_text()

            raws = self.cont.ny.get_raw_text() + self.cont.bd.get_raw_text()
            self.settings.save_settings(
                plot_var={s: r for s, r in zip(("ny_x", "ny_y", "bd_x", "bd_y1", "bd_y2"), raws)}
            )
            is_logs = self.cont.ny.is_log() + self.cont.bd.is_log()
            self.settings.save_settings(
                is_log={
                    s: il for s, il in zip(("ny_x", "ny_y", "bd_x", "bd_y1", "bd_y2"), is_logs)
                }
            )

            cmaps = StylizedPlot.get_cmaps(
                [self.ax1, self.ax2, self.ax3],
                ("diverging_named_cmap", "diverging_clr_cmap"),
                True,
            )
            markers = StylizedPlot.marker_list()

            # Parse data
            checked_names = []
            if self.data.primary() in self.data.raw:
                main_dataset = self.data.primary()
                self.data.set_var(main_dataset)

                checked_names = self.data.get_checked()

                if main_dataset in checked_names:
                    checked_names.remove(main_dataset)
                    checked_names.insert(0, main_dataset)

            else:
                main_dataset = "Model"

            if self.model_check.isChecked() or not self.data.raw:
                self.plotted_data.append("Model")

            # Clear the axes in preparation
            if clear_axis:
                self.nyquist_plot.clear()
                self.bode_plot.clear()

            self.nyquist_plot.title = str(main_dataset)

            self.nyquist_plot.xlabel = self.settings.parse_label(
                self.cont.ny.ax["x"].axisRawText()
            )
            self.nyquist_plot.ylabels = self.settings.parse_label(
                self.cont.ny.ax["y"].axisRawText()
            )

            self.nyquist_plot.x_scale = "log" if self.cont.ny.ax["x"].is_log() else ny_lin_type
            self.nyquist_plot.y_scales = ["log" if self.cont.ny.ax["y"].is_log() else ny_lin_type]

            bode_labels = self.settings.parse_label(self.cont.bd.ax["x"].axisRawText())
            bode_label1 = self.settings.parse_label(self.cont.bd.ax["y1"].axisRawText())
            bode_label2 = self.settings.parse_label(self.cont.bd.ax["y2"].axisRawText())

            space = " "

            self.bode_plot.title = (
                f"{bode_label1.split(space)[0]} & {bode_label2.split(space)[0]} of {main_dataset}"
            )
            self.bode_plot.xlabel = bode_labels
            self.bode_plot.x_scale = "log" if self.cont.bd.ax["x"].is_log() else "linear"

            self.bode_plot.ylabels = [bode_label1, bode_label2]

            self.bode_plot.y_scales = [
                "log" if self.cont.bd.ax["y1"].is_log() else "linear",
                "log" if self.cont.bd.ax["y2"].is_log() else "linear",
            ]
        except Exception as exc:
            message = construct_error_message(exc, "while preparing labels/formatting")
            raise GraphGUIError(message) from exc

        key = ""
        try:
            self.line = []

            ### Begin Plots
            ## Begin Scatter plots
            if self.data.raw:
                self.update_error()
                n_cmap = 0
                line_kwargs_init = self.options["plotting"]["connect points"].copy()
                show_line = (
                    line_kwargs_init.pop("show", True)
                    if isinstance(line_kwargs_init, dict)
                    else False
                )
                for key in checked_names:
                    system_data = self.data.raw[key]
                    system_data.savgol_kwargs = self.options["simulation"]["savgol_kwargs"]
                    system_data.interp_kwargs = self.options["simulation"]["interp_kwargs"]
                    system_data.norm_kwargs = self.options["simulation"]["norm_kwargs"]

                    kwargs = {
                        "label": self.data.get_label(key, self.legend_check.isChecked()),
                        # "color": "freq",
                        "color": bode_x,
                        "marker": markers[n_cmap],
                    }

                    if isinstance(self.options["plotting"]["Data options"], dict):
                        kwargs.update(self.options["plotting"]["Data options"])

                    kwargs.update(
                        StylizedPlot.DecadeCmapNorm(system_data[bode_x], cmap=cmaps[0][n_cmap])
                    )

                    # Plot scatter
                    self.nyquist_plot.plot(
                        "scatter",
                        # system_data.get_df(ny_x, ny_y, "freq"),
                        system_data.get_df(ny_x, ny_y, bode_x),
                        name=key,
                        add_to_annotations=self.data.get_mark(key),
                        **kwargs,
                    )

                    kwargs["color"] = bode_x
                    kwargs.update(
                        StylizedPlot.DecadeCmapNorm(system_data[bode_x], cmap=cmaps[1][n_cmap])
                    )

                    self.bode_plot.plot(
                        "scatter",
                        system_data.get_df(bode_x, bode_y1, bode_y2),
                        name=key,
                        **kwargs,
                    )

                    for i, scatter in enumerate(StylizedPlot.get_scatter(self.ax3)):
                        data = scatter.get_array()
                        if data is None:
                            data = scatter.get_offsets().data  # type: ignore
                        cmap_norm = StylizedPlot.DecadeCmapNorm(data, cmaps[2][i])
                        scatter.set_cmap(cmap_norm["cmap"])
                        scatter.set_norm(cmap_norm["norm"])

                    if show_line:
                        line_kwargs = line_kwargs_init.copy()

                        self.nyquist_plot.plot(
                            "line",
                            # system_data.get_df(ny_x, ny_y, "freq"),
                            system_data.get_df(ny_x, ny_y, bode_x),
                            name=f"_{key}_conn",
                            label="_" + line_kwargs.pop("label", key),
                            exclude_from_all=True,
                            **line_kwargs,
                        )
                        self.bode_plot.plot(
                            "line",
                            system_data.get_df(bode_x, bode_y1, bode_y2),
                            name=f"_{key}_conn",
                            label="_" + line_kwargs.pop("label", key),
                            exclude_from_all=True,
                            **line_kwargs,
                        )

                    self.plotted_data.append(key)
                    n_cmap += 1
                    if n_cmap >= min(len(cmap) for cmap in cmaps):
                        n_cmap = 0
        except Exception as exc:
            name = " " + key if key else "s"
            message = construct_error_message(exc, "while plotting the scatter plot", name, "")
            raise GraphGUIError(message) from exc

        name = "Model"
        try:

            # model_names = []
            ### Plot Model and Sub models
            if self.model_check.isChecked() or not self.data.raw:
                ## Plot Model
                # Generate a list of colors for all datasets
                fit_kwargs = {}
                if isinstance(self.options["plotting"]["Model options"], dict):
                    fit_kwargs.update(self.options["plotting"]["Model options"].copy())

                kwargs = {
                    "label": "_model",
                    "color": fit_kwargs.pop("color", "r"),
                    "annotator": "decade",
                    "a_kwargs": {"color": fit_kwargs.pop("color", "r")},
                }

                kwargs.update(fit_kwargs)

                # Plot simulation
                self.nyquist_plot.plot(
                    "line",
                    generated_df.get_df(ny_x, ny_y, bode_x),
                    name="Model",
                    add_to_annotations=True,
                    **kwargs,
                )

                kwargs.pop("annotator", None)
                kwargs.pop("a_kwargs", None)

                self.bode_plot.plot(
                    "line",
                    generated_df.get_df(bode_x, bode_y1, bode_y2),
                    name="Model",
                    **kwargs,
                )

                self.line.append(self.ax1.lines[-1])
                self.line.append(self.ax2.lines[-1])
                self.line.append(self.ax3.lines[-1])

                fit_kwargs.pop("label", None)
                fit_kwargs.pop("color", None)
                if isinstance(self.options["plotting"]["Sub Model options"], dict):
                    fit_kwargs.update(self.options["plotting"]["Sub Model options"].copy())
                cmap = fit_kwargs.pop("cmap", "autumn")
                # Plot Sub models
                if generated_sub_dfs is not None:
                    colors = StylizedPlot.DynamicColor(
                        total=len(generated_sub_dfs), cmap=cmap, return_list=True
                    )

                    for i, line_data in enumerate(generated_sub_dfs):
                        name = line_data.attrs.get("model", f"Sub_model_{i}")
                        label = name if self.legend_check.isChecked() else f"_sub_model_{i}"
                        kwargs = {
                            "label": label,
                            "color": colors[i],
                            "annotator": "topn",
                            "a_kwargs": {"color": colors[i]},
                        }

                        kwargs.update(fit_kwargs)

                        add_anno = kwargs.pop("annotate_all", False)

                        # Plot simulation
                        self.nyquist_plot.plot(
                            "line",
                            line_data.get_df(ny_x, ny_y, bode_x),
                            name=name,
                            add_to_annotations=add_anno,
                            **kwargs,
                        )

                        kwargs.pop("annotator", None)
                        kwargs.pop("a_kwargs", None)

                        self.bode_plot.plot(
                            "line",
                            line_data.get_df(bode_x, bode_y1, bode_y2),
                            name=name,
                            **kwargs,
                        )

                        self.line.append(self.ax1.lines[-1])
                        self.line.append(self.ax2.lines[-1])
                        self.line.append(self.ax3.lines[-1])
                        # model_names.append(f"Sub_model_{i}")
        except Exception as exc:
            name = " " + name if name else "s"
            message = construct_error_message(exc, "while plotting the model plot", name, "")
            raise GraphGUIError(message) from exc

        self._block_update = False

        try:
            ## Begin Band plots
            if self.cont.ny_band_check.isChecked() or self.cont.bd_band_check.isChecked():
                if isinstance(self.options["bands"]["std_devs"], (int, float)):
                    std_devs = [
                        self.options["bands"]["std_devs"] * param for param in params_values
                    ]
                else:
                    std_devs = [v[0] for v in self.parameters_std.values]

                ci_inputs = {
                    "percentile": self.options["bands"]["percentile"],
                    "params_values": params_values,
                    "std_devs": std_devs,
                    "model": self.settings.model,
                    "freq": generated_df["freq"],
                    "num_freq_points": self.options["bands"]["band_freq_num"],
                    "target_form": ["freq", bode_y1, bode_y2],
                    "thickness": self.options["simulation"]["thickness"],
                    "area": self.options["simulation"]["area"],
                }

                if not self.ci_inputs:
                    self.ci_inputs = ci_inputs
                    remake_ci = True
                else:
                    # Check if any of the inputs have changed
                    remake_ci = False
                    for key, value in ci_inputs.items():
                        if self.ci_inputs[key] != value:
                            self.ci_inputs = ci_inputs
                            remake_ci = True
                            break

                ci_dfs_dict = self.ci_inputs.get("ci_dfs_dict")
                if remake_ci or ci_dfs_dict is None:
                    # Generate confidence interval data self.parameters_std
                    ci_analysis = ImpedanceConfidence(
                        self.options["bands"]["percentile"],  # 5 converts to 97.5 and 2.5
                        params_values,  # popt
                        std=std_devs,  # standard
                    )

                    # is actually a dict of dataframes
                    ci_dfs_dict = ci_analysis.gen_conf_band(
                        generated_df[bode_x],
                        # func=wrapCircuit(self.settings.model, {}),
                        func=ImpedanceFunc(self.settings.model, {}),
                        num_x_points=self.options["bands"]["band_freq_num"],
                        main_col="real",
                        target_form=[bode_x, bode_y1, bode_y2],
                        thickness=self.options["simulation"]["thickness"],
                        area=self.options["simulation"]["area"],
                    )
                    self.ci_inputs["ci_dfs_dict"] = ci_dfs_dict

                perc_cols = [col for col in list(ci_dfs_dict.values())[0].columns if "%" in col]
                min_col = perc_cols[0] if perc_cols else "min"
                max_col = perc_cols[-1] if perc_cols else "max"

                if self.cont.ny_band_check.isChecked():
                    self.nyquist_plot.plot(
                        "band",
                        ci_dfs_dict["nyquist"]["real", min_col, max_col],
                        label="_nyquist_band",  # keys
                        color=self.options["bands"]["band_color"],
                        alpha=self.options["bands"]["band_alpha"],
                    )
                if self.cont.bd_band_check.isChecked():
                    ci_df = [
                        ci_dfs_dict[bode_y1][bode_x, min_col, max_col],
                        ci_dfs_dict[bode_y2][bode_x, min_col, max_col],
                    ]
                    self.bode_plot.plot(
                        "band",
                        ci_df,
                        label="_bode_band",  # keys,  # keys
                        color=self.options["bands"]["band_color"],
                        alpha=self.options["bands"]["band_alpha"],
                    )
        except Exception as exc:
            message = construct_error_message(exc, "while plotting the band plot")
            raise GraphGUIError(message) from exc

        try:
            name = ""
            ## Begin pinned line plots
            for _, row in self.pinned.df.iterrows():
                if row["Show"] != "" and row["Model"]:
                    name = row["Name"]
                    model = row["Model"]

                    # Extract parameter values and stds
                    params_values = [
                        row[f"{p}_values"] for p in self.generator.parse_parameters(model)
                    ]

                    pinned_df = self.generator.get(params_values, model=model)

                    self.plotted_data.append(name)
                    # Plot pinned results using StylizedPlot objects
                    self.nyquist_plot.plot(
                        "line",
                        pinned_df.get_df(ny_x, ny_y, bode_x),
                        fmt=row["Show"],
                        label=name,
                    )
                    self.bode_plot.plot(
                        "line",
                        pinned_df.get_df(bode_x, bode_y1, bode_y2),
                        fmt=row["Show"],
                        label=name,
                    )

        except Exception as exc:
            name = " " + name if name else "s"
            message = construct_error_message(exc, "while plotting the pinned plot", name, "")
            raise GraphGUIError(message) from exc

        try:
            if any(self.pinned.df["Show"] != "") or self.legend_check.isChecked():
                self.ax1.legend()
                self.ax2.legend()

            self.ax1 = self.nyquist_plot.ax[0]
            self.ax2, self.ax3 = self.bode_plot.ax

            self.cursor_var.blockSignals(True)
            old_cursor = self.cursor_var.currentText()
            self.cursor_var.clear()
            self.cursor_var.addItems(self.plotted_data)

            if old_cursor in self.plotted_data:
                self.cursor_var.setCurrentText(old_cursor)
            elif main_dataset in self.plotted_data:
                self.cursor_var.setCurrentText(str(main_dataset))
            else:
                self.cursor_var.setCurrentText("All")
            self.cursor_var.blockSignals(False)

            self.update_format()

        except Exception as exc:
            message = construct_error_message(exc, "while concluding the update")
            raise GraphGUIError(message) from exc

    @graceful_error_handler(popup=True, title="Line Update")
    def update_line_data(self, *_, **__):
        """Update the line data in real-time."""
        if self._block_update:
            return
        try:
            if len(self.parameters.names) != len(
                self.generator.parse_parameters(self.settings.model)
            ):
                return

            param_vals = [entry.values[0] for entry in self.parameters]

            if not all(isinstance(val, (int, float)) for val in param_vals):
                return

            sim_freq = None
            if (
                self.data.primary() in self.data.raw
                and not self.options["simulation"]["sim_param_freq"]
            ):
                sim_freq = self.data.raw[self.data.primary()]["freq"]

            generated_df = self.generator.get(param_vals, freq=sim_freq)
        except Exception as exc:
            message = construct_error_message(exc, "while generating the updated model")
            raise GraphGUIError(message) from exc
        try:

            # Generate sub datasets
            generated_sub_dfs = None
            if self.sub_model_check.isChecked():
                generated_sub_dfs = self.generator.get_sub_groups(
                    param_vals,
                    freq=sim_freq,
                    shift=self.options["plotting"]["suplots: shift by real"],
                    re_order_shift=self.options["plotting"]["suplots: shift using max"],
                    use_numbers=self.options["plotting"]["suplots: use numbers"],
                    sub_groups=self.model_entry.sub_models,
                )
        except Exception as exc:
            message = construct_error_message(exc, "while generating the updated sub-models")
            raise GraphGUIError(message) from exc
        try:
            # Update history and intervals
            self.parameters.update_history()
            self.parameters_std.update_history()

            ny_x, ny_y = self.cont.ny.get_text()

            bode_x, bode_y1, bode_y2 = self.cont.bd.get_text()

            # Update the Nyquist plot using the new `update_data` method
            self.nyquist_plot.update_data("Model", generated_df.get_df(ny_x, ny_y, bode_x))

            # Update the Bode plot using the new `update_data` method
            self.bode_plot.update_data("Model", generated_df.get_df(bode_x, bode_y1, bode_y2))

            keys = [[ny_x, ny_y], [bode_x, bode_y1], [bode_x, bode_y2]]

            # Update main dataset lines
            if self.line:
                if self.data.raw:
                    self.update_error()

                for line, key in zip(
                    self.line[:3], keys
                ):  # First 3 lines are for the main dataset
                    line.set_xdata(generated_df[key[0]])
                    line.set_ydata(generated_df[key[1]])
        except Exception as exc:
            message = construct_error_message(exc, "while plotting the updated model")
            raise GraphGUIError(message) from exc
        try:
            # Update sub dataset lines
            if generated_sub_dfs is not None and len(self.line) > 3:
                for i, sub_df in enumerate(generated_sub_dfs):
                    self.nyquist_plot.update_data(
                        sub_df.attrs.get("model", f"Sub_model_{i}"),
                        # sub_df.get_df(ny_x, ny_y, "freq"),
                        sub_df.get_df(ny_x, ny_y, bode_x),
                    )
                    self.bode_plot.update_data(
                        sub_df.attrs.get("model", f"Sub_model_{i}"),
                        sub_df.get_df(bode_x, bode_y1, bode_y2),
                    )
                    for j, key in enumerate(keys):
                        line_index = (
                            3 + i * 3 + j
                        )  # Offset by main dataset lines and previous sub datasets
                        if line_index < len(self.line):
                            self.line[line_index].set_xdata(sub_df[key[0]])
                            self.line[line_index].set_ydata(sub_df[key[1]])
        except Exception as exc:
            message = construct_error_message(exc, "while plotting the updated sub-models")
            raise GraphGUIError(message) from exc

        self.update_format()

    @graceful_error_handler(popup=True, title="Format Update")
    def update_format(self, *_, **__):
        """Update the format of the plots and scale them based on the selected label."""
        try:
            self.param_stats.refresh()
            # Get the selected dataset key from cursor_var
            selected_data_key = self.cursor_var.currentText()
            if selected_data_key not in self.nyquist_plot.directory:
                logger.warning("Selected data '%s' not found in plotted data.", selected_data_key)
                return

            # Use the label associated with the selected dataset
            # label = "_model" if selected_data_key == "Model" else self.data.get_label(selected_data_key, self.legend_check.isChecked())

            # Update Nyquist plot
            if self.nyquist_plot is not None:
                self.nyquist_plot.formatting()

                self.nyquist_plot.scale(
                    selected_data_key,
                    pad=self.options["plotting"]["window padding"],
                    quantile=self.options["plotting"]["outlier percent"],
                    # use_prior=self.lock_scale_check.isChecked(),
                    allow_invert=self.options["plotting"]["allow inv log"],
                    invert_threshold=self.options["plotting"]["inversion threshold"],
                )

                if self.options["plotting"]["plot1 as nyquist"]:
                    self.nyquist_plot.square()
                self.ax1.set_xlim(self.cont.ny.ax["x"].axisLims(self.ax1.get_xlim()))
                self.ax1.set_ylim(self.cont.ny.ax["y"].axisLims(self.ax1.get_ylim()))
                self.nyquist_plot.update_annotation()

            # Update Bode plot
            if self.bode_plot is not None:
                self.bode_plot.formatting()
                self.bode_plot.scale(
                    selected_data_key,
                    pad=self.options["plotting"]["window padding"],
                    quantile=self.options["plotting"]["outlier percent"],
                    # use_prior=self.lock_scale_check.isChecked(),
                    allow_invert=self.options["plotting"]["allow inv log"],
                    invert_threshold=self.options["plotting"]["inversion threshold"],
                )
                self.ax2.set_xlim(self.cont.bd.ax["x"].axisLims(self.ax2.get_xlim()))
                self.ax2.set_ylim(self.cont.bd.ax["y1"].axisLims(self.ax2.get_ylim()))
                self.ax3.set_ylim(self.cont.bd.ax["y2"].axisLims(self.ax3.get_ylim()))

            # Redraw the canvases
            # self.canvas1.draw()
            # self.canvas2.draw()
            self.canvas1.draw_idle()
            self.canvas2.draw_idle()

            # self.fig1.tight_layout()
            # self.fig2.tight_layout()set_layout_engine
            self.fig1.set_layout_engine("constrained")
            self.fig2.set_layout_engine("constrained")

        except Exception as exc:
            message = construct_error_message(exc, "while reformatting the sub_dfs")
            raise GraphGUIError(message) from exc

    def _get_point_info(self, event, threshold=0.05):
        """
        Extract point information from Bode plot.

        Args:
            event: MouseEvent from matplotlib

        Returns:
            tuple: (nearest_index, x, y, None) if point found, (None, None, None, None) otherwise
        """
        x_in, y_in = event.xdata, event.ydata
        res = {
            "index": np.nan,
            "x": x_in if x_in is not None else np.nan,
            "y": y_in if y_in is not None else np.nan,
            "z": np.nan,
        }
        if (
            x_in is None
            or y_in is None
            or self.cursor_var.currentText() not in self.bode_plot.directory
        ):
            return res

        ax = event.inaxes

        # Convert to log scale if needed for distance calculation
        x_in = res["x"] = np.log10(x_in) if ax.get_xscale() == "log" else x_in
        y_in = res["y"] = np.log10(y_in) if ax.get_yscale() == "log" else y_in

        # Determine which subplot we're in
        if ax == self.ax1:
            x_col, y_col = self.cont.ny.get_text()
            z_col = self.cont.bd.ax["x"].axisText()
            df = self.nyquist_plot.directory[self.cursor_var.currentText()]["data"][0]
        elif ax == self.ax2:
            x_col, y_col, z_col = self.cont.bd.get_text()
            df = self.bode_plot.directory[self.cursor_var.currentText()]["data"][0]
        elif ax == self.ax3:
            x_col, z_col, y_col = self.cont.bd.get_text()
            df = self.bode_plot.directory[self.cursor_var.currentText()]["data"][1]
        else:
            return res

        res = {
            "index": np.nan,
            x_col: x_in,
            y_col: y_in,
            z_col: np.nan,
        }

        # Handle log scale conversions for data points
        x_arr = np.log10(abs(df[x_col])) if ax.get_xscale() == "log" else df[x_col]
        y_arr = np.log10(abs(df[y_col])) if ax.get_yscale() == "log" else df[y_col]

        # Get axis limits for normalization
        xlim = (
            np.diff(np.log10(ax.get_xlim()))[0]
            if ax.get_xscale() == "log"
            else np.diff(ax.get_xlim())[0]
        )
        ylim = (
            np.diff(np.log10(ax.get_ylim()))[0]
            if ax.get_yscale() == "log"
            else np.diff(ax.get_ylim())[0]
        )

        # Calculate normalized distances
        distances = np.sqrt((x_arr / xlim - x_in / xlim) ** 2 + (y_arr / ylim - y_in / ylim) ** 2)
        res["index"] = np.argmin(distances)

        # Check if point is within threshold
        if not threshold:
            res[x_col] = df[x_col][res["index"]]
            res[y_col] = df[y_col][res["index"]]
            res[z_col] = df[z_col][res["index"]]

        elif distances[res["index"]] < threshold:
            res[x_col] = df[x_col][res["index"]]
            res[y_col] = df[y_col][res["index"]]
            res[z_col] = df[z_col][res["index"]]

        return res

    @graceful_error_handler(popup=True, title="Save Point")
    def save_current_point(self, event):
        """Add a data point at the clicked position."""
        try:
            # Get current dataset
            data_name = self.data.primary()
            if data_name not in self.data.raw:
                QMessageBox.warning(self, "Warning", "No nearby dataset points.")
                return

            point_info = self._get_point_info(event)

            info_str = "\n".join(f"{k}: {v}" for k, v in point_info.items())

            # Ask for a point name
            point_name, ok = QInputDialog.getText(
                self, "Point Name", f"Enter a name for this point:\n{info_str}", text="point"
            )

            if ok:
                # Store point with custom name in the dataset attributes
                self.data.raw[data_name].attrs |= {
                    f"{point_name}_{k}": v for k, v in point_info.items()
                }

        except Exception as exc:
            message = construct_error_message(exc, "while adding a point to the dataset")
            raise GraphGUIError(message) from exc

    @graceful_error_handler(title="Nyquist Cursor Position")
    def nyquist_cursor_position(self, event):
        """Update the cursor position display with x, y, and z values."""
        try:
            point_info = self._get_point_info(event)
            if not any(np.isnan(val) for val in point_info.values()):
                nearest_index, x, y, z = point_info.values()
                self.cursor_printout.setText(
                    f"Point {int(nearest_index)} at:\n({x:.3e}, {y:.3e}, {z:.3e})"
                )

            else:
                self.cursor_printout.setText("Point:\nN/A")
        except Exception as exc:
            self.cursor_printout.setText(str(exc.__class__.__name__))
            message = construct_error_message(exc, "while updating cursor position")
            raise GraphGUIError(message) from exc

    @graceful_error_handler(title="Bode Cursor Position")
    def bode_cursor_position(self, event):
        """Update the cursor position display with x, y, and z values."""
        try:
            point_info = self._get_point_info(event)
            if not any(np.isnan(val) for val in point_info.values()):
                nearest_index, x, y, z = point_info.values()
                self.cursor_printout.setText(f"Point {int(nearest_index)} at:\n({x:.3e}, {y:.3e})")
            else:
                self.cursor_printout.setText("Point:\nN/A")
        except Exception as exc:
            self.cursor_printout.setText(str(exc.__class__.__name__))
            message = construct_error_message(exc, "while updating cursor position")
            raise GraphGUIError(message) from exc

    def auto_fit_datasets(self):
        """Automatically fit datasets based on pinned results."""

        auto_fit_inputs, ok = FormDialog.getResult(
            self,
            title="Auto Fit Datasets",
            content=self.settings.afit_inputs.copy(),
        )
        if not ok:
            return

        fit_type = auto_fit_inputs["Fit Type"].lower()

        iterations = 1 if "norm" in fit_type else int(auto_fit_inputs["Iterations"])

        if auto_fit_inputs["Fit All Datasets"]:
            valid_rows = pd.DataFrame(
                [(n, n, self.settings.model) for n in self.data.raw],
                columns=["Name", "Dataset", "Model"],
            )
            auto_fit_inputs["Use Current Model"] = True
        else:
            # Step 2: Search for valid rows before the loop
            match_string = auto_fit_inputs["Match String"]
            valid_rows = pd.DataFrame(
                [
                    row
                    for _, row in self.pinned.df.iterrows()
                    if re.search(match_string, row["Name"]) and row["Model"].lower() != "linkk"
                ]
            )

        if valid_rows.empty:
            QMessageBox.information(self, "No Matches", "No datasets matched the criteria.")
            return

        suffix = auto_fit_inputs["New Name Suffix"]
        names = self.pinned.df[self.pinned.df["Model"].str.lower() != "linkk"]["Name"].tolist()
        n = 1
        while any(nm.endswith(f"{suffix}{n}") for nm in names):
            n += 1
        suffix += f"{n}"

        self.settings.save_settings(
            afit_inputs={k: {"default": v} for k, v in auto_fit_inputs.items()}
        )
        try:
            if glbl_str := auto_fit_inputs["Value Override"]:
                glbl_str = "{" + glbl_str + "}" if not glbl_str.startswith("{") else glbl_str
                global_vals = ContainerEvaluator().parse(glbl_str)
                global_vals = {} if not isinstance(global_vals, dict) else global_vals
                for key, value in global_vals.items():
                    if f"{key}_values" in valid_rows.columns:
                        valid_rows[f"{key}_std"] = (
                            value * valid_rows[f"{key}_std"] / valid_rows[f"{key}_values"]
                        )
                        valid_rows[f"{key}_values"] = value
        except Exception as exc:
            # Log and continue
            logger.warning("Error parsing Value Override: %s", exc)

        # Prepare auto_fit_params
        self.auto_fit_params = {
            "df": valid_rows,
            "fit_type": fit_type,
            "iterations": iterations,
            "suffix": suffix,
            "run_order": self.data.run_order(valid_rows["Dataset"]),
            "lock_bound": auto_fit_inputs["Respect Bound Locks"],
            "sequential_fit": auto_fit_inputs["Fit Parameters Separately"],
            "use_pin_model": not bool(auto_fit_inputs["Use Current Model"]),
            "use_pin_const": not bool(auto_fit_inputs["Use Locked Parameter Value"]),
            "temp_save_path": str(
                Path(self.options["general"]["backup path ('~' == home dir)"]).expanduser()
                / "temp_fit_results.csv"
            ),
        }
        # self.settings.save_settings(afit_inputs=auto_fit_inputs)

        # match_string = auto_fit_inputs["Match String (regex compatable)"]
        # suffix = auto_fit_inputs["New Name Suffix (appended to dataset)"]

        # fit_type = auto_fit_inputs["Fit Type (Normal, Iterative, or Bootstrap)"]
        # run_type = fit_type.lower().split()[0]
        # if run_type not in ["normal", "iterative", "bootstrap"]:
        #     run_type = "normal"

        # iterations = (
        #     1
        #     if "norm" in fit_type.lower()
        #     else int(float(auto_fit_inputs["Iterations (for Iterative or Bootstrap)"]))
        # )
        # self.auto_fit_params = {
        #     "df": valid_rows,
        #     "fit_type": run_type,
        #     "iterations": iterations,
        #     "suffix": suffix,
        #     "run_order": self.data.run_order(valid_rows["Dataset"]),
        #     "lock_bound": auto_fit_inputs["Prioritize bound over initial guess"],
        #     "sequential_fit": auto_fit_inputs["Fit parameters sequentially"],
        #     "use_pin_model": not bool(auto_fit_inputs["Use Current (Primary) Model"]),
        #     "temp_save_path": str(
        #         Path(self.options["general"]["backup path ('~' == home dir)"]).expanduser()
        #         / "temp_fit_results.csv"
        #     ),
        # }

        # self.run_fit(iterations=len(valid_rows) * (1 + iterations), run_type="auto")
        self.run_fit(iterations=len(valid_rows) * iterations, run_type="auto")

    def run_bootstrap(self):
        """Run the bootstrap fit based on the selected model."""
        # Create a popup to get the number of desired iterations
        if self.settings.model.lower() == "linkk":
            self.run_fit()
        else:
            iterations, ok = QInputDialog.getInt(
                self,
                "Bootstrap Fit",
                "Enter the number of iterations:",
                500,  # Default value
                1,  # Minimum value
                100000,  # Maximum value
                1,  # Step
            )

            if ok:
                self.run_fit(iterations=iterations, run_type="bootstrap")

    def run_iterative(self):
        """Run the bootstrap fit based on the selected model."""
        # Create a popup to get the number of desired iterations
        if self.settings.model.lower() == "linkk":
            self.run_fit()
        else:
            iterations, ok = QInputDialog.getInt(
                self,
                "Iterative Fit",
                "Enter the number of iterations:",
                10,  # Default value
                1,  # Minimum value
                100000,  # Maximum value
                1,  # Step
            )

            if ok:
                self.run_fit(iterations=iterations, run_type="iterative")

    # Method to handle fitting and status updates
    def run_fit(self, iterations=1, run_type="fit"):
        """Run the circuit fit based on the selected model."""
        if self.data.primary() not in self.data.raw:
            QMessageBox.critical(self, "Error", "No data to fit.")
            return

        if self.settings.model.lower() == "linkk":
            self._block_update = True
            param_values = self.generator.get_linkk(
                self.data.get(self.data.primary()),
                **self.options["linkk"],
            )[1]
            self.parameters.values = param_values

            self.update_graphs()
            # self.progress_dialog.close()
            # self.kill_operation = False
            self.cancel_event.clear()
            return

        if iterations <= 1:
            self.create_progress_dialog(
                self,
                title="Running Single Fit",
                label_text="Fitting data...",
                cancel="Cancel Operation",
                maximum=0,
            )
        else:
            self.create_progress_dialog(
                None,
                title="Running Multiple Fits",
                label_text="Fitting data...",
                cancel="Cancel Operation",
            )

        self.worker = FittingWorker(self, iterations, run_type)

        self.run_in_thread(
            self.on_fitting_finished,
            self.on_worker_error,
            self.update_progress,
            self.progress_dialog,
        )

    @graceful_error_handler(popup=True, title="Fitting Finished")
    def on_fitting_finished(self, fit_results):
        """Handle the completion of the fitting operation."""
        try:
            # print("in main")
            if fit_results is None or fit_results.empty:
                # if not self.suppress_window:
                try:
                    self.progress_dialog.close()
                    self.progress_dialog.deleteLater()
                except (RuntimeError, AttributeError) as exc:
                    show_error_message(
                        exc, title="Fitting", message="while closing progress dialog", popup=False
                    )

                # self.kill_operation = False
                self.cancel_event.clear()
                return

            if all(col in fit_results.columns for col in ["Name", "Dataset", "Model"]):
                self.pinned.append_df(fit_results)

                try:
                    self.progress_dialog.close()
                    self.progress_dialog.deleteLater()
                except (
                    KeyError,
                    ValueError,
                    TypeError,
                    AttributeError,
                    FloatingPointError,
                ) as exc:
                    show_error_message(
                        exc, title="Fitting", message="while closing progress dialog", popup=False
                    )
                    # print(f"Error closing progress dialog: {exc}")

                # self.kill_operation = False
                self.cancel_event.clear()
                self.backup_fit_results()
                return
            # print("past second close type")
            if "value" not in fit_results.columns or "std" not in fit_results.columns:
                return

            self.fit_results = fit_results
            # if not self.suppress_window:
            try:
                self.progress_dialog.close()
                self.progress_dialog.deleteLater()
            except RuntimeError as exc:
                show_error_message(
                    exc, title="Fitting", message="while closing progress dialog", popup=False
                )

            # self.kill_operation = False
            self.cancel_event.clear()
            self.apply_fit_results()
        except Exception as exc:
            message = construct_error_message(exc, "while concluding the fit")
            raise GraphGUIError(message) from exc

    def apply_fit_results(self):
        """Apply the fit results to the parameters."""
        if self.fit_results is None:
            QMessageBox.warning(
                self,
                "Warning",
                "No fit results to apply.",
            )
            return
        try:
            fit_vals = "<br>".join(
                [
                    "{}: {:.2e} ± {:.2e}".format(
                        name,
                        self.fit_results.loc[name, "value"],
                        self.fit_results.loc[name, "std"],
                    )
                    for name in self.fit_results.index
                ]
            )
            # if not self.suppress_window:
            reply = QMessageBox.question(
                self,
                "Update Parameters",
                f"<b>Do you want to update the parameters with the fit results?</b><br><b>Results:</b> ({self.fit_results.attrs.get('error', 'Error: N/A')})<br>"
                + fit_vals,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )

            if reply == QMessageBox.Yes:
                self._block_update = True

                df_attrs = self.fit_results.attrs
                if (
                    self.settings.model != df_attrs["model"]
                    or self.data.primary() != df_attrs["dataset_var"]
                ):
                    self.settings.model = df_attrs["model"]
                    self.model_entry.setText(self.settings.model)
                    self.data.set_var(df_attrs["dataset_var"])
                    self.update_param_section(True, False)

                self.options["simulation"]["thickness"] = df_attrs.get(
                    "thickness", self.options["simulation"]["thickness"]
                )
                self.options["simulation"]["area"] = df_attrs.get(
                    "area", self.options["simulation"]["area"]
                )

                for name in self.fit_results.index:
                    self.parameters[name] = [self.fit_results.loc[name, "value"]]
                    self.parameters_std[name] = [self.fit_results.loc[name, "std"]]

                self.update_graphs()

        except (KeyError, IndexError, ValueError, TypeError) as exc:
            self._block_update = False
            show_error_message(
                exc, title="Fitting", message="while applying fit results", popup=True, parent=self
            )
            # QMessageBox.critical(self, "Error", f"Error in fit wrap-up: {exc}.")


if __name__ == "__main__":
    shell = get_ipython()
    in_spyder = shell is not None and "SPYDER" in shell.__class__.__name__.upper()
    is_debug = hasattr(sys, "gettrace") and sys.gettrace() is not None
    if is_debug:
        # print("Running in debug mode.")
        if shell is not None:
            shell.run_line_magic("matplotlib", "inline")
            # shell.run_line_magic("gui", "qt")
        app = QApplication(sys.argv)
        window = GraphGUI(debug=is_debug)
        window.show()
        sys.exit(app.exec_())
    elif in_spyder:
        # print("Running in Spyder.")
        shell.run_line_magic("matplotlib", "inline")  # type: ignore
        app = QApplication(sys.argv)
        window = GraphGUI(debug=is_debug)
        window.show()
        try:
            from IPython.lib.guisupport import start_event_loop_qt4, is_event_loop_running_qt4

            if not is_event_loop_running_qt4():
                shell.run_line_magic("gui", "qt")  # type: ignore
            start_event_loop_qt4(app)
        except ImportError:
            app.exec_()
        # sys.exit(app.exec_())

    elif "--terminal" in sys.argv:
        # print("Running in terminal mode.")
        import subprocess

        subprocess.run(["python", __file__])
    else:
        # print("Running in normal mode.")
        app = QApplication(sys.argv)
        window = GraphGUI()
        window.show()
        sys.exit(app.exec_())
