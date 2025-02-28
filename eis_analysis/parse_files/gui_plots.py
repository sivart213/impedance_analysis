# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney
pound !usr/bin/env python3
General function file
"""
from collections import namedtuple, OrderedDict
import sys
from IPython import get_ipython
import numpy as np
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)

import sip
# from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QLabel,
    QCheckBox,
    QPushButton,
    QFrame,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QSizePolicy,
    # QDialog,
    QWidget,
    QMainWindow,
    QMessageBox,
)


from ..utils.plot_factory import GeneratePlot


from ..z_fit.gui_widgets import (
    AlignComboBox,
)


class GraphBase:
    """Base class for handling graph updates."""

    def __init__(
        self,
        central_widget=None,
        control_frame=None,
        control_frame_layout=None,
        plot_format="scatter",
        loaded_data=None,
    ):
        """Initialize GraphBase variables."""
        self.central_widget = central_widget
        self.control_frame = control_frame
        self.control_frame_layout = control_frame_layout
        self.plot_format = plot_format
        self.loaded_data = loaded_data
        self.set_base_variables()
    
    def set_base_variables(self):
        self.cmaps = None
        self.plot_styles = None
        self.n_cmap = 0
        self.plotted_data = {}
        self.reset_cols = True

        self.figures = []
        self.axes = []
        self.graph_frames = []
        self.canvases = []
        self.toolbars = []
        self.plot_objects = []
        self.combo_boxes = {}
        self.checkboxes = {}

    def create_plot_controls(self):
        """Create the plot controls layout for GraphBase."""
        self.graph_frame = QFrame(self.central_widget)
        self.graph_frame_layout = QHBoxLayout()
        self.graph_frame.setLayout(self.graph_frame_layout)

        cols_layout = QGridLayout()
        button_layout = QHBoxLayout()
        add_checks_layout = QHBoxLayout()

        # Create combo boxes for each axis
        self.combo_boxes["x_axis"] = AlignComboBox(self.control_frame)
        self.combo_boxes["x_axis"].setTextAlignment(Qt.AlignCenter)
        self.combo_boxes["x_axis"].addItem("Select Column")

        self.combo_boxes["y_axis"] = AlignComboBox(self.control_frame)
        self.combo_boxes["y_axis"].setTextAlignment(Qt.AlignCenter)
        self.combo_boxes["y_axis"].addItem("Select Column")

        self.combo_boxes["z_axis"] = AlignComboBox(self.control_frame)
        self.combo_boxes["z_axis"].setTextAlignment(Qt.AlignCenter)
        self.combo_boxes["z_axis"].addItem("Select Column")

        # Create checkboxes for log scale
        self.checkboxes["x_log"] = QCheckBox("X Log Scale", self.control_frame)
        self.checkboxes["y_log"] = QCheckBox("Y Log Scale", self.control_frame)
        self.checkboxes["z_log"] = QCheckBox("Z Log Scale", self.control_frame)
        self.checkboxes["anno"] = QCheckBox("Annotate", self.control_frame)
        self.checkboxes["square"] = QCheckBox("Square", self.control_frame)
        self.checkboxes["inv_x"] = QCheckBox("-X", self.control_frame)
        self.checkboxes["inv_y"] = QCheckBox("-Y", self.control_frame)

        # Create buttons
        update_graphs_button = QPushButton("Update Graphs", self.control_frame)
        update_graphs_button.setFixedWidth(80)
        update_graphs_button.clicked.connect(self.update_graphs)

        update_format_button = QPushButton("Update Format", self.control_frame)
        update_format_button.setFixedWidth(80)
        update_format_button.clicked.connect(self.update_format)

        clear_graphs_button = QPushButton("Clear Graphs", self.control_frame)
        clear_graphs_button.setFixedWidth(80)
        clear_graphs_button.clicked.connect(self.clear_graphs)

        # Create the plot controls layout
        self.control_frame_layout.addLayout(cols_layout)
        cols_layout.addWidget(QLabel("X Axis"), 0, 0)
        cols_layout.addWidget(self.combo_boxes["x_axis"], 0, 1)
        cols_layout.addWidget(self.checkboxes["x_log"], 0, 2)
        cols_layout.addWidget(QLabel("Y Axis"), 1, 0)
        cols_layout.addWidget(self.combo_boxes["y_axis"], 1, 1)
        cols_layout.addWidget(self.checkboxes["y_log"], 1, 2)
        cols_layout.addWidget(QLabel("Z Axis"), 2, 0)
        cols_layout.addWidget(self.combo_boxes["z_axis"], 2, 1)
        cols_layout.addWidget(self.checkboxes["z_log"], 2, 2)

        self.control_frame_layout.addLayout(add_checks_layout)
        add_checks_layout.addWidget(self.checkboxes["anno"])
        add_checks_layout.addWidget(self.checkboxes["square"])
        add_checks_layout.addWidget(self.checkboxes["inv_x"])
        add_checks_layout.addWidget(self.checkboxes["inv_y"])

        self.control_frame_layout.addLayout(button_layout)
        button_layout.addWidget(update_graphs_button)
        button_layout.addWidget(update_format_button)
        button_layout.addWidget(clear_graphs_button)

    def clear_graphs(self, clear_axis=True):
        """Clear the axes if clear_axis is True."""
        self.plotted_data = {}
        self.loaded_data = {}
        self.reset_cols = True
        # self.update_graphs()

        self.clear_axes(clear_axis)
        for canvas in self.canvases:
            canvas.draw()
    
    def set_combobox_variables(self, *items_list):
        """Set the items for the X and Y axis comboboxes."""
        if not isinstance(items_list[0], (list, tuple)):
            items_list = [str(item) for item in items_list] * len(self.combo_boxes)

        if (n_diff := (len(self.combo_boxes) - len(items_list))) > 0:
            # Collect all unique values from the lists in items_list
            unique_values = OrderedDict()
            for items in items_list:
                for item in items:
                    unique_values[item] = None
            unique_values = list(unique_values.keys())
            
            # Make up the difference by using unique values
            items_list = list(items_list) + [unique_values] * n_diff

        for combobox, items in zip(self.combo_boxes.values(), items_list):
            current_value = combobox.currentText()
            if self.reset_cols:
                current_items = OrderedDict({"Select Column": None})
            else:
                current_items = OrderedDict((combobox.itemText(i), None) for i in range(combobox.count()))
            for item in items:
                current_items[item] = None
            new_items = list(current_items.keys())
            
            combobox.clear()
            combobox.addItems(new_items)

            if current_value in new_items:
                index = combobox.findText(current_value)
                combobox.setCurrentIndex(index)
        if self.reset_cols:
            self.reset_cols = False

    
    def create_cursor_layout(self):
        freq_layout = QVBoxLayout()
        freq_sub_layout = QHBoxLayout()
        freq_layout.addLayout(freq_sub_layout)

        self.cursor_var = AlignComboBox(self.control_frame)
        self.cursor_var.setTextAlignment(Qt.AlignCenter)
        self.cursor_var.addItems(["Model"])
        self.cursor_var.setCurrentText("Model")

        cursor_label = QLabel("Point at cursor of ", self.control_frame)
        font = cursor_label.font()
        font.setPointSize(10)
        font.setBold(True)
        cursor_label.setFont(font)

        self.cursor_printout = QLabel("Points: N/A")

        self.control_frame_layout.addLayout(freq_layout)
        freq_sub_layout.addWidget(cursor_label)
        freq_sub_layout.addWidget(self.cursor_var)
        freq_layout.addWidget(self.cursor_printout)

    def create_graph_frames(self):
        """Create the graph frames for the plot."""
        # Generate and append components for the plot
        self.generate_graph_components(
            (10, 10),
            1,
            None,
            # 600,
        )

        self.toolbars[0].update()

        self.canvases[0].mpl_connect("motion_notify_event", self.cursor_position)

    def generate_graph_components(self, fig_size=(6,6), subplot=1, sharex_val=None, max_size=None):
        # Generate figures and axes
        if subplot == 1:
            fig, ax = GeneratePlot.subplots(figsize=fig_size)
            self.figures.append(fig)
            self.axes.append(ax)
        else:
            fig, ax = GeneratePlot.subplots(
                subplot, 1, figsize=fig_size, sharex=sharex_val
            )
            self.figures.append(fig)
            self.axes.append(ax)

        # Create frame
        frame = QFrame(self.graph_frame)
        frame_layout = QVBoxLayout()
        frame.setLayout(frame_layout)
        self.graph_frame.layout().addWidget(frame)
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        if max_size is not None:
            frame.setMaximumSize(max_size, 16777215)
        self.graph_frames.append(frame)

        # Create canvas
        canvas = FigureCanvasQTAgg(fig)
        frame_layout.addWidget(canvas)
        self.canvases.append(canvas)

        # Create toolbar
        toolbar = NavigationToolbar2QT(canvas, frame)
        frame_layout.addWidget(toolbar)
        self.toolbars.append(toolbar)

    def create_plot_objects(self, axes, title, scales, square=False, **kwargs):
        """Create the plot."""
        formats = ["scale", "format"]
        if square:
            formats.append("square")

        self.plot_objects.append(
            GeneratePlot(
                axes,
                title=title,
                scales=scales,
                init_formats=formats,
                **kwargs
            )
        )

    def create_plots(self):
        """Create the plot."""
        x_scale = "log" if self.checkboxes["x_log"].isChecked() else "lin"
        y_scale = "log" if self.checkboxes["y_log"].isChecked() else "lin"

        skwargs = {"allow_invert": True, "invert_threshold": 80}

        self.create_plot_objects(
            self.axes[0], "plot", [x_scale, y_scale], False, fkwargs=dict(power_lim=2), skwargs=skwargs
        )

    def plot_scatter_data(self, dataset, key, annotate=False):
        """Plot scatter data for a single system."""
        data_list = self.select_data(dataset)

        if not isinstance(data_list, (list, tuple)):
            data_list = [data_list]
    
        if len(data_list) == 0 or not isinstance(data_list[0], dict):
            return
        
        # Ensure the returned value is a list of length self.plot_objects
        while len(data_list) < len(self.plot_objects):
            data_list.extend(data_list)
        data_list = data_list[:len(self.plot_objects)]
            # data_list = data_list * int(np.ceil(len(self.plot_objects)/len(data_list)))
        
        for i, plot_obj in enumerate(self.plot_objects):
            data_dict = data_list[i]
            data = data_dict["data"]
            zcol = list(data.columns)[2] if data.shape[1] >= 3 else None
            styling = {
                # "cmap": self.cmaps[i][self.n_cmap],
                "styling": self.plot_styles["colors"][self.n_cmap],
                "marker": self.plot_styles["markers"][self.n_cmap],
                }

            plot_obj.xscale = data_dict["x_scale"]
            plot_obj.yscales = data_dict["y_scale"]
            zscale = data_dict.get("z_scale", "")

            if zcol is not None:
                styling["cmap"] = self.cmaps[i][self.n_cmap]
                styling["styling"] = zcol
                if "log" in zscale:
                    styling = {**styling,  **GeneratePlot.DecadeCmapNorm(
                        dataset[zcol], cmap=styling["cmap"]
                    )}

            plot_obj.plot(
                "scatter",
                data,
                label=key,
                # styling=cmap,
                # marker=self.plot_styles["markers"][self.n_cmap],
                **styling,
            )
            
            if annotate and i == 0:
                try:
                    plot_obj.annotate(data)
                except (ValueError, IndexError):
                    pass
        
        for i, scatter in enumerate(GeneratePlot.get_scatter(self.axes[-1])):
            try:
                cmap_norm = GeneratePlot.DecadeCmapNorm(
                    scatter.get_array(), self.cmaps[-1][i]
                )
                scatter.set_cmap(cmap_norm["cmap"])
                scatter.set_norm(cmap_norm["norm"])
            except AttributeError:
                pass

        self.n_cmap += 1
        if self.n_cmap >= min(len(cmap) for cmap in self.cmaps):
            self.n_cmap = 0
        self.plotted_data[key] = dataset

    def plot_line_data(self, dataset, name, style):
        """Plot pinned dataset if any."""
        data_list = self.select_data(dataset)

        for i, plot_obj in enumerate(self.plot_objects):
            data = data_list[i]
            plot_obj.plot(
                "line",
                data,
                styling=style,
                label=name,
            )
        self.plotted_data[name] = dataset

    def prepare_colormaps_and_markers(self):
        """Prepare colormaps and markers."""
        self.cmaps = []
        for ax in self.axes:
            if isinstance(ax, (list, tuple)):
                for a in ax:
                    self.cmaps.append(
                        [
                            scatter.get_cmap().name
                            for scatter in GeneratePlot.get_scatter(a)
                        ]
                    )
            else:
                self.cmaps.append(
                    [
                        scatter.get_cmap().name
                        for scatter in GeneratePlot.get_scatter(ax)
                    ]
                )

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

        for i, ax_cmaps in enumerate(self.cmaps):
            for cmap in base_cmaps:
                if cmap not in ax_cmaps:
                    ax_cmaps.append(cmap)
            self.cmaps[i] = ax_cmaps

        # List of colors
        colors = [
            "blue", "green", "red", "cyan", "magenta", "yellow", "black", "gold",
            "orange", "purple", "brown", "pink", "gray", "olive", "maroon", "navy",
            "teal", "lime", "indigo", "violet"
        ]

        # List of markers
        markers = [
            "o", "v", "^", "<", ">", "8", "s", "p", "*", ".", "h", "H", "D", "d", "P", "X",
            "1", "2", "3", "4", "|", "_", "+", "x"
        ]

        # Ensure markers and colors lists are longer than any of the self.cmaps lists
        max_cmap_length = max(len(cmap) for cmap in self.cmaps)
        while len(colors) <= max_cmap_length:
            colors.extend(colors)
        while len(markers) <= max_cmap_length:
            markers.extend(markers)

        self.n_cmap = 0

        # Combine markers and colors into a dictionary
        self.plot_styles = {
            "colors": colors,
            "markers": markers
        }
        return self.plot_styles

    def update_format(self):
        """Update the format of the plots."""
        for plot in self.plot_objects:
            if plot is not None:
                plot.formatting()
                plot.scale()
                # formats = ["scale", "format"]
                if self.checkboxes["square"].isChecked():
                    plot.square()
        for canvas in self.canvases:
            canvas.draw()

    def clear_axes(self, clear_axis=True):
        """Clear the axes if clear_axis is True."""
        if clear_axis:
            for plot in self.plot_objects:
                plot.clear()

    def select_data(self, dataset):
        """Select data from the dataset based on the current parameters."""
        x_col = self.combo_boxes["x_axis"].currentText()
        y_col = self.combo_boxes["y_axis"].currentText()
        z_col = self.combo_boxes["z_axis"].currentText()

        x_scale = "log" if self.checkboxes["x_log"].isChecked() else "lin"
        y_scale = "log" if self.checkboxes["y_log"].isChecked() else "lin"
        z_scale = "log" if self.checkboxes["z_log"].isChecked() else "lin"

        if x_col not in dataset.columns or y_col not in dataset.columns:
            return []
        
        if self.checkboxes["inv_x"].isChecked():
            dataset[x_col] = -dataset[x_col]
        if self.checkboxes["inv_y"].isChecked():
            dataset[y_col] = -dataset[y_col]
        
        if z_col in dataset.columns:
            return [{
                "data": dataset[[x_col, y_col, z_col]] if z_col != x_col else dataset[[x_col, y_col]],
                "x_scale": x_scale,
                "y_scale": y_scale,
                "z_scale": z_scale
            }]
        return [{
            "data": dataset[[x_col, y_col]],
            "x_scale": x_scale,
            "y_scale": y_scale
        }]

    def finalize_plots(self, show_legend):
        """Apply formats and draw canvas, update cursor variable."""
        if show_legend:
            for ax in self.axes:
                if isinstance(ax, (list, tuple)):
                    ax[0].legend()
                else:
                    ax.legend()

        for n, plot in enumerate(self.plot_objects):
            if not isinstance(self.axes[n], (list, tuple)):
                self.axes[n] = plot.ax[0]
            else:
                self.axes[n] = plot.ax
            plot.apply_formats()

        for canvas in self.canvases:
            canvas.draw()

        self.cursor_var.clear()
        items = list(self.plotted_data.keys())
        if "Model" in items:
            items.remove("Model")
            items.append("Model")
        self.cursor_var.addItems(items)

    def cursor_position(self, event):
        """Update the cursor position display with x, y, and z values."""
        if not self.plotted_data:
            return

        x_in, y_in = event.xdata, event.ydata
        if x_in is not None and y_in is not None and self.plotted_data is not None:
            ax = event.inaxes
            x_in = np.log10(x_in) if ax.get_xscale() == "log" else x_in
            y_in = np.log10(y_in) if ax.get_yscale() == "log" else y_in

            df = self.plotted_data[self.cursor_var.currentText()]
            x_col = self.combo_boxes["x_axis"].currentText()
            y_col = self.combo_boxes["y_axis"].currentText()

            if y_col is not None:
                x_arr = (
                    np.log10(abs(df[x_col])) if ax.get_xscale() == "log" else df[x_col]
                )
                y_arr = (
                    np.log10(abs(df[y_col])) if ax.get_yscale() == "log" else df[y_col]
                )
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

                distances = np.sqrt(
                    (x_arr / xlim - x_in / xlim) ** 2
                    + (y_arr / ylim - y_in / ylim) ** 2
                )
                nearest_index = np.argmin(distances)
                if distances[nearest_index] < 0.05:
                    x = df[x_col][nearest_index]
                    y = df[y_col][nearest_index]
                    self.cursor_printout.setText(
                        f"Point {int(nearest_index)} at:\n({x:.3e}, {y:.3e})"
                    )
                else:
                    self.cursor_printout.setText("Point: N/A")
            else:
                self.cursor_printout.setText("Point: N/A")
        else:
            self.cursor_printout.setText("Point: N/A")

    def update_graphs(self):
        """Update the plots based on the loaded data and plot format."""
        if not self.loaded_data:
            return
        step = "preparation"
        name = "N/A"
        try:
            self.prepare_colormaps_and_markers()
            self.clear_axes(True)
            step = "plotting"
            # Determine the plot format and call the appropriate plotting functions
            for name, dataset in self.loaded_data.items():
                if self.plot_format == "scatter":
                    self.plot_scatter_data(dataset, name, annotate=self.checkboxes["anno"].isChecked())
                elif self.plot_format == "line":
                    self.plot_line_data(dataset, name, style="r")
            step = "finalization"
            if self.plotted_data:
                # Finalize the plots
                self.finalize_plots(show_legend=True)
        except (TypeError, ValueError, IndexError, KeyError, ZeroDivisionError, OverflowError) as e:
            # self.show_error_message(f"An error occurred while updating: {str(e)}")
            error_type = type(e).__name__
            QMessageBox.critical(
            self.central_widget, "Error", f"An error of type {error_type} occurred during {step} of plot {name}: {str(e)}"
        )
        
    # def show_error_message(self, message):
    #     """Display an error message in a popup window."""
    #     error_popup = QErrorMessage(None)
    #     error_popup.showMessage(message)


class PopupGraph(GraphBase):
    """Subclass of GraphBase to create a popup window for plotting."""

    def __init__(self, *args, **kwargs):
        """Initialize PopupGraph variables."""
        super().__init__(*args, **kwargs)
        self.init_args = args
        self.init_kwargs = kwargs
        self.popup_window = None

    def create_popup_window(self):
        """Create a popup window for plotting."""
        # super().__init__(*self.init_args, **self.init_kwargs)
        self.set_base_variables()
        self.popup_window = QMainWindow()
        self.popup_window.setWindowTitle("Plot Window")
        
        central_widget = QWidget()
        layout = QHBoxLayout(central_widget)
        
        # Create control frame and add it to the popup window
        self.control_frame = QFrame(central_widget)
        self.control_frame_layout = QVBoxLayout(self.control_frame)
        self.control_frame_layout.setAlignment(Qt.AlignTop)
        self.control_frame.setLayout(self.control_frame_layout)
        self.control_frame.setMinimumSize(300, 100)
        self.control_frame.setMaximumSize(500, 16777215)
        
        self.create_plot_controls()
        self.create_cursor_layout()
        
        layout.addWidget(self.control_frame)

        # Create graph frame and add it to the popup window
        self.graph_frame = QFrame(central_widget)
        layout.addWidget(self.graph_frame)
        
        self.graph_frame_layout = QVBoxLayout(self.graph_frame)
        self.graph_frame.setLayout(self.graph_frame_layout)
        
        self.create_graph_frames()
        self.create_plots()
        
        central_widget.setLayout(layout)
        self.popup_window.setCentralWidget(central_widget)
        self.popup_window.closeEvent = self.closeEvent
        self.popup_window.show()

    def handle_trigger(self, name=None, data=None):
        """Handle the trigger to add data or create the popup window."""
        if data is not None:
            # if self.popup_window and self.popup_window.isVisible():
            if self.popup_window is not None and not sip.isdeleted(self.popup_window):
                # Append data if the popup is already open
                self.loaded_data.update({name: data})
                self.set_combobox_variables(list(data.columns), list(data.columns))
                self.update_graphs()
            else:
                # Load data and create the popup window
                self.loaded_data = {name: data}
                self.create_popup_window()
                self.set_combobox_variables(list(data.columns), list(data.columns))
                self.update_graphs()
        else:
            self.create_popup_window()

    def closeEvent(self, event):
        """Close the popup window"""
        event.accept()
        if not sip.isdeleted(self.popup_window):
            self.popup_window.deleteLater()
    
    def set_combobox_variables(self, *items_list, defaults=None):
        """Set the items for the X and Y axis comboboxes."""
        # Call the original method from GraphBase
        super().set_combobox_variables(*items_list)
        
        # Set the current item to defaults if provided
        if defaults is not None:
            if isinstance(defaults, (str, int)):
                defaults = [defaults] * len(self.combo_boxes)
            for combobox, default in zip(self.combo_boxes.values(), defaults):
                index = combobox.findText(default) if isinstance(default, str) else default
                if 0 <= index < combobox.count():
                    combobox.setCurrentIndex(index)

# class ImpedGraphBase(GraphBase):
#     """Base class for handling graph updates."""

#     def __init__(self, central_widget=None, control_frame=None, control_frame_layout=None, plot_format=None, loaded_data=None):
#         """Initialize GraphBase variables."""
#         self.central_widget = central_widget
#         self.control_frame = control_frame
#         self.control_frame_layout = control_frame_layout
#         self.plot_format = plot_format
#         self.loaded_data = loaded_data
#         self.cmaps = None
#         self.plot_styles["markers"] = None
#         self.n_cmap = 0
#         self.nyquist_plot = None
#         self.bode_plot = None
#         self.plotted_data = {}
#         self.ny_type = None
#         self.bode1_type = None
#         self.bode2_type = None
#         self.bode1_mode = None
#         self.bode2_mode = None

#         self.figures = []
#         self.axes = []
#         self.graph_frames = []
#         self.canvases = []
#         self.toolbars = []
#         self.plot_objects = []

#         PlotVars = namedtuple('PlotVars', ['var_val', 'var_scale', 'var_units'])
#         self.plot_vars = PlotVars(
#             var_val={
#                 "Z": "impedance",
#                 "Y": "admittance",
#                 "M": "modulus",
#                 "ε": "permittivity",
#                 "εᵣ": "relative_permittivity",
#                 "σ": "conductivity",
#                 "ρ": "resistivity",
#                 "Real": "real",
#                 "Imag": "imag",
#                 "+Imag": "pos_imag",
#                 "Mag": "mag",
#                 "Θ": "phase",
#                 "tan(δ)": "tan",
#             },
#             var_scale={
#                 "Real": "lin",
#                 "Imag": "lin",
#                 "+Imag": "log",
#                 "Mag": "log",
#                 "Θ": "deg",
#                 "tan(δ)": "lin",
#             },
#             var_units={
#                 "Z": r"[$\Omega$]",
#                 "Y": "[S]",
#                 "M": "[cm/F]",
#                 "ε": "[F/cm]",
#                 "εᵣ": "[1]",
#                 "σ": "[S/cm]",
#                 "ρ": r"[$\Omega$ cm]",
#             }
#         )

#     def create_plot_controls(self):
#         """Create the plot controls layout."""
#         self.graph_frame = QFrame(self.central_widget)
#         self.graph_frame_layout = QHBoxLayout()
#         self.graph_frame.setLayout(self.graph_frame_layout)

#         cols_layout = QGridLayout()


#         button_layout = QHBoxLayout()

#         plot_types = [
#             "Z",
#             "Y",
#             "M",
#             "ε",
#             "εᵣ",
#             "σ",
#             "ρ",
#         ]

#         plot_modes = [
#             "Real",
#             "Imag",
#             "Mag",
#             "Θ",
#             "+Imag",
#             "tan(δ)",
#         ]

#         # Create combo boxes
#         self.ny_type_var = AlignComboBox(self.control_frame)
#         self.ny_type_var.setTextAlignment(Qt.AlignCenter)
#         self.ny_type_var.addItems(plot_types)
#         self.ny_type_var.setCurrentIndex(0)

#         self.top_type_var = AlignComboBox(self.control_frame)
#         self.top_type_var.setTextAlignment(Qt.AlignCenter)
#         self.top_type_var.addItems(plot_types)
#         self.top_type_var.setCurrentIndex(0)

#         self.top_mode_var = AlignComboBox(self.control_frame)
#         self.top_mode_var.setTextAlignment(Qt.AlignCenter)
#         self.top_mode_var.addItems(plot_modes)
#         self.top_mode_var.setCurrentIndex(0)

#         self.bot_type_var = AlignComboBox(self.control_frame)
#         self.bot_type_var.setTextAlignment(Qt.AlignCenter)
#         self.bot_type_var.addItems(plot_types)
#         self.bot_type_var.setCurrentIndex(0)

#         self.bot_mode_var = AlignComboBox(self.control_frame)
#         self.bot_mode_var.setTextAlignment(Qt.AlignCenter)
#         self.bot_mode_var.addItems(plot_modes)
#         self.bot_mode_var.setCurrentIndex(1)

#         self.cursor_var = AlignComboBox(self.control_frame)
#         self.cursor_var.setTextAlignment(Qt.AlignCenter)
#         self.cursor_var.addItems(["Model"])
#         self.cursor_var.setCurrentText("Model")

#         update_graphs_button = QPushButton("Update Graphs", self.control_frame)
#         update_graphs_button.setFixedWidth(100)
#         # update_graphs_button.clicked.connect(self.update_graphs)

#         update_format_button = QPushButton("Update Format", self.control_frame)
#         update_format_button.setFixedWidth(100)
#         update_format_button.clicked.connect(self.update_format)

#         # Create checkboxes
#         self.nyquist_checkbox = QCheckBox("Nyquist", self.control_frame)
#         self.bode_checkbox = QCheckBox("Bode", self.control_frame)

#         self.top_log_checkbox = QCheckBox("Top", self.control_frame)
#         self.bot_log_checkbox = QCheckBox("Bottom", self.control_frame)


#         nyq_label = QLabel("Nyquist")
#         font = nyq_label.font()
#         font.setBold(True)
#         nyq_label.setFont(font)

#         bode1_label = QLabel("BodeTop")
#         font = bode1_label.font()
#         font.setBold(True)
#         bode1_label.setFont(font)

#         bode2_label = QLabel("BodeBot")
#         font = bode2_label.font()
#         font.setBold(True)
#         bode2_label.setFont(font)

#         bands_label = QLabel("Bands")
#         font = bands_label.font()
#         font.setBold(True)
#         bands_label.setFont(font)

#         bode_log_label = QLabel("Y Log Scale")
#         font = bode_log_label.font()
#         font.setBold(True)
#         bode_log_label.setFont(font)


#         # Create the plot controls layout
#         self.control_frame_layout.addLayout(cols_layout)
#         cols_layout.addWidget(bands_label, 0, 0)
#         cols_layout.addWidget(self.nyquist_checkbox, 0, 1)
#         cols_layout.addWidget(self.bode_checkbox, 0, 2)
#         cols_layout.addWidget(nyq_label, 1, 0)
#         cols_layout.addWidget(self.ny_type_var, 1, 1)
#         cols_layout.addWidget(bode_log_label, 2, 0)
#         cols_layout.addWidget(self.top_log_checkbox, 2, 1)
#         cols_layout.addWidget(self.bot_log_checkbox, 2, 2)
#         cols_layout.addWidget(bode1_label, 3, 0)
#         cols_layout.addWidget(self.top_type_var, 3, 1)
#         cols_layout.addWidget(self.top_mode_var, 3, 2)
#         cols_layout.addWidget(bode2_label, 4, 0)
#         cols_layout.addWidget(self.bot_type_var, 4, 1)
#         cols_layout.addWidget(self.bot_mode_var, 4, 2)

#         self.control_frame_layout.addLayout(button_layout)
#         button_layout.addWidget(update_graphs_button)
#         button_layout.addWidget(update_format_button)

#     def create_cursor_layout(self):
#         freq_layout = QVBoxLayout()
#         freq_sub_layout = QHBoxLayout()
#         freq_layout.addLayout(freq_sub_layout)

#         cursor_label = QLabel("Point at cursor of ", self.control_frame)
#         font = cursor_label.font()
#         font.setPointSize(10)
#         font.setBold(True)
#         cursor_label.setFont(font)

#         self.cursor_printout = QLabel("Points: N/A")

#         self.control_frame_layout.addLayout(freq_layout)
#         freq_sub_layout.addWidget(cursor_label)
#         freq_sub_layout.addWidget(self.cursor_var)
#         freq_layout.addWidget(self.cursor_printout)

#     # def create_graph_frames(self):
#     #     # Add parts for the graph frame
#     #     self.fig1, self.ax1 = GeneratePlot.subplots(figsize=(6, 6))
#     #     self.fig1.subplots_adjust(left=0.175)
#     #     self.fig2, (self.ax2, self.ax3) = GeneratePlot.subplots(2, 1, figsize=(10, 10), sharex=True)
#     #     self.ax2.set_label("Bode Top Figure")
#     #     self.ax3.set_label("Bode Bottom Figure")

#     #     self.nyquist_frame = QFrame(self.graph_frame)
#     #     self.nyquist_frame_layout = QVBoxLayout()
#     #     self.nyquist_frame.setLayout(self.nyquist_frame_layout)
#     #     self.graph_frame_layout.addWidget(self.nyquist_frame)
#     #     self.nyquist_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
#     #     self.nyquist_frame.setMaximumSize(600, 16777215)

#     #     self.bode_frame = QFrame(self.graph_frame)
#     #     self.bode_frame_layout = QVBoxLayout()
#     #     self.bode_frame.setLayout(self.bode_frame_layout)
#     #     self.graph_frame_layout.addWidget(self.bode_frame)

#     #     self.canvas1 = FigureCanvasQTAgg(self.fig1)
#     #     self.canvas2 = FigureCanvasQTAgg(self.fig2)
#     #     self.nyquist_frame_layout.addWidget(self.canvas1)
#     #     self.bode_frame_layout.addWidget(self.canvas2)

#     #     self.toolbar1 = NavigationToolbar2QT(self.canvas1, self.nyquist_frame)
#     #     self.toolbar1.update()
#     #     self.nyquist_frame_layout.addWidget(self.toolbar1)

#     #     self.canvas1.mpl_connect("motion_notify_event", self.nyquist_cursor_position)
#     #     self.canvas2.mpl_connect("motion_notify_event", self.bode_cursor_position)

#     #     self.toolbar2 = NavigationToolbar2QT(self.canvas2, self.bode_frame)
#     #     self.toolbar2.update()
#     #     self.bode_frame_layout.addWidget(self.toolbar2)

#     def create_graph_frames(self):
#         """Create the graph frames for the Nyquist and Bode plots."""
#         # Generate and append components for Nyquist plot
#         self.generate_graph_components(
#             (6, 6),
#             1,
#             None,
#             600,
#         )

#         # Generate and append components for Bode plot
#         self.generate_graph_components(
#             (10, 10),
#             2,
#             True,
#             16777215,
#         )

#         # Add specifics for the graph frame
#         self.axes[1].set_label("Bode Top Figure")
#         self.axes[2].set_label("Bode Bottom Figure")

#         self.toolbars[0].update()
#         self.toolbars[1].update()

#         self.canvases[0].mpl_connect("motion_notify_event", self.nyquist_cursor_position)
#         self.canvases[1].mpl_connect("motion_notify_event", self.bode_cursor_position)


#     def generate_graph_components(self, fig_size, subplot, sharex_val, max_size):
#         # Generate figures and axes
#         if subplot == 1:
#                 fig, ax = GeneratePlot.subplots(figsize=fig_size)
#                 self.figures.append(fig)
#                 self.axes.append(ax)
#         else:
#                 fig, ax = GeneratePlot.subplots(subplot, 1, figsize=fig_size, sharex=sharex_val)
#                 self.figures.append(fig)
#                 self.axes.append(ax)

#         # Create frame
#         frame = QFrame(self.graph_frame)
#         frame_layout = QVBoxLayout()
#         frame.setLayout(frame_layout)
#         self.graph_frame.layout().addWidget(frame)
#         frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
#         frame.setMaximumSize(max_size, 16777215)
#         self.graph_frames.append(frame)

#         # Create canvas
#         canvas = FigureCanvasQTAgg(fig)
#         frame_layout.addWidget(canvas)
#         self.canvases.append(canvas)

#         # Create toolbar
#         toolbar = NavigationToolbar2QT(canvas, frame)
#         frame_layout.addWidget(toolbar)
#         self.toolbars.append(toolbar)

#     def create_plots(self):
#         """Create the Nyquist and Bode plots."""
#         self.plot_objects = []
#         self.plot_objects.append(GeneratePlot(
#             self.ax1,
#             title=f"{self.ny_type_var.currentText()} Nyquist of Model",
#             scales="LinFrom0Scaler",
#             init_formats=["scale", "format", "square"],
#             fkwargs=dict(power_lim=2),
#         ))

#         self.plot_objects.append(GeneratePlot(
#             [self.ax2, self.ax3],
#             labels=[
#                 "Frequency [Hz]",
#                 self.top_type_var.currentText(),
#                 self.top_type_var.currentText(),
#             ],
#             title=f"{self.top_type_var.currentText()} & {self.top_type_var.currentText()} of Model",
#             scales=[
#                 "log",
#                 self.plot_vars.var_scale[self.top_mode_var.currentText()],
#                 self.plot_vars.var_scale[self.bot_mode_var.currentText()],
#             ],
#             init_formats=["scale", "format"],
#         ))

#     def plot_scatter_data(self, dataset, key, annotate=False):
#         """Plot scatter data for a single system."""
#         data_list = self.select_data(dataset)
#         for i, plot_obj in enumerate(self.plot_objects):
#             data = data_list[i]
#             plot_obj.plot(
#                 "scatter",
#                 data,
#                 label=key,
#                 styling="freq",
#                 marker=self.plot_styles["markers"][self.n_cmap],
#                 **GeneratePlot.DecadeCmapNorm(
#                     dataset["freq"], cmap=self.cmaps[i][self.n_cmap]
#                 ),
#             )
#             if annotate and i == 0:
#                 plot_obj.annotate(data)

#         for i, scatter in enumerate(GeneratePlot.get_scatter(self.ax3)):
#             cmap_norm = GeneratePlot.DecadeCmapNorm(
#                 scatter.get_array(), self.cmaps[2][i]
#             )
#             scatter.set_cmap(cmap_norm["cmap"])
#             scatter.set_norm(cmap_norm["norm"])

#         self.n_cmap += 1
#         if self.n_cmap >= min(len(cmap) for cmap in self.cmaps):
#             self.n_cmap = 0
#         self.plotted_data[key] = dataset

#     def plot_line_data(self, dataset, name, style):
#         """Plot pinned dataset if any."""
#         data_list = self.select_data(dataset)

#         for i, plot_obj in enumerate(self.plot_objects):
#             data = data_list[i]
#             plot_obj.plot(
#                 "line",
#                 data,
#                 styling=style,
#                 label=name,
#             )
#         self.plotted_data[name] = dataset


#     def prepare_colormaps_and_markers(self):
#         """Prepare colormaps and markers."""
#         self.cmaps = [
#             [
#                 scatter.get_cmap().name
#                 for scatter in GeneratePlot.get_scatter(ax)
#             ]
#             for ax in [self.ax1, self.ax2, self.ax3]
#         ]

#         base_cmaps = [
#             "Spectral_r",
#             "coolwarm",
#             "Blues_r",
#             "Greens_r",
#             "Oranges_r",
#             "Reds_r",
#             "YlOrBr_r",
#             "YlOrRd_r",
#             "OrRd_r",
#             "PuRd_r",
#             "RdPu_r",
#             "BuPu_r",
#             "GnBu_r",
#             "PuBu_r",
#             "YlGnBu_r",
#             "PuBuGn_r",
#             "BuGn_r",
#             "YlGn_r",
#             "Greys_r",
#             "Purples_r",
#         ]

#         for i, ax_cmaps in enumerate(self.cmaps):
#             for cmap in base_cmaps:
#                 if cmap not in ax_cmaps:
#                     ax_cmaps.append(cmap)
#             self.cmaps[i] = ax_cmaps

#         self.plot_styles["markers"] = (
#             "o",
#             "v",
#             "^",
#             "<",
#             ">",
#             "8",
#             "s",
#             "p",
#             "*",
#             ".",
#             "h",
#             "H",
#             "D",
#             "d",
#             "P",
#             "X",
#         )
#         self.n_cmap = 0
#         return self.cmaps, self.plot_styles["markers"]

#     def update_format(self):
#         """Update the format of the plots."""
#         for plot in self.plot_objects:
#             if plot is not None:
#                 plot.formatting()
#                 plot.scale()
#                 # plot.square()
#         for canvas in self.canvases:
#             canvas.draw()


#     def initialize_parameters(self):
#         """Initialize parameters and generate data."""
#         self.plotted_data = {}

#         self.ny_type = self.plot_vars.var_val[self.ny_type_var.currentText()]
#         self.bode1_type = self.plot_vars.var_val[self.top_type_var.currentText()]
#         self.bode2_type = self.plot_vars.var_val[self.bot_type_var.currentText()]
#         self.bode1_mode = self.plot_vars.var_val[self.top_mode_var.currentText()]
#         self.bode2_mode = self.plot_vars.var_val[self.bot_mode_var.currentText()]

#     def clear_axes(self, clear_axis=True):
#         """Clear the axes if clear_axis is True."""
#         if clear_axis:
#             self.nyquist_plot.clear()
#             self.bode_plot.clear()

#     def parse_label(self, main, mode):
#         """Parse the label based on the selected option."""
#         units = self.plot_vars.var_units[main]
#         if mode.lower() == "real":
#             return f"{main}' {units}"
#         elif mode.lower() == "imag":
#             return f"{main}'' {units}"
#         elif mode.lower() == "+imag":
#             return f"{main}'' {units}"
#         elif mode.lower() == "mag":
#             return f"|{main}| {units}"
#         elif mode == "phase" or self.plot_vars.var_val[mode] == "phase":
#             return "θ [deg]"
#         elif mode == "tan" or self.plot_vars.var_val[mode] == "tan":
#             return "tan(δ) [1]"

#     def set_plot_titles_and_labels(self, name, dataset):
#         """Set titles and labels for Nyquist and Bode plots."""
#         def get_charge_and_label(type_var, type_var_text, mode_var_text):
#             """Helper function to get charge and label."""
#             label = self.parse_label(type_var_text, mode_var_text)
#             if "imag" in mode_var_text:
#                 charge = dataset[type_var].sign * -1
#                 charge = "-" if charge < 0 else ""
#                 label = charge + label
#             return label

#         # Set Nyquist plot titles and labels
#         charge = dataset[self.ny_type].sign * -1
#         charge = "-" if charge < 0 else ""
#         self.nyquist_plot.title = f"{self.ny_type_var.currentText()} Nyquist of {name}"
#         self.nyquist_plot.xlabel = self.parse_label(self.ny_type_var.currentText(), "real")
#         self.nyquist_plot.ylabels = charge + self.parse_label(self.ny_type_var.currentText(), "imag")

#         # Set Bode plot titles and labels
#         bode_label1 = get_charge_and_label(self.bode1_type, self.top_type_var.currentText(), self.top_mode_var.currentText())
#         bode_label2 = get_charge_and_label(self.bode2_type, self.bot_type_var.currentText(), self.bot_mode_var.currentText())

#         space = " "

#         def scale_check(mode_var, check):
#             mode_text = mode_var.currentText().lower()
#             if check.isChecked():
#                 if mode_text == "imag":
#                     mode_var.setCurrentText("+Imag")
#                     return "log"
#                 elif mode_text == "θ":
#                     check.setChecked(False)
#                     return self.plot_vars.var_scale[mode_var.currentText()]
#                 else:
#                     return "log"
#             else:
#                 if mode_text == "mag" or mode_text == "+imag":
#                     check.setChecked(True)
#                 return self.plot_vars.var_scale[mode_var.currentText()]

#         self.bode_plot.title = f"{bode_label1.split(space)[0]} & {bode_label2.split(space)[0]} of {name}"
#         self.bode_plot.ylabels = [bode_label1, bode_label2]
#         self.bode_plot.yscales = [
#             scale_check(self.top_mode_var, self.top_log_checkbox),
#             scale_check(self.bot_mode_var, self.bot_log_checkbox),
#         ]

#     def select_data(self, dataset):
#         """Select data from the dataset based on the current parameters."""
#         nyquist_data = dataset.get_df(f"{self.ny_type}.real", f"{self.ny_type}.pos_imag", "freq")
#         bode_data = dataset.get_df("freq", f"{self.bode1_type}.{self.bode1_mode}", f"{self.bode2_type}.{self.bode2_mode}")
#         return nyquist_data, bode_data


#     def plot_band_data(self, dataset, min_col, max_col, band_color="blue", band_alpha=0.3):
#         """Plot confidence intervals."""
#         nyquist_data, bode_data = self.select_data(dataset)
#         if self.nyquist_checkbox.isChecked():
#             self.nyquist_plot.plot(
#                 "band",
#                 nyquist_data,
#                 ["nyquist"],
#                 ["real", min_col, max_col],
#                 styling=band_color,
#                 alpha=band_alpha,
#             )
#         if self.bode_checkbox.isChecked():
#             self.bode_plot.plot(
#                 "band",
#                 bode_data,
#                 [f"{self.bode1_type}.{self.bode1_mode}", f"{self.bode2_type}.{self.bode2_mode}"],
#                 ["freq", min_col, max_col],
#                 styling=band_color,
#                 alpha=band_alpha,
#             )

#     # def finalize_plots(self, show_legend):
#     #     """Apply formats and draw canvas, update cursor variable."""
#     #     if show_legend:
#     #         self.ax1.legend()
#     #         self.ax2.legend()

#     #     self.ax1 = self.nyquist_plot.ax[0]
#     #     self.ax2, self.ax3 = self.bode_plot.ax

#     #     self.nyquist_plot.apply_formats()
#     #     self.bode_plot.apply_formats()

#     #     self.canvas1.draw()
#     #     self.canvas2.draw()

#     #     self.cursor_var.clear()
#     #     items = list(self.plotted_data.keys())
#     #     if "Model" in items:
#     #         items.remove("Model")
#     #         items.append("Model")
#     #     self.cursor_var.addItems(items)
#     def finalize_plots(self, show_legend):
#         """Apply formats and draw canvas, update cursor variable."""
#         if show_legend:
#             for ax in self.axes:
#                 if isinstance(ax, (list, tuple)):
#                     ax[0].legend()
#                 else:
#                     ax.legend()

#         for n, plot in enumerate(self.plot_objects):
#             if len(self.axes[n]) == 1:
#                 self.axes[n] = plot.ax[0]
#             else:
#                 self.axes[n] = plot.ax
#             plot.apply_formats()

#         for canvas in self.canvases:
#             canvas.draw()

#         self.cursor_var.clear()
#         items = list(self.plotted_data.keys())
#         if "Model" in items:
#             items.remove("Model")
#             items.append("Model")
#         self.cursor_var.addItems(items)

#     def nyquist_cursor_position(self, event):
#         """Update the cursor position display with x, y, and z values."""

#         x_in, y_in = event.xdata, event.ydata
#         if x_in is not None and y_in is not None and self.plotted_data is not None:
#             df = self.plotted_data[self.cursor_var.currentText()]
#             x_col = f"{self.ny_type}.real"
#             y_col = f"{self.ny_type}.pos_imag"
#             z_col = "freq"

#             distances = np.sqrt((df[x_col] - x_in) ** 2 + (df[y_col] - y_in) ** 2)
#             nearest_index = np.argmin(distances)
#             if distances[nearest_index] < 0.05 * self.ax1.get_xlim()[1]:
#                 x = df[x_col][nearest_index]
#                 y = df[y_col][nearest_index]
#                 z = df[z_col][nearest_index]
#                 self.cursor_printout.setText(f"Point {int(nearest_index)} at:\n({x:.3e}, {y:.3e}, {z:.3e})")
#             else:
#                 self.cursor_printout.setText("Point: N/A")
#         else:
#             self.cursor_printout.setText("Point: N/A")

#     def bode_cursor_position(self, event):
#         """Update the cursor position display with x, y, and z values."""

#         x_in, y_in = event.xdata, event.ydata
#         if x_in is not None and y_in is not None and self.plotted_data is not None:
#             ax = event.inaxes
#             x_in = np.log10(x_in) if ax.get_xscale() == "log" else x_in
#             y_in = np.log10(y_in) if ax.get_yscale() == "log" else y_in

#             df = self.plotted_data[self.cursor_var.currentText()]
#             x_col = "freq"
#             y_col = None
#             if ax == self.ax2:
#                 y_col = f"{self.bode1_type}.{self.bode1_mode}"
#             elif ax == self.ax3:
#                 y_col = f"{self.bode2_type}.{self.bode2_mode}"
#             if y_col is not None:
#                 x_arr = np.log10(abs(df[x_col])) if ax.get_xscale() == "log" else df[x_col]
#                 y_arr = np.log10(abs(df[y_col])) if ax.get_yscale() == "log" else df[y_col]
#                 xlim = np.diff(np.log10(ax.get_xlim()))[0] if ax.get_xscale() == "log" else np.diff(ax.get_xlim())[0]
#                 ylim = np.diff(np.log10(ax.get_ylim()))[0] if ax.get_yscale() == "log" else np.diff(ax.get_ylim())[0]

#                 distances = np.sqrt((x_arr/xlim - x_in/xlim) ** 2 + (y_arr/ylim - y_in/ylim) ** 2)
#                 nearest_index = np.argmin(distances)
#                 if distances[nearest_index] < 0.05:
#                     x = df[x_col][nearest_index]
#                     y = df[y_col][nearest_index]
#                     self.cursor_printout.setText(f"Point {int(nearest_index)} at:\n({x:.3e}, {y:.3e})")
#                 else:
#                     self.cursor_printout.setText("Point: N/A")
#             else:
#                 self.cursor_printout.setText("Point: N/A")
#         else:
#             self.cursor_printout.setText("Point: N/A")

#     def update_graphs(self):
#         """Update the plots based on the loaded data and plot format."""
#         if not self.loaded_data:
#             return

#         self.initialize_parameters()
#         self.prepare_colormaps_and_markers()
#         self.clear_axes(True)
#         self.set_plot_titles_and_labels("Model", self.loaded_data["Model"])


#         # Determine the plot format and call the appropriate plotting functions
#         for name, dataset in self.loaded_data.items():
#             if self.plot_format == "scatter":
#                 self.plot_scatter_data(dataset, name, annotate=True)
#             elif self.plot_format == "line":
#                 self.plot_line_data(dataset, name, style="r")
#             elif self.plot_format == "confidence_intervals":
#                 self.plot_band_data(dataset, min_col="min", max_col="max", band_color="blue", band_alpha=0.3)

#         # Finalize the plots
#         self.finalize_plots(show_legend=True)
