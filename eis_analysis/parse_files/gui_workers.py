# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""

from pathlib import Path

import pandas as pd
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QMessageBox,
    QProgressDialog,
)

from ..data_treatment import CachedColumnSelector, modify_sub_dfs, dataframe_manager
from ..equipment.mfia_ops import convert_mfia_data
from ..system_utilities.file_io import save, load_hdf, load_file
from ..system_utilities.io_tools import nest_dict, flatten_dict

CommonExceptions = (
    TypeError,
    ValueError,
    IndexError,
    KeyError,
    AttributeError,
    IOError,
    OSError,
)


class WorkerError(Exception):
    """Custom exception class for handling unexpected errors."""


class WorkerFunctions:
    """Mix-in class for GUI classes to handle worker functions."""

    worker: QObject | None = None
    thread: QThread | None = None
    progress_dialog: QProgressDialog | None = None
    kill_operation: bool = False
    # suppress_window: bool = False
    thread_finished: bool = True
    _is_debugging: bool = False
    # worker_lock: bool = False

    def create_progress_dialog(
        self,
        parent,
        title="Progress",
        label_text="Processing...",
        cancel=None,
        minimum=0,
        maximum=100,
        cancel_func=None,
    ):
        """Create and return a QProgressDialog."""
        self.kill_operation = False
        self.progress_dialog = QProgressDialog(label_text, cancel, minimum, maximum, parent)
        assert self.progress_dialog is not None, "Progress dialog creation failed"
        self.progress_dialog.setWindowTitle(title)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        if cancel:
            cancel_func = self.cancel_operation if cancel_func is None else cancel_func
            self.progress_dialog.canceled.connect(cancel_func)
        self.progress_dialog.show()

    def run_in_main(
        self,
        finished_slot=None,
        error_slot=None,
        progress_slot=None,
        progress_dialog=None,
    ):
        """Helper function to run a worker in the main thread."""
        self.kill_operation = False
        if finished_slot is None:
            finished_slot = self.finished_default
        if error_slot is None:
            error_slot = self.on_worker_error
        assert self.worker is not None, "Worker is not set"
        self.worker.finished.connect(finished_slot)
        self.worker.error.connect(error_slot)
        self.worker.finished.connect(self.worker.deleteLater)

        # Connect progress signal if provided
        if progress_slot:
            self.worker.progress.connect(progress_slot)

        self.worker.run()

    def run_in_thread(
        self,
        finished_slot=None,
        error_slot=None,
        progress_slot=None,
        progress_dialog=None,
    ):
        """Helper function to run a worker in a separate thread with optional progress dialog."""
        self.kill_operation = False
        if finished_slot is None:
            finished_slot = self.finished_default
        if error_slot is None:
            error_slot = self.on_worker_error

        if self._is_debugging:
            return self.run_in_main(finished_slot, error_slot)

        thread = QThread()  # Use a local variable for the thread
        assert self.worker is not None, "Worker is not set"
        self.worker.moveToThread(thread)

        thread.started.connect(self.worker.run)
        self.worker.finished.connect(finished_slot)
        self.worker.error.connect(error_slot)
        self.worker.finished.connect(thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        # Connect progress signal if provided
        if progress_slot:
            self.worker.progress.connect(progress_slot)

        # Start the thread
        thread.start()

        # Keep a reference to the thread to prevent garbage collection
        if not hasattr(self, "_threads"):
            self._threads = []
        self._threads.append(thread)

        # Clean up finished threads
        thread.finished.connect(lambda: self._threads.remove(thread))

    def update_progress(self, value, sub_value=None):
        """Update the progress bar."""
        assert self.progress_dialog is not None, "Progress dialog is not set"
        if sub_value is not None:
            sub_val = (value / 100) * sub_value
            main_progress = int(sub_val)  # /sub_value*100
            sub_progress = (sub_val - int(sub_val)) * 100
            self.progress_dialog.setLabelText(
                f"Step {main_progress} of {sub_value}; Step Progress: {sub_progress:.2f}%\n\nTotal Progress..."
            )

        self.progress_dialog.setValue(int(value))

    def on_worker_error(self, error_message):
        """Handle errors from worker functions."""

        try:
            assert self.progress_dialog is not None, "Progress dialog is not set"
            self.progress_dialog.close()
            self.progress_dialog.deleteLater()
        except RuntimeError:
            pass
        self.thread_finished = True

        # if not self.suppress_window:
        self.kill_operation = False
        # self.worker_lock = False
        QMessageBox.critical(self, "Error", f"Operation failed: {error_message}")

    def cancel_operation(self):
        """Cancel the bootstrap fit."""
        self.kill_operation = True  # Set cancellation flag

    def finished_default(self, *_, **__):
        """Handle the completion of data I/O operations."""

        try:
            assert self.progress_dialog is not None, "Progress dialog is not set"
            self.progress_dialog.close()
            self.progress_dialog.deleteLater()
        except RuntimeError:
            pass
        self.thread_finished = True
        self.kill_operation = False


class LoadDatasetsWorker(QObject):
    """Worker to load datasets from files."""

    finished = pyqtSignal(dict)  # Signal to emit the results
    error = pyqtSignal(str)  # Signal to emit errors

    def __init__(self, t_files, tree, get_all):
        """Initialize the worker."""
        super().__init__()
        self.t_files = t_files
        self.tree = tree
        self.get_all = get_all

    def run(self):
        """Load the datasets from the files."""
        name = "-"
        try:
            # load the data
            name_str = self.t_files.columns[0]
            path_str = self.t_files.columns[1]
            loaded_data = {}
            for ind in self.t_files.index:
                try:
                    name = self.t_files[name_str][ind]
                    pth = self.t_files[path_str][ind]

                    if self.t_files[name_str].value_counts().get(name, 0) > 1:
                        name = f"{name}_{ind}"

                    if ".h" in pth.suffix:
                        if self.get_all:
                            name = pth.parent.stem
                        raw_data = load_hdf(pth, key_sep=True, attach_file_stats=True)
                        loaded_data[name] = convert_mfia_data(raw_data, flip=False, flatten=2)
                    elif ".xls" in pth.suffix:
                        loaded_data[name] = load_file(pth, index_col=0, attach_file_stats=True)[0]
                    else:
                        if pth.suffix == ".csv" and "header" in name.lower():
                            continue
                        else:
                            if self.get_all:
                                name = pth.parent.stem
                            raw_data = load_file(pth, attrs_file="header", attach_file_stats=True)[
                                0
                            ]
                            loaded_data[name] = convert_mfia_data(
                                raw_data,
                                flip=False,
                                flatten=0,
                                transpose_check=True,
                            )
                except CommonExceptions as e:
                    self.error.emit(
                        f"{e.__class__.__name__} occurred while loading dataset {str(name)}: {str(e)}"
                    )

            loaded_data = flatten_dict(loaded_data)

            self.finished.emit(loaded_data)
        except CommonExceptions as e:
            self.error.emit(f"{e.__class__.__name__} occurred while loading: {str(e)}")


class SaveDatasetsWorker(QObject):
    """Worker to save datasets to files."""

    finished = pyqtSignal()  # Signal to emit when done
    error = pyqtSignal(str)  # Signal to emit errors

    def __init__(self, path, data, columns, save_format, t_files, get_all, **kwargs):
        """Initialize the worker."""
        super().__init__()
        self.path = Path(path)
        self.data = data
        self.columns = columns
        self.save_format = save_format
        self.t_files = t_files
        self.get_all = get_all

        if self.save_format == "Freq, Real, & Imag":
            self.columns = ["freq", "real", "imag"]

    def run(self):
        """Save the datasets to files."""
        try:
            try:
                if self.save_format == "Freq, Real, & Imag":
                    cached_keys = CachedColumnSelector(["freq", "real", "imag"])
                    cached_keys.cache = ["freq", "Z'", "Z''"]
                    data = modify_sub_dfs(
                        self.data,
                        (cached_keys.get_valid_columns, ["imps"]),
                        lambda df: (
                            df.reset_index(drop=True)
                            if isinstance(df.index, pd.MultiIndex)
                            else df
                        ),
                    )
                else:
                    data = dataframe_manager(self.data, columns=self.columns, allow_merge=True)

            except CommonExceptions as e:
                raise WorkerError("Error occurred while converting the data.") from e

            if self.path.suffix:
                self.save_to_file(self.path, data)
            else:
                self.save_to_dir(self.path, data)
            self.finished.emit()
        except WorkerError as e:
            # breakpoint()
            self.error.emit(str(e))
            self.finished.emit()

    def save_to_file(self, out_path, data):
        """Saves the converted data."""

        try:
            save(
                data,
                Path(out_path).parent,
                name=Path(out_path).stem,
                file_type=Path(out_path).suffix,
                mult_to_single=True,
                attrs=True,
            )
        except PermissionError as e:
            # breakpoint()
            raise WorkerError(
                f"Permission error: {str(e)}. Please check the file is closed or not in use."
            ) from e
        except CommonExceptions as e:
            # breakpoint()
            raise WorkerError(f"Error occurred while processing {out_path.stem}") from e

    def save_to_dir(self, out_path, data):
        """Saves the converted data."""
        try:
            # Construct the path and file name
            suffix = ""
            if self.save_format == "Selected Columns":
                suffix = "_red"

            elif self.save_format == "Freq, Real, & Imag":
                suffix = "_imps"

            save(
                nest_dict(data),  # data
                out_path,  # path
                merge_cells=True,  # for excel save
                mult_to_single=True,  # directs recursive save
                attrs=True,
                file_modifier=suffix,
            )
        except PermissionError as e:

            raise WorkerError(
                f"Permission error: {str(e)}. Please check the file is closed or not in use."
            ) from e
        except CommonExceptions as e:
            raise WorkerError(f"Error occurred while saving data to {out_path.stem}") from e


# ARCHIVE
