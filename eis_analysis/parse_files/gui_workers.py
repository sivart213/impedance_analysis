# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""

from typing import Optional
from pathlib import Path

import pandas as pd

from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread
from PyQt5.QtWidgets import (
    QMessageBox,
    QProgressDialog,
)

from ..data_treatment import simplify_multi_index
from ..system_utilities.file_io import load_file, save, load_hdf
from ..equipment.mfia_ops import (
    convert_mfia_data,
    convert_mfia_df_for_fit,
)
from ..dict_ops import flatten_dict

CommonExceptions = (
    TypeError,
    ValueError,
    IndexError,
    KeyError,
    AttributeError,
)


class WorkerError(Exception):
    """Custom exception class for handling unexpected errors."""


class WorkerFunctions:
    """Mix-in class for GUI classes to handle worker functions."""

    worker: Optional[object] = None
    thread: Optional[QThread] = None
    progress_dialog: Optional[object] = None
    kill_operation: bool = False

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
        self.progress_dialog = QProgressDialog(
            label_text, cancel, minimum, maximum, parent
        )
        self.progress_dialog.setWindowTitle(title)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        if cancel:
            cancel_func = (
                self.cancel_operation if cancel_func is None else cancel_func
            )
            self.progress_dialog.canceled.connect(cancel_func)
        self.progress_dialog.show()

    def run_in_thread(
        self,
        finished_slot=None,
        error_slot=None,
        progress_slot=None,
        progress_dialog=None,
    ):
        """Helper function to run a worker in a separate thread with optional progress dialog."""
        if finished_slot is None:
            finished_slot = self.thread_finished
        if error_slot is None:
            error_slot = self.on_worker_error

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

    def update_progress(self, value):
        """Update the progress bar."""
        self.progress_dialog.setValue(value)

    def on_worker_error(self, error_message):
        """Handle errors from worker functions."""
        breakpoint()
        self.progress_dialog.close()
        self.progress_dialog.deleteLater()
        self.kill_operation = False
        QMessageBox.critical(
            self, "Error", f"Operation failed: {error_message}"
        )

    def cancel_operation(self):
        """Cancel the bootstrap fit."""
        self.kill_operation = True  # Set cancellation flag

    def thread_finished(self, *_, **__):
        """Handle the completion of data I/O operations."""
        self.progress_dialog.close()
        self.progress_dialog.deleteLater()
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
                name = self.t_files[name_str][ind]
                pth = self.t_files[path_str][ind]

                if ".h" in pth.suffix:
                    if self.get_all:
                        name = pth.parent.stem
                    raw_data = load_hdf(pth, key_sep=True)
                    loaded_data[name] = convert_mfia_data(
                        raw_data, flip=False, flatten=2
                    )
                elif ".xls" in pth.suffix:
                    loaded_data[name] = load_file(pth, index_col=0)[0]

            loaded_data = flatten_dict(loaded_data)

            # self.tree.set_all_data(loaded_data, True)
            # self.tree.set_data_org()

            self.finished.emit(loaded_data)
        except CommonExceptions as e:
            self.error.emit(f"Error occurred while loading dataset {str(name)}: {str(e)}")


class SaveDatasetsWorker(QObject):
    """Worker to save datasets to files."""

    finished = pyqtSignal()  # Signal to emit when done
    error = pyqtSignal(str)  # Signal to emit errors

    def __init__(
        self, path, data, columns, save_format, t_files, get_all
    ):
        """Initialize the worker."""
        super().__init__()
        self.path = Path(path)
        self.data = data
        self.columns = columns
        self.save_format = save_format
        self.t_files = t_files
        self.get_all = get_all

    def run(self):
        """Save the datasets to files."""
        try:
            if self.path.suffix:
                self.save_to_file(self.path)
            else:
                self.save_to_dir(self.path)
            self.finished.emit()
        except WorkerError as e:
            # breakpoint()
            self.error.emit(str(e))
            self.finished.emit()

    def save_to_file(self, out_path):
        """Saves the converted data."""
        try:
            
            if self.save_format == "Selected Columns":
                for k, v in self.data.items():
                    self.data[k] = simplify_multi_index(
                        v[[c for c in self.columns if c in v.columns]],
                        allow_merge = True,
                    )
            else:
                self.data = convert_mfia_df_for_fit(self.data)
        except CommonExceptions as e:
            breakpoint()
            raise WorkerError(
                "Error occurred while converting the data."
            ) from e

        try:
            save(
                self.data,
                Path(out_path).parent,
                name=Path(out_path).stem,
                file_type=Path(out_path).suffix,
            )
        except CommonExceptions as e:
            breakpoint()
            raise WorkerError(
                f"Error occurred while processing {out_path.stem}"
            ) from e

    def save_to_dir(self, out_path):
        """Saves the converted data."""
        try:
            pkeys = pd.DataFrame(
                [
                    [
                        k,
                        Path(k).parent.parent,
                        Path(k).parent.stem,
                        Path(k).stem,
                    ]
                    for k in self.data.keys()
                ],
                columns=["key", "ppart", "fname", "sname"],
            )
            # Group by 'fname'
            grouped = pkeys.groupby("fname")
        except CommonExceptions as e:
            breakpoint()
            raise WorkerError(
                "Error occurred while preparing the data."
            ) from e

        # Iterate through the groups
        for fname, group in grouped:
            try:
                # Create a dictionary for the current group
                data = {
                    row.sname: self.data[row.key]
                    for row in group.itertuples(index=False)
                }
                # Construct the path
                suffix = ""
                if self.save_format == "Selected Columns":
                    for k, v in data.items():
                        data[k] = simplify_multi_index(
                            v[[c for c in self.columns if c in v.columns]]
                        )
                    suffix = "_reduced"

                elif self.save_format == "Freq, Real, & Imag":
                    for k, v in data.items():
                        data[k] = simplify_multi_index(
                            v[[c for c in self.columns if c in v.columns]]
                        )
                    suffix = "_imps"

                if self.get_all and fname in self.t_files.name.tolist():
                    tmp = self.t_files.loc[
                        self.t_files.name == fname, "path"
                    ].values[0]
                    fname = tmp.parent.stem
                    out_path = out_path / tmp.parent.parent.stem

                save(
                    data,
                    out_path / group.iloc[0, 1],
                    fname + suffix,
                    merge_cells=True,
                )
            except CommonExceptions as e:
                breakpoint()
                raise WorkerError(
                    f"Error occurred while processing {fname}"
                ) from e
