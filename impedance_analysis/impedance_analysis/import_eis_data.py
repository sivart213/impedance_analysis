# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 22:12:37 2022

@author: j2cle
"""
import numpy as np
import pandas as pd

from pathlib import Path

from research_tools.functions import load
from research_tools.impedance_analysis.impedance_analysis.fit_eis_data import Impedance


class DataImport(object):
    def __init__(self, file=None, path=None, tool="Agilent", read_type="full"):
        if isinstance(file, Path):
            path = str(file.parent)
            file = file.name
        if path is None:
            path = Path.cwd()
        self.file = file
        self.path = path
        self.tool = tool
        self.read_type = read_type

    def __getitem__(self, item):
        """Add indexing methods for Data"""
        if isinstance(item, (int, slice, np.integer)):
            return self.data[self._keys[item]]
        elif isinstance(item, str) and item in self.keys():
            return self.data[item]
        elif hasattr(self, item):
            return getattr(self, item)

    @property
    def sheets(self):
        """Pulls the actual data from the excel document."""
        if not hasattr(self, "_sheets"):
            self._sheets = load(self.file, self.path)[0]
            self._sheets.pop("Main", None)
            self._sheets.pop("MasterSheet", None)
        return self._sheets

    @sheets.setter
    def sheets(self, val):
        """Set input data into the sheets."""
        if not isinstance(val, dict):
            if isinstance(val, pd.Dataframe):
                self._sheets = {"Data": val}
                self._keys = ["Data"]
            elif isinstance(val, (list, np.ndarray)):
                self._sheets = {"Data": pd.DataFrame(val)}
                self._keys = ["Data"]
            else:
                raise TypeError(
                    "Data must be DataFrame, NP array, or list (or dict of said types)"
                )
        else:
            self._sheets = val
            self._keys = []
            for key, dat in val.items():
                self._keys.append(key)
                if isinstance(val, (list, np.ndarray)):
                    self._sheets[key] = pd.Dataframe(val)
                if not isinstance(val, pd.Dataframe):
                    raise TypeError("Data must be dict of DataFrames")

    @property
    def column_info(self):
        """Gets the column names from Agilent xls docs."""
        if not hasattr(self, "_column_info"):
            self._column_info = {
                key: pd.Series(val.loc[3:5, 11].to_numpy(), index=val.loc[3:5, 10])
                for key, val in self.sheets.items()
            }
        return self._column_info

    @property
    def raw_data(self):
        """Gets all data columns."""
        if not hasattr(self, "_raw_data"):
            if self.read_type.lower() == "shallow":
                return {}
            self._raw_data = {}
            for key, val in self.sheets.items():
                tmp = val.dropna(axis=1, thresh=int(val.shape[0] * 0.25)).dropna(
                    thresh=int(val.shape[1] * 0.25)
                )
                if tmp.isna().sum().sum() == 1 and np.isnan(tmp.iloc[0, 0]):
                    tmp = tmp.dropna(axis=1)
                if (
                    sum([isinstance(s, str) for s in tmp.iloc[0, :]])
                    > tmp.shape[1] * 0.75
                    and sum([isinstance(s, str) for s in tmp.iloc[0, :]])
                    > sum([isinstance(s, str) for s in tmp.iloc[1, :]]) * 2
                ):
                    self._raw_data[key] = pd.DataFrame(
                        tmp.iloc[1:, :].to_numpy(), columns=tmp.iloc[0, :], dtype=float
                    )
                else:
                    self._raw_data[key] = tmp
        return self._raw_data

    @property
    def data(self):
        """Returns the desired data in the form of freq and impedance (|Z| and phase)."""
        if not hasattr(self, "_data"):
            if self.read_type.lower() == "shallow":
                return {}
            if self.tool.lower() == "agilent":
                self._data = {
                    key: pd.DataFrame(
                        {
                            "freq": val.iloc[:, 1].to_numpy(),
                            self.column_info[key]["Y1"]: Impedance(
                                val.iloc[:, [2, 3]]
                            ).Z,
                            self.column_info[key]["Y2"]: Impedance(
                                val.iloc[:, [4, 5]]
                            ).Z,
                        }
                    )
                    for key, val in self.raw_data.items()
                }
            elif self.tool.lower() == "mfia":
                self._data = {
                    key: pd.DataFrame(
                        {
                            "freq": val["frequency"].to_numpy(),
                            "real": val["realz"].to_numpy(),
                            "imag": val["imagz"].to_numpy(),
                        }
                    )
                    for key, val in self.raw_data.items()
                }

        for key, val in self._data.items():
            for col in val.iloc[:, 1:].columns:
                if val[col].to_numpy().imag.sum() == 0:
                    val[col] = val[col].to_numpy().real
        return self._data

    def keys(self):
        """Return sheet names of excel file."""
        if not hasattr(self, "_keys"):
            self._keys = list(self.sheets.keys())
        return self._keys


# %% Testing
if __name__ == "__main__":
    from research_tools.functions import f_find, p_find

    my_folder_path = p_find("Dropbox (ASU)", "Work Docs", "Data", "Raw", "MFIA", base="home")
    files = f_find(my_folder_path, re_filter="topcon")

    file = files[0]

    data_in = DataImport(file, tool="MFIA", read_type="full")
