# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 22:12:37 2022

@author: j2cle
"""
import os
import numpy as np
import pandas as pd

from scipy import fft
from dataclasses import dataclass
from pathlib import Path
from research_tools.functions import load
from research_tools.functions import Complex_Imp, p_find

        
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
            self.parse()
        return self._data
    
    def parse(self):
        if self.read_type.lower() == "shallow":
            return {}
        if self.tool.lower() == "agilent":
            self._data = {
                key: pd.DataFrame(
                    {
                        "freq": val.iloc[:, 1].to_numpy(),
                        self.column_info[key]["Y1"]: Complex_Imp(
                            val.iloc[:, [2, 3]]
                        ).Z,
                        self.column_info[key]["Y2"]: Complex_Imp(
                            val.iloc[:, [4, 5]]
                        ).Z,
                    }
                )
                for key, val in self.raw_data.items()
            }
        if self.tool.lower() == "mfia":
            self._data = {
                key: pd.DataFrame(
                    {
                        "freq": val["frequency"].to_numpy(),
                        "real": val["realz"].to_numpy(),
                        "imag": val["imagz"].to_numpy(),
                    }
                ).sort_values("freq", ignore_index=True)
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

@dataclass
class TiePieData(object):
    folder: str = "RefSweep"
    path: str = str(p_find("Data", "Raw", "TiePie", "Decade_Sweep"))

    def __getitem__(self, item, read_type="full"):
        """Return sum of squared errors (pred vs actual)."""
        if isinstance(item, (int, slice, np.integer)):
            return self.raw_data[self.names[item]]
        elif isinstance(item, str) and item in self.names:
            return self.raw_data[item]
        elif hasattr(self, item):
            return getattr(self, item)

    @property
    def log(self):
        """Return sum of squared errors (pred vs actual)."""
        if not hasattr(self, "_log"):
            self._log = pd.read_excel(
                os.sep.join((self.path, self.folder, "Sample_log.xlsx"))
            )
        return self._log

    @property
    def names(self):
        """Return sum of squared errors (pred vs actual)."""
        return self.log["Identifier"].unique()

    @property
    def data(self):
        """Return sum of squared errors (pred vs actual)."""
        if not hasattr(self, "_data"):
            self._data = {ident: self.merge(ident) for ident in self.names}
        return self._data

    def merge(self, ident, alt=False):
        focus = self.log[self.log["Identifier"] == ident]

        files = [name for name in focus["File names"] if "alt" not in name]
        if alt:
            files[0] = focus["File names"][
                focus["File names"].str.contains("alt")
            ].to_numpy()[0]

        for file in files:
            data = pd.read_csv(
                os.sep.join((self.path, self.folder, f"{file}.csv")), skiprows=[0, 1, 8]
            )
            freq = fft.fftfreq(len(data), data["Relative time"].diff().mean())
            dec = np.unique(np.floor(np.log10(freq)), return_index=True)[1]
            volt_1 = fft.fft(data["Average1"].to_numpy())

            dec_std = np.array(
                [volt_1[int(dec[d]) : int(dec[d]) + 10].std() for d in range(len(dec))]
            )
            dec = dec[dec_std > 1.25]

            freq = freq[dec[-2] : dec[-1]]
            volt_1 = volt_1[dec[-2] : dec[-1]]
            volt_2 = fft.fft(data["Average2"].to_numpy())[dec[-2] : dec[-1]]

            z_calc = (
                volt_1
                / (volt_2)
                * self.log["Reference R"][self.log["File names"] == file].to_numpy()[0]
            )

            if file == files[0]:
                z_arrays = z_calc
                freqs = freq
            else:
                z_arrays = np.concatenate((z_arrays, z_calc))
                freqs = np.concatenate((freqs, freq))

        z_df = pd.DataFrame(z_arrays, columns=["complex"])
        z_df.insert(0, "freq", value=freqs)
        return z_df
# %% Testing
if __name__ == "__main__":
    from research_tools.functions import f_find, p_find
    
    my_folder_path = p_find("impedance_analysis", "testing", "Data", "Raw", base="cwd")
    files = f_find(my_folder_path, re_filter="mfia_pv")

    data_in = DataImport(files[0], tool="MFIA", read_type="full")
