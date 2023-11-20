# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 22:12:37 2022.

@author: j2cle
"""
import os
import numpy as np
import pandas as pd

from scipy import fft
from dataclasses import dataclass

from research_tools.functions import p_find


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
    # examples
    from research_tools.functions import f_find

    my_folder_path = p_find("Dropbox (ASU)", "Work Docs", "Data", "Raw", "MFIA", base="home")
    files = f_find(my_folder_path, re_filter="topcon")

    file = files[0]
