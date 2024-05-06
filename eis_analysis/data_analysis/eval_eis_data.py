# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 22:12:37 2022.

@author: j2cle
"""
import os
import numpy as np
import pandas as pd
import h5py
import itertools

from pathlib import Path
from IPython import get_ipython
from scipy.optimize import Bounds

from impedance.models.circuits import CustomCircuit

from research_tools.functions import (
    save,
    slugify,
    gen_bnds,
    nyquist,
    bode,
    Complex_Imp,
)
from eis_analysis.data_analysis.import_eis_data import DataImport


class IS_Ckt(object):
    def __init__(
        self, data, guess, constants={}, model="R_0-p(R_1,C_1)", conf=None
    ):
        self.data = data

        self.guess = [
            self.Z.real.max()
            if isinstance(x, str) and x.lower() == "max"
            else x
            for x in guess
        ]

        self.const = constants

        self.model = model

        self.reviewed = False

        self.cost = np.inf

        self.fit_obj = None

        self.ckt = CustomCircuit(
            initial_guess=self.guess, constants=self.const, circuit=self.model
        )
        self.ckt.parameters_ = self.guess

        if conf is not None and len(conf) == len(guess):
            self.ckt.conf_
        else:
            self.ckt.conf_ = [10] * len(self.guess)

    def __getitem__(self, item):
        """Return sum of squared errors (pred vs actual)."""
        if hasattr(self, item):
            return getattr(self, item)
        elif hasattr(self.ckt, item):
            return getattr(self.ckt, item)
        elif hasattr(self.Z, item):
            return getattr(self.Z, item)

    def __repr__(self):
        return repr(print(self.ckt))

    @property
    def freq(self):
        return self.data["freq"].to_numpy()

    @freq.setter
    def freq(self, val):
        if not hasattr(self, "_data"):
            self._data = pd.DataFrame(columns=["freq", "real", "imag"])
        self._data["freq"] = val

    @property
    def Z(self):
        if not hasattr(self, "_Z"):
            self._Z = Complex_Imp(self.data[["real", "imag"]])
        return self._Z

    @Z.setter
    def Z(self, val):
        if not hasattr(self, "_data"):
            self._data = pd.DataFrame(columns=["freq", "real", "imag"])
        if not isinstance(val, Complex_Imp):
            self._Z = Complex_Imp(val)
        else:
            self._Z = val
        self._data["real"] = self._Z.real
        self._data["imag"] = self._Z.imag

    @property
    def data(self):
        # Wraps results in a useful dataframe format
        if self._data.shape[1] != 9:
            self._data["inv_imag"] = -1 * self.Z.X
            self._data["mag"] = self.Z.mag
            self._data["phase"] = self.Z.phase
            self._data["inv_phase"] = -1 * self.Z.phase
            self._data["decade1"] = 10 ** np.floor(
                np.log10(self._data["freq"])
            ).astype(float)
            self._data["decade2"] = self._data["decade1"].where(
                np.log10(self._data["decade1"]).diff() == 1, np.nan
            )
        return self._data

    @data.setter
    def data(self, val):
        # Wraps results in a useful dataframe format
        self._data = pd.DataFrame(columns=["freq", "real", "imag"])
        if not isinstance(val, pd.DataFrame):
            val = pd.DataFrame(np.array(val))
            if val.shape[1] > val.shape[0]:
                val = val.T
        self.freq = val.iloc[:, 0].to_numpy()
        self.Z = val.iloc[:, 1:]

    @property
    def sim_f(self):
        if not hasattr(self, "_sim_f"):
            self._sim_f = np.logspace(-1, 7, len(self.Z.Z))
        return self._sim_f

    @sim_f.setter
    def sim_f(self, val):
        if len(val) == 2:
            self._sim_f = np.logspace(val[0], val[1], len(self.Z.Z))
        elif len(val) == 3:
            self._sim_f = np.logspace(val[0], val[1], val[2])
        else:
            self._sim_f = val

    @property
    def guess(self):
        return self._guess

    @guess.setter
    def guess(self, val):
        if isinstance(val, (list, np.ndarray)):
            self._guess = list(val)
            if not hasattr(self, "_init_params"):
                self._init_params = list(val)
            elif any(val != self._init_params):
                self._fit_params = list(val)
        elif isinstance(val, str):
            if "init" in val.lower():
                self._guess = self._init_params
            else:
                self._guess = self._fit_params
        if hasattr(self, "ckt"):
            self.ckt.initial_guess = list(self._guess)

    @property
    def guess_conf(self):
        if hasattr(self.ckt, "conf_"):
            return self.ckt.conf_
        else:
            return [10] * len(self.guess)

    @property
    def fit_params(self):
        if not hasattr(self, "_fit_params"):
            if not hasattr(self, "ckt"):
                return self._guess
            else:
                return self.ckt.parameters_
        return self._fit_params

    @property
    def pred(self):
        return Complex_Imp(self.ckt.predict(self.freq, use_initial=False))

    @property
    def sim(self):
        return Complex_Imp(self.ckt.predict(self.sim_f, use_initial=True))

    @property
    def data_pred(self):
        vals = [
            self.freq,
            self.pred.R,
            self.pred.X,
            -1 * self.pred.X,
            self.pred.mag,
            self.pred.phase,
            -1 * self.pred.phase,
            self.sim_f,
            self.sim.R,
            self.sim.X,
            -1 * self.sim.X,
            self.sim.mag,
            self.sim.phase,
            -1 * self.sim.phase,
        ]
        columns = [
            "pr_freq",
            "pr_real",
            "pr_imag",
            "pr_inv_imag",
            "pr_mag",
            "pr_phase",
            "pr_inv_phase",
            "si_freq",
            "si_real",
            "si_imag",
            "si_inv_imag",
            "si_mag",
            "si_phase",
            "si_inv_phase",
        ]
        self._data_pred = pd.DataFrame(dict(zip(columns, vals)))
        return self._data_pred

    @property
    def data_all(self):
        return pd.concat((self.data, self.data_pred), axis=1)

    @property
    def fit_res(self):
        """Return pd.DataFrame of the parameter, conf, and std dev of the results"""
        _fit_res = pd.DataFrame(
            [
                self.ckt.parameters_,
                self.ckt.conf_,
                # np.array(self.ckt.conf_) / np.array(self.ckt.parameters_),
            ],
            columns=self.ckt.get_param_names()[0],
        )
        for col in _fit_res.columns:
            if "CPE" in col:
                elem = col.split("_")
                if elem[2] == "0":
                    _fit_res = _fit_res.rename(columns={col: f"Qy_{elem[1]}"})
                elif elem[2] == "1":
                    _fit_res = _fit_res.rename(columns={col: f"Qn_{elem[1]}"})
        return _fit_res

    @property
    def fit_res_alt(self):
        _fit_res = (
            self.fit_res.iloc[:2, :].T.stack().to_numpy().reshape((1, -1))
        )
        new_cols = []
        for col in self.fit_res.columns:
            new_cols.append(col)
            new_cols.append(f"{col}_err")
        return pd.DataFrame(_fit_res, columns=new_cols)

    def base_fit(self, f=None, Z=None, mask=None, conf_bounds=False, **kwargs):
        if f is None:
            f = self.freq
        if Z is None:
            Z = self.Z.Z
        if mask is not None:
            if callable(mask):
                mask = mask(self)
            f = f[mask]
            Z = Z[mask]
        if conf_bounds:
            kwargs["bounds"] = gen_bnds(self.fit_params, self.guess_conf)
        if "bounds" in kwargs and not isinstance(kwargs["bounds"], (Bounds)):
            kwargs["bounds"] = Bounds(
                kwargs["bounds"][0], kwargs["bounds"][1], keep_feasible=True
            )

        self.ckt.fit(
            f,
            Z,
            weight_by_modulus=kwargs.pop("weight_by_modulus", True),
            **kwargs,
        )

        self.guess = self.ckt.parameters_

    def nyquist(self, title="nyquist", pad=1.25, **kwargs):
        # if not hasattr(self, "bounds"):
        #     self.conf_int(0.25)
        data = self.Z.df.copy()
        data.insert(0, "freq", self.freq)

        fit = self.pred.df.copy()
        fit.insert(0, "freq", self.freq)

        return nyquist(
            data=data,
            fit=fit,
            bmin="min",
            bmax="max",
            title=title,
            pad=pad,
            **kwargs,
        )

    def bode(self, top="mag", bot="phase", title="bode", **kwargs):
        data = self.Z.df.copy()
        data.insert(0, "freq", self.freq)

        fit = self.pred.df.copy()
        fit.insert(0, "freq", self.freq)

        return bode(
            data,
            top=top,
            bot=bot,
            fit=fit,
            bmin="min",
            bmax="max",
            title=title,
            **kwargs,
        )


# %% Testing
if __name__ == "__main__":
    from research_tools.functions import f_find, p_find
    from eis_analysis import DataImport
    from pathlib import Path

    # Import data using by first getting the appropriate filename.  f_find and p_find
    # search for the desired files given a list of folder names. DataImport handles the actual
    # importing of data
    my_folder_path = p_find(
        "Dropbox (ASU)", "Work Docs", "Data", "Raw", "MFIA", base="home"
    )

    files = f_find(my_folder_path)
    file = files[0]
    data_in = DataImport(file, tool="MFIA", read_type="full")

    # The impedance class wraps the complex class with terms common to impedance.  Used internally
    # by several of the eis modules/classes.
    imp_data = Complex_Imp(data_in[data_in.keys()[0]])

    # Begin fitting of impedance data by first declaring the initial conditions needed by
    # impedance.py
    model = "R_0-p(R_1,C_1)"
    guess = [1, 1e2, 1e-6]
    constants = {}
    conf = [1.0, 1e1, 1e-14]

    # Establish the ckt object. Data is retained in the object for fitting and refitting as well as
    # exporting.
    ckt = IS_Ckt(data_in[data_in.keys()[0]], guess, constants, model, conf)

    # Call base fit (which uses impedance.py fit, which in turn uses least squares) on the data
    # contained within the object.
    ckt.base_fit(bounds=gen_bnds(guess, [2, 4, 6], "log"))

    from research_tools.functions import f_find, p_find

    ckt_model = "L_1-p(R_1,C_1)-p(R_2,CPE_1)-p(R_3,CPE_2)"

    init_position = [1e-6, 0.5, 1e-10, "max", 5e-6, 1, 50, 5e-6, 0.95]

    uni_bands = Bounds(
        [1e-7, 1e-2, 1e-16, 1, 1e-12, 0.75, 15, 1e-12, 0.5],
        [5e-6, 10, 1e-8, 5e5, 1e-3, 1, 200, 1e3, 1],
        keep_feasible=True,
    )
    ls_kwargs = dict(
        ftol=1e-14,
        xtol=1e-6,
        maxfev=1e6,
        jac="3-point",
        x_scale="jac",
        bounds=uni_bands,
    )
    # names_all = names_base+names_base_r2+names_hot_base+names_hot_insitu