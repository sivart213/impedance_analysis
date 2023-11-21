# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 22:12:37 2022

@author: j2cle
"""

import numpy as np
import pandas as pd

from dataclasses import dataclass, InitVar

from scipy.optimize import least_squares, Bounds
from impedance.models.circuits import CustomCircuit, fitting, elements

from research_tools.equations import Statistics
from research_tools.functions import gen_bnds, gen_mask, nyquist, bode, Complexer


@dataclass
class Complex_Imp(Complexer):
    data: InitVar[np.ndarray] = np.ndarray(0)
    name: str = "Z"

    def __post_init__(self, data):
        if isinstance(data, (pd.DataFrame, pd.Series)):
            if data.columns.isin(["theta"]).any() and "pol" not in self.name:
                self.name = "polar"
            if data.iloc[:, 0].name == "Y" and "Y" not in self.name:
                self.name = "Y"
        self.array = data
        if "Y" in self.name:
            self.array = 1 / self.array

        
    def __getitem__(self, item):
        """Return sum of squared errors (pred vs actual)."""
        if hasattr(self, item.upper()):
            return getattr(self, item.upper())
        elif hasattr(self, item.lower()):
            return getattr(self, item.lower())
        elif "y" in item.lower() and "real" in item.lower():
            return self.Y.real
        elif "y" in item.lower() and "imag" in item.lower():
            return self.Y.imag
        elif "real" in item.lower():
            return self.real
        elif "imag" in item.lower():
            return self.imag
        elif "y" in item.lower() and "mag" in item.lower():
            return np.abs(self.Y)
        elif "y" in item.lower() and "phase" in item.lower():
            return np.angle(self.Y, deg=True)
        elif "mag" in item.lower():
            return self.mag
        elif "phase" in item.lower():
            return self.phase
    
    @property
    def Z(self):
        """Calculate. generic discription."""
        return self.array

    @Z.setter
    def Z(self, _): pass

    @property
    def R(self):
        """Calculate. generic discription."""
        return self.Z.real

    @R.setter
    def R(self, _): pass

    @property
    def X(self):
        """Calculate. generic discription."""
        return self.Z.imag

    @X.setter
    def X(self, _): pass

    @property
    def Y(self):
        """Calculate. generic discription."""
        return 1 / self.array

    @Y.setter
    def Y(self, _): pass

    @property
    def G(self):
        """Calculate. generic discription."""
        return self.Y.real

    @G.setter
    def G(self, _): pass

    @property
    def B(self):
        """Calculate. generic discription."""
        return self.Y.imag

    @B.setter
    def B(self, _): pass
    
        # setattr(self, "Z", self.array)
        # setattr(self, "R", self.Z.real)
        # setattr(self, "X", self.Z.imag)
        # setattr(self, "Y", 1 / self.array)
        # setattr(self, "G", self.Y.real)
        # setattr(self, "B", self.Y.imag)

class IS_Ckt(object):
    def __init__(self, data, guess, constants={}, model="R_0-p(R_1,C_1)", conf=None):
        self.data = data

        self.guess = [
            self.Z.real.max() if isinstance(x, str) and x.lower() == "max" else x
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
            self._data["decade1"] = 10 ** np.floor(np.log10(self._data["freq"])).astype(
                float
            )
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
            self._guess = val
            if not hasattr(self, "_init_params"):
                self._init_params = val
            elif any(val != self._init_params):
                self._fit_params = val
        elif isinstance(val, str):
            if "init" in val.lower():
                self._guess = self._init_params
            else:
                self._guess = self._fit_params
        if hasattr(self, "ckt"):
            self.ckt.initial_guess = self._guess

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
    def stats(self):
        # return self.error_calc(forms=["Z", "real", "imag", "mag", "phase"])
        return self.error_calc(forms=["real", "imag"])

    @property
    def func(self):
        # return fitting.wrapCircuit(self.ckt.circuit, self.ckt.constants)
        def zwrap(params, freq):
            if isinstance(freq, (float, np.float, int, np.integer)) or (
                isinstance(freq, (np.ndarray, pd.Series)) and len(freq) == 1
            ):
                freq = [float(freq)]
            res_str = fitting.buildCircuit(
                self.ckt.circuit,
                freq,
                *params,
                constants=self.ckt.constants,
                eval_string="",
                index=0,
            )[0]
            return eval(res_str, elements.circuit_elements)

        return zwrap

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
                np.array(self.ckt.conf_) / np.array(self.ckt.parameters_),
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
        _fit_res = self.fit_res.iloc[:2, :].T.stack().to_numpy().reshape((1, -1))
        new_cols = []
        for col in self.fit_res.columns:
            new_cols.append(col)
            new_cols.append(f"{col}_err")
        return pd.DataFrame(_fit_res, columns=new_cols)

    def rel_freq(self, params=None, solve=False, **kwargs):
        # get the appropriate freq to get the desired value
        if params is None:
            params = self.ckt.parameters_
        dfz = self.Z.df.copy()

        dfz["freq"] = self.freq

        # store real values at:
        # real min, real @ imag min for image > 0, real @ imag min for real & imag > 0,
        # real max, real @ imag max for real & imag > 0 (max for imag>0 excluded due to neg results)
        bns = np.array(
            [
                dfz["real"][dfz["real"] >= 0].min(),
                dfz["real"][dfz["inv_imag"][dfz["inv_imag"] >= 0].argmin()],
                dfz["real"][
                    dfz["inv_imag"][
                        (dfz["real"] >= 0) & (dfz["inv_imag"] >= 0)
                    ].argmin()
                ],
                dfz["real"].max(),
                dfz["real"][
                    dfz["inv_imag"][
                        (dfz["real"] >= 0) & (dfz["inv_imag"] >= 0)
                    ].argmax()
                ],
            ]
        )
        bns = bns[bns > 0]

        lfunc = (
            lambda freq, params, val: Complex_Imp(self.func(params, freq))["real"] - val
        )
        kwargs = {**{"jac": "3-point", "xtol": 1e-12}, **kwargs}
        if solve:
            freq = np.array(
                [
                    least_squares(
                        lfunc,
                        float(dfz.loc[b, "freq"]),
                        args=(params, dfz.loc[b, "real"]),
                        bounds=(0, 1e12),
                        **kwargs,
                    )["x"][0]
                    for b in dfz.index
                ]
            )
        else:
            freq = np.array(
                [
                    least_squares(
                        lfunc,
                        float(dfz["freq"][dfz["real"].isin([b])]),
                        args=(params, b),
                        bounds=(0, 1e12),
                        **kwargs,
                    )["x"][0]
                    for b in bns
                ]
            )
            freq = np.logspace(
                np.log10(freq.min()), np.log10(freq.max()), len(self.freq)
            )
        return freq

    def loss_func(self, params, **kwargs):
        if "error" not in kwargs.keys() or kwargs["error"] is None:
            kwargs["error"] = "mape"
        if len(params) == len(self.ckt.parameters_):
            return self.error_calc(params=params, **kwargs)
        else:
            return np.array(
                [
                    self.error_calc(params=params[n, :], **kwargs)
                    for n in range(len(params))
                ]
            )

    def error_calc(
        self,
        pred=None,
        params=None,
        mask=None,
        forms=["mag", "phase"],
        msolve=False,
        f_weight=None,
        error=None,
        e_weight=None,
        **kwargs,
    ):
        true = self.Z

        if pred is None and params is not None:
            pred = Complex_Imp(self.func(params, self.freq))
        elif pred is None:
            pred = self.pred
        elif isinstance(pred, str):
            pred = self[pred]
        elif not isinstance(pred, Complex_Imp):
            pred = Complex_Imp(pred)
        if not isinstance(forms, list):
            forms = [forms]
        pred_freq = self.rel_freq(params, solve=msolve, **kwargs)
        fmask = gen_mask(mask, self.freq)

        if e_weight is not None:
            weight1 = e_weight(self.freq)[fmask]
        else:
            weight1 = e_weight
        errs = {
            fr: Statistics(
                true[fr][fmask],
                pred[fr][fmask],
                weight1,
                indep=self.freq[fmask],
                n_param=len(self.ckt.parameters_),
            )
            for fr in forms
        }

        if error is None:
            return errs
        else:
            try:
                return Statistics(
                    np.concatenate([true[fr][fmask] for fr in forms]),
                    np.concatenate([pred[fr][fmask] for fr in forms]),
                    np.concatenate([weight1 for fr in forms]),
                    indep=np.concatenate([self.freq[fmask] for fr in forms]),
                    n_param=len(self.ckt.parameters_),
                )[error]
            except ValueError:
                return Statistics(
                    np.concatenate([true[fr][fmask] for fr in forms]),
                    np.concatenate([pred[fr][fmask] for fr in forms]),
                    indep=np.concatenate([self.freq[fmask] for fr in forms]),
                    n_param=len(self.ckt.parameters_),
                )[error]

    def base_fit(self, f=None, Z=None, mask=None, **kwargs):
        if f is None:
            f = self.freq
        if Z is None:
            Z = self.Z.Z
        if mask is not None:
            f = f[mask]
            Z = Z[mask]
        if "bounds" in kwargs and not isinstance(kwargs["bounds"], (tuple, Bounds)):
            tmp_bnds = gen_bnds(self.fit_params, kwargs.pop("bounds"))
            kwargs["bounds"] = Bounds(tmp_bnds[0], tmp_bnds[1], keep_feasible=True)

        self.ckt.fit(
            f, Z, weight_by_modulus=kwargs.pop("weight_by_modulus", True), **kwargs
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
    from impedance_analysis import DataImport
    from pathlib import Path

    # Import data using by first getting the appropriate filename.  f_find and p_find
    # search for the desired files given a list of folder names. DataImport handles the actual
    # importing of data
    my_folder_path = p_find("Dropbox (ASU)", "Work Docs", "Data", "Raw", "MFIA", base="home")

    files = f_find(my_folder_path)
    file = files[0]
    data_in = DataImport(file, tool="MFIA", read_type="full")

    # The impedance class wraps the complex class with terms common to impedance.  Used internally
    # by several of the eis modules/classes.
    imp_data = Complex_Imp(data_in[data_in.keys()[0]])

    # Begin fitting of impedance data by first declaring the initial conditions needed by
    # impedance.py
    model = "R_0-p(R_1,C_1)"
    guess = [1e4, 1e8, 1e-12]
    constants = {}

    # Establish the ckt object. Data is retained in the object for fitting and refitting as well as
    # exporting.
    ckt = IS_Ckt(data_in[data_in.keys()[0]], guess, constants, model)

    # Call base fit (which uses impedance.py fit, which in turn uses least squares) on the data
    # contained within the object.
    ckt.base_fit()
    