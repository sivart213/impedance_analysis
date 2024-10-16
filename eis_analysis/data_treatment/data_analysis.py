# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 22:12:37 2022.

@author: j2cle
"""
# import os
import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import Bounds

from impedance.models.circuits import CustomCircuit, fitting, elements


from ..utils import (
    nyquist,
    bode,
)

# from ..equipment import (
#     hz_label,
# )


from .dataset_ops import (
    Complex_Imp,
    hz_label, 
)

def gen_bounds(
    arr, dev=0.1, dev_type="infer", abs_bnd=None, max_bnd=False
):
    """
    Generate upper and lower boundaries for an array of values. Boundary range
    can be set directly via dev_type or is inferred by the dtype of the dev.
    Dev is inferred as follows: int -> Log scale, float ->

    Parameters
    ----------
    arr : np.array
        An array of the the initial values
    dev : [int, float, list]
        The expected deviation or error. A single numerical value will be applied
        to all values of the array. List inputs must be of the same length as
        the input array.
    abs_bnd : [tuple, list of tuples]
        The expected deviation or error.
    max_bnd : bool
        The expected deviation or error.


    Returns
    -------
    config_file : dict
        Returns a dict containing all settings imported from the .ini file
    """
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr, float)
    arr = arr[~np.isnan(arr)]

    if isinstance(dev, (float, int, np.number, sp.Number)) or len(dev) == 1:
        dev = np.array([dev] * len(arr), float)
    if not isinstance(dev, np.ndarray):
        dev = np.array(dev)

    if isinstance(dev_type, str):
        dev_type = [dev_type]
    if len(dev_type) != len(dev):
        dev_type = [dev_type[0]] * len(dev)

    if abs_bnd is None:
        abs_bnd = [(0, np.inf)]

    if len(abs_bnd) != len(dev):
        abs_bnd = [abs_bnd[0]] * len(dev)

    dev = abs(dev)
    a_min = arr * 0.9
    a_max = arr * 1.1
    if len(dev) != len(arr):
        print("Error: bad bound input")
        return (a_min, a_max)
    for n, val in enumerate(arr):
        if (
            int(dev[n]) == dev[n] and dev_type[n] == "infer"
        ) or "log" in dev_type[n].lower():
            # integers are assumed to be log variation
            an = val * 10 ** float(-dev[n])
            ax = val * 10 ** float(dev[n])
        elif (
            dev[n] in sp.Interval(0.1, 100)
            and abs(np.log10(abs(val / dev[n]))) > 2
        ) or "perc" in dev_type[n].lower():
            an = val * (1 - dev[n])
            ax = val * (1 + dev[n])
        else:
            # assumes the value is a std dev if vals are w/in 2 orders of magnitude
            an = val - dev[n]
            ax = val + dev[n]
        a_min[n] = an if an >= abs_bnd[n][0] else abs_bnd[n][0]
        a_max[n] = ax if ax < abs_bnd[n][1] else abs_bnd[n][1]

        if max_bnd:
            a_min[n] = 0 if val > 1 else an
            a_max[n] = 1 if val in sp.Interval(0, 1, True, True) else ax

    return (a_min, a_max)


def ode_bounds(f=None, x=None, ind=0, dep=0, deg=0, **kwargs):
    """
    Generate boundary conditions for an ordinary differential equation (ODE).

    Parameters:
    f (sympy.Function, optional): The function representing the ODE. Defaults to a symbolic function 'f'.
    x (sympy.Symbol, optional): The independent variable. Defaults to a symbolic variable 'x'.
    ind (int or float, optional): The value of the independent variable at the boundary. Defaults to 0.
    dep (int or float, optional): The value of the dependent variable at the boundary. Defaults to 0.
    deg (int, optional): The degree of the derivative for the boundary condition. Defaults to 0.
    **kwargs: Additional keyword arguments.
        bounds (dict, tuple, or list, optional): Predefined boundary conditions. Can be a dictionary of sympy expressions,
                                               a tuple/list of numeric values, or a tuple/list of dictionaries/tuples/lists.

    Returns:
    dict: A dictionary representing the boundary conditions for the ODE.
    """
    if f is None:
        f = sp.symbols("f", cls=sp.Function)
    if x is None:
        x = sp.Symbol("x", real=True)
    bounds = kwargs.get("bounds", kwargs.get("bounds"))
    if bounds is None:
        if deg >= 1:
            return {f(x).diff(*[x] * int(deg)).subs(x, ind): dep}
        return {f(ind): dep}

    if isinstance(bounds, dict) and all(
        isinstance(m, sp.Basic) for m in bounds.keys()
    ):
        return bounds
    elif isinstance(bounds, (tuple, list)):
        if all(isinstance(n, (int, float, np.number)) for n in bounds):
            return ode_bounds(f, x, *bounds[:3])
        if all(isinstance(n, (tuple, list, dict, np.ndarray)) for n in bounds):
            res = {}
            for bnd in bounds:
                if isinstance(bnd, dict):
                    res = {**res, **ode_bounds(f, x, **bnd)}
                else:
                    res = {**res, **ode_bounds(f, x, *bnd[:3])}
            return res
    return None





class IS_Ckt(object):
    """
    A class to represent an Impedance Spectroscopy Circuit (IS_Ckt).

    This class is used to model and analyze impedance spectroscopy data using a custom circuit model.

    Attributes:
    data (array-like): The impedance spectroscopy data.
    guess (list): Initial guess for the circuit parameters.
    constants (dict): Constants used in the circuit model.
    model (str): The string representation of the circuit model.
    reviewed (bool): Flag indicating whether the model has been reviewed.
    cost (float): The cost function value (sum of squared errors).
    fit_obj (object): The fitting object used for optimization.
    ckt (CustomCircuit): The custom circuit object used for modeling.

    Methods:
    __getitem__(item): Return the attribute specified by 'item' from the class, circuit, or data.
    __repr__(): Return a string representation of the IS_Ckt object.
    """
    def __init__(
        self, data, guess, constants=None, model="R_0-p(R_1,C_1)", conf=None
    ):
        self.data = data
        self.guess = [
            self.Z.real.max()
            if isinstance(x, str) and x.lower() == "max"
            else x
            for x in guess
        ]

        if constants is None:
            constants = {}
            
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
            self.ckt.conf_ = conf
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
        else:
            return None

    # # def __repr__(self):
    # #     return repr(print(self.ckt))

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
            self._data["flabel"] = hz_label(self._data["freq"], kind="exp", label_rc=False, postfix="")
            # self._data["decade1"] = 10 ** np.floor(
            #     np.log10(self._data["freq"])
            # ).astype(float)
            # self._data["decade2"] = self._data["decade1"].where(
            #     np.log10(self._data["decade1"]).diff() == 1, np.nan
            # )
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
   
    
    # @property
    # def ckt(self):
    #     if not hasattr(self, "_ckt"):
    #         self._ckt = CustomCircuit(
    #             initial_guess=self.guess, constants=self.const, circuit=self.model
    #         )
    #         self.fit()
    #     return self._ckt    
   
    
    # @property
    # def func(self):
    #     # return fitting.wrapCircuit(self.ckt.circuit, self.ckt.constants)
    #     def zwrap(params, freq):
    #         if isinstance(freq, (float, np.float, int, np.integer)) or (
    #             isinstance(freq, (np.ndarray, pd.Series)) and len(freq) == 1
    #         ):
    #             freq = [float(freq)]
    #         res_str = fitting.buildCircuit(
    #             self.ckt.circuit,
    #             freq,
    #             *params,
    #             constants=self.ckt.constants,
    #             eval_string="",
    #             index=0,
    #         )[0]
    #         return eval(res_str, elements.circuit_elements)

    #     return zwrap

    # @property
    # def sim_f(self):
    #     if not hasattr(self, "_sim_f"):
    #         self._sim_f = np.logspace(-1, 7, len(self.Z.Z))
    #     return self._sim_f

    # @sim_f.setter
    # def sim_f(self, val):
    #     if len(val) == 2:
    #         self._sim_f = np.logspace(val[0], val[1], len(self.Z.Z))
    #     elif len(val) == 3:
    #         self._sim_f = np.logspace(val[0], val[1], val[2])
    #     else:
    #         self._sim_f = val

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

    # @property
    # def sim(self):
    #     return Complex_Imp(self.ckt.predict(self.sim_f, use_initial=True))

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
            # self.sim_f,
            # self.sim.R,
            # self.sim.X,
            # -1 * self.sim.X,
            # self.sim.mag,
            # self.sim.phase,
            # -1 * self.sim.phase,
        ]
        columns = [
            "pr_freq",
            "pr_real",
            "pr_imag",
            "pr_inv_imag",
            "pr_mag",
            "pr_phase",
            "pr_inv_phase",
            # "si_freq",
            # "si_real",
            # "si_imag",
            # "si_inv_imag",
            # "si_mag",
            # "si_phase",
            # "si_inv_phase",
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
    def fit_res_flat(self):
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
            kwargs["bounds"] = gen_bounds(self.fit_params, self.guess_conf)
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
            scatter_data=data,
            line_data=fit,
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
            scatter_data=data,
            top=top,
            bot=bot,
            line_data=fit,
            bmin="min",
            bmax="max",
            title=title,
            **kwargs,
        )


# # %% Testing
# if __name__ == "__main__":
#     from eis_analysis.system_utilities import find_files, find_path, DataImport

#     # Import data using by first getting the appropriate filename.  f_find and find_path
#     # search for the desired files given a list of folder names. DataImport handles the actual
#     # importing of data
#     my_folder_path = find_path("Work Docs", "Data", "Raw", "MFIA",  base=find_path("ASU Dropbox", base="drive"))

#     files = find_files(my_folder_path)
#     file = files[0]
#     data_in = DataImport(file, tool="MFIA", read_type="full")

#     # The impedance class wraps the complex class with terms common to impedance.  Used internally
#     # by several of the eis modules/classes.
#     imp_data = Complex_Imp(data_in[data_in.keys()[0]])

#     # Begin fitting of impedance data by first declaring the initial conditions needed by
#     # impedance.py
#     model = "R_0-p(R_1,C_1)"
#     guess = [1, 1e2, 1e-6]
#     constants = {}
#     conf = [1.0, 1e1, 1e-14]

#     # Establish the ckt object. Data is retained in the object for fitting and refitting as well as
#     # exporting.
#     ckt = IS_Ckt(data_in[data_in.keys()[0]], guess, constants, model, conf)

#     # Call base fit (which uses impedance.py fit, which in turn uses least squares) on the data
#     # contained within the object.
#     ckt.base_fit(bounds=gen_bounds(guess, [2, 4, 6], "log"))

#     ckt_model = "L_1-p(R_1,C_1)-p(R_2,CPE_1)-p(R_3,CPE_2)"

#     init_position = [1e-6, 0.5, 1e-10, "max", 5e-6, 1, 50, 5e-6, 0.95]

#     uni_bands = Bounds(
#         [1e-7, 1e-2, 1e-16, 1, 1e-12, 0.75, 15, 1e-12, 0.5],
#         [5e-6, 10, 1e-8, 5e5, 1e-3, 1, 200, 1e3, 1],
#         keep_feasible=True,
#     )
#     ls_kwargs = dict(
#         ftol=1e-14,
#         xtol=1e-6,
#         maxfev=1e6,
#         jac="3-point",
#         x_scale="jac",
#         bounds=uni_bands,
#     )
    # names_all = names_base+names_base_r2+names_hot_base+names_hot_insitu