# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""

import numpy as np
import pandas as pd
from sklearn import metrics
from scipy import stats, optimize, linalg

# import functions as gf
# import eqns as ge

from collections import deque
from utilities.functions import pathify, slugify, eqn_sets
from utilities.eqns import np_cost
from utilities.constants import K_B__EV

import os
import dill
import difflib
import logging

from dataclasses import dataclass, InitVar
from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.backend.operators import compute_pbest, compute_objective_function
from pyswarms.backend.handlers import BoundaryHandler, VelocityHandler, OptionsHandler
from pyswarms.backend.topology import Ring, VonNeumann, Pyramid, Random

from pyswarms.utils.search import GridSearch, RandomSearch


class Statistics:
    """Return sum of squared errors (pred vs actual)."""

    def __init__(
        self, true, pred, weight=None, indep=None, func=None, n_param=1, resid_type="", **kwargs
    ):

        if not isinstance(true, np.ndarray):
            true = np.array(true)
        if not isinstance(pred, np.ndarray):
            pred = np.array(pred)

        self.true = true
        self.pred = pred
        self.indep = indep
        self.weight = weight
        self.n_param = n_param

        self.kwargs = kwargs

        if "int" in resid_type.lower():
            self._resid = self.int_std_res
        elif "ext" in resid_type.lower():
            self._resid = self.ext_std_res
        else:
            self._resid = self.true - self.pred

    def __getitem__(self, name):
        """Calculate. generic discription."""
        shorthand = {
            "mape": "mean_abs_perc_err",
            "msle": "mean_sq_log_err",
            "rmse": "root_mean_sq_err",
            "mse": "mean_sq_err",
            "rmae": "root_mean_abs_err",
            "mae": "mean_abs_err",
            "medae": "median_abs_err",
            "r2": "r_sq",
            "var": "explained_var_score",
        }

        if name.lower() in shorthand.keys():
            return getattr(self, shorthand[name.lower()])
        else:
            return getattr(self, name)

    @property
    def indep(self):
        """Return SIMS data in log or normal form."""
        return self._indep

    @indep.setter
    def indep(self, value):
        """Set SIMS data."""
        if value is None:
            self._indep = np.arange(0, len(self.true))
        else:
            if not isinstance(value, np.ndarray):
                value = np.array(value)
            self._indep = value

    @property
    def dft(self):
        """Return degrees of freedom population dep. variable variance."""
        return self.indep.shape[0] - self.n_param

    @property
    def df(self):
        """Return degrees of freedom."""
        return self.dft

    @property
    def dfe(self):
        """Return degrees of freedom population error variance."""
        return self.indep.shape[0]

    @property
    def residuals(self):
        """Return calculated external standardized residual.."""
        return self.true - self.pred

    @property
    def sq_err(self):
        """Return sum of squared errors (pred vs actual)."""
        return np.square(self.residuals)

    @property
    def sse(self):
        """Return sum of squared errors (pred vs actual)."""
        return np.sum(self.sq_err)

    @property
    def sst(self):
        """Return total sum of squared errors (actual vs avg(actual))."""
        return np.sum((self.true - np.average(self.true, weights=self.weight, axis=0)) ** 2)

    @property
    def r_sq(self):
        """Return calculated r2_score."""
        if not np.iscomplexobj(self.true) and not np.iscomplexobj(self.pred):
            return metrics.r2_score(self.true, self.pred, sample_weight=self.weight)
        else:
            return 1 - self.sse / self.sst

    @property
    def adj_r_sq(self):
        """Return calculated value of adjusted r^2."""
        return 1 - (self.sse / self.dfe) / (self.sst / self.dft)

    @property
    def int_std_res(self):
        """Return calculated internal standardized residual."""
        n = len(self.indep)
        diff_mean_sqr = np.dot(
            (self.indep - np.average(self.indep, weights=self.weight, axis=0)),
            (self.indep - np.average(self.indep, weights=self.weight, axis=0)),
        )
        h_ii = (
            self.indep - np.average(self.indep, weights=self.weight, axis=0)
        ) ** 2 / diff_mean_sqr + (1 / n)
        Var_e = np.sqrt(sum((self.residuals) ** 2) / (n - 2))
        return (self.residuals) / (Var_e * ((1 - h_ii) ** 0.5))

    @property
    def ext_std_res(self):
        """Return calculated external standardized residual.."""
        r = self.int_std_res
        n = len(r)
        return [r_i * np.sqrt((n - 2 - 1) / (n - 2 - r_i**2)) for r_i in r]

    @property
    def normal_test(self):
        """Return calculated value from ks."""
        try:
            self.stat, self.p = stats.normaltest(self._resid)
        except (ValueError, TypeError):
            self.p = np.nan
            self.stat = np.nan
        return self.p

    @property
    def shap_test(self):
        """Return calculated value of the shap test."""
        try:
            self.shap_stat, self.shap_p = stats.shapiro(self._resid)
        except (ValueError, TypeError):
            self.shap_p = np.nan
            self.shap_stat = np.nan
        return self.shap_p

    @property
    def ks_test(self):
        """Return calculated value from ks."""
        self.ks_stat, self.ks_p = stats.ks_2samp(self.pred, self.true)
        return self.ks_p

    @property
    def chi_sq(self):
        """Return calculated value from ks."""
        try:
            self.chi_stat, self.chi_p = stats.chisquare(self.pred, self.true)
        except (ValueError, TypeError):
            self.chi_p = np.nan
            self.chi_stat = np.nan
        return self.chi_p

    @property
    def explained_var_score(self):
        """Return calculated explained_variance_score."""
        if not np.iscomplexobj(self.true) and not np.iscomplexobj(self.pred):
            return metrics.explained_variance_score(self.true, self.pred, sample_weight=self.weight)
        else:
            return 1 - np.var(self.residuals) / np.var(self.true)

    @property
    def max_err(self):
        """Return calculated max_error."""
        if not np.iscomplexobj(self.true) and not np.iscomplexobj(self.pred):
            return metrics.max_error(self.true, self.pred, sample_weight=self.weight)
        else:
            return abs(self.residuals)

    @property
    def mean_abs_err(self):
        """Return calculated mean_absolute_error."""
        if not np.iscomplexobj(self.true) and not np.iscomplexobj(self.pred):
            return metrics.mean_absolute_error(self.true, self.pred, sample_weight=self.weight)
        else:
            return np.average(self.max_err, weights=self.weight, axis=0)

    @property
    def median_abs_err(self):
        """Return calculated median_absolute_error."""
        if not np.iscomplexobj(self.true) and not np.iscomplexobj(self.pred):
            return metrics.median_absolute_error(self.true, self.pred)
        else:
            return np.median(self.max_err)

    @property
    def mean_sq_err(self):
        """Return calculated mean_squared_error."""
        if not np.iscomplexobj(self.true) and not np.iscomplexobj(self.pred):
            return metrics.mean_squared_error(self.true, self.pred, sample_weight=self.weight)
        else:
            return np.average(self.sq_err, weights=self.weight, axis=0)

    @property
    def root_mean_sq_err(self):
        """Return calculated mean_squared_error."""
        return np.sqrt(self.mean_sq_err)

    @property
    def root_mean_abs_err(self):
        """Return calculated root mean_squared_error."""
        return np.sqrt(self.mean_abs_err)

    @property
    def mean_abs_perc_err(self):
        """Return calculated mean_absolute_percentage_error."""
        if not np.iscomplexobj(self.true) and not np.iscomplexobj(self.pred):
            return metrics.mean_absolute_percentage_error(
                self.true, self.pred, sample_weight=self.weight
            )
        else:
            array = self.max_err / abs(
                np.where(self.true != 0, self.true, np.finfo(np.float64).eps)
            )
            return np.average(array, weights=self.weight, axis=0)

    @property
    def mean_sq_log_err(self):
        """Return calculated mean_squared_log_error."""
        if (
            not np.iscomplexobj(self.true)
            and not np.iscomplexobj(self.pred)
            and (self.true > 0).all()
            and (self.pred > 0).all()
        ):
            return metrics.mean_squared_log_error(self.true, self.pred, sample_weight=self.weight)
        else:
            array = np.square(np.log(1 + self.true) - np.log(1 + self.pred))
            return np.average(array, weights=self.weight, axis=0)

    @property
    def mean_poisson_dev(self):
        """Return calculated mean_poisson_deviance."""
        if not np.iscomplexobj(self.true) and not np.iscomplexobj(self.pred):
            return metrics.mean_poisson_deviance(self.true, self.pred, sample_weight=self.weight)
        else:
            array = 2 * (self.true * np.log(self.true / self.pred) + self.pred - self.true)
            return np.average(array, weights=self.weight, axis=0)

    @property
    def mean_gamma_dev(self):
        """Return calculated mean_gamma_deviance."""
        if not np.iscomplexobj(self.true) and not np.iscomplexobj(self.pred):
            return metrics.mean_gamma_deviance(self.true, self.pred, sample_weight=self.weight)
        else:
            array = 2 * (np.log(self.pred / self.true) + self.true / self.pred - 1)
            return np.average(array, weights=self.weight, axis=0)

    def pcov_calc(self, params, func=None, eps=None, args=[]):
        """Calculate. generic discription."""
        if eps is None:
            eps = np.sqrt(np.finfo(float).eps)

        if func is None and hasattr(self, "func"):
            func = self.func
        elif func is callable:
            self.func = func

        self.n_param = len(params)
        # params = self.ckt.parameters_
        if isinstance(eps, (float, np.float, int, np.integer)) or len(eps) == self.n_param:
            self.jac = np.array(
                [optimize.approx_fprime(params, func, eps, indep, *args) for indep in self.indep]
            )
        elif len(eps) == len(self.indep):
            self.jac = np.array(
                [
                    optimize.approx_fprime(params, func, eps[n], self.indep[n], *args)
                    for n in range(len(eps))
                ]
            )
        else:
            self.jac = np.array(
                [optimize.approx_fprime(params, func, eps[0], indep, *args) for indep in self.indep]
            )
        _, s, vh = linalg.svd(self.jac)
        threshold = np.finfo(float).eps * max(self.jac.shape) * s[0]
        s = s[s > threshold]
        vh = vh[: s.size]

        self.pcov = np.dot(vh.T / s**2, vh)
        self.perr = np.sqrt(np.diag(self.pcov))

        return self.perr

    def bands(self, params, intervals=0.1):
        """Calculate. generic discription."""
        if isinstance(intervals, (float, np.float)):
            intervals = params * intervals
        if intervals.shape == params.shape:
            intervals = np.array([params + intervals, params - intervals])

        inters = [intervals[n, :].copy() for n in range(2)]
        for n in range(len(params)):
            high = intervals[0, :].copy()
            low = intervals[1, :].copy()
            for m in range(len(params)):
                high[n - m] = intervals[1, n - m]
                low[n - m] = intervals[0, n - m]
                if not np.array([n.all() for n in list(np.isin(inters, high))]).any():
                    inters.append(high.copy())
                if not np.array([n.all() for n in list(np.isin(inters, low))]).any():
                    inters.append(low.copy())
        return inters


class PickleJar:
    """Calculate. generic discription."""

    def __init__(self, data=None, folder="Auto", path=None, history=False, **kwargs):
        """Calculate. generic discription."""
        self.history = history
        self.folder = folder
        if path is not None:
            self.path = path
        if data is not None:
            self.append(data)

    @property
    def database(self):
        """Return sum of squared errors (pred vs actual)."""
        for _database in os.walk(self.path):
            break
        return pd.Series(_database[2])

    @property
    def path(self):
        """Return sum of squared errors (pred vs actual)."""
        if not hasattr(self, "_path"):
            self._path = pathify("work", "Data", "Analysis", "Pickles", self.folder)
            if not os.path.exists(self._path):
                os.makedirs(self._path)
        return self._path

    @path.setter
    def path(self, value):
        self._path = value
        if not os.path.exists(self._path):
            os.makedirs(self._path)

    def __setitem__(self, name, data):
        """Calculate. generic discription."""
        db = self.database
        name = slugify(name)
        if self.history and len(self.database) != 0:
            self.shift(name)

        with open(os.sep.join((self.path, name)), "wb") as dill_file:
            dill.dump(data, dill_file)

    def __getitem__(self, name):
        """Calculate. generic discription."""
        if isinstance(name, (int, np.integer, float, np.float)) and int(name) < len(self.database):
            name = self.database[int(name)]
        else:
            name = slugify(name)

        if not self.database.isin([name]).any():
            name = difflib.get_close_matches(name, self.database)[0]
        with open(os.sep.join((self.path, slugify(name))), "rb") as dill_file:
            data = dill.load(dill_file)
        return data

    def shift(self, name):
        """Calculate. generic discription."""
        if len(self.database) == 0:
            return

        db = self.database[self.database.str.startswith(name)]
        itr = len(db[db.str.startswith(name)])
        if itr > 0:
            old = self.__getitem__(name)
            self.__setitem__(f"{name} ({itr})", old)

    def pickler(self, value):
        """Calculate. generic discription."""
        db = self.database

        if isinstance(value, (tuple, list, np.ndarray)) and len(value) == 2:
            name = value[0]
            data = value[1]
        elif isinstance(value, dict) and len(value) == 1:
            name = list(value.keys())[0]
            data = list(value.values())[0]
        else:
            data = value
            if len(db) == 0:
                itr = 0
            else:
                itr = len(db[db.str.startswith("data")])
            name = f"data ({itr})"

        self.__setitem__(name, data)

    def append(self, value):
        """Calculate. generic discription."""
        db = self.database
        if isinstance(value, dict):
            [self.pickler((key, val)) for key, val in value.items()]
        elif (
            isinstance(value, (tuple, list, np.ndarray, pd.Series)) and len(np.array(value)[0]) == 2
        ):
            [self.pickler(val) for val in value]
        else:
            self.pickler(value)

    def to_dict(self, value):
        """Calculate. generic discription."""
        if isinstance(value, dict):
            val_dict = {key: self.__getitem__(key) for key in value.keys()}
        elif isinstance(value, (tuple, list, np.ndarray, pd.Series)):
            if np.array(value).ndim == 1:
                val_dict = {val: self.__getitem__(val) for val in value}
            else:
                val_dict = {val[0]: self.__getitem__(val[0]) for val in value}
        else:
            val_dict = {value: self.__getitem__(value)}
        return val_dict

    def queary(self, value):
        """Calculate. generic discription."""
        if not isinstance(value, (tuple, list, np.ndarray)):
            value = [value]

        if len(self.database) == 0:
            return []
        res = self.database
        for val in value:
            res = res[res.str.contains(val)]
        return res


class NernstPlanck(object):
    """Calculate. generic discription."""

    def __init__(self, df):
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        self.df = df

    def char_eq(self, target="time", as_set=False):
        """Return sum of squared errors (pred vs actual)."""
        param_values = self.df.loc[0, :].to_dict()
        param_values["k_b"] = K_B__EV

        char = "2*(dif*time)**(1/2)+dif/(k_b*temp)*efield*time-thick"

        x0 = eqn_sets(param_values, target=target, eqns=char, as_set=False)

        return x0

    def np_sim(
        self,
        target="time",
        ratio=None,
        scale="lin",
        ls_params={},
        bound=0.5,
        **pre_kwargs,
    ):
        """Return sum of squared errors (pred vs actual)."""
        ls_params = {**{"jac": "3-point", "xtol": 1e-12}, **ls_params}

        x0 = self.char_eq(target, **pre_kwargs)
        bounds = (x0 * (1 - bound), x0 * (1 + bound))

        val = {**{"x0": x0, "bounds": bounds}, **ls_params}

        if ratio is None and "sourceC" in self.df.columns:
            ratio = self.df.conc / self.df.sourceC
        else:
            ratio = 0.08
        try:
            results = optimize.least_squares(
                np_cost,
                args=(
                    self.df.copy(),
                    ratio,
                    target,
                    scale,
                ),
                **val,
            ).x[0]
        except ValueError:
            results = 0
        return results


@dataclass
class Complexer(object):
    """Calculate. generic discription."""

    data: InitVar[np.ndarray] = np.ndarray(0)
    name: str = "Z"

    def __post_init__(self, data):
        """Calculate. generic discription."""
        self.array = data

    def __getitem__(self, item):
        """Return sum of squared errors (pred vs actual)."""
        if hasattr(self, item.upper()):
            return getattr(self, item.upper())
        elif hasattr(self, item.lower()):
            return getattr(self, item.lower())

    @property
    def array(self):
        """Calculate. generic discription."""
        return self._array  # .reshape((-1, 1))

    @array.setter
    def array(self, arr):
        if isinstance(arr, np.ndarray):
            self._array = arr
        else:
            self._array = np.array(arr).squeeze()

        if not self._array.dtype == "complex128":
            if len(self._array.shape) == 2 and self._array.shape[1] >= 2:
                if "pol" in self.name.lower():
                    if (abs(self._array[:, 1]) > np.pi / 2).any():
                        self._array[:, 1] = np.deg2rad(self._array[:, 1])

                    self._array = self._array[:, 0] * (
                        np.cos(self._array[:, 1]) + 1j * np.sin(self._array[:, 1])
                    )
                else:
                    self._array = self._array[:, 0] + 1j * self._array[:, 1]
            else:
                self._array = self._array + 1j * 0
        elif len(self._array.shape) == 2 and self._array.shape[1] >= 2:
            self._array = self._array[:, 0]

    @property
    def real(self):
        """Calculate. generic discription."""
        return self.array.real

    @real.setter
    def real(self, _):
        pass

    @property
    def imag(self):
        """Calculate. generic discription."""
        return self.array.imag

    @imag.setter
    def imag(self, _):
        pass

    @property
    def mag(self):
        """Calculate. generic discription."""
        return np.abs(self.array)

    @mag.setter
    def mag(self, _):
        pass

    @property
    def phase(self):
        """Calculate. generic discription."""
        return np.angle(self.array, deg=True)

    @phase.setter
    def phase(self, _):
        pass

    @property
    def df(self):
        """Calculate. generic discription."""
        vals = [self.real, self.imag, -1 * self.imag, self.mag, self.phase, -1 * self.phase]
        columns = ["real", "imag", "inv_imag", "mag", "phase", "inv_phase"]
        # self._data = pd.DataFrame(dict(zip(columns, vals)))
        return pd.DataFrame(dict(zip(columns, vals)))

    @df.setter
    def df(self, _):
        pass


class PSO(object):
    """Calculate. generic discription."""

    def __init__(self, func, particles=15, dims=5, options=None, oh_strategy=None, pso_kwargs=None):

        self.func = func
        self.particles = particles
        self.dims = dims

        self.options = options
        self.oh_strategy = oh_strategy
        self.pso_kwargs = pso_kwargs

    @property
    def dims(self):
        """Calculate. generic discription."""
        if "bounds" in self.pso_kwargs.keys():
            return len(self.pso_kwargs["bounds"][0])
        else:
            return self._dims

    @dims.setter
    def dims(self, value):
        self._dims = int(value)

    @property
    def ci(self):
        """Calculate. generic discription."""
        if not hasattr(self, "hist_df"):
            return None
        else:
            if not hasattr(self, "_alpha"):
                self._alpha = 90
            bnd = self.cost * (1 + (100 - self._alpha) / 100)
            self.ci_hist = self.hist_df[self.hist_df["cost"] <= bnd]
            return self.ci_hist.loc[:, "p0":].describe()

    @ci.setter
    def ci(self, value):
        self._alpha = value

    @property
    def oh_strategy(self):
        """Calculate. generic discription."""
        return self._pso_kwargs["oh_strategy"]

    @oh_strategy.setter
    def oh_strategy(self, value):
        opts = ["c1", "c2", "w", "k", "p"]
        strats = self.strat_opts["oh_strategy"]
        if isinstance(value, dict) and all(k in opts and v in strats for k, v in value.items()):
            self._pso_kwargs["oh_strategy"] = {**self.pso_kwargs["oh_strategy"], **value}

    @property
    def strat_opts(self):
        """Calculate. generic discription."""
        return {
            "oh_strategy": list(OptionsHandler(None).strategies.keys()),
            "vh_strategy": list(VelocityHandler(None).strategies.keys()),
            "bh_strategy": list(BoundaryHandler(None).strategies.keys()),
        }

    @property
    def pso_kwargs(self):
        """Calculate. generic discription."""
        if not hasattr(self, "_pso_kwargs"):
            self._pso_kwargs = {
                "oh_strategy": {"w": "exp_decay"},
                "vh_strategy": "invert",
                "bh_strategy": "periodic",
            }
        return self._pso_kwargs

    @pso_kwargs.setter
    def pso_kwargs(self, value):
        if isinstance(value, dict):
            for key, val in value.items():
                if key == "oh_strategy":
                    self.oh_strategy = val
                elif key in self.strat_opts.keys() and val in self.strat_opts[key]:
                    self._pso_kwargs = {**self.pso_kwargs, **{key: val}}
                elif key not in self.strat_opts.keys():
                    self._pso_kwargs = {**self.pso_kwargs, **{key: val}}

    @property
    def options(self):
        """Calculate. generic discription."""
        if not hasattr(self, "_options"):
            self._options = {
                "start": {"c1": 2, "c2": 2, "w": 0.8},
                "stop": {},
            }
        return self._options

    @options.setter
    def options(self, value):
        if isinstance(value, dict):
            for key, val in value.items():
                opt = self.options
                if key in ["start", "stop"]:
                    # val is a dict
                    if all(n in val.keys() for n in ["c1", "c2", "w"]):
                        opt[key] = {**opt[key], **val}
                        self._options = {**self.options, **opt}
                elif key in ["c1", "c2", "w", "k", "p"]:
                    opt["start"][key] = val
                    self._options = {**self.options, **opt}

    def gbest_init(self):
        """Calculate. generic discription."""
        return GlobalBestPSO(
            n_particles=self.particles,
            dimensions=self.dims,
            options=self.options["start"],
            **self.pso_kwargs,
        )

    def optimize_opts(
        self, options, search="random", iters=25, n_iters=10, set_best=True, **kwargs
    ):
        """Calculate. generic discription."""
        if "rand" in search.lower():
            dev = [0, 2]
            opts = {**{"c1": dev, "c2": dev, "w": dev, "k": dev, "p": 1}, **options}
            g = RandomSearch(
                GlobalBestPSO,
                n_particles=self.particles,
                dimensions=self.dims,
                options=opts,
                objective_func=self.func,
                iters=iters,
                n_selection_iters=n_iters,
            )
        else:
            g = GridSearch(
                GlobalBestPSO,
                n_particles=self.particles,
                dimensions=self.dims,
                options=options,
                objective_func=self.func,
                iters=min(iters, n_iters),
            )

        best_score, best_options = g.search()
        if set_best:
            self.options = {
                "start": {key: val for key, val in best_options.items() if key in options.keys()},
                "stop": {},
            }
        return best_score, best_options

    def optimize(self, iters=1000, verbose=True, new=True, **kwargs):
        """Calculate. generic discription."""
        if hasattr(self, "hist_df"):
            del self.hist
        opt = self.gbest_init()
        top = opt.top
        for key, item in kwargs.items():
            if "topo" in key.lower():
                if "pyramid" in item.lower():
                    top = Pyramid()
                elif "von" in item.lower():
                    top = Pyramid()
                elif "pyramid" in item.lower():
                    top = Pyramid()
                elif "rand" in item.lower():
                    top = Random()
                kwargs.pop(key)
                break

        if verbose:
            log_level = logging.INFO
        else:
            log_level = logging.NOTSET

        opt.rep.log("Obj. func. args: {}".format(kwargs), lvl=logging.DEBUG)
        opt.rep.log(
            "Optimize for {} iters with {}".format(iters, self.options),
            lvl=log_level,
        )

        self.swarm = opt.swarm
        # Populate memory of the handlers
        opt.bh.memory = self.swarm.position
        opt.vh.memory = self.swarm.position

        self.swarm.pbest_cost = np.full(opt.swarm_size[0], np.inf)
        ftol_history = deque(maxlen=opt.ftol_iter)
        for i in opt.rep.pbar(iters, opt.name) if verbose else range(iters):
            # Compute cost for current position and personal best
            # fmt: off
            self.swarm.current_cost = compute_objective_function(self.swarm, self.func, **kwargs)
            self.swarm.pbest_pos, self.swarm.pbest_cost = compute_pbest(self.swarm)
            # Set best_cost_yet_found for ftol
            best_cost_yet_found = self.swarm.best_cost
            self.swarm.best_pos, self.swarm.best_cost = top.compute_gbest(self.swarm)
            # fmt: on
            if verbose:
                opt.rep.hook(best_cost=self.swarm.best_cost)
            # Save to history
            hist = opt.ToHistory(
                best_cost=self.swarm.best_cost,
                mean_pbest_cost=np.mean(self.swarm.pbest_cost),
                mean_neighbor_cost=self.swarm.best_cost,
                position=self.swarm.position,
                velocity=self.swarm.velocity,
            )
            opt._populate_history(hist)
            # local hist
            df = pd.DataFrame(self.swarm.position, columns=[f"p{x}" for x in range(self.dims)])
            df = df.reset_index().rename(columns={"index": "particle"})
            # df2 = pd.DataFrame(self.swarm.velocity, columns=[f"v{x}" for x in range(self.swarm.n_particles)])
            # df = pd.concat([df, df2], axis=1).reset_index().rename(columns={"index": "particle"})
            df.insert(1, "cost", self.swarm.current_cost)
            df.insert(0, "pbest", self.swarm.pbest_cost)
            df.insert(0, "gbest", [self.swarm.best_cost] * self.swarm.n_particles)
            if hasattr(self, "hist_df"):
                self.hist_df = pd.concat([self.hist_df, df], axis=0, ignore_index=True)
            else:
                self.hist_df = df.copy()
            # Verify stop criteria based on the relative acceptable cost ftol
            relative_measure = opt.ftol * (1 + np.abs(best_cost_yet_found))
            delta = np.abs(self.swarm.best_cost - best_cost_yet_found) < relative_measure
            if i < opt.ftol_iter:
                ftol_history.append(delta)
            else:
                ftol_history.append(delta)
                if all(ftol_history):
                    break
            # Perform options update
            self.swarm.options = opt.oh(
                opt.options, iternow=i, itermax=iters, end_opts=self.options["stop"]
            )
            # Perform velocity and position updates
            self.swarm.velocity = top.compute_velocity(
                self.swarm, opt.velocity_clamp, opt.vh, opt.bounds
            )
            self.swarm.position = top.compute_position(self.swarm, opt.bounds, opt.bh)
        # Obtain the final best_cost and the final best_position
        self.cost = self.swarm.best_cost.copy()
        self.pos = self.swarm.pbest_pos[self.swarm.pbest_cost.argmin()].copy()
        # Write report in log and return final cost and position
        opt.rep.log(
            "Optimization finished | best cost: {}, best pos: {}".format(self.cost, self.pos),
            lvl=log_level,
        )

        return (self.cost, self.pos)
