# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 22:12:37 2022

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

from research_tools.functions import save, slugify, gen_bnds, nyquist, bode, Complex_Imp
from eis_analysis.data_analysis.import_eis_data import DataImport

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
        _fit_res = self.fit_res.iloc[:2, :].T.stack().to_numpy().reshape((1, -1))
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
            f = f[mask]
            Z = Z[mask]
        if conf_bounds:
            kwargs["bounds"] = gen_bnds(self.fit_params, self.guess_conf)
        if "bounds" in kwargs and not isinstance(kwargs["bounds"], (Bounds)):
            kwargs["bounds"] = Bounds(kwargs["bounds"][0], kwargs["bounds"][1], keep_feasible=True)

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
    

class IS_Data(object):
    """
    Top level class to contain and operate on impedance data.

    Maintains a hdf5 file of the data for rapid access to data. Calls importing classes as needed
    for files not already included in the hdf. Runs fiting method of IS_Ckt to get results

    ...

    Attributes
    ----------
    super_set : str
        The name of the group name used to combine all related data.
        A super_set must consist of files that are to be fit with the same settings
        Used also as the hdf5 filename
    names : list (of strings)
        A list of all filenames to be included in the super_set
    hdf_pth : Path
        The path to the hdf5 file which contains all data in the super_set
    model : str
        A string representing the model to be used for fitting. R, C, and L represent Resistors,
        Capacitors, and Inductors respectfully. Give each element a unique number
        and use 'R1-R2' and 'p(R1,R2)' for series and parallel elements respectfully.
        Visit https://impedancepy.readthedocs.io/en/latest/index.html for full documentation.
        Default : 'Randles circuit'
    init_pos : list (of float)
        The initial positions of each circuit element used for fitting.
        Exclude values for circuit elements which are constant.
        Default : [1e1, 1e6, 1e-12]
    consts : dict, optional
        Dictionary of constant circuit elements. Format -> {circuit_element: value}
        Default : {}
    area : Union[int, float], optional
        The area of the sample to convert to Ohms/cm\ :sup:`2`\.
        Default : 1
    eis_file : dict
    results : dict
    data_obj : h5py object

    Properties
    ----------
    weight : function
        Function to be used for weighing fit
        TODO evaluate for removal here or in IS_Ckt
    hdf_data : dict
        Return a dict of the data from the super_set hdf5 file
    hdf_attr : dict
        Return a dict of the attributes from the super_set hdf5 file
    # hdf : h5py object
    #     TODO evaluate
    comb_fit_res : pd.DataFrame
        Return a DataFrame of the fit results suitable for exporting to excel


    Methods
    -------
    update(files, path)
    get_raw(file, store)
    hdf_status(status)
    to_hdf(group, ckt_obj)
    base_fitter(fit_list, thresh, refit, bounds_by_conf, **fit_kwargs)
    update_obj()
    export_data(item, path, name, ftype)
    export_figs(path, name, ftype)

    """

    def __init__(
        self,
        super_set,
        hdf_folder,
        model="R_0-p(R_1, C_1)",
        init_pos=[1e1, 1e6, 1e-12],
        constants={},
        area=1,
        **kwargs,
    ):
        self.super_set = super_set
        self.hdf_pth = Path(hdf_folder) / f"{super_set}.h5"

        self.init_pos = init_pos
        self.consts = constants
        self.model = model
        self.area = area

        self.get_raw(kwargs.pop("raw_files", None), **kwargs)


        self.results

    def __getitem__(self, item):
        """Return sum of squared errors (pred vs actual)."""
        if hasattr(self, item):
            return getattr(self, item)
        if hasattr(self, "results"):
            if hasattr(self.results, item):
                return getattr(self.results, item)
            else:
                return {
                    key: value[item]
                    for key, value in self.results.items()
                    if hasattr(value, item)
                }

    @property
    def weight(self):
        return lambda f: np.flip(f) / np.flip(f).sum()

    @property
    def hdf_keys(self):
        """List of keys in the hdf file. Updates with hdf_opperator toggle"""
        return self.get_hdf_names(self.hdf_pth)

    @property
    def results(self):
        if not hasattr(self, "_results"):
            self.load_hdf()
        return self._results

    @results.setter
    def results(self, val):
        if not hasattr(self, "_results"):
            self._results = {}
        if isinstance(val, dict):
            self._results = {**self._results, **val}
        elif isinstance(val, (tuple, list)):
            self._results[val[0]] = val[1]

    @property
    def comb_fit_res(self):
        """Contains formated Dataframe for easy export to excel, sorted by category"""
        for key in self.results.keys():
            if not hasattr(self, "_comb_fit_res"):
                cols = ["bias"] + list(self.results[key].fit_res_alt.columns)
                self._comb_fit_res = pd.DataFrame(columns=cols, dtype=float)
            try:
                fkey = float(key)
            except ValueError:
                continue
            if fkey not in self._comb_fit_res["bias"].to_numpy():
                self._comb_fit_res = pd.concat(
                    [self._comb_fit_res, self.results[key].fit_res_alt]
                )
                self._comb_fit_res.iloc[-1, 0] = fkey
        return self._comb_fit_res.sort_values("bias").reset_index(drop=True)

    def get_raw(
        self,
        file,
        store=False,
        **kwargs,
    ):
        """
        Imports the data from excel or csv using DataImport (or TiePieData)

        Parameters
        ----------
        file : Union[str, Path]
            The path to the raw data to be imported and analyzed
        store : bool
            Stores the imported data as an object for further evaluation if desired

        Returns
        -------
        data : dict
            Returns a dict of the dataframes generated by the file
        """

        if isinstance(file, (tuple, list)):
            return {Path(f).stem: self.get_raw(f, store, **kwargs) for f in file}
        if file is None:
            return

        file = Path(file)

        data_in = DataImport(
            file, tool=kwargs.get("tool", "Agilent"), read_type="shallow"
        )

        if all([f"{file.stem}/{name}" in self.hdf_keys for name in data_in.keys()]):
            return

        data_in.read_type = "full"

        if store:
            if not hasattr(self, "data_obj"):
                self.data_obj = {}
            self.data_obj[file.stem] = data_in

        # for name in self.zip_keys(file.stem, data_in.keys()): # name.split("/")[1]
        for name in data_in.keys():
            if f"{file.stem}/{name}" not in self.hdf_keys:
                print(f"Added {file.stem}/{name} to hdf")
                self.to_hdf(
                    f"{file.stem}/{name}",
                    IS_Ckt(
                        data_in[name],
                        self.init_pos,
                        self.consts,
                        self.model,
                    ),
                )
        return

    def load_hdf(self, reset=False):
        """
        Load contents of hdf into results
        """
        get_ipython().run_line_magic("matplotlib", "inline")
        results = {}
        with h5py.File(self.hdf_pth, "a") as hf:
            names = self.get_hdf_names(hf)
            for name in names:
                data = hf[f"{name}/data"][()]
                if reset:
                    guess = self.init_pos
                    constants = self.consts
                    model = self.model
                else:
                    guess = hf[f"{name}/guess"][()]
                    constants = hf[name].attrs.get("consts", {})
                    model = hf[name].attrs.get("model", "R_0-p(R_1, C_1)")

                results[name] = IS_Ckt(
                    data=data,
                    guess=guess,
                    constants=constants,
                    model=model,
                )

        if not hasattr(self, "_results"):
            self._results = results
        else:
            self.results = results

        return results

    def to_hdf(self, group, ckt_obj):
        # must be sent the group from outside
        """
        Saves the raw data in an hdf file with data, guess, and conf datasets

        If the group dos not exist, it creates the group

        Parameters
        ----------
        group : Union[str, h5py.Group]
            The relavent group to store the data. Typically the filename of the raw data
        ckt_obj : IS_Ckt
            The IS_Ckt object containing the important data

        """
        if isinstance(group, str):
            with h5py.File(self.hdf_pth, "a") as hf:
                group = hf.require_group(group)
                self.to_hdf(group, ckt_obj)
            return
        else:
            # add attrs
            group.attrs["model"] = self.model
            for key, val in ckt_obj.const.items():
                group.attrs[key] = val

            dst_data = group.require_dataset(
                "data",
                shape=(len(ckt_obj.data), 3),
                dtype=float,
                maxshape=(None, None),
            )
            dst_data[()] = ckt_obj.data.iloc[:, :3].to_numpy()

            dst_guess = group.require_dataset(
                "guess",
                shape=len(ckt_obj.guess),
                dtype=float,
                maxshape=(None,),
            )
            dst_guess[()] = ckt_obj.guess

            dst_conf = group.require_dataset(
                "conf",
                shape=len(ckt_obj.guess_conf),
                dtype=float,
                maxshape=(None,),
            )

            dst_conf[()] = ckt_obj.guess_conf

        return

    def compile_results(self, key, *obj):
        if key == "clear_results":
            del self._results
        if len(obj) == 1:
            obj = obj[0]
        if isinstance(obj, dict) and "data" not in obj.keys():
            {k: self.compile_results(v) for k, v in obj.items()}
        elif isinstance(obj, tuple):
            data_dict = dict(
                data=obj[0],
                guess=obj[1],
                consts=obj[2],
                model=obj[3],
            )
        else:
            data_dict = dict(
                data=obj["data"],
                guess=obj["guess"],
                consts=obj["const"],
                model=obj["model"],
            )
        res_dict = {key: data_dict}
        if self._results_as_ckt_obj:
            res_dict[key] = IS_Ckt(**data_dict)

        self._results = {**self._results, **res_dict}

        return

    def base_fitter(
        self, fit_list, thresh=1, refit=False, bounds_by_conf=False, **fit_kwargs
    ):
        """
        Iteravely Runs base_fit of the IS_Ckt object for each data_set

        Parameters
        ----------
        fit_list : list
            List of all names to be fit
        thresh : Union[int, float]
            Error threshold for skipping the fit
        refit : bool
            TODO Not used, eval for removal
        bounds_by_conf : bool
            If True, create boundaries from the confidence interval sent to Complex_Imp
        **fit_kwargs : dict
            weight_by_modulus : bool
                Fit the result using the modulus to apply weighting
            bounds : Union[tuple, Bounds]
                The boundaries of the fit
            ftol : float
                least_squares parameter.See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
            xtol : float
                least_squares parameter. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
            maxfev : float
                least_squares parameter. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
            jac : str
                least_squares parameter. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
            x_scale : str

        """
        get_ipython().run_line_magic("matplotlib", "inline")

        hdf_keys = self.hdf_keys

        with h5py.File(self.hdf_pth, "a") as hf:
            # get list of files to fit
            if fit_list == []:
                fit_list = hdf_keys
            elif isinstance(fit_list, str):
                fit_list = [x for x in hdf_keys if fit_list in x]

            for sheet in hdf_keys:
                if any(name == sheet for name in fit_list):
                    fname = sheet.split("/")[0]
                    sname = sheet.split("/")[1]
                    vals = self.results[
                        sheet
                    ].fit_res.copy()  # assumes results already exists
                    if all(vals.iloc[1, :] / vals.iloc[0, :] < thresh):
                        print(f"Skipping {fname}, run: {sname}")
                        continue
                    print(f"Fitting {fname}, run: {sname}")
                    # print(f"Previous best cost: {self.results[sheet].stats}")

                    if bounds_by_conf:
                        minb = -1 * vals.diff().iloc[1, :]
                        maxb = vals.iloc[:2, :].sum()
                        fit_kwargs["bounds"] = Bounds(
                            [
                                min(max(minb[n], 1e-24), abs(vals.loc[0, n] * 0.5))
                                for n in minb.index
                            ],
                            [
                                1
                                if "Qn" in n
                                else max(maxb[n], abs(vals.loc[0, n] * 1.5))
                                for n in maxb.index
                            ],
                            keep_feasible=True,
                        )

                    self.results[sheet].base_fit(**fit_kwargs)

                    # self.results[sheet].nyquist(title=f"{fname}, bias: {sname}")
                    # self.results[sheet].bode(title=f"{fname}, bias: {sname}")
                    # print(f"Final cost: {self.results[sheet].stats}")

                    print(f"Final fit: \n{self.results[sheet].fit_res.T}\n")
                    if (self.results[sheet].fit_res.iloc[1, :]<hf[sheet]["conf"][()]).mean() > 0.5:
                        self.to_hdf(hf[sheet], self.results[sheet])

    def export_data(self, item="data_all", path=None, name=None, ftype="xlsx"):
        """
        Exports data and fit results via dataframe

        Parameters
        ----------
        item : str
        path : Path
        name : str
        ftype : str

        """
        if isinstance(item, list):
            if not isinstance(path, list) or len(path) != len(item):
                path = [path] * len(item)
            if not isinstance(name, list):
                name = [name] * len(item)
            if not isinstance(ftype, list):
                ftype = [ftype] * len(item)
            for n in range(len(item)):
                self.export_data(item[n], path[n], name[n], ftype[n])

        if path is None:
            path = self.hdf_pth.parent.parent/"IS"/self.super_set

        if isinstance(item, str):
            save(self[item], path, name, ftype)
        else:
            save(item, path, name, ftype)

    def export_figs(self, path=None, name=None, ftype="nyquist"):
        """
        Export figures of data and fit results.

        Parameters
        ----------
        path : Path
        name : str
        ftype : str

        """
        # get_ipython().run_line_magic('matplotlib', 'qt5')
        if path is None:
            path = self.hdf_pth.parent.parent/"IS"/"figs"/self.super_set
        if not os.path.exists(path):
            os.makedirs(path)

        for sheet in list(self.eis_file.keys()):
            fname = self.eis_file.file.split(".")[0] + f", bias: {sheet}"
            if name is None:
                pname = fname
            elif isinstance(name, str):
                pname = name + f", bias: {sheet}"
            elif callable(name):
                pname = name(fname, sheet)

            if "bode" in ftype.lower():
                plt = self.results[sheet].bode(title=f"{pname}_bode", return_fig=True)
                sname = slugify(pname.replace(": ", "_")) + "_bode"
            else:
                plt = self.results[sheet].nyquist(
                    title=f"{pname}_nyquist", return_fig=True
                )
                sname = slugify(pname.replace(": ", "_")) + "_nyquist"

            plt.savefig(os.sep.join((path, f"{sname}.png")))

    @staticmethod
    def zip_keys(*args):
        """Creates a list of keys in full path format from args"""
        return ["/".join(val) for val in itertools.product(*args)]

    @staticmethod
    def get_hdf_names(arg):
        """Gets the group names in full path format"""
        names = []
        if isinstance(arg, h5py.File):
            arg.visititems(
                lambda n, obj: names.append(n)
                if isinstance(obj, h5py.Group)
                and any([isinstance(m, h5py.Dataset) for m in obj.values()])
                else None
            )
        elif isinstance(arg, (str, Path)):
            with h5py.File(Path(arg), "a") as hf:
                hf.visititems(
                    lambda n, obj: names.append(n)
                    if isinstance(obj, h5py.Group)
                    and any([isinstance(m, h5py.Dataset) for m in obj.values()])
                    else None
                )
        return names


# %% Testing
if __name__ == "__main__":
    from research_tools.functions import f_find, p_find
    from eis_analysis import DataImport
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

    uni_bands = Bounds([1e-7, 1e-2, 1e-16, 1, 1e-12, 0.75, 15, 1e-12, 0.5],
                       [5e-6, 10, 1e-8, 5e5, 1e-3, 1, 200, 1e3, 1],
                       keep_feasible=True)
    ls_kwargs = dict(ftol=1e-14, xtol=1e-6, maxfev=1e6, jac="3-point", x_scale="jac", bounds=uni_bands)
    # names_all = names_base+names_base_r2+names_hot_base+names_hot_insitu

    # Import the data from the raw files exported from the meeasurement tool
    # This provides the 1st filter, "re_filter" which will only get the filenames which match
    # the filter name.

    my_hdf_path = p_find("Dropbox (ASU)", "Work Docs", "Data", "Analysis", "HDFs", base="home")
    my_folder_path = p_find("Dropbox (ASU)", "Work Docs", "Data", "Raw", "MFIA", base="home")

    files = f_find(my_folder_path, re_filter="polar")

    # Create an object to operate on all of the available data.  This will also save the
    # data into an hdf for persistant storage
    test_group = IS_Data("polarized_tc",
                       my_hdf_path,
                       model=ckt_model,
                       init_pos=init_position,
                       raw_files=files,
                       tool="MFIA",
                       )

    # Data can also be loaded after initialization directly via get_raw
    test_group.get_raw(
        f_find(my_folder_path, re_filter="polar"),
        tool="MFIA",
        )

    # This will run the fitting function
    test_group.base_fitter("ntc_postpid_r1", thresh=1, refit=False, bounds_by_conf=False, **ls_kwargs)
    
    
