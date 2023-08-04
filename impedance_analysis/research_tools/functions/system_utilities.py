# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
import re
import os
import h5py
import dill
import difflib
import numpy as np
import pandas as pd
import unicodedata
import configparser

from pathlib import Path
from datetime import datetime as dt
from inspect import getmembers

from research_tools.functions.data_treatment import dict_df, dict_flat, dict_key_sep

# warnings.simplefilter("ignore", np.RankWarning)
# warnings.filterwarnings("ignore")


# %% Path resolving functions
def pathify(*dir_in, target=None):
    cwd = os.getcwd()
    sep = os.sep
    if len(dir_in) == 0:
        top_dir = "work"
        sub_dir = "python"
        folder = "python"
    elif len(dir_in) == 1:
        if sep not in dir_in[0]:
            return cwd
        top_dir = dir_in[0].split(sep)[0]
        sub_dir = dir_in[0].split(sep)[1]
        folder = sep.join(dir_in[0].split(sep)[1:])
    elif len(dir_in) == 2:
        top_dir = dir_in[0]
        sub_dir = dir_in[1].split(sep)[0]
        folder = dir_in[1]
    else:
        top_dir = dir_in[0]
        sub_dir = dir_in[1]
        folder = sep.join(dir_in[1:])

    top_names = ["top_dir", "directory", "parent", "path"]
    sub_names = ["sub_dir", "sub_directory", "child", "sub_path"]
    folder_names = ["folder", "fold", "target"]
    if target is None or target.lower() in folder_names:
        target = "target"
    elif target.lower() in top_names:
        target = top_dir
    elif target.lower() in sub_names:
        target = sub_dir

    if target == "cwd":
        return cwd

    result = {"cwd": cwd, top_dir: None, sub_dir: None, "target": None}

    root_dir = None
    for n in range(len(cwd.split(sep)) - 1):
        if top_dir.lower() in cwd.split(sep)[n].lower() and root_dir is None:
            root_dir = sep.join(cwd.split(sep)[:n])
        if (
            top_dir.lower() in cwd.split(sep)[n].lower()
            and sub_dir == cwd.split(sep)[n + 1]
        ):
            result[top_dir] = sep.join(cwd.split(sep)[: n + 1])
            result[sub_dir] = sep.join(cwd.split(sep)[: n + 2])
            if sep in folder:
                # folder = result[sub_dir] + sep + sep.join(folder.split(sep)[1:])
                folder = sep.join((result[sub_dir], *folder.split(sep)[1:]))
            else:
                folder = result[sub_dir]
            result["target"] = folder

    if root_dir is None:
        root_dir = cwd

    if result[top_dir] is None:
        exact_top = []
        exact_sub = []
        approx_top = []
        approx_sub = []
        exact = False
        for dirpaths, dirnames, files in os.walk(os.path.abspath(root_dir)):
            exact_dir = np.array(dirnames)[[sub_dir == item for item in dirnames]]
            approx_dir = np.array(dirnames)[
                [sub_dir.lower() in item.lower() for item in dirnames]
            ]
            if (
                top_dir.lower() in dirpaths.split(sep)[-1].lower()
                and len(exact_dir) > 0
            ):
                exact_top.append(dirpaths)
                exact_sub.append(exact_dir.tolist())
                exact = True
            elif (
                top_dir.lower() in dirpaths.split(sep)[-1].lower()
                and len(approx_dir) > 0
                and not exact
            ):
                approx_top.append(dirpaths)
                approx_sub.append(approx_dir.tolist())
        if exact:
            pathlist = exact_top
            dirlist = exact_sub
        else:
            pathlist = approx_top
            dirlist = approx_sub

        if len(pathlist) != 1:
            result[sub_dir] = sep.join((cwd, sub_dir))
            result["target"] = sep.join((cwd, folder))
        else:
            result[top_dir] = pathlist[0]
            if len(dirlist[0]) != 1:
                result[sub_dir] = sep.join((pathlist[0], sub_dir))
                result["target"] = sep.join((pathlist[0], folder))
            else:
                result[sub_dir] = sep.join((pathlist[0], dirlist[0][0]))
                if sep in folder:
                    folder = sep.join((dirlist[0][0], *folder.split(sep)[1:]))
                else:
                    folder = dirlist[0][0]
                result["target"] = sep.join((pathlist[0], folder))

    try:
        return result[target]
    except KeyError:
        return result


def p_find(*dir_in, as_list=False, **kwargs):
    if len(dir_in) == 1 and isinstance(dir_in[0], (list, np.ndarray)):
        dir_in = list(dir_in[0])

    if Path(*dir_in).exists():
        return Path(*dir_in)

    if isinstance(dir_in, Path):
        dir_in = dir_in[0].parts

    if as_list:
        return [p_find(d, **kwargs) for d in dir_in]

    base_path = kwargs.get("base", Path.home() / "Documents")
    if isinstance(base_path, str) and base_path in ["cwd", "home"]:
        base_path = getattr(Path, base_path)()
    if base_path is None:
        return Path(pathify(*dir_in, target=kwargs.get("target", None)))

    if not base_path.exists() and Path("D:\\", *base_path.parts[1:]).exists():
        base_path = Path("D:\\", *base_path.parts[1:])

    if len(dir_in) >= 1 and Path(*dir_in).parts[0] in base_path.parts:
        for b in base_path.parents:
            if dir_in[0] not in b.parts:
                base_path = b
                break

    dir_path = base_path / Path(*dir_in)

    if not dir_path.exists() and Path("D:\\", *dir_path.parts[1:]).exists():
        dir_path = Path("D:\\", *dir_path.parts[1:])

    if not dir_path.exists():
        filesurvey = []
        for row in os.walk(base_path):  # Walks through current path
            for foldname in row[1]:  # row[2] is the file name
                full_path: Path = Path(row[0]) / Path(
                    foldname
                )  # row[0] ist der Ordnerpfad
                if (full_path / Path(*dir_in)).exists():
                    filesurvey.append(full_path / Path(*dir_in))

        dir_path = filesurvey[0]
        for f in filesurvey:
            dir_path = f if len(f.parts) < len(dir_path.parts) else dir_path

    if not dir_path.exists():
        for p in dir_path.parents:
            if len(p.parts) >= len(Path.home().parts):
                print("pathified")
                return Path(
                    pathify(*Path(*dir_in).parts, target=kwargs.get("target", None))
                )
            elif p.exists():
                break

    return dir_path


def f_find(path, search=False, res_type="path", re_filter=None):
    path = Path(path)
    if search:
        res = f_find(path.parent)
        return [r for r in res if r.parent == path.parent and r.stem == path.stem][0]

    filesurvey = []
    for row in os.walk(Path(path)):  # Walks through current path
        for filename in row[
            2
        ]:  # row[2] is the file name re.search(r"(.h5|.hdf5)$", str(file))
            full_path: Path = Path(row[0]) / Path(filename)  # row[0]
            if re_filter is None or re.search(re_filter, full_path.name):
                # if re_filter is None or re_filter in full_path.name:
                filesurvey.append(
                    dict(
                        path=full_path,
                        file=filename,
                        date=full_path.stat().st_mtime,
                        size=full_path.stat().st_size,
                    )
                )

    if res_type == "all" or filesurvey == []:
        return filesurvey
    else:
        try:
            return [f[res_type] for f in filesurvey]
        except KeyError:
            return filesurvey


def pathlib_mk(dir_path):
    dir_file = ""

    if dir_path.suffix != "":
        dir_file = dir_path.name
        dir_path = dir_path.parent

    if not dir_path.exists():
        if (
            dir_path.parent.exists()
            or dir_path.parent.parent.exists()
            or dir_path.parent.parent.parent.exists()
        ):
            dir_path.mkdir(parents=True)
        elif (Path.home() / "Desktop").exists():
            dir_path = Path.home() / "Desktop"
        else:
            dir_path = Path.home()

    if dir_file != "":
        dir_path.touch()

    return dir_path / dir_file


# %% File I/O functions
def slugify(value, allow_unicode=False, sep="-"):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py.

    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores. Replace whitespace with desired
    separator such as '-', '_', or ' '.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    # return re.sub(r"[-\s]+", "-", value).strip("-_")
    return re.sub(r"[-\s]+", sep, value).strip("-_")


def save(data, path=None, name=None, ftype="xls"):
    """Save data into excel file."""
    if isinstance(path, Path):
        path = str(path)
    if path is None:
        path = pathify(
            "work", "Data", "Analysis", "Autosave", dt.now().strftime("%Y%m%d")
        )
    if name is None:
        name = "data_" + dt.now().strftime("%H_%M")
    if not os.path.exists(path):
        os.makedirs(path)

    if isinstance(data, (list, np.ndarray)):
        if isinstance(data[0], (pd.DataFrame, pd.Series)):
            data = {x: data[x] for x in range(len(data))}
        else:
            data = pd.DataFrame(data)

    if isinstance(data, (dict)):
        if not isinstance(data[list(data.keys())[0]], (pd.DataFrame, pd.Series)):
            data = pd.DataFrame(data)

    if isinstance(data, (pd.DataFrame, pd.Series)) and "xls" in ftype.lower():
        data.to_excel(os.sep.join((path, f"{slugify(name)}.xlsx")), merge_cells=False)
    elif isinstance(data, (dict)) and "xls" in ftype.lower():
        with pd.ExcelWriter(os.sep.join((path, f"{slugify(name)}.xlsx"))) as writer:
            for key, df in data.items():
                df.to_excel(writer, sheet_name=key, merge_cells=False)
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        data.to_csv(os.sep.join((path, f"{slugify(name)}.{ftype}")), index=False)
    elif isinstance(data, (dict)):
        for key, df in data.items():
            df.to_csv(
                os.sep.join((path, f"{slugify(name)}_{key}.{ftype}")), index=False
            )


def load(file, path=None, pdkwargs={}, hdfkwargs={}, **kwargs):
    """
    Loads data from excel or hdf5
    kwargs:
        flat_df
        target
    """
    file = Path(file)
    if isinstance(path, list):
        path = pathify(path)
    if isinstance(path, str):
        path = Path(path)
    if path is not None:
        file = path / file

    data = {}
    attrs = {}

    if re.search(r"(.xls|.xls\w)$", str(file)):
        names = pd.ExcelFile(file).sheet_names
        data = pd.read_excel(
            file,
            sheet_name=names,
            header=pdkwargs.pop("header", None),
            **pdkwargs,
        )

    elif re.search(r"(.h5|.hdf5)$", str(file)):
        data, attrs = load_hdf(
            file,
            None,
            kwargs.get("target", "/"),
            kwargs.get("key_sep", True),
            **hdfkwargs,
        )
        if kwargs.get("flat_df", False):
            data = dict_df(dict_flat(data))
    elif file.exists() and file.is_dir():
        filelist = f_find(
            file, re_filter=kwargs.get("file_filter", kwargs.get("re_filter", ""))
        )
        return [load(f, None, pdkwargs, hdfkwargs, **kwargs) for f in filelist]
    return data, attrs


def load_hdf(file, path=None, target="/", key_sep=False, **kwargs):
    if isinstance(path, list):
        path = pathify(path)
    if isinstance(path, Path):
        path = str(path)
    if path is not None:
        file = os.sep.join((path, file))

    def get_ds_dictionaries(name, node):
        if target in name:
            if isinstance(node, h5py.Dataset):
                ds_dict[node.name] = np.array(node[()])
            if any(node.attrs):
                for key, val in node.attrs.items():
                    attr_dict[node.name + "/" + key] = val

    with h5py.File(file, "r") as hf:
        ds_dict = {}
        attr_dict = {}
        hf.visititems(get_ds_dictionaries, **kwargs)
    if key_sep:
        return dict_key_sep(ds_dict), dict_key_sep(attr_dict)
    return ds_dict, attr_dict


def get_config(file, sections=["base"], **kwargs):
    """
    Get the necessary information from a configuration .ini file.

    Parameters
    ----------
    file : [str, Path]
        The path to the .ini file containing the configuration settings to be imported. If file is
        None, any kwargs are returned as the settings. This is only usefull for external functions
        which may pass None for other reasons.
    sections : list, optional
        Defines what sections of the .ini to import. If "all" is passed, function will create a
        dict of dicts, separating each section into its own dict of settings.
        If no section match, attempts to find sections which include the values provided by the
        list. If there are still no matches, the first section will be called and returned.
    kwargs : function, optional
        Pass a function for plotting the data. Function must be limited to single data input
        of type dict or dataframe with keys appropriate to the native output of the sweeper obj.
        i.e. "frequency", "realz", "imagz", "absz", "phasez", "param0", or "param1"
        Default : None

    Returns
    -------
    config_file : dict
        Returns a dict containing all settings imported from the .ini file
    """
    if file is None:
        return kwargs
    cp = configparser.ConfigParser()
    cp.optionxform = lambda option: option
    # Load the configuration file
    if Path(file).is_file():
        cp.read(Path(file))
    elif (Path.cwd() / file).is_file():
        cp.read(Path.cwd() / file)
    elif (Path(file) / "config.ini").is_file():
        cp.read(Path(file) / "config.ini")
    else:
        cp.read(Path.cwd() / "config.ini")

    if isinstance(sections, str):
        if sections == "all":
            config_file = {}
            for sec in cp.sections():
                config_file[sec] = get_config(file, sections=[sec])
                config_file[sec] = {
                    k: kwargs.get(k, v) for k, v in config_file[sec].items()
                }
            return config_file
        sections = [sections]

    checked_sec = [s_in for s_in in sections if s_in in cp.sections()]
    if checked_sec == []:
        checked_sec = [
            s_file
            for s_in in sections
            for s_file in cp.sections()
            if s_in.lower() in s_file.lower()
        ]
    if checked_sec == []:
        checked_sec = [cp.sections()[0]]
    config_file = {
        k: eval(v)
        if not bool(re.findall("[a-df-zA-DF-Z\\\\@!&^]|^[/eE]|[/eE]$|^\\..+\\.$", v))
        else v
        for sc in checked_sec
        for k, v in dict(cp.items(sc)).items()
    }
    included = []
    if "include_sections" in config_file.keys():
        included = [
            get_config(file, [sec])
            for sec in config_file.pop("include_sections").split(", ")
        ]
        for inc in included:
            config_file = {**config_file, **inc}

    return {**config_file, **kwargs}


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
        if isinstance(name, (int, np.integer, float)) and int(name) < len(
            self.database
        ):
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
            isinstance(value, (tuple, list, np.ndarray, pd.Series))
            and len(np.array(value)[0]) == 2
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


if __name__ == "__main__":
    from inspect import getmembers
