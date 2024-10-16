# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
import numpy as np
import pandas as pd

# from collections.abc import Mapping
from dataclasses import (
    dataclass,
    InitVar,
)


from ..string_operations import (
    str_in_list,
    eng_not,
)

def hz_label(
    data,
    test_arr=None,
    prec=2,
    kind="eng",
    space=" ",
    postfix="Hz",
    label_rc=True,
    targ_col="frequency",
    test_col="imag",
    new_col="flabel",
):
    """
    Generates frequency labels for MFIA data.

    This function creates a new column in the provided DataFrame or processes a numpy array
    to generate frequency labels based on the specified parameters. It uses the target column
    for frequency values and the test column for additional calculations.

    Parameters:
    data (pd.DataFrame or np.ndarray): The data to process. If a DataFrame, it should contain
                                       the target and test columns.
    test_arr (np.ndarray, optional): An array for additional calculations. If not provided,
                                     it is computed using a moving average of the test column.
    prec (int, optional): The precision for the frequency labels. Default is 2.
    kind (str, optional): The format kind for the labels ('eng' for engineering notation). Default is "eng".
    space (str, optional): The space between the number and the postfix. Default is " ".
    postfix (str, optional): The postfix for the frequency labels. Default is "Hz".
    label_rc (bool, optional): If True, labels are generated in reverse order. Default is True.
    targ_col (str, optional): The name of the target column for frequency values in the DataFrame. Default is "frequency".
    test_col (str, optional): The name of the test column for additional calculations in the DataFrame. Default is "imag".
    new_col (str, optional): The name of the new column to store the generated labels in the DataFrame. Default is "flabel".

    Returns:
    pd.DataFrame or np.ndarray: The DataFrame with the new column of frequency labels, or the processed numpy array.
    """
    if isinstance(data, pd.DataFrame):
        targ_col = str_in_list(targ_col, data.columns)
        test_col = str_in_list(test_col, data.columns)
        if targ_col not in data.columns or test_col not in data.columns:
            return data
        data[new_col] = hz_label(
            data[targ_col].to_numpy(),
            test_arr=moving_average(-1 * data[test_col].to_numpy(), 5, True),
            prec=prec,
            kind=kind,
            space=space,
            postfix=postfix,
            label_rc=label_rc,
        )
        return data
    # if isinstance(data, pd.Series)
    base = [float(10 ** (np.floor(np.log10(a)))) if a > 0 else 0 for a in data]
    base_diff = np.diff(base)

    res = [np.nan] * len(data)

    for n, value in enumerate(data):
        if value == 0:
            continue
        elif n == 0 or base_diff[n - 1] != 0:
            res[n] = str(eng_not(base[n], 0, kind, space)) + postfix
            if (
                label_rc
                and isinstance(test_arr, (list, np.ndarray))
                and test_arr[n] == max(abs(np.array(test_arr)))
            ):
                res[n] = res[n] + " (RC)"
        elif (
            label_rc
            and isinstance(test_arr, (list, np.ndarray))
            and test_arr[n] == max(abs(np.array(test_arr)))
        ):
            try:
                if len(kind) > 2 and "exp" in kind.lower():
                    res[n] = "RC (f=" + eng_not(data[n], prec, "eng", " ") + "Hz)"
                else:
                    res[n] = "RC (f=" + eng_not(data[n], prec, kind, space) + "Hz)"
            except TypeError:
                return res
    return res

def extendspace(start, stop, num=50, ext=0, logscale=True, as_exp=False):
    if logscale:
        start = np.log10(start)
        stop = np.log10(stop)

    delta = np.diff(np.linspace(start, stop, num)).mean()

    new_start = start - delta * ext
    new_stop = stop + delta * ext

    if logscale and not as_exp:
        return 10**new_start, 10**new_stop, int(num + 2 * ext)

    return new_start, new_stop, int(num + 2 * ext)


def range_maker(start, stop, points_per_decade=24, ext=0, is_exp=False):
    if not is_exp:
        start = np.log10(start)
        stop = np.log10(stop)
    count = int(1 + points_per_decade * abs(start - stop))
    start, stop, count = extendspace(start, stop, count, ext, False, True)
    return {"start": 10**start, "stop": 10**stop, "samplecount": count}


def most_frequent(arg):
    """
    Finds the most frequent element in an array.

    This function takes an array-like input and returns the most frequently occurring
    element. If there are multiple elements with the same highest frequency, it returns
    the first one encountered.

    Parameters:
    arg (array-like): The input array to analyze.

    Returns:
    int: The most frequent element in the array.
    """
    unique, counts = np.unique(arg, return_counts=True)
    index = np.argmax(counts)
    return int(unique[index])


def moving_average(arr, w=2, logscale=False):
    """
    Computes the moving average of an array.

    This function calculates the moving average of the input array `arr` with a specified
    window size `w`. If `logscale` is True, the logarithm (base 10) of the array values is
    used for the calculation. The function handles edge cases by adjusting the window size
    and returns the result as a list of floats.

    Parameters:
    arr (list or np.ndarray): The input array to compute the moving average for.
    w (int, optional): The window size for the moving average. Default is 2.
    logscale (bool, optional): If True, computes the moving average on the logarithm (base 10)
                               of the array values. Default is False.

    Returns:
    list of float: The moving average of the input array.
    """
    if logscale:
        arr = np.log10([a if a > 0 else 1e-30 for a in arr])
    res = list(np.convolve(arr, np.ones(w), "valid") / w)
    w -= 1
    while w >= 1:
        if w % 2:
            res.append((np.convolve(arr, np.ones(w), "valid") / w)[-1])
        else:
            res.insert(0, (np.convolve(arr, np.ones(w), "valid") / w)[0])
        w -= 1
    if logscale:
        return [float(10**f) for f in res]
    return [float(f) for f in res]


def insert_inverse_col(df, name):
    """
    Inserts a new column in the DataFrame with the inverse of the specified column.

    This function searches for a column in the DataFrame that matches the given name
    (or a close match if the exact name is not found). It then creates a new column
    with the inverse values of the specified column and inserts it immediately after
    the original column.

    Parameters:
    df (pd.DataFrame): The DataFrame to modify.
    name (str): The name of the column to invert.

    Returns:
    pd.DataFrame: The modified DataFrame with the new inverse column inserted.
    """
    name = str_in_list(name, df.columns)
    if name in df.columns:
        if isinstance(name, tuple):
            new_name = tuple(
                [
                    name[n] if n < len(name) - 1 else "inv_" + name[n]
                    for n in range(len(name))
                ]
            )
        else:
            new_name = "inv_" + name
        df.insert(df.columns.get_loc(name) + 1, new_name, -1 * df[name])
    return df


def modify_sub_dfs(data, *functions):
    """
    Applies a series of functions to a DataFrame or nested DataFrames within a dictionary.

    This function takes a DataFrame or a dictionary containing DataFrames and applies
    a series of functions to each DataFrame. Each function can be provided as a callable
    or as a tuple containing a callable and its arguments. The function modifies the
    DataFrame in place and returns the modified DataFrame.

    Parameters:
    data (pd.DataFrame or dict): The DataFrame or dictionary of DataFrames to modify.
    *functions (callable or tuple): A series of functions or tuples of functions and their arguments
                                to apply to the DataFrame(s).

    Returns:
    pd.DataFrame or dict: The modified DataFrame or dictionary of DataFrames.
    """
    if isinstance(data, pd.DataFrame):
        for f in functions:
            if isinstance(f, (tuple, list)):
                if len(f) == 1:
                    res = f[0](data)
                elif len(f) == 2:
                    res = (
                        f[0](data, **f[1])
                        if isinstance(f[1], dict)
                        else f[0](data, f[1])
                    )
                else:
                    res = f[0](data, *f[1:])
            else:
                res = f(data)
            if res is not None:
                data = res
        return data
    elif isinstance(data, dict):
        return {k: modify_sub_dfs(d, *functions) for k, d in data.items()}
    elif isinstance(data, (list, tuple)):
        return [modify_sub_dfs(d, *functions) for d in data]

    return data


@dataclass
class Complexer(object):
    """Calculate. generic discription."""

    data: InitVar[np.ndarray] = np.ndarray(0)
    name: str = "Z"

    def __post_init__(self, data):
        """Calculate. generic discription."""
        self.array = data

    # def __getitem__(self, item):
    #     """Return sum of squared errors (pred vs actual)."""
    #     if hasattr(self, item.upper()):
    #         return getattr(self, item.upper())
    #     elif hasattr(self, item.lower()):
    #         return getattr(self, item.lower())
    #     else:
    #         return None

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
                        np.cos(self._array[:, 1])
                        + 1j * np.sin(self._array[:, 1])
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
        vals = [
            self.real,
            self.imag,
            -1 * self.imag,
            self.mag,
            self.phase,
            -1 * self.phase,
        ]
        columns = ["real", "imag", "inv_imag", "mag", "phase", "inv_phase"]
        # self._data = pd.DataFrame(dict(zip(columns, vals)))
        return pd.DataFrame(dict(zip(columns, vals)))

    @df.setter
    def df(self, _):
        pass


@dataclass
class Complex_Imp(Complexer):
    # data: InitVar[np.ndarray] = np.ndarray(0)
    # name: str = "Z"

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
        else:
            return None

    @property
    def Z(self):
        """Calculate. generic discription."""
        return self.array

    @Z.setter
    def Z(self, _):
        pass

    @property
    def R(self):
        """Calculate. generic discription."""
        return self.Z.real

    @R.setter
    def R(self, _):
        pass

    @property
    def X(self):
        """Calculate. generic discription."""
        return self.Z.imag

    @X.setter
    def X(self, _):
        pass

    @property
    def Y(self):
        """Calculate. generic discription."""
        return 1 / self.array

    @Y.setter
    def Y(self, _):
        pass

    @property
    def G(self):
        """Calculate. generic discription."""
        return self.Y.real

    @G.setter
    def G(self, _):
        pass

    @property
    def B(self):
        """Calculate. generic discription."""
        return self.Y.imag

    @B.setter
    def B(self, _):
        pass

@dataclass
class ComplexSystem(Complex_Imp):
    thickness: float = 1.0
    area: float = 1.0

    def __post_init__(self, data):
        if isinstance(data, pd.DataFrame):
            if "freq" in data.columns:
                self.frequency = data["freq"].values
                data = data.drop(columns=["freq"])
        elif isinstance(data, np.ndarray):
            if data.ndim == 2:
                if np.all(data[:, 0] >= 0) and not np.iscomplexobj(data[:, 0]):
                    self.frequency = data[:, 0]
                    data = data[:, 1:]
                elif np.all(data[:, -1] >= 0) and not np.iscomplexobj(data[:, -1]):
                    self.frequency = data[:, -1]
                    data = data[:, :-1]
        super().__post_init__(data)
        # Additional initialization if needed

    def __getitem__(self, item):
        raise TypeError(f"'{self.__class__.__name__}' object is not subscriptable")

    @property
    def omega(self):
        """Calculate. generic discription."""
        return 2*np.pi*self.frequency
    
    @omega.setter
    def omega(self, _):
        pass

    @property
    def A_D(self):
        """Calculate. generic discription."""
        return self.area/self.thickness
    
    @A_D.setter
    def A_D(self, _):
        pass

    @property
    def e_0(self):
        """Calculate. generic discription. """
        return 8.85418782e-14 # F/cm
    
    @e_0.setter
    def e_0(self, _):
        pass

    @property
    def C_0(self):
        """Calculate. generic discription. """
        return self.e_0*self.A_D
    
    @C_0.setter
    def C_0(self, _):
        pass
    
    @property
    def M(self):
        """Calculate the modulus. (M+jM) generic discription."""
        return (self.array * self.omega * self.C_0 * 1j)
        # return (self.array * self.omega * self.A_D * 1j)

    @M.setter
    def M(self, _):
        pass
    
    @property
    def permittivity(self):
        """Calculate the complex permittivity. (e-je) generic discription. """
        return 1/self.M * (self.e_0)  
    
    @permittivity.setter
    def permittivity(self, _):
        pass

    @property
    def permittivity_r(self):
        """Calculate the complex permittivity. (e-je) generic discription. """
        return 1/self.M
    
    @permittivity_r.setter
    def permittivity_r(self, _):
        pass

    @property
    def conductivity(self):
        """Calculate complex conductivity. (sigma + jsigma) generic discription."""
        return (self.Y / self.A_D)

    @conductivity.setter
    def conductivity(self, _):
        pass

    @property
    def resistivity(self):
        """Calculate complex resistivity. (rho - jrho) generic discription."""
        return (1 / self.conductivity)

    @resistivity.setter
    def resistivity(self, _):
        pass

    @property
    def loss_factor(self):
        """Calculate complex resistivity. (rho - jrho) generic discription."""
        return -1* self.permittivity_r.imag

    @loss_factor.setter
    def loss_factor(self, _):
        pass

    @property
    def loss_tangent(self):
        """Calculate complex resistivity. (rho - jrho) generic discription."""
        return -1* self.permittivity_r.imag/self.permittivity_r.real

    @loss_tangent.setter
    def loss_tangent(self, _):
        pass

    def get_real(self, complex_str: str) -> float:
        """Get the real part of the complex value."""
        complex_value = getattr(self, complex_str)
        return complex_value.real

    def get_imag(self, complex_str: str) -> float:
        """Get the imaginary part of the complex value."""
        complex_value = getattr(self, complex_str)
        sign = -1 if "resis" in complex_str or "perm" in complex_str else 1
        return sign * complex_value.imag

    def get_mag(self, complex_str: str) -> float:
        """Get the magnitude of the complex value."""
        complex_value = getattr(self, complex_str)
        return np.abs(complex_value)

    def get_phase(self, complex_str: str) -> float:
        """Get the phase of the complex value in degrees."""
        complex_value = getattr(self, complex_str)
        return np.angle(complex_value, deg=True)

    def get_target_df(self, complex_str: str, x_axis: str = None) -> pd.DataFrame:
        """Create a DataFrame with specified complex value components."""
        complex_value = getattr(self, complex_str)
        sign = -1 if "resis" in complex_str or "perm" in complex_str else 1

        data = {
            "real": complex_value.real,
            "imag": sign * complex_value.imag,
            "inv_imag": -1 * sign * complex_value.imag,
            "mag": np.abs(complex_value),
            "phase": np.angle(complex_value, deg=True),
            "inv_phase": -1 * np.angle(complex_value, deg=True),
        }

        df = pd.DataFrame(data)

        if x_axis:
            if "freq" in x_axis:
                x_data = self.frequency
                x_name = "freq"
            elif "omega" in x_axis:
                x_data = self.omega
                x_name = "omega"
            else:
                raise ValueError("x_axis must contain 'freq' or 'omega'")

            if "log" in x_axis:
                x_data = np.log10(x_data)

            df.insert(0, x_name, x_data)
            # df.rename(columns={"x_axis": x_axis}, inplace=True)

        return df
    
    def supplement_df(self) -> pd.DataFrame:
        """Create a DataFrame with residuals: freq, omega, loss_factor, and loss_tangent."""
        data = {
            "freq": self.frequency,
            "omega": self.omega,
            "loss_factor": self.loss_factor,
            "loss_tangent": self.loss_tangent,
        }

        return pd.DataFrame(data)