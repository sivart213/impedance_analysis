# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
import re
from dataclasses import dataclass, InitVar

import numpy as np
import pandas as pd

from .data_ops import convert_val

@dataclass
class Complexer(object):
    """Calculate. generic discription."""

    data: InitVar[np.ndarray] = np.array(0)
    name: str = "Z"
    sign: int = 1
    long_name: str = "impedance"
    latex: str = "$Z$"
    units: str = "$\Omega$"

    def __post_init__(self, data):
        """Calculate. generic discription."""
        self.array = data

    def __add__(self, other):
        if isinstance(other, Complexer):
            return Complexer(self.array + other.array)
        return Complexer(self.array + other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Complexer):
            return Complexer(self.array - other.array)
        return Complexer(self.array - other)

    def __rsub__(self, other):
        if isinstance(other, Complexer):
            return Complexer(other.array - self.array)
        return Complexer(other - self.array)

    def __mul__(self, other):
        if isinstance(other, Complexer):
            return Complexer(self.array * other.array)
        return Complexer(self.array * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Complexer):
            return Complexer(self.array / other.array)
        return Complexer(self.array / other)

    def __rtruediv__(self, other):
        if isinstance(other, Complexer):
            return Complexer(other.array / self.array)
        return Complexer(other / self.array)

    def __pow__(self, power, modulo=None):
        return Complexer(self.array**power)

    def __neg__(self):
        return Complexer(-self.array)

    def __abs__(self):
        return Complexer(abs(self.array))

    def __eq__(self, other):
        if isinstance(other, Complexer):
            return np.array_equal(self.array, other.array)
        return np.array_equal(self.array, other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return len(self._array)

    def __iter__(self):
        return iter(self.array)

    def __getitem__(self, index):
        """Allow slicing and indexing."""
        if isinstance(index, str) and hasattr(self, index):
            return getattr(self, index)
        return self.array[index]

    def __repr__(self):
        """Custom repr to include self.array."""
        return f"{self.__class__.__name__}({self._array})"

    @property
    def array(self):
        """Calculate. generic discription."""
        return self._array

    @array.setter
    def array(self, arr):
        if isinstance(arr, np.ndarray):
            self._array = arr
        if isinstance(arr, type(self)):
            self._array = arr.array
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

        self._sign = (
            1 if np.sum(self.sign * self._array.imag > 0) > len(self._array) / 2 else -1
        )

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
        return self.sign * self.array.imag

    @imag.setter
    def imag(self, _):
        pass

    @property
    def pos_imag(self):
        """Calculate. generic discription."""
        return self._sign * self.sign * self.array.imag

    @pos_imag.setter
    def pos_imag(self, _):
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
        return self.sign * np.angle(self.array, deg=True)

    @phase.setter
    def phase(self, _):
        pass

    @property
    def tan(self):
        """Calculate. generic discription."""
        return self.sign * np.tan(np.angle(self.array, deg=False))

    @tan.setter
    def tan(self, _):
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
            self.tan,
            -1 * self.tan,
        ]
        columns = [
            "real",
            "imag",
            "inv_imag",
            "mag",
            "phase",
            "inv_phase",
            "tan",
            "inv_tan",
        ]
        # self._data = pd.DataFrame(dict(zip(columns, vals)))
        return pd.DataFrame(dict(zip(columns, vals)))

    @df.setter
    def df(self, _):
        pass


class ComplexSystem:
    aliases = {
        "freq": "frequency",
        "w": "angular_frequency",
        "ω": "angular_frequency",
        "omega": "angular_frequency",
        "e_0": "permittivity_constant",
        "ε_0": "permittivity_constant",
        "ε₀": "permittivity_constant",
        "a_d": "area_over_thickness",
        "c_0": "characteristic_capacitance",
        "c₀": "characteristic_capacitance",
        "vacuum_capacitance": "characteristic_capacitance",
        "z": "impedance",
        "y": "admittance",
        "m": "modulus",
        "e": "permittivity",
        "ε": "permittivity",
        "epsilon": "permittivity",
        "perm": "permittivity",
        "e_r": "relative_permittivity",
        "ε_r": "relative_permittivity",
        "εᵣ": "relative_permittivity",
        "epsilon_r": "relative_permittivity",
        "perm_r": "relative_permittivity",
        "permittivity_r": "relative_permittivity",
        "cond": "conductivity",
        "sigma": "conductivity",
        "resis": "resistivity",
        "rho": "resistivity",
        "ρ": "resistivity",
    }

    def __init__(self, data, frequency=None, thickness=1.0, area=1.0):
        self.thickness = thickness
        self.area = area

        if frequency is not None:
            self.frequency = frequency
        elif isinstance(data, pd.DataFrame):
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
        else:
            self.frequency = np.ones(len(data))

        self.complexer = Complexer(data)

    def __setattr__(self, name, value):
        name = self.aliases.get(name, name)
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name == "aliases":
            raise AttributeError
        name = self.aliases.get(name.lower(), name)
        return object.__getattribute__(self, name)

    def __getitem__(self, index):
        """Allow slicing and indexing."""
        if isinstance(index, str):
            if hasattr(self, index):
                return getattr(self, index)
            elif hasattr(self.complexer, index):
                return getattr(self.complexer, index)
            elif "." in index or "/" in index:
                param, part = re.split(r"[./]", index)
                if hasattr(self, param):
                    complexer_obj = getattr(self, param)
                    if hasattr(complexer_obj, part):
                        return getattr(complexer_obj, part)
                    else:
                        raise AttributeError(
                            f"'{param}' object has no attribute '{part}'"
                        )
                else:
                    raise AttributeError(
                        f"'{self.__class__.__name__}' object has no attribute '{param}'"
                    )
        else:
            raise TypeError("Index must be a string")

    @property
    def thickness(self):
        """Calculate. generic discription."""
        return self._thickness

    @thickness.setter
    def thickness(self, value):
        if isinstance(value, str):
            self._thickness = convert_val(value, funit="cm")
        elif isinstance(value, (tuple, list)):
            self._thickness = convert_val(*value, funit="cm")
        elif isinstance(value, dict):
            self._thickness = convert_val(**value, funit="cm")
        elif isinstance(value, (int, float)):
            self._thickness = value

    @property
    def area(self):
        """Calculate. generic discription."""
        return self._area

    @area.setter
    def area(self, value):
        if isinstance(value, str):
            self._area = convert_val(value, funit="cm", expon=1)
        elif isinstance(value, (tuple, list)):
            self._area = convert_val(*value, funit="cm", expon=1)
        elif isinstance(value, dict):
            self._area = convert_val(**value, funit="cm", expon=1)
        elif isinstance(value, (int, float)):
            self._area = value

    @property
    def array(self):
        return self.complexer.array

    @property
    def df(self):
        return self.complexer.df

    @property
    def angular_frequency(self):
        """Calculate. generic discription."""
        return 2 * np.pi * self.frequency

    @property
    def area_over_thickness(self):
        """Calculate. generic discription."""
        return self.area / self.thickness

    @property
    def permittivity_constant(self):
        """Calculate. generic discription."""
        return 8.85418782e-14  # F/cm

    @property
    def characteristic_capacitance(self):
        """Calculate. generic discription."""
        return self.e_0 * self.a_d

    @property
    def impedance(self):
        """Calculate. generic discription."""
        return Complexer(self.array)

    @property
    def admittance(self):
        """Calculate. generic discription."""
        arr = 1 / self.array
        return Complexer(arr)

    @property
    def modulus(self):
        """Calculate the modulus. (M+jM) generic discription."""
        arr = self.array * self.angular_frequency * self.C_0 * 1j
        return Complexer(arr)

    @property
    def permittivity(self):
        """Calculate the complex permittivity. (e-je) generic discription."""
        arr = 1 / self.M * self.e_0
        return Complexer(arr, sign=-1)

    @property
    def relative_permittivity(self):
        """Calculate the complex permittivity. (e-je) generic discription."""
        arr = 1 / self.M
        return Complexer(arr, sign=-1)

    @property
    def conductivity(self):
        """Calculate complex conductivity. (sigma + jsigma) generic discription."""
        return Complexer((1 / self.array) / self.a_d)

    @property
    def resistivity(self):
        """Calculate complex resistivity. (rho - jrho) generic discription."""
        return Complexer(self.array * self.a_d, sign=-1)

    @property
    def loss_tangent(self):
        """Calculate complex resistivity. (rho - jrho) generic discription."""
        return -1 * self.relative_permittivity.imag / self.relative_permittivity.real

    @property
    def loss_factor(self):
        """Calculate complex resistivity. (rho - jrho) generic discription."""
        return -1 * self.relative_permittivity.imag

    def base_df(self, complex_str: str = None, x_axis: str = None) -> pd.DataFrame:
        """Create a DataFrame with specified complex value components."""
        if complex_str is None:
            complex_str = "impedance"

        data = {
            "real": self[f"{complex_str}.real"],
            "imag": self[f"{complex_str}.imag"],
            "inv_imag": self[f"{complex_str}.imag"] * -1,
            "mag": self[f"{complex_str}.mag"],
            "phase": self[f"{complex_str}.phase"],
            "inv_phase": self[f"{complex_str}.phase"] * -1,
            "tan": self[f"{complex_str}.tan"],
            "inv_tan": self[f"{complex_str}.tan"] * -1,
        }

        df = pd.DataFrame(data)

        if x_axis:
            if "freq" in x_axis:
                x_data = self.frequency
                x_name = "freq"
            elif "omega" in x_axis:
                x_data = self.angular_frequency
                x_name = "omega"
            else:
                raise ValueError("x_axis must contain 'freq' or 'omega'")

            if "log" in x_axis:
                x_data = np.log10(x_data)

            df.insert(0, x_name, x_data)

        return df

    def plot_df(self, complex_str: str = None, x_axis: str = None) -> pd.DataFrame:
        """Create a DataFrame with specified complex value components."""
        if complex_str is None:
            complex_str = "impedance"

        data = {
            "real": self[f"{complex_str}.real"],
            "imag": self[f"{complex_str}.imag"],
            "inv_imag": self[f"{complex_str}.imag"] * -1,
            "mag": self[f"{complex_str}.mag"],
            "phase": self[f"{complex_str}.phase"],
            "inv_phase": self[f"{complex_str}.phase"] * -1,
            "tan": self[f"{complex_str}.tan"],
            "inv_tan": self[f"{complex_str}.tan"] * -1,
        }

        df = pd.DataFrame(data)

        if x_axis:
            if "freq" in x_axis:
                x_data = self.frequency
                x_name = "freq"
            elif "omega" in x_axis:
                x_data = self.angular_frequency
                x_name = "omega"
            else:
                raise ValueError("x_axis must contain 'freq' or 'omega'")

            if "log" in x_axis:
                x_data = np.log10(x_data)

            df.insert(0, x_name, x_data)

        return df

    def get_custom_df(self, *args) -> pd.DataFrame:
        """
        Create a DataFrame with specified non-complex value components.

        Parameters:
        *args (str): Strings specifying the parameters and their parts, e.g., "rho.real".

        Returns:
        pd.DataFrame: The resulting DataFrame.
        """
        if len(args) == 2 and hasattr(self, args[1]):
            return self.base_df(args[1], args[0])

        data = {}

        # Parse the arguments to get the target columns
        for arg in args:
            value = self[arg]
            data[arg] = value

        return pd.DataFrame(data)

    def supplement_df(self) -> pd.DataFrame:
        """Create a DataFrame with residuals: freq, omega, loss_factor, and loss_tangent."""
        data = {
            "freq": self.frequency,
            "omega": self.angular_frequency,
            "loss_factor": self.loss_factor,
            "loss_tangent": self.loss_tangent,
        }

        return pd.DataFrame(data)


@dataclass
class Complex_Imp(Complexer):

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


# def extendspace(start, stop, num=50, ext=0, logscale=True, as_exp=False):
#     if logscale:
#         start = np.log10(start)
#         stop = np.log10(stop)

#     delta = np.diff(np.linspace(start, stop, num)).mean()

#     new_start = start - delta * ext
#     new_stop = stop + delta * ext

#     if logscale and not as_exp:
#         return 10**new_start, 10**new_stop, int(num + 2 * ext)

#     return new_start, new_stop, int(num + 2 * ext)


# def range_maker(start, stop, points_per_decade=24, ext=0, is_exp=False):
#     if not is_exp:
#         start = np.log10(start)
#         stop = np.log10(stop)
#     count = int(1 + points_per_decade * abs(start - stop))
#     start, stop, count = extendspace(start, stop, count, ext, False, True)
#     return {"start": 10**start, "stop": 10**stop, "samplecount": count}

