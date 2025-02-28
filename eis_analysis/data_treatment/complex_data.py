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

    data: InitVar[np.ndarray] = np.array([complex(0,0)])
    name: str = "Z"
    sign: int = 1
    long_name: str = "impedance"
    latex: str = "$Z$"
    units: str = "$\Omega$"

    def __post_init__(self, data):
        """Calculate. generic discription."""
        self._sign = 1
        self._array = None
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
        if self._array is None:
            return np.array([complex(0,0)])
        return self._array

    @array.setter
    def array(self, value):
        # if isinstance(arr, np.ndarray):
        #     arr = arr
        if isinstance(value, type(self)):
            arr = value.array
        else:
            arr = np.array(value).squeeze()
        
            if not arr.shape and arr != complex(0,0):
                arr = np.array(value)
        
        if arr.shape:
            if arr.dtype == "complex128":
                if len(arr.shape) == 2 and arr.shape[1] >= 2:
                    arr = arr[:, 0]
                self._array = arr
                self._set_sign()

            else:
                if len(arr.shape) == 2 and arr.shape[1] >= 2:
                    if "pol" in self.name.lower():
                        if (abs(arr[:, 1]) > np.pi / 2).any():
                            arr[:, 1] = np.deg2rad(arr[:, 1])

                        self._array = arr[:, 0] * (
                            np.cos(arr[:, 1]) + 1j * np.sin(arr[:, 1])
                        )
                        self._set_sign()
                    else:
                        self._array = arr[:, 0] + 1j * arr[:, 1]
                        self._set_sign()
                else:
                    self._array = arr + 1j * 0
                    self._set_sign()
        

        # self._sign = (
        #     1 if np.sum(self.sign * arr.imag > 0) > len(arr) / 2 else -1
        # )
    def _set_sign(self):
        if self._array is not None:
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
            # -1 * self.imag,
            self.mag,
            self.phase,
            # -1 * self.phase,
            self.tan,
            # -1 * self.tan,
        ]
        columns = [
            "real",
            "imag",
            # "inv_imag",
            "mag",
            "phase",
            # "inv_phase",
            "tan",
            # "inv_tan",
        ]
        # self._data = pd.DataFrame(dict(zip(columns, vals)))
        return pd.DataFrame(dict(zip(columns, vals)))

    @df.setter
    def df(self, _):
        pass


class ComplexSystem:
    aliases = {
        "f": "frequency",
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
        "imp": "impedance",
        "y": "admittance",
        "adm": "admittance",
        "c": "capacitance",
        "cap": "capacitance",
        "m": "modulus",
        "mod": "modulus",
        "e": "permittivity",
        "ε": "permittivity",
        "epsilon": "permittivity",
        "perm": "permittivity",
        "e_r": "relative_permittivity",
        "e_r0": "relative_permittivity_0",
        "e_rinf": "relative_permittivity_inf",
        "ε_r": "relative_permittivity",
        "εᵣ": "relative_permittivity",
        "epsilon_r": "relative_permittivity",
        "perm_r": "relative_permittivity",
        "permittivity_r": "relative_permittivity",
        "cond": "conductivity",
        "sigma": "conductivity",
        "σ": "conductivity",
        "sigma_dc": "dc_conductivity",
        "σ_dc": "dc_conductivity",
        "resis": "resistivity",
        "rho": "resistivity",
        "ρ": "resistivity",
    }

    def __init__(self, data, frequency=None, thickness=1.0, area=1.0, form=None):
        self._area=None
        self._thickness=None
        self._frequency=None
        self.complexer= Complexer()

        self.thickness = thickness
        self.area = area
        
        

        if isinstance(data, type(self)):
            self.frequency = data.frequency
            self.complexer = data.complexer
            self.update(data)
        else:
            freq = frequency
            if freq is None:
                if isinstance(data, pd.DataFrame):
                    if "freq" in data.columns:
                        freq = data["freq"].values
                        data = data.drop(columns=["freq"])
                    elif "frequency" in data.columns:
                        freq = data["frequency"].values
                        data = data.drop(columns=["frequency"])
                elif isinstance(data, np.ndarray):
                    if data.ndim == 2:
                        if np.all(data[:, 0] >= 0) and not np.iscomplexobj(data[:, 0]):
                            freq = data[:, 0]
                            data = data[:, 1:]
                        elif np.all(data[:, -1] >= 0) and not np.iscomplexobj(data[:, -1]):
                            freq = data[:, -1]
                            data = data[:, :-1]
            self.complexer = Complexer(data)
            if freq is not None:
                freq, data = self.ensure_frequency_order(freq, self.complexer.array)
                self._frequency = freq
                self.complexer.array = data
            
        
        if isinstance(form, str) and self[form] != self.complexer:
            self.complexer = self[form]
    
    def update(self, data=None, frequency=None, thickness=None, area=None, form="impedance"):
        """Update the ComplexSystem with new data or parameters."""
        if isinstance(data, self.__class__):
            frequency = data.frequency
            thickness = data.thickness if data.thickness != 1.0 else thickness
            area = data.area if data.area != 1.0 else area
            data = data[form].array

        if data is not None:
            new_system = self.__class__(data, frequency, self.thickness, self.area)
            if all(new_system.frequency == 1) and len(data) == len(self.frequency):
                new_system.frequency = self.frequency
            self.__dict__.update(new_system.__dict__)
        elif frequency is not None and len(frequency) == len(self.frequency):
            self._frequency = self.ensure_frequency_order(frequency)[0]
        
        if form != "impedance" or self[form] != self.complexer:
            self.complexer = self[form]

        if thickness is not None:
            self.thickness = float(thickness)
        if area is not None:
            self.area = float(area)

    
    def ensure_frequency_order(self, frequency=None, data=None):
        """Ensure frequency is ordered and data is aligned."""
        try:
            freq = frequency if frequency is not None else self._frequency
            if freq is None:
                return freq, data

            b_freq = [freq[i] <= freq[i+1] for i in range(len(freq)-1)] # True if ascending
            # ratio = sum(b_freq) / len(b_freq)
            freq_is_incr = sum(b_freq) / len(b_freq) >= 0.5
            if self._frequency is not None:
                # b_old = [self.frequency[i] <= self.frequency[i+1] for i in range(len(self.frequency)-1)]
                freq_is_incr = all([self.frequency[i] <= self.frequency[i+1] for i in range(len(self.frequency)-1)])
            if data is not None:
                # Data needs to be updated and frequency is not ordered
                sorted_indices = np.argsort(freq) if freq_is_incr else np.argsort(freq)[::-1]
                freq = freq[sorted_indices]
                data = data[sorted_indices]
            else:
                freq = np.array(sorted(freq, reverse=not freq_is_incr))
            return freq, data
        except IndexError as e:
            breakpoint()
            raise e
        
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
    
    def __len__(self):
        return len(self.complexer)

    @property
    def frequency(self):
        """Calculate. generic discription."""
        if self._frequency is None:
            return np.ones(len(self.complexer.array))
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        if value is not None:
            self._frequency = self.ensure_frequency_order(np.array(value))[0]

    @property
    def thickness(self):
        """Calculate. generic discription."""
        if self._thickness is None:
            return 1.0
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
        if self._area is None:
            return 1.0
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
    def capacitance(self):
        """Calculate the capacitance. (C-jC) generic discription."""
        arr = 1 / (1j * self.angular_frequency * self.array)
        return Complexer(arr, sign=-1)

    @property
    def modulus(self):
        """Calculate the modulus. (M+jM) generic discription."""
        arr = self.array * self.angular_frequency * self.C_0 * 1j
        return Complexer(arr)

    @property
    def permittivity(self):
        """Calculate the complex permittivity. (e-je) generic discription."""
        arr = self.e_0 / self.M
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
    
    @property
    def dc_conductivity(self):
        """Calculate the DC conductivity."""
        return self.val_towards_zero(self.conductivity.real)
    
    @property
    def relative_permittivity_0(self):
        """Calculate the relative permittivity at 0 Hz."""
        return self.val_towards_zero(self.relative_permittivity.real)
    
    @property
    def relative_permittivity_inf(self):
        """Calculate the relative permittivity at infinite Hz."""
        return self.val_towards_infinity(self.relative_permittivity.real)



    def base_df(self, complex_str: str = None, x_axis: str = None) -> pd.DataFrame:
        """Create a DataFrame with specified complex value components."""
        if complex_str is None:
            complex_str = "impedance"

        data = {
            "real": self[f"{complex_str}.real"],
            "imag": self[f"{complex_str}.imag"],
            # "inv_imag": self[f"{complex_str}.imag"] * -1,
            "mag": self[f"{complex_str}.mag"],
            "phase": self[f"{complex_str}.phase"],
            # "inv_phase": self[f"{complex_str}.phase"] * -1,
            "tan": self[f"{complex_str}.tan"],
            # "inv_tan": self[f"{complex_str}.tan"] * -1,
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

    def get_df(self, *args, cartesian:bool=True, as_complex:bool=False) -> pd.DataFrame:
        """
        Create a DataFrame with specified non-complex value components.

        Parameters:
        *args (str): Strings specifying the parameters and their parts, e.g., "rho.real".

        Returns:
        pd.DataFrame: The resulting DataFrame.
        """
        # if len(args) == 2 and hasattr(self, args[1]):
        #     return self.base_df(args[1], args[0])
        if not args:
            args = ("impedance",)

        data = {}
        sub_keys = ["real", "imag"] if cartesian else ["mag", "phase"]
        # Parse the arguments to get the target columns
        for arg in args:
            value = self[arg]
            if not as_complex and isinstance(value, Complexer):
                data[f"{arg}.{sub_keys[0]}"] = value[sub_keys[0]]
                data[f"{arg}.{sub_keys[1]}"] = value[sub_keys[1]]
            else:
                data[arg] = value
        df = pd.DataFrame(data)
        # for col in df.columns:
        #     parts = re.split(r'[./]', col)
        #     if parts[0].lower() in ["z", "imp", "impedance"]:
        #         df = df.rename(columns={col: parts[-1]})
        

        df.attrs = {
            "area": self.area,
            "thickness": self.thickness,
            "a_d": self.area_over_thickness,
            "c_0": self.characteristic_capacitance,
            "e_0": self.permittivity_constant,
            # "dc_cond": self.dc_conductivity,
            # "e_r0": self.relative_permittivity_0,
            # "e_rinf": self.relative_permittivity_inf,
        }
        return df


    def val_towards_zero(self, array, perc=5):
        """
        Returns the average value of the array close to the lower end of self.frequency.
        """
        # Determine the number of elements to consider based on the percentage
        num_elements = max(1, int(len(array) * perc / 100))
        
        # Determine which end of the array corresponds to the lower end of self.frequency
        if self.frequency[0] < self.frequency[-1]:
            # Lower end is at the beginning of the array
            return np.mean(array[:num_elements])
        else:
            # Lower end is at the end of the array
            return np.mean(array[-num_elements:])

    def val_towards_infinity(self, array, perc=5):
        """
        Returns the average value of the array close to the higher end of self.frequency.
        """
        # Determine the number of elements to consider based on the percentage
        num_elements = max(1, int(len(array) * perc / 100))
        
        # Determine which end of the array corresponds to the higher end of self.frequency
        if self.frequency[0] > self.frequency[-1]:
            # Higher end is at the beginning of the array
            return np.mean(array[:num_elements])
        else:
            # Higher end is at the end of the array
            return np.mean(array[-num_elements:])



# class ComplexSystemDx:
#     """Class to handle complex system with derivative calculations."""

#     def __init__(self, data, frequency=None, thickness=1.0, area=1.0, form=None, dx=0):
#         self.complex_system = ComplexSystem(data, frequency, thickness, area, form)
#         self.derivative_calculator = DerivativeCalculator(dx)

#     def transform(self):
#         """Transform the data using ComplexSystem and calculate derivatives."""
#         data = self.complex_system.array
#         frequency = self.complex_system.frequency
#         transformed_data = self.derivative_calculator.calculate_derivative(data, frequency)
#         return Complexer(transformed_data)




# class ComplexSystemDx(ComplexSystem):
#     def __init__(self, *args, dx=0, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.dx = dx


#     def calculate_derivative(self, data, level=None):
#         """Calculate the derivative of the data based on the specified level."""
#         level = level or self.dx
#         if level == 0:
#             return data
        
#         # df = pd.DataFrame({'freq': self.frequency, 'data': data})
#         # df = df.groupby('freq').mean().reset_index()
#         # data = df['data'].values
#         epsilon = 1e-10
#         freq = []
#         for f in self.frequency:
#             if f not in freq:
#                 freq.append(f)
#             else:
#                 freq.append(freq[-1]+f*epsilon)

#         data = np.log10(data)
#         freq = np.log10(freq)
#         for _ in range(level):
#             try:
#                 data = np.gradient(data, freq, edge_order=2)
#                 # data = np.diff(data) / np.diff(freq)
#                 # data = np.diff(data)
#                 # data = np.append(data, data[-1])
#             except FloatingPointError as e:
#                 breakpoint()
#                 raise FloatingPointError from e
        
        
#         return data
#         # if level == 1:
#         #     return np.gradient(data, self.frequency, edge_order=2)
#         # elif level == 2:
#         #     return np.gradient(np.gradient(data, self.frequency, edge_order=2), edge_order=2)
#         # elif level == 3:
#         #     return np.gradient(np.gradient(np.gradient(data, self.frequency, edge_order=2), edge_order=2), edge_order=2)
#         # return data
    
    

#     @property
#     def impedance(self):
#         """Calculate. generic description."""
#         data = super().impedance.array
#         data = self.calculate_derivative(data, self.dx)
#         return Complexer(data)

#     @property
#     def admittance(self):
#         """Calculate. generic description."""
#         data = super().admittance.array
#         data = self.calculate_derivative(data, self.dx)
#         return Complexer(data)

#     @property
#     def capacitance(self):
#         """Calculate the capacitance. (C-jC) generic description."""
#         data = super().capacitance.array
#         data = self.calculate_derivative(data, self.dx)
#         return Complexer(data, sign=-1)

#     @property
#     def modulus(self):
#         """Calculate the modulus. (M+jM) generic description."""
#         data = super().modulus.array
#         data = self.calculate_derivative(data, self.dx)
#         return Complexer(data)

#     @property
#     def permittivity(self):
#         """Calculate the complex permittivity. (e-je) generic description."""
#         data = super().permittivity.array
#         data = self.calculate_derivative(data, self.dx)
#         return Complexer(data, sign=-1)

#     @property
#     def relative_permittivity(self):
#         """Calculate the complex permittivity. (e-je) generic description."""
#         data = super().relative_permittivity.array
#         data = self.calculate_derivative(data, self.dx)
#         return Complexer(data, sign=-1)

#     @property
#     def conductivity(self):
#         """Calculate complex conductivity. (sigma + jsigma) generic description."""
#         data = super().conductivity.array
#         data = self.calculate_derivative(data, self.dx)
#         return Complexer(data)

#     @property
#     def resistivity(self):
#         """Calculate complex resistivity. (rho - jrho) generic description."""
#         data = super().resistivity.array
#         data = self.calculate_derivative(data, self.dx)
#         return Complexer(data, sign=-1)