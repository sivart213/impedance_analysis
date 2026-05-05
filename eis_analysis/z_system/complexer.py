# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray, ArrayLike

from eis_analysis.z_system.array_parsing import ArraySignature, parse_z_array


class FreqArray:
    """Array wrapper with frequency semantics."""

    __slots__ = ("_array", "_start", "_stop", "signatures")
    __array_priority__ = 1000

    def __init__(self, data: ArrayLike = tuple(), writeable=True):
        # Default to empty array; user may populate via gen_array
        self._array: NDArray[np.floating] = np.array([], dtype=float)
        self._start: float = 1.0
        self._stop: float = 1.0

        data = np.array(data, dtype=float)
        if data.size > 0:
            self.gen_array(*data, update=True)

        self.signatures = ArraySignature.from_array(self._array)

        self.writeable = writeable

    # ---------------- Constructors ----------------
    @classmethod
    def _from_valid(cls, arr, copy=None):
        """Bypass __init__ for fastest construction."""
        obj = cls.__new__(cls)
        obj._array = np.asarray(arr, dtype=float, copy=copy)
        obj.signatures = ArraySignature.from_array(obj._array)
        return obj

    def gen_array(
        self,
        *data,
        update: bool = False,
        logspace: bool = True,
    ) -> NDArray[np.floating]:
        """
        Generate an array using linspace/logspace semantics.

        Parameters
        ----------
        start : float
            The starting value of the sequence.
        stop : float
            The end value of the sequence.
        num : int, optional
            Number of samples to generate. Default is 50.

        update : bool
            If True, replace internal array with the generated one.

        logspace : bool
            If True, use np.logspace; otherwise use np.linspace.

        Returns
        -------
        np.ndarray
            The generated frequency array.
        """
        start = np.log10(self.start) if logspace else self.start
        stop = np.log10(self.stop) if logspace else self.stop
        num = 50 if self.empty() else self._array.size
        direct = False
        if len(data) == 1:
            if isinstance(data[0], int):
                num = data[0]
            else:
                direct = True
                data = data[0]
        elif len(data) == 2:
            start, stop = data
        elif len(data) == 3 and isinstance(data[2], int):
            start, stop, num = data
        else:
            direct = True

        if direct:
            arr = np.asarray(data, dtype=float)
        elif logspace:
            arr = np.logspace(start, stop, num, dtype=float)
        else:
            arr = np.linspace(start, stop, num, dtype=float)

        if update and arr.size > 0:
            self.array = arr
            self.start = arr[0]
            self.stop = arr[-1]

        return arr

    # ---------------- Arithmetic ----------------
    def __add__(self, other):
        return self._from_valid(self._array + other)

    def __radd__(self, other):
        return self._from_valid(other + self._array)

    def __sub__(self, other):
        return self._from_valid(self._array - other)

    def __rsub__(self, other):
        return self._from_valid(other - self._array)

    def __mul__(self, other):
        return self._from_valid(self._array * other)

    def __rmul__(self, other):
        return self._from_valid(other * self._array)

    def __truediv__(self, other):
        return self._from_valid(self._array / other)

    def __rtruediv__(self, other):
        return self._from_valid(other / self._array)

    def __pow__(self, power, modulo=None):
        return self._from_valid(self._array**power)

    def __neg__(self):
        return self._from_valid(-self._array)

    def __abs__(self):
        return self._from_valid(abs(self._array))

    def __eq__(self, other):
        return bool(np.array_equal(self._array, np.asarray(other)))

    def __ne__(self, other):
        return not self.__eq__(other)

    # ---------------- Container protocol ----------------
    def __len__(self):
        return len(self._array)

    def __iter__(self):
        return iter(self._array)

    def __getitem__(self, index):
        if not isinstance(index, str):
            return self._array[index]
        if hasattr(self, index):
            return getattr(self, index)
        raise KeyError(f"'{index}' is not a valid index or attribute.")

    def __setattr__(self, name, value):
        if name != "writeable" and not getattr(self, "writeable", True):
            raise AttributeError(f"FreqArray is not writeable; cannot modify '{name}'")
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._array, name)

    # ---------------- NumPy interop ----------------
    def __array__(self, dtype=None, copy=None):
        if dtype is None and not copy:
            return self._array
        return np.array(self._array, dtype=dtype or float, copy=copy)

    # ---------------- Properties ----------------
    @property
    def writeable(self) -> bool:
        return self._array.flags.writeable

    @writeable.setter
    def writeable(self, value: bool):
        self._array.flags.writeable = bool(value)

    @property
    def array(self) -> NDArray[np.floating]:
        return self._array

    @array.setter
    def array(self, value: ArrayLike):
        if isinstance(value, FreqArray):
            value = value.array.copy()
        else:
            value = np.asarray(value, dtype=float)
        if any(value < 0):
            raise ValueError("Input values must be positive.")
        self._array = value

    @property
    def start(self) -> float:
        try:
            return self._array[0]
        except IndexError:
            return self._start

    @start.setter
    def start(self, value: float):
        if value < 0:
            raise ValueError("Start value must be non-negative.")
        self._start = float(value)

    @property
    def stop(self) -> float:
        try:
            return self._array[-1]
        except IndexError:
            return self._stop

    @stop.setter
    def stop(self, value: float):
        if value < 0:
            raise ValueError("Stop value must be non-negative.")
        self._stop = float(value)

    @property
    def angular(self):
        """ω = 2πf"""
        return 2 * np.pi * self._array

    def empty(self) -> bool:
        """Check if the array is empty."""
        return self._array.size == 0

    # ---------------- Representation ----------------
    def __repr__(self):
        arr_str = np.array2string(self._array, threshold=6)
        return f"{self.__class__.__name__}(array={arr_str})"


class Complexer:
    """Array wrapper with impedance semantics."""

    __slots__ = ("_array", "_sign")
    __array_priority__ = 1000

    def __init__(
        self,
        data: ArrayLike | None = None,
        sign: int = 1,
        eval_polar: bool = False,
        writeable: bool = True,
    ):
        # self._array: NDArray[np.complexfloating] = np.array([complex(1, 1)], dtype=complex)
        self._array: NDArray[np.complexfloating] = np.array([], dtype=complex)
        self._sign = 1
        self.sign = sign
        if data is not None:
            self._array = parse_z_array(data, eval_polar)[0]

        self.writeable = writeable

    def resolve(self, array: ArrayLike, update: bool = True) -> NDArray[np.complexfloating]:
        """Re-parse the internal array or a new one."""
        array = parse_z_array(array, True, self.sign)[0]
        if update:
            self._array = array
        return array

    def empty(self) -> bool:
        """Check if the array is empty."""
        return self._array.size == 0

    # ---------------- Arithmetic ----------------
    def __add__(self, other) -> Self:
        return self._from_valid(self._array + other, self._sign)

    def __radd__(self, other) -> Self:
        return self._from_valid(other + self._array, self._sign)

    def __sub__(self, other) -> Self:
        return self._from_valid(self._array - other, self._sign)

    def __rsub__(self, other) -> Self:
        return self._from_valid(other - self._array, self._sign)

    def __mul__(self, other) -> Self:
        return self._from_valid(self._array * other, self._sign)

    def __rmul__(self, other) -> Self:
        return self._from_valid(other * self._array, self._sign)

    def __truediv__(self, other) -> Self:
        return self._from_valid(self._array / other, self._sign)

    def __rtruediv__(self, other) -> Self:
        return self._from_valid(other / self._array, self._sign)

    def __pow__(self, power, modulo=None) -> Self:
        return self._from_valid(self._array**power, self._sign)

    def __neg__(self) -> Self:
        return self._from_valid(-self._array, self._sign)

    def __abs__(self) -> Self:
        return self._from_valid(abs(self._array), self._sign)

    def __eq__(self, other) -> bool:
        return bool(np.array_equal(self._array, np.asarray(other)))

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    # ---------------- Container protocol ----------------
    def __len__(self) -> int:
        return len(self._array)

    def __iter__(self):
        return iter(self._array)

    def __getitem__(self, index) -> np.ndarray | int | float | complex:
        if not isinstance(index, str):
            return self._array[index]
        if hasattr(self, index):
            return getattr(self, index)
        raise KeyError(f"'{index}' is not a valid index or attribute.")

    def __setattr__(self, name, value):
        if name != "writeable" and not getattr(self, "writeable", True):
            raise AttributeError(f"Complexer is not writeable; cannot modify '{name}'")
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        """
        Delegate unknown attributes to the underlying NumPy array.
        Called only if normal lookup fails, so it won't override
        defined properties like .real, .imag, etc.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._array, name)

    # ---------------- Representation ----------------
    def __repr__(self):
        arr_str = np.array2string(self._array, threshold=6)
        return f"{self.__class__.__name__}(array={arr_str}, sign={self.sign})"

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        if dtype is None and not copy:
            return self._array
        return np.array(self._array, dtype=dtype or complex, copy=copy)

    # ---------------- Properties ----------------
    @property
    def writeable(self) -> bool:
        return self._array.flags.writeable

    @writeable.setter
    def writeable(self, value: bool):
        self._array.flags.writeable = bool(value)

    @property
    def sign(self) -> int:
        return self._sign

    @sign.setter
    def sign(self, value: int):
        self._sign = int(np.sign(value)) if value != 0 else 1

    @property
    def array(self) -> NDArray[np.complexfloating]:
        return self._array

    @array.setter
    def array(self, value: Any):
        if isinstance(value, Complexer):
            self._array = value.array.copy()
            self._sign = value.sign
        else:
            self._array = parse_z_array(value, False, self.sign)[0]

    @property
    def real(self) -> NDArray[np.floating]:
        return self._array.real

    @property
    def imag(self) -> NDArray[np.floating]:
        return self.sign * self._array.imag

    @property
    def mag(self) -> NDArray[np.floating]:
        return np.abs(self._array)

    @property
    def phase(self) -> NDArray[np.floating]:
        return self.sign * np.angle(self._array, deg=True)

    @property
    def slope(self) -> NDArray[np.floating]:
        return self.sign * np.tan(np.angle(self._array, deg=False))

    @classmethod
    def _from_valid(cls, arr: np.ndarray, sign: int = 1, copy: bool | None = None) -> Self:
        obj = cls.__new__(cls)  # bypasses __init__
        obj._array = np.asarray(arr, dtype=complex, copy=copy)
        obj._sign = sign
        return obj


if __name__ == "__main__":
    import pandas as pd

    from testing.rc_ckt_sim import RCCircuit
    from eis_analysis.z_system.testing.helpers import (
        get_data_df,
        get_data_form,
        baseline_parse_timings,  # noqa: F401
    )

    # from eis_analysis.z_system.testing.test_complex_supports import (
    #     run_single_case,
    #     test_parse_z_array_all_variations,
    # )

    time_res = baseline_parse_timings(100, False, 0)
    time_res_ms = {k: np.median(v) * 1e3 for k, v in time_res.items()}

    rc_data = RCCircuit(freq=(-4, 7, 100), true_values=[24, 1e9, 1e-11], noise=0.01)

    comp = Complexer(rc_data.Z)

    val = abs(comp)

    data_form = "Y"
    sign = 1
    form = "polar"  # "complex", "1j*rect", "rect", "polar"
    noise_opt = True
    freq_opt = True
    extra_opt = 1
    etype = "a"

    data = get_data_form(rc_data, noise_opt, data_form=data_form)
    df = get_data_df(data, rc_data.freq, form, freq_opt, extra_opt, sign, etype)
    # data_x = get_data_form(rc_data, noise_opt, data_form="e_r")
    # df["x-real"] = data_x.real
    # df["x-imag"] = data_x.imag
    # df["x-mag"] = np.abs(data_x)
    # df["x-phase"] = np.angle(data_x, True)

    parsed, freq = parse_z_array(df.to_numpy(), eval_polar=True, sign=sign)

    result = pd.DataFrame()
    result["data.real"] = data.real
    result["data.imag"] = data.imag
    result["res.real"] = parsed.real
    result["res.imag"] = parsed.imag
    if freq is not None:
        result["freq"] = freq

    # run_single_case(rc_data, sign, form, noise_opt, freq_opt, extra_opt, etype, data_form, True)

    # test_parse_z_array_all_variations()
