# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 14:25:57 2023

@author: j2cle
"""


import numpy as np
import pandas as pd

from collections.abc import Mapping
from dataclasses import (
    dataclass,
    fields,
    _FIELDS,
    _FIELD,
    InitVar,
)


class BaseClass(object):
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, val):
        setattr(self, key, val)

    def update(self, *args, **kwargs):
        for k, v in dict(args).items():
            if k in self.__dict__.keys():
                self[k] = v
        for k, v in kwargs.items():
            (
                kwargs.pop(k)
                for k in kwargs.keys()
                if k not in self.__dict__.keys()
            )
            self[k] = v
        return self

    def copy(self, **kwargs):
        """Return a new instance copy of obj"""
        (kwargs.pop(k) for k in kwargs.keys() if k not in self.__dict__.keys())
        kwargs = {**self.__dict__, **kwargs}
        return self.__class__(**kwargs)

    def inputs(self):
        return list(self.__dict__.keys())

    def sanitize(
        self, raw, key_names=None, create_instance=False, **init_kwargs
    ):
        """
        dicta would be the old kwargs (key: value)
        dictb would be the renaming dict (old K: new K)
        dictc would be the cls input args
        """
        if isinstance(key_names, (list, tuple)):
            try:
                key_names = {k: v for k, v in key_names}
            except ValueError:
                pass
        if isinstance(key_names, dict):
            raw = {key_names.get(k, k): v for k, v in raw.items()}

        kwargs = {k: raw.get(k, v) for k, v in self.__dict__.items()}
        if create_instance:
            return self.__class__(**init_kwargs, **kwargs)
        return kwargs


class DictMixin(Mapping, BaseClass):
    def __iter__(self):
        return (f.name for f in fields(self))

    def __getitem__(self, key):
        if not isinstance(key, str):
            return [self[k] for k in key]
        field = getattr(self, _FIELDS)[key]
        if field._field_type is not _FIELD:
            raise KeyError(f"'{key}' is not a dataclass field.")
        return getattr(self, field.name)

    def __setitem__(self, key, val):
        setattr(self, key, val)

    def __len__(self):
        return len(fields(self))


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

    # setattr(self, "Z", self.array)
    # setattr(self, "R", self.Z.real)
    # setattr(self, "X", self.Z.imag)
    # setattr(self, "Y", 1 / self.array)
    # setattr(self, "G", self.Y.real)
    # setattr(self, "B", self.Y.imag)


# %% Testing
if __name__ == "__main__":
    from pathlib import Path
