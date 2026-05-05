# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
import copy
from enum import Enum
from types import MappingProxyType
from typing import Any, Self, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray, ArrayLike

try:
    from .convert import ZDataOps, convert
    from .complexer import Complexer, FreqArray, parse_z_array
    from .definitions import (
        MOD_GRPS,
        ARR_ALIASES,
        COMP_ALIASES,
        OFFSET_FORMS,
        CONST_ALIASES,
        NEG_IMAG_FORMS,
    )
    from .imped_parsing import HasArray, ItemTransforms
    from ..data_treatment.dataset_ops import KeyMatcher
except ImportError:
    from eis_analysis.z_system.convert import ZDataOps, convert
    from eis_analysis.z_system.complexer import Complexer, FreqArray, parse_z_array
    from eis_analysis.z_system.definitions import (
        MOD_GRPS,
        ARR_ALIASES,
        COMP_ALIASES,
        OFFSET_FORMS,
        CONST_ALIASES,
        NEG_IMAG_FORMS,
    )
    from eis_analysis.z_system.imped_parsing import HasArray, ItemTransforms
    from eis_analysis.data_treatment.dataset_ops import KeyMatcher

import warnings

V = TypeVar("V", int, float, complex)

ATTR_TYPES = (str, int, float, np.number, bool)


class SortOrder(Enum):
    """Tri-state sort order for frequency axis."""

    NONE = None  # do not sort
    ASCENDING = True  # sort ascending
    DESCENDING = False  # sort descending

    def __bool__(self):
        """
        Allow SortOrder to behave like a bool in contexts where
        True = ascending, False = descending, None = no sort.
        """
        if self is SortOrder.NONE:
            raise ValueError("SortOrder.NONE cannot be coerced to bool")
        return self.value

    @property
    def is_ascending(self):
        """Check if the order is ascending."""
        return self is SortOrder.ASCENDING

    @property
    def is_descending(self):
        """Check if the order is descending."""
        return self is SortOrder.DESCENDING

    @property
    def is_none(self):
        """Check if the order is none."""
        return self is SortOrder.NONE

    @classmethod
    def _missing_(cls, value=None):
        # Normalize bool/None into the right member
        if isinstance(value, str):
            value = value.lower()
            if value[:3] == "asc":
                return cls.ASCENDING
            if value[:4] == "desc":
                return cls.DESCENDING
            if "none" in value or not value:
                return cls.NONE
        raise ValueError(f"{value!r} is not a valid {cls.__name__}")


class ArrayAt0(Enum):
    NONE = None
    IS_MAX = True
    IS_MIN = False

    @property
    def expect_max(self):
        """Check if the order is ascending."""
        return self is ArrayAt0.IS_MAX

    @property
    def expect_min(self):
        """Check if the order is descending."""
        return self is ArrayAt0.IS_MIN

    @property
    def expect_nothing(self):
        """Check if the expectation is not set."""
        return self is ArrayAt0.NONE

    @classmethod
    def _missing_(cls, value=None):
        # Normalize bool/None into the right member
        if isinstance(value, str):
            value = value.lower()
            if "max" in value or "high" in value:
                return cls.IS_MAX
            if "min" in value or "low" in value:
                return cls.IS_MIN
            if "none" in value or not value:
                return cls.NONE
        raise ValueError(f"{value!r} is not a valid {cls.__name__}")


def ensure_order(
    data: np.ndarray,
    primary_freq: ArrayLike | None,
    backup_freq: ArrayLike | None = None,
    order: str | bool | None | SortOrder = None,
    expected_z_at_dc: str | bool | None | ArrayAt0 = ArrayAt0.IS_MAX,
    tolerance: float = 0.1,
) -> tuple[np.ndarray | None, np.ndarray]:
    """
    Ensure frequency is ordered according to self.order and align data rows.
    If primary_freq is None, fall back to backup_freq.
    If backup_freq is used (external source), then z_size_at_dc may enforce
    a re-sort based on impedance expectation at DC.
    """
    # Resolve frequency source
    data = np.array(data)
    freq = np.array([1]) if primary_freq is None else np.array(primary_freq)
    if len(freq) != len(data) or np.all(freq == 1):
        freq = np.array([1]) if backup_freq is None else np.array(backup_freq)
        if len(freq) != len(data) or np.all(freq == 1):
            return None, data

    order = SortOrder(order)
    z_at_dc = ArrayAt0(expected_z_at_dc)

    # Decide sorting based on self.order
    if order.is_none:  # SortOrder.NONE
        # self.order is NONE → check monotonicity (ie mostly increasing or decreasing)
        diffs = np.diff(freq)
        if max(np.sum(diffs >= 0), np.sum(diffs <= 0)) / len(diffs) < 1 - tolerance:
            return freq, data
    else:
        idx = np.argsort(freq)
        if order.is_descending:  # SortOrder.DESCENDING
            idx = idx[::-1]
        freq, data = freq[idx], data[idx]

    # 2) If external frequency was used, enforce DC expectation on data only
    #    (invert data if |data| at DC does not meet expectation).
    if primary_freq is None and not z_at_dc.expect_nothing:
        dc_idx = int(np.argmin(freq))  # DC = minimal frequency
        mag = Complexer(data).mag
        tol = max(tolerance, 0.05)

        idx = np.argmax(mag) if z_at_dc.expect_max else np.argmin(mag)

        # relative difference checks (of value and index)
        val_diff = abs(mag[dc_idx] - mag[idx]) / mag[idx] if mag[idx] else 0.0
        idx_diff = abs(dc_idx - idx) / len(mag)

        if val_diff > tol and idx_diff > tol:
            data = data[::-1]

        # target_idx = np.argmax(mag) if z_at_dc.expect_max else np.argmin(mag)
        # if abs(dc_idx - target_idx) > len(mag) * max(tolerance, 0.05):
        #     data = data[::-1]

    return freq, data


class ZDataParser:
    """
    Mixin class for parsing
    """

    @staticmethod
    def _dissect_system(
        system: "ComplexSystem", form: str = "impedance"
    ) -> tuple[np.ndarray | None, np.ndarray, float | None, float | None, dict[str, Any]]:
        """
        Parse another ComplexSystem instance.
        Returns frequency, data, thickness, area, attrs.
        Thickness/area return None to signal 'use default=1'.
        Frequency defaults to ones of same length as data if not provided.
        """
        data = system.array
        freq = system.frequency
        if len(freq) != len(data) or np.all(freq == 1):
            freq = None
        thickness = system.thickness if system.thickness != 1.0 else None
        area = system.area if system.area != 1.0 else None
        return freq, data, thickness, area, dict(system.info)

    @staticmethod
    def _dissect_df(
        df: pd.DataFrame, form: str = "impedance"
    ) -> tuple[np.ndarray | None, np.ndarray, float | None, float | None, dict[str, Any]]:
        """
        Parse a pandas DataFrame into components.
        Returns frequency (if found), data, thickness, area, attrs.
        """
        form = COMP_ALIASES.get(str(form).lower(), "impedance")
        sign = -1 if form in NEG_IMAG_FORMS else 1
        attrs = {}
        attrs |= df.attrs
        thickness = attrs.pop("thickness", None)
        area = attrs.pop("area", None)

        keys = ["freq", "real", "imag", f"{form}.real", f"{form}.imag", form]
        cols = KeyMatcher(df, keys).result_dict

        freq = None
        data = None

        # order-dependent: 0=freq, 1=real, 2=imag
        if "freq" in cols:
            freq = df[cols["freq"]].to_numpy()
            df = df.drop(columns=[cols["freq"]])

        if form in cols and np.iscomplex(df[cols[form]]).any():
            data = df[cols[form]].to_numpy()
        elif keys[1] in cols and keys[2] in cols:
            data = df[cols[keys[1]]].to_numpy() + 1j * sign * df[cols[keys[2]]].to_numpy()
        elif keys[3] in cols and keys[4] in cols:
            data = df[cols[keys[3]]].to_numpy() + 1j * sign * df[cols[keys[4]]].to_numpy()

        # fallback if no usable data
        if data is None or freq is None:
            data2, freq2 = parse_z_array(df.to_numpy(), sign=sign)
            freq = freq2 if freq is None else freq
            data = data2 if data is None else data

        return freq, data, thickness, area, attrs

    @staticmethod
    def _dissect_array(
        arr: np.ndarray, form: str = ""
    ) -> tuple[np.ndarray | None, np.ndarray, float | None, float | None, dict[str, Any]]:
        """
        Parse a numpy array into components.
        Returns frequency (if detected), data, thickness=None, area=None, attrs={}.
        """
        freq = None
        form = str(form).lower()
        # check_polar = any(s in form for s in ["polar", "phase", "mag"])
        if arr.ndim <= 2:
            arr, freq = parse_z_array(
                arr,
                True,
                -1 if COMP_ALIASES.get(form, "impedance") in NEG_IMAG_FORMS else 1,
            )
        return freq, arr, None, None, {}

    def dissect_data(
        self, data: "ComplexSystem | pd.DataFrame | ArrayLike", form: str = "impedance"
    ) -> tuple[np.ndarray | None, np.ndarray, float | None, float | None, dict[str, Any]]:
        """
        Director function: dispatch to the appropriate parser.
        Always returns (freq, data, thickness, area, attrs).
        """
        if isinstance(data, ComplexSystem):
            return self._dissect_system(data, form)
        elif isinstance(data, pd.DataFrame):
            return self._dissect_df(data, form)
        else:
            return self._dissect_array(np.array(data), form)


class ComplexSystem(ZDataOps, ZDataParser, ItemTransforms[Complexer | np.ndarray | float | int]):
    """
    Class object representing a complex impedance system, with associated frequency, thickness,
    area, and metadata attributes.

    Primarily used to provide a single interface for parsing and converting various data formats
    into a standardized form (impedance array with frequency axis), while also storing relevant
    metadata (thickness, area, custom attributes).
    """

    aliases = MappingProxyType(CONST_ALIASES | ARR_ALIASES | COMP_ALIASES)
    _order: SortOrder
    _ast_attr_types: tuple[type,] = (Complexer,)

    def __init__(
        self,
        data: ArrayLike | Complexer | Self | None = None,
        frequency: ArrayLike | None = None,
        thickness: int | float | None = None,
        area: int | float | None = None,
        form: str = "impedance",
        order: str | bool | None | SortOrder = None,
        writeable=True,
        **kwargs,
    ):
        """
        Initialize a new ComplexSystem from raw data, another ComplexSystem,
        or a Complexer object.

        Parameters
        ----------
        data : ArrayLike | Complexer | ComplexSystem
            Input data or object to construct from.
        frequency : ArrayLike, optional
            External frequency array. Used only if no frequency is embedded in data.
        thickness : float, optional
            External thickness value. Acts as a default if not provided in data.
        area : float, optional
            External area value. Acts as a default if not provided in data.
        form : str, default="impedance"
            Data representation form. Conversion to impedance is performed if needed.
        **kwargs : dict
            Additional attributes to attach.

        Notes
        -----
        - It is recommended that all data be provided in impedance form,
        rather than relying on internal conversion.
        - Precedence rules:
            * Frequency: parsed from data > explicit `frequency` kwarg > default ones.
            * Thickness: parsed from data > `thickness` kwarg > default (1.0).
            * Area: parsed from data > `area` kwarg > default (1.0).
        """
        self._writeable = True
        self._set_data()
        self._order = SortOrder.NONE
        self._attrs: dict[str, str | int | float | np.number | bool] = {}
        # self._info = MappingProxyType(self._attrs)
        super().__init__(default_x=kwargs.pop("default_x", "frequency"))

        if data is None:
            data = Complexer()

        # self.thickness = thickness
        # self.area = area
        form = COMP_ALIASES.get(str(form).lower(), "impedance")
        self.order = order

        sort_kwargs = {}
        sort_kwargs["expected_z_at_dc"] = kwargs.pop("expected_z_at_dc", True)
        sort_kwargs["tolerance"] = kwargs.pop("tolerance", 0.1)

        for key in list(kwargs.keys()):
            try:
                if key in [
                    "attrs",
                    "form_kwargs",
                    "savgol_kwargs",
                    "interp_kwargs",
                    "norm_kwargs",
                ]:
                    setattr(self, key, kwargs.pop(key))
            except AttributeError:
                # If the attribute cannot be set, just pass
                pass

        # Use the attrs setter to sanitize and store the remaining kwargs
        self.attrs = self.info | kwargs

        freq, data, thick, ar, attrs = self.dissect_data(data, form)

        self.attrs |= attrs
        self.thickness = thick if thick is not None else thickness
        self.area = ar if ar is not None else area

        if form != "impedance":
            sort_kwargs["expected_z_at_dc"] = None
            freq, data = ensure_order(data, freq, frequency, self.order, **sort_kwargs)
            self._set_data(convert(data, form, system=self, frequency=freq, strict=False), freq)
            warnings.warn(
                "Conversion inside __init__ is deprecated; please pass impedance data.",
                FutureWarning,
                stacklevel=2,
            )
        else:
            freq, data = ensure_order(data, freq, frequency, self.order, **sort_kwargs)
            self._set_data(Complexer(data), freq)

        self.writeable = writeable

    def __setattr__(self, name, value):
        if name != "writeable" and not getattr(self, "writeable", True):
            raise AttributeError(f"ComplexSystem is not writeable; cannot modify '{name}'")
        name = self.aliases.get(str(name), str(name))
        object.__setattr__(self, name, value)

    def __getattr__(self, name) -> Complexer | np.ndarray | float | int:
        if name == "aliases":
            raise AttributeError
        name = self.aliases.get(name.lower(), name)
        return object.__getattribute__(self, str(name))

    def __getitem__(self, index) -> Complexer | np.ndarray | float | int:
        """Allow slicing and indexing."""
        if isinstance(index, str):
            if hasattr(self, index.lower()):
                return getattr(self, index.lower())
            elif hasattr(self.complexer_obj, index.lower()):
                return getattr(self.complexer_obj, index.lower())
            try:
                return self._parse_and_transform(index)
            except Exception as exc:
                raise AttributeError(
                    f"'{index}' is not a valid index for '{self.__class__.__name__}'\nError: {exc}"
                ) from exc
        else:
            raise TypeError("Index must be a string")

    def __repr__(self) -> str:
        """Custom repr to include self.array."""
        msg = f"{self.__class__.__name__}"
        data = ""
        if hasattr(self, "complexer_obj"):
            arr_str = np.array2string(self.complexer_obj.array, threshold=6, precision=4)
            data += f"Z = {arr_str}"
        if hasattr(self, "frequency"):
            freq_str = np.array2string(self.frequency, threshold=6, precision=4)
            data += f", freq = {freq_str}"
        if data:
            msg += f"({data})"
        return msg

    def __len__(self) -> int:
        return len(self.complexer_obj.array)

    def __copy__(self) -> Self:
        cls = type(self)  # self.__class__
        result = cls.__new__(cls)
        # result.__dict__.update(self.__dict__)
        for key, val in self.__dict__.items():
            if key == "_writeable":
                continue
            copied = copy.copy(val)
            # Defensive unlock for child objects
            if hasattr(copied, "writeable"):
                try:
                    copied.writeable = True
                except Exception:
                    pass
            result.__dict__[key] = copied
        return result

    def __deepcopy__(self, memo) -> Self:
        cls = type(self)
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, val in self.__dict__.items():
            if key == "_writeable":
                continue
            copied = copy.deepcopy(val, memo)
            # Defensive unlock for child objects
            if hasattr(copied, "writeable"):
                try:
                    copied.writeable = True
                except Exception:
                    pass
            result.__dict__[key] = copied
            # setattr(result, key, copy.deepcopy(val, memo))
        return result

    def copy(self, deep: bool = True) -> Self:
        """Create a copy of the ComplexSystem."""
        if deep:
            return copy.deepcopy(self)
        return copy.copy(self)

    def clone(self, data: ArrayLike | Complexer | Self, form: str = "impedance", **kwargs) -> Self:
        """
        Create a new ComplexSystem instance with new data, while preserving
        this instance's thickness, area, and attrs. Frequency from the new data
        (if available) continues to dominate.

        Parameters
        ----------
        data : ArrayLike | Complexer | ComplexSystem
            Input data or object to construct the clone from.
        form : str, default="impedance"
            Data representation form. Conversion to impedance is performed if needed.
        **kwargs : dict
            Additional attributes to attach.

        Returns
        -------
        ComplexSystem
            A new instance with new data but preserved thickness and area.

        Notes
        -----
        - It is recommended that all data be provided in impedance form,
        rather than relying on internal conversion.
        - Precedence rules:
            * Frequency: parsed from new data > explicit `frequency` kwarg > default ones.
            * Thickness: always preserved from the source instance (this object).
            * Area: always preserved from the source instance (this object).
        - This method ensures that any internal reversion/normalization
        during construction uses provided kwargs, but the final state
        re‑asserts the current instance’s thickness and area.
        """
        # Step 1: create a fresh instance with the same class
        kwargs.setdefault("frequency", self.frequency)
        kwargs.setdefault("thickness", self.thickness)
        kwargs.setdefault("area", self.area)
        new = self.__class__(data, form=form, **kwargs)

        # Step 2: copy over thickness/area from *this* instance
        new.thickness = self.thickness
        new.area = self.area
        new.attrs |= self.info

        return new

    def update(
        self, data: ArrayLike | Complexer | Self, form: str = "impedance", **kwargs
    ) -> None:
        """
        Update this ComplexSystem with new data or parameters.

        Parameters
        ----------
        data : ArrayLike | Complexer | ComplexSystem
            Input data or object to merge into this system.
        form : str, default="impedance"
            Data representation form. Conversion to impedance is performed if needed.
        **kwargs : dict
            Explicit overrides for metadata (e.g., thickness, area).

        Notes
        -----
        - It is recommended that all data be provided in impedance form,
        rather than relying on internal conversion.
        - Precedence rules:
            * Frequency: parsed from data > keep existing frequency > ignore kwargs.
            * Thickness: parsed from data > keep existing value > final override from `thickness` kwarg.
            * Area: parsed from data > keep existing value > final override from `area` kwarg.
        - Kwargs are applied *after* conversions, ensuring that updates do not
        interfere with internal reversion/normalization steps.
        """
        if not self.writeable:
            raise AttributeError("ComplexSystem instance is locked; cannot update")

        internal = {}
        if isinstance(data, ComplexSystem):
            internal = data.__dict__.copy()
        freq, data, thick, ar, attrs = self.dissect_data(data, form)

        # Precedence: parsed val > provided value > existing value
        self.__dict__.update(internal)

        thickness = kwargs.pop("thickness", None)
        area = kwargs.pop("area", None)

        self.thickness = thick if thick is not None else thickness
        self.area = ar if ar is not None else area
        self.attrs |= attrs

        # self.complexer_obj = Complexer(data)
        # self.complexer_obj = self.to_impedance(form)
        sort_kwargs = {}
        frequency = kwargs.pop("frequency", self.frequency)
        sort_kwargs["expected_z_at_dc"] = kwargs.pop("expected_z_at_dc", True)
        sort_kwargs["tolerance"] = kwargs.pop("tolerance", 0.1)

        if form != "impedance":
            sort_kwargs["expected_z_at_dc"] = None
            freq, data = ensure_order(data, freq, frequency, self.order, **sort_kwargs)
            kwargs.setdefault("strict", True)
            self._set_data(
                convert(
                    data, form, frequency=freq, area=self.area, thickness=self.thickness, **kwargs
                ),
                freq,
            )
        else:
            freq, data = ensure_order(data, freq, frequency, self.order, **sort_kwargs)
            self._set_data(Complexer(data), freq)

        self.thickness = thickness
        self.area = area

    @property
    def writeable(self) -> bool:
        """Flag indicating whether the ComplexSystem is writeable (modifiable)."""
        return self._writeable

    @writeable.setter
    def writeable(self, value: bool):
        value = bool(value)
        object.__setattr__(self, "_writeable", value)
        for obj in self.__dict__.values():
            if isinstance(obj, np.ndarray):
                obj.flags.writeable = value

            elif hasattr(obj, "writeable"):
                try:
                    obj.writeable = value
                    # setattr(obj, "writeable", value)
                except Exception:
                    pass

    @property
    def order(self) -> SortOrder:
        """Get the full attributes dictionary."""
        return self._order

    @order.setter
    def order(self, value: str | bool | None | SortOrder) -> None:
        """
        Setter for attrs. Ensures only valid kwargs for savgol_filter are updated.
        """
        old_order = self._order
        self._order = SortOrder(value)
        # NOTE: Check relies on old(?) default freq
        if not self._order.is_none and old_order != self._order and not all(self.frequency == 1):
            freq, data = ensure_order(
                self.complexer_obj.array, self.frequency, None, self._order, None
            )
            self._set_data(data, freq)

    @property
    def _base_attrs(self) -> dict[str, float]:
        """Get the auto-calculated portion of the attrs dictionary."""
        init = {
            "area": self.area,
            "thickness": self.thickness,
            "c_0": self.characteristic_capacitance,
        }
        # NOTE: Check relies on old(?) default Z
        if len(self.complexer_obj.array) <= 1:
            return init
        return {
            **init,
            "dc_Z": self.get_value("impedance.real", towards="zero"),
            "dc_cond": self.get_value("conductivity.real", towards="zero"),
            "e_r0": self.get_value("relative_permittivity.real", towards="zero"),
            "e_rinf": self.get_value("relative_permittivity.real", towards="infinity"),
        }

    @property
    def info(self) -> dict[str, str | int | float | np.number | bool]:
        """Get the updatable portion of the attrs dictionary."""
        return self._attrs.copy()
        # return self._info

    @property
    def attrs(self) -> dict[str, str | int | float | np.number | bool]:
        """Get the full attributes dictionary."""
        return self._attrs | self._base_attrs

    @attrs.setter
    def attrs(self, value: dict) -> None:
        """
        Setter for attrs. Ensures only valid kwargs for savgol_filter are updated.
        """
        if not isinstance(value, dict):
            raise TypeError("input args must be a dictionary")
        self._attrs = {
            str(k): v
            for k, v in value.items()
            if isinstance(v, ATTR_TYPES) and str(k) not in self._base_attrs
        }
        # self._info = MappingProxyType(self._attrs)

    def get_value(
        self,
        name: str | ArrayLike,
        target_type: type[V] = float,
        towards: str = "",
        perc: int | float = 5,
    ) -> V:
        """
        Retrieve a value by name.

        Parameters
        ----------
        name : str
            The name or alias of the desired property (e.g., "impedance", "admittance").
        target_type : type, default=float
            The desired return type.  Options: int, float, or complex.
        towards : str, optional
            If the property is an array, specify "zero" or "infinity" to
            retrieve the value towards the low or high end of the frequency spectrum.
        perc : int | float, default=5
            Percentage of data points to consider when calculating the value towards zero or infinity.

        Returns
        -------
        value : target_type (e.g., float, int)
            The corresponding value converted to the specified target type.

        Raises
        ------
        AttributeError
            If the specified name does not correspond to a valid property.
        """
        if isinstance(name, str):
            try:
                data = self[name]
                if isinstance(data, Complexer):
                    data = data.array
            except (AttributeError, TypeError):
                raise AttributeError(
                    f"'{name}' is not a valid property for '{self.__class__.__name__}'"
                )
        else:
            data = np.array(name)

        if isinstance(data, np.ndarray):
            if towards and len(data) == len(self.frequency):
                towards = str(towards).lower()
                if "inf" in towards:
                    data = self.val_towards_infinity(np.asarray(data), perc=perc)
                else:
                    data = self.val_towards_zero(np.asarray(data), perc=perc)
            elif len(data) >= 1:
                data = np.median(data)
            else:
                raise ValueError(f"Array for '{name}' is empty and cannot provide a value.")

        try:
            return target_type(data)
        except Exception:
            raise TypeError(
                f"Form '{name}' is invalid and/or cannot be returned as a {target_type.__name__}."
            )

    def get_array(
        self, name: str, allow_default: bool = True, as_complex: bool = False
    ) -> np.ndarray:
        """
        Retrieve a numpy array by name, using aliases if necessary.

        Parameters
        ----------
        name : str
            The name or alias of the desired property (e.g., "impedance", "admittance").

        Returns
        -------
        np.ndarray
            The corresponding numpy array.

        Raises
        ------
        AttributeError
            If the specified name does not correspond to a valid property.
        """
        data = None
        if name.lower() not in ARR_ALIASES:
            try:
                data = self[name]
            except (AttributeError, TypeError):
                data = self.Z.array if allow_default else None
        else:
            data = self[ARR_ALIASES[name.lower()]]

        if isinstance(data, Complexer):
            data = data.array
        if isinstance(data, np.ndarray):
            if not as_complex and data.dtype.kind == "c":
                return np.column_stack((data.real, data.imag))
            return data

        raise TypeError(f"Form '{name}' is invalid and/or cannot be returned as a numpy array.")

    def get_complexer(self, name: str, allow_default: bool = True, raw: bool = False) -> Complexer:
        """
        Retrieve a Complexer object by name, using aliases if necessary.

        Parameters
        ----------
        name : str
            The name or alias of the desired Complexer property (e.g., "impedance", "admittance").
        allow_default : bool, optional
            If True, fall back to impedance if the name is invalid.
        raw : bool, optional
            If True, bypass any "cleaned" properties (such as susceptibility) and convert directly
            from impedance. Only applicable to forms in OFFSET_FORMS.

        Returns
        -------
        Complexer
            The corresponding Complexer object.

        Raises
        ------
        TypeError
            If the resolved data cannot be returned as a Complexer.
        """
        base = COMP_ALIASES.get(name.lower(), name)

        if raw and base in OFFSET_FORMS:
            data = convert(self.Z, from_form="impedance", to_form=base, system=self, strict=False)
        else:
            try:
                data = self[base]
            except (AttributeError, TypeError):
                data = self.Z if allow_default else None

        if isinstance(data, Complexer):
            return data
        elif isinstance(data, np.ndarray) and (data.dtype.kind == "c" or data.ndim == 2):
            return Complexer(data)
        else:
            raise TypeError(
                f"Form '{name}' is invalid and/or cannot be returned as a Complexer object."
            )

    def get_df(
        self,
        *args: str,
        cartesian: bool = True,
        as_complex: bool = False,
        f_of_point: int | float | list[int | float] = 0,
    ) -> pd.DataFrame:
        """
        Create a DataFrame with specified value components from the ComplexSystem.

        Parameters
        ----------
        *args : str
            Strings specifying the parameters and their parts to include as columns in the DataFrame.
            Each argument can be a property name (e.g., "impedance", "admittance") or a property part
            (e.g., "impedance.real", "capacitance.phase"). If no arguments are provided, defaults to
            ("impedance",) and includes the real and imaginary parts of the impedance.

        cartesian : bool, default=True
            If True, for each property argument that is a Complexer and as_complex is False,
            include the real and imaginary parts as separate columns (".real", ".imag").
            If False, include the magnitude and phase as separate columns (".mag", ".phase").

        as_complex : bool, default=False
            If True, include the raw Complexer object or array for each argument as a single column.
            If False, split Complexer values into their respective parts as determined by `cartesian`.

        f_of_point : int, float, or list, default=0
            If nonzero, for each value in f_of_point (or the single value if not a list),
            the value of each column at the nearest frequency is added to the DataFrame's attrs
            dictionary with a key of the form "<column>_<frequency>".

        Returns
        -------
        pd.DataFrame
            DataFrame containing the requested columns and attributes. The DataFrame's attrs
            dictionary is updated with the ComplexSystem's attrs and, if f_of_point is set,
            with the values at the specified frequencies.

        Notes
        -----
        - If a requested argument is a Complexer and as_complex is False, it is split into two columns
        according to the `cartesian` flag.
        - If a requested argument is not a Complexer or as_complex is True, it is included as a single column.
        - The DataFrame's attrs dictionary is a copy of the ComplexSystem's attrs, possibly extended
        with values at specific frequencies if f_of_point is set.
        - This method does not include an x-axis column (such as "freq" or "omega"); use `base_df`
        if you want an x-axis included automatically.
        - To replicate the output of `base_df`, call `get_df` with all desired property parts
        (e.g., "impedance.real", "impedance.imag", "impedance.mag", "impedance.phase", "impedance.tan").

        Examples
        --------
        >>> cs.get_df("impedance")
        # Returns DataFrame with columns "impedance.real" and "impedance.imag"

        >>> cs.get_df("capacitance", cartesian=False)
        # Returns DataFrame with columns "capacitance.mag" and "capacitance.phase"

        >>> cs.get_df("impedance.real", "impedance.imag", "impedance.mag", "impedance.phase", "impedance.tan")
        # Returns DataFrame with all five columns, similar to base_df

        >>> cs.get_df("impedance", f_of_point=[1e3, 1e4])
        # Adds values at frequencies 1e3 and 1e4 to DataFrame.attrs

        """
        if not args:
            args = ("impedance",)

        data = {}
        sub_keys = MOD_GRPS["cartesian"] if cartesian else MOD_GRPS["polar"]
        # Parse the arguments to get the target columns
        for arg in args:
            value = self[arg]
            if not as_complex and isinstance(value, Complexer):
                data[f"{arg}.{sub_keys[0]}"] = value[sub_keys[0]]
                data[f"{arg}.{sub_keys[1]}"] = value[sub_keys[1]]
            else:
                data[arg] = value
        df = pd.DataFrame(data)

        df.attrs |= self.attrs

        if f_of_point:
            if isinstance(f_of_point, (int, float)):
                f_of_point = [f_of_point]
            for f in f_of_point:
                f_index = self.get_value(f"(freq @ {f}).index", target_type=int)
                f_actual = "{:.1E}".format(self.frequency[f_index])
                # for each non freq column, add the value at the f_index to the attrs
                for col in df.columns:
                    if "freq" not in col:
                        try:
                            df.attrs[f"{col}_{f_actual}"] = df[col][f_index]
                        except (IndexError, TypeError):
                            pass

        return df

    @staticmethod
    def is_valid_key(key: str, dummy_test: bool = True) -> bool:
        """
        Check if a string is a valid key for ComplexSystem.

        Parameters:
        key (str): The key to check.

        Returns:
        bool: True if the key is valid, False otherwise.
        """

        try:
            if (
                hasattr(ComplexSystem, key)
                or hasattr(Complexer, key)
                or key in ComplexSystem.aliases
            ):
                return True

            if not dummy_test:
                return False
            # Create a dummy instance for validation
            dummy_instance = ComplexSystem(
                data=np.array([1 + 1j, 2 + 2j]), frequency=np.array([1, 2])
            )

            # Attempt to access the key
            _ = dummy_instance[key]
            return True
        except (AttributeError, TypeError, KeyError):
            return False

    def _ensure_array(self, value: ArrayLike | HasArray | Self) -> np.ndarray:
        """
        Helper function to ensure the input is converted to a numpy array.

        Parameters:
        value: The input value to be converted.

        Returns:
        np.ndarray: The converted numpy array.
        """
        if isinstance(value, str):
            val = getattr(self, value)
            if val is None:
                raise ValueError(f"Attribute '{value}' not found in ComplexSystem.")
            return np.array(getattr(val, "array", val))
        return np.asarray(getattr(value, "array", value))

    def cirith_ungol(
        self, freq_array: NDArray[np.floating], z_array: NDArray[np.complexfloating]
    ) -> None:
        """
        Fast-path update for fitting loops or similar constrained cases.
        Typically only used in controlled contexts (e.g., SciPy optimizers).
        Skips validation and assumes:
        - z_array is a 1d complex numpy array.
        - freq_array is a 1d real numpy array.
        - length of z_array matches length of freq_array.

        Parameters
        ----------
        freq_array : NDArray[np.floating]
            1D array of frequency values.
        z_array : NDArray[np.complexfloating]
            1D array of complex impedance values.
        """
        self.complexer_obj = Complexer._from_valid(z_array)
        self._frequency = FreqArray._from_valid(freq_array)
        # self._frequency = np.asarray(freq_array)
        return


if __name__ == "__main__":
    # from local.pickle_testing import check_deep_picklability, check_picklability, check_attr_picklability
    import timeit  # noqa: F401

    from testing.rc_ckt_sim import RCCircuit  # noqa: F401
    from eis_analysis.impedance_supplement.ops import get_impedance  # noqa: F401

    ckt = RCCircuit(true_values=[24, 1e9, 1e-11], noise=0.01)
    rc_system = ComplexSystem(data=ckt.Z_noisy, frequency=ckt.freq, area=25, thickness=500e-4)

    # rc_system.writeable = False

    # rc_system.writeable=True

    # rc_system.order = True

    # freq = rc_system.frequency
    # ones = np.ones_like(freq, dtype=complex)

    # # Case 1: Update with raw array, explicit area/thickness
    # cs1 = ComplexSystem(data=ones, frequency=freq, area=1.0, thickness=1.0)
    # cs1.update(
    #     rc_system.e_r.array,
    #     form="relative_permittivity",
    #     area=rc_system.area,
    #     thickness=rc_system.thickness,
    # )

    from_form = "relative_permittivity_corrected"
    data = rc_system.get_complexer(from_form, raw=True)

    kwargs = {}
    if from_form == "relative_permittivity_corrected":
        kwargs["phys_const"] = rc_system.val_towards_zero(rc_system.conductivity.real)
    elif from_form == "susceptibility":
        kwargs["phys_const"] = rc_system.val_towards_infinity(rc_system.relative_permittivity.real)

    result = convert(
        data.array.real - 1j * data.array.imag,
        from_form=from_form,
        to_form="impedance",
        system=rc_system,
        **kwargs,
    )

    # np.testing.assert_allclose(
    #     result.array,
    #     rc_system.impedance.array,
    #     rtol=1e-6,
    #     atol=1e-8,
    #     err_msg=f"Conversion from {from_form} to impedance failed for",
    # )

    # ckt = RCCircuit(freq=(-4, 7, 200), true_values=[24, 1e9, 1e-11], noise=0.01)
    # # Z = ckt.Z_noisy
    # Z = get_impedance(ckt.freq, 24, 1e-3, 1e9, 1e-11, model="R0-L0-p(R1,C1)")
    # Z_n = ckt.get_noisy_z(arr=Z)
    # rc_system_0 = ComplexSystem(data=Z, frequency=ckt.freq, area=25, thickness=500e-4)
    # rc_system = ComplexSystem(data=Z_n, frequency=ckt.freq, area=25, thickness=500e-4)

    # rel_perm = rc_system.get_array("relative_permittivity", as_complex=True)
    # freq = rc_system.frequency
    # ones = np.ones_like(freq, dtype=complex)

    # # Case 1: Update with raw array, explicit area/thickness
    # # cs1 = ComplexSystem(data=ones, frequency=freq, area=1.0, thickness=1.0)
    # # cs1.update(
    # #     rc_system.e_r.array,
    # #     form="relative_permittivity",
    # #     area=rc_system.area,
    # #     thickness=rc_system.thickness,
    # # )
    # # print("Case 1: raw array, explicit area/thickness")
    # # # np.testing.assert_allclose(cs1.impedance.array, rc_system.array, rtol=1e-6, atol=1e-8)

    # # # Case 2: Update with Complexer, explicit area/thickness
    # # cs2 = ComplexSystem(data=ones, frequency=freq, area=1.0, thickness=1.0)
    # # cs2.update(
    # #     rc_system.e_r,
    # #     form="relative_permittivity",
    # #     area=rc_system.area,
    # #     thickness=rc_system.thickness,
    # # )
    # # print("Case 2: Complexer, explicit area/thickness")
    # # # np.testing.assert_allclose(cs2.impedance.array, rc_system.array, rtol=1e-6, atol=1e-8)

    # # Case 3: Update with another ComplexSystem (should use its area/thickness)
    # cs3 = ComplexSystem(data=ones, frequency=freq, area=1.0, thickness=1.0)
    # rel_df = rc_system.get_df("relative_permittivity", cartesian=False, f_of_point=11)
    # cs3.update(rel_df.to_numpy(), form="relative_permittivity", area=1.5, thickness=1.5)
    # print("Case 3: DataFrame, area/thickness from data")
    # # np.testing.assert_allclose(cs3.array, rc_system.array, rtol=1e-6, atol=1e-8)

    # cs4 = ComplexSystem(data=ones, frequency=freq, area=1.0, thickness=1.0)
    # rel_perm_system = ComplexSystem(rc_system.e_r.array, freq, rc_system.thickness, rc_system.area)
    # cs4.update(rel_perm_system, form="relative_permittivity", area=1.5, thickness=1.5)
    # print("Case 4: ComplexSystem, area/thickness from data")

    # # Create a new system with different area/thickness
    # cs = ComplexSystem(data=np.ones_like(rel_perm), frequency=freq, area=1.0, thickness=1.0)

    # # Update with relative_permittivity data and explicit area/thickness
    # cs.update(rel_perm, form="relative_permittivity", area=0.5, thickness=0.5)

    # simple = np.array([[1, 2], [3, -4], [-1, 0.5]])
    # s_complexer = Complexer(simple)

    # # # fmt: off
    # # df = pd.DataFrame()
    # # df["freq"] = ckt.freq
    # # df["Z"] = Z
    # # df["Z.real"] = Z.real
    # # df["Z.imag"] = Z.imag
    # # df["1j*Z.imag"] = 1j * Z.imag
    # # df["Z.mag"] = np.abs(Z)
    # # df["Z.phase"] = np.angle(Z, deg=True)
    # # df["Z.rad"] = np.angle(Z, deg=False)
    # # df["freq bad"] = ckt.freq
    # # df.loc[0, "freq bad"] = 1e-12

    # # test = ZDataParser._dissect_df(df)

    # # results = {}
    # # results["1) f & Z"] = df[["freq", "Z"]].to_numpy()
    # # results["2) f & Z rect"] = df[["freq", "Z.real", "Z.imag"]].to_numpy()
    # # results["3) f & Z pol deg"] = df[["freq", "Z.mag", "Z.phase"]].to_numpy()
    # # results["4) f & Z pol rad"] = df[["freq", "Z.mag", "Z.rad"]].to_numpy()
    # # results["5) f & Z j rect"] = df[["freq", "Z.real", "1j*Z.imag"]].to_numpy()
    # # results["6) f & Z j rect 2"] = df[["freq", "Z.real", "1j*Z.imag", "Z.imag",]].to_numpy()
    # # results["7) f & Z rect/pol"] = df[["freq", "Z.real", "Z.imag", "Z.mag", "Z.rad"]].to_numpy()
    # # results["8) Z"] = df[["Z"]].to_numpy()
    # # results["9) Z rect"] = df[["Z.real", "Z.imag"]].to_numpy()
    # # results["10) Z j rect"] = df[["Z.real", "1j*Z.imag"]].to_numpy()
    # # results["11) f b & Z"] = df[["freq bad", "Z", "Z.real", "freq"]].to_numpy()
    # # results["12) f b & Z rect"] = df[["freq bad", "Z.real", "Z.imag"]].to_numpy()
    # # # fmt: on
    # # for cond, data in results.items():
    # #     print("Condition:", cond)
    # #     # # define callables that close over `data`
    # #     # runs = 10000
    # #     # t1 = timeit.timeit(lambda: Complexer(ZDataParser._dissect_array(data)[1]), number=runs)
    # #     # print("Timeit r1:", t1 / runs * 1e6, "μs")  # , "s")

    # #     # t2 = timeit.timeit(lambda: ZDataParser.parse_z_array(data), number=runs)
    # #     # print("Timeit r2:", t2 / runs * 1e6, "μs")  # , "s")

    # #     res = ZDataParser.parse_z_array(data)

    # #     if " f " in cond:
    # #         assert isinstance(res[1], np.ndarray)
    # #         assert np.allclose(res[1][1:], df["freq"][1:])
    # #     else:
    # #         assert res[1] is None
    # #     assert np.allclose(res[0], Z)  # , rtol=1e-1)
