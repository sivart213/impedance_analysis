# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray, ArrayLike


def spacing_consistency(
    arr: np.ndarray, axis: int = 0, eps: float = 1e-12, alpha=0.9
) -> np.ndarray:
    """
    Measure consistency of spacing along an axis.
    Returns values in [0,1], where 1 = perfectly consistent.

    Parameters
    ----------
    arr : ndarray
        Input array (1D or 2D).
    axis : int
        Axis along which to compute differences.
    eps : float
        Small constant to avoid division by zero and rounding tolerance.
    """
    diffs = np.diff(arr, axis=axis)

    med = np.median(diffs, axis=0)
    mad = np.median(np.abs(diffs - med), axis=0)
    frac = 1 - np.clip(mad / np.maximum(np.abs(med), eps), 0, 1)

    frac_r = np.abs(med) / np.maximum(np.max(np.abs(diffs), axis=0), eps)
    return frac * (alpha + (1 - alpha) * frac_r)


def _unique_peak(arr: np.ndarray, val: float | None = None) -> int | None:
    """
    Return the index of a unique peak in a 1D array.
    If `val` is supplied, return it directly. (mask := scores[0] >= 3.0 - 1e-12).sum() == 1
    If the maximum is not unique, return None.
    """

    ref = arr.max() if val is None else val
    peak_idxs = np.flatnonzero(arr == ref)

    if len(peak_idxs) == 1:
        return int(peak_idxs[0])
    return None


def resolve_rect(
    data, avail, col1=None, col2=None, scores=None, eval_polar=True, sign=1, **_
) -> NDArray[np.complex128] | None:
    """Try to resolve rectangular (real/imag) representation."""
    sign = int(np.sign(sign)) if sign != 0 else 1
    if col1 is not None and col2 is not None:
        return col1 + 1j * sign * col2

    if avail.sum() == 1:  # Case: exactly one column left
        if col1 is not None:
            return col1 + 1j * sign * data[:, avail][:, 0]
        if col2 is not None:
            return data[:, avail][:, 0] + 1j * sign * col2

    if scores is not None:  # Case: fill using scores (if provided)
        if col1 is not None:
            return col1 + 1j * sign * data[:, np.argmax(scores[3])]  # 3: imag
        if col2 is not None:
            return data[:, np.argmax(scores[1])] + 1j * sign * col2  # 1: real
        if avail.sum() >= 2:
            REAL, MAG, IMAG, PHASE = range(1, 5)
            idx_rng = np.arange(data.shape[1])
            i = r = p = m = None
            i = _unique_peak(scores[IMAG])
            p = _unique_peak(scores[PHASE])
            if i is not None:
                r = _unique_peak(scores[REAL] * (idx_rng != i) + 0.01 * (np.abs(idx_rng - i) == 1))
            if p is not None:
                m = _unique_peak(scores[MAG] * (idx_rng != p) + 0.01 * (np.abs(idx_rng - p) == 1))

            if not eval_polar or p is None or m is None:
                if i is None or r is None:
                    return data[:, idx_rng[avail][0]] + 1j * sign * data[:, idx_rng[avail][1]]
                return data[:, r] + 1j * sign * data[:, i]

            if i is not None and r is not None:
                if scores[MAG][m] * scores[PHASE][p] <= scores[REAL][r] * scores[IMAG][i]:
                    return data[:, r] + 1j * sign * data[:, i]
            if abs(data[:, p]).max() > np.pi / 2:
                return data[:, m] * np.exp(1j * np.deg2rad(data[:, p]))
            return data[:, m] * np.exp(1j * data[:, p])
    return None


def parse_z_array(
    value: Any,
    eval_polar: bool = True,
    sign: int = 1,
) -> tuple[NDArray[np.complexfloating], NDArray[np.floating] | None]:
    """
    Normalize input into (data, freq).
    - data: always complex dtype
    - freq: frequency array if detected, else None
    """
    # Normalize value to np.array with shape (n, k), n>=k
    arr0 = np.squeeze(np.array(value))
    if arr0.ndim == 0:
        arr0 = arr0[None]
    if arr0.ndim == 1:
        if np.iscomplex(arr0).all():
            return arr0, None  # 8
        elif arr0.size == 1:
            return arr0.astype(complex), None
        arr0 = arr0[:, None]
    elif arr0.ndim > 2:
        raise ValueError("Expected <= 2D array-like input.")
    if arr0.size == 0:
        return np.array([], dtype=complex), None
    if arr0.shape[0] < arr0.shape[1]:
        arr0 = arr0.T

    arr = np.real(arr0)
    need = 2
    freq = real = imag = comp_arr = None
    sign = int(np.sign(sign)) if sign != 0 else 1
    avail = np.ones(arr0.shape[1], dtype=bool)
    # --- Eval of columns with true imaginary components ---
    if (imask := (np.imag(arr0) != 0).any(axis=0)).any():
        imag, avail[imask] = np.imag(arr0[:, imask][:, 0]), False

        if (mask := (np.real(arr0) != 0).any(axis=0) & imask).any():
            comp_arr = arr0[:, np.argmax(mask)]
            if avail.sum() <= 1:
                return comp_arr, (arr[:, avail][:, 0] if avail.sum() == 1 else freq)
            real, imag = np.real(comp_arr), np.imag(comp_arr)
        arr = arr.copy()
        arr[:, imask] = np.imag(arr0[:, imask])
        if real is not None:
            avail &= ~np.isclose(arr, real[:, None]).all(axis=0)
        avail &= ~np.isclose(arr, imag[:, None]).all(axis=0)
        if avail.sum() <= 1:
            if comp_arr is not None:  # 1
                return comp_arr, (arr[:, avail][:, 0] if avail.sum() == 1 else freq)
            comp_arr = resolve_rect(arr, avail, real, imag, None, eval_polar, sign)
            return (0 + 1j * sign * imag if comp_arr is None else comp_arr), freq  # 10
        comp_arr = None
        need = 1 if real is None else 0

    # fixed trait order: 0=freq, 1=real, 2=mag, 3=imag, 4=phase
    scores = np.zeros((5, arr.shape[1]))

    # 1) Positivity fraction for all
    is_pos_sum = (arr > 0).sum(axis=0) / arr.shape[0]
    scores[:3] += is_pos_sum  # freq, real, mag
    scores[3:] += 1.0 - is_pos_sum  # imag, phase

    # 2) Monotonicity fraction for freq/real/mag
    mono_frac = np.abs(np.diff(np.argsort(arr, 0), axis=0).sum(0)) / (arr.shape[0] - 1)
    scores[:3] += mono_frac  # freq, real, mag

    # 3) Range case for phase/imag
    if imag is None:
        range_max = np.max(np.abs(arr), axis=0)
        scores[3, range_max > 90] += 1.0  # 3=imag
        scores[3, range_max > 360] += 1.0  # 3=imag
        scores[4, range_max <= 90] += 1.0  # 4=phase (0 < range_max) & ()
        scores[4, range_max <= np.pi / 2] += 1.0  # 4=phase (0 < range_max) & ()

    if avail.sum() <= need:
        comp_arr = resolve_rect(arr, avail, real, imag, scores * avail, eval_polar, sign)
        if comp_arr is not None:
            return comp_arr, freq  # 9
    else:  # 4) log spacing fraction
        with np.errstate(divide="ignore", invalid="ignore"):
            scores[0] += spacing_consistency(np.log10(abs(arr), where=(arr != 0)))

        if (mask := scores[0] >= 3.0 - 1e-12).sum() == 1:  # 0=freq  # 2, 3, 4, 5, 6, 7, 11
            freq, avail[mask] = np.real(arr[:, mask][:, 0]).astype(float), False

    if freq is None and avail.sum() > need:
        f = np.argmax(scores[0] * avail)  # 0=freq
        freq, avail[f] = np.real(arr[:, f]).astype(float), False  # 12

    if avail.sum() > need:
        col_ranges = np.abs(arr.sum(axis=0)) * avail  # 7, 11
        col_order = np.argsort(-col_ranges)[:4]
        # trait order: 2=mag, 1=real, 3=imag, 4=phase
        scores[[2, 1, 3, 4][: len(col_order)], col_order] += 1.0

    comp_arr = resolve_rect(arr, avail, real, imag, scores * avail, eval_polar, sign)
    if comp_arr is not None:
        return comp_arr, freq  # 2, 3, 4, 5, 6, 7, 11, 12

    data = arr[:, np.flatnonzero(avail)[0] if avail.any() else 0] + 1j * 0
    return data.astype(complex), freq


class Complexer:
    """Array wrapper with impedance semantics."""

    __slots__ = ("_array", "_sign")
    __array_priority__ = 1000

    def __init__(self, data: ArrayLike | None = None, sign: int = 1, eval_polar: bool = False):
        self._array: NDArray[np.complexfloating] = np.array([complex(1, 1)], dtype=complex)
        self._sign = 1
        self.sign = sign
        if data is not None:
            self._array = parse_z_array(data, eval_polar)[0]

    def resolve(self, array: ArrayLike, update: bool = True) -> NDArray[np.complexfloating]:
        """Re-parse the internal array or a new one."""
        array = parse_z_array(array, True, self.sign)[0]
        if update:
            self._array = array
        return array

    # ---------------- Arithmetic ----------------
    def __add__(self, other):
        return Complexer._from_valid(self._array + other, self._sign)

    def __radd__(self, other):
        return Complexer._from_valid(other + self._array, self._sign)

    def __sub__(self, other):
        return Complexer._from_valid(self._array - other, self._sign)

    def __rsub__(self, other):
        return Complexer._from_valid(other - self._array, self._sign)

    def __mul__(self, other):
        return Complexer._from_valid(self._array * other, self._sign)

    def __rmul__(self, other):
        return Complexer._from_valid(other * self._array, self._sign)

    def __truediv__(self, other):
        return Complexer._from_valid(self._array / other, self._sign)

    def __rtruediv__(self, other):
        return Complexer._from_valid(other / self._array, self._sign)

    def __pow__(self, power, modulo=None):
        return Complexer._from_valid(self._array**power, self._sign)

    def __neg__(self):
        return Complexer._from_valid(-self._array, self._sign)

    def __abs__(self):
        return Complexer._from_valid(abs(self._array), self._sign)

    def __eq__(self, other):
        return bool(np.array_equal(self._array, np.asarray(other)))

    def __ne__(self, other):
        return not self.__eq__(other)

    # ---------------- Container protocol ----------------
    def __len__(self):
        return len(self._array)

    def __iter__(self):
        return iter(self._array)

    def __getitem__(self, index) -> np.ndarray | int | float | complex:
        if not isinstance(index, str):
            return self._array[index]
        if hasattr(self, index):
            return getattr(self, index)
        raise KeyError(f"'{index}' is not a valid index or attribute.")

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
    def real(self):
        return self._array.real

    @property
    def imag(self):
        return self.sign * self._array.imag

    @property
    def mag(self):
        return np.abs(self._array)

    @property
    def phase(self):
        return self.sign * np.angle(self._array, deg=True)

    @property
    def slope(self):
        return self.sign * np.tan(np.angle(self._array, deg=False))

    @classmethod
    def _from_valid(cls, arr: np.ndarray, sign: int = 1, copy: bool | None = None) -> Self:
        obj = cls.__new__(cls)  # bypasses __init__
        obj._array = np.asarray(arr, dtype=complex, copy=copy)
        obj._sign = sign
        return obj
