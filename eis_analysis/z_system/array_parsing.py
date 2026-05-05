# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
import re
from enum import Flag, auto, nonmember
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray, ArrayLike

_SPLIT_RE = re.compile(r"\b(and|or|not)\b")

# class ArrayView:
#     __slots__ = ("arr", "_cached", "_diff", "_signs", "_nz_mask")

#     _cached = {"_diff", "_signs", "_nz_mask"}


class ArrayView:
    """
    Minimal, demand-driven analysis helper for numeric arrays.
    Stores the array and provides cached structural transforms
    along a chosen axis.
    """

    __slots__ = (
        "arr",
        "_axis",
        "_n",
        "_cached",
        "_abs",
        "_diff",
        "_log",
        "_log10",
        "_alog",
        "_alog10",
    )

    def __init__(self, arr: ArrayLike, axis: int | None = -1):
        arr = np.asarray(arr)
        self.arr = arr
        self._axis = -1
        self._n = arr.size

        self._cached = set()

        # lazy fields
        self._abs = None
        self._diff = None
        self._log = None
        self._log10 = None
        self._alog = None
        self._alog10 = None

        self.axis = axis
        if self.axis == -1:  # Case where check deg would not have been triggered
            self._check_deg()
        # # total size or size along the chosen axis if defined
        # if axis is not None and axis >= 0:
        #     self._axis = axis
        #     if arr.ndim > 0:
        #         self._n = arr.shape[axis]

        # self._check_deg()

    def _check_deg(self):
        # handle degenerate cases
        if self._n < 2:
            null = np.array([], float)
            self._diff = null
            if self._n < 1:
                self._abs = null
                self._log = null
                self._log10 = null
                self._alog = null
                self._alog10 = null

    def __getitem__(self, idx):
        try:
            arr = self.arr[idx]
        except Exception as exc:
            raise IndexError(
                f"Indexing with {idx} failed for array of shape {self.arr.shape}"
            ) from exc

        new = type(self)(arr, axis=self.axis if arr.ndim == self.arr.ndim else None)

        if new.n < 1:
            return new

        for name in self._cached:
            value = getattr(self, name)
            if isinstance(value, np.ndarray):  # and value.size:
                try:
                    if value.ndim == self.arr.ndim:
                        value = value[idx]
                        new._cached.add(name)
                    else:
                        value = None
                except Exception:
                    value = None
            setattr(new, name, value)

            # if value is not None:
            #     setattr(new, name, value[idx])

        new._check_deg()

        return new

    # ---------------- NumPy interop ----------------
    def __array__(self, dtype=None, copy=None):
        if dtype is None and not copy:
            return self.arr
        return np.array(self.arr, dtype=dtype or float, copy=copy)

    # --------------------------------------------------------------
    @property
    def axis(self):
        return self._axis

    @axis.setter
    def axis(self, value: int | None):
        if value is not None and value != self._axis:
            self._axis = value
            if self.arr.ndim > 0:
                self.n = self.arr.shape[value]  # if self.arr.ndim > 0 else self.arr.size
            else:
                self._check_deg()

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        self._n = value
        self._check_deg()

    @classmethod
    def asview(cls, obj: Any, axis: int | None = None) -> Self:
        """Return obj if it's already an ArrayView, else wrap it."""
        if isinstance(obj, cls):
            return obj
        return cls(obj, axis=axis)

    def abs(self) -> NDArray[Any]:
        if self._abs is None:
            self._abs = np.abs(self.arr)
            self._cached.add("_abs")
        return self._abs

    def diff(self) -> NDArray[Any]:
        if self._diff is None:
            self._diff = np.diff(self.arr, axis=self.axis)
            self._cached.add("_diff")
        return self._diff

    def log(self) -> NDArray[Any]:
        if self._log is None:
            arr = self.arr
            self._log = np.zeros_like(arr)
            if self.arr.dtype.kind == "c":
                np.log(arr, out=self._log, where=arr != 0)
            else:
                np.log(arr, out=self._log, where=arr > 0)
            self._cached.add("_log")
        return self._log

    def log10(self) -> NDArray[Any]:
        if self._log10 is None:
            arr = self.arr
            self._log10 = np.zeros_like(arr)
            if self.arr.dtype.kind == "c":
                np.log10(arr, out=self._log10, where=arr != 0)
            else:
                np.log10(arr, out=self._log10, where=arr > 0)
            self._cached.add("_log10")
        return self._log10

    def log_abs(self) -> NDArray[Any]:
        if self._alog is None:
            arr = self.abs()
            self._alog = np.zeros_like(arr)
            np.log(arr, out=self._alog, where=arr != 0)
            self._cached.add("_alog")
        return self._alog

    def log10_abs(self) -> NDArray[Any]:
        if self._alog10 is None:
            arr = self.abs()
            self._alog10 = np.zeros_like(arr)
            np.log10(arr, out=self._alog10, where=arr != 0)
            self._cached.add("_alog10")
        return self._alog10


class SigArrayView(ArrayView):
    """
    Extends ArrayView with sign-based transforms of the diff.
    """

    __slots__ = (
        "_dsigns",
        "_dsigns_nz",
    )

    def __init__(self, arr: ArrayLike, axis: int = -1):
        self._dsigns = None
        self._dsigns_nz = None

        super().__init__(arr, axis=axis)

    def _check_deg(self):
        super()._check_deg()
        if self._n < 2:
            null = np.array([], int)
            self._dsigns = null
            self._dsigns_nz = null

    # --------------------------------------------------------------

    def diff_signs(self) -> NDArray[np.integer]:
        if self._dsigns is None:
            self._dsigns = np.sign(self.diff())
            self._cached.add("_dsigns")
        return self._dsigns

    def diff_signs_nonzero(self) -> NDArray[np.integer]:
        if self._dsigns_nz is None:
            arr = self.diff_signs()
            self._dsigns_nz = arr[arr != 0]
            self._cached.add("_dsigns_nz")
        return self._dsigns_nz


class SigFlag(Flag):
    """
    Base class for multi-state structural signatures using bitwise flags.

    Subclassing
    -----------
    Subclasses define:
    - A set of concrete Flag members (via class attributes).
      These represent atomic structural truths (e.g., POSITIVE, MONOTONIC).
    - Optional *pseudo-flags*: derived boolean properties or string expressions
      that combine real flags. Pseudo-flags must be defined as lowercase
      attributes or `_name` attributes. They may be:
        • bool-returning properties or methods
        • string expressions using flag names and `and`, `or`, `not`


    Subclass Requirements
    ---------------------
    - Flag names must be valid Python identifiers.
    - Pseudo-flag attributes must not mutate state.
    - Pseudo-flag expressions must reference only valid flags/pseudo-flags.
    - Pseudo-flag expressions must not contain circular references as this will
      lead to infinite recursion.
    - Subclasses should treat flags as *evaluated truths*, not as instructions
      for computation. Computation belongs in classmethods such as `from_array`.

    Invariants
    ----------
    - `is_(...)` never mutates the instance.
    - Pseudo-flag evaluation is lazy and recursive but terminates.
    - Expressions are sandboxed: only boolean operators and known tokens allowed.

    This class provides a unified interface for querying structural state while
    allowing subclasses to define their own atomic and derived semantics.
    """

    def __repr__(self):
        base = str(super().__repr__())
        # name = type(self).__name__ + "."
        return base.replace(f"{type(self).__name__}.", "")
        # lines = []
        # for flag in ArraySignature:
        #     if flag is ArraySignature.NONE:
        #         continue
        #     lines.append(f"  {flag.name} : {bool(self & flag)}")
        # return "\n".join(lines)

    def is_(self, flag: str | Self) -> bool:
        """
        Check if the given flag or pseudo-flag is active in the current state.

        Accepts either a single flag name, a flag expression, or a pseudo-flag defined by the class.
        Flag expressions can combine multiple flags using standard combinators `and`, `or`, and `not`.
        Logical operators (&, |, ~) can also be used.


        Resolution Rules
        ----------------
        The `is_(...)` method accepts:
        - A Flag member
        - A flag name (case-insensitive)
        - A pseudo-flag name
        - A boolean expression combining flags and pseudo-flags

        Resolution proceeds as:
        1. Direct flag lookup (case-insensitive)
        2. Attribute lookup for pseudo-flags (`name` or `_name`)
        3. If the attribute is:
            • bool → returned directly
            • callable → invoked with no arguments
            • str → parsed as a boolean expression over flags/pseudo-flags
        4. Expressions are evaluated with Python's logical operators
        (`and`, `or`, `not`) after mapping each token to its boolean value.

        Parameters
        ----------
        flag : str | Self
            The name of the flag to check, or a pseudo-flag expression combining multiple flags.

        Returns
        -------
        bool
            True if the specified flag(s) are active, False otherwise.
        """
        if not isinstance(flag, str):
            return bool(self & flag)

        name = flag.strip()
        if name.upper() in self._member_map_:
            return bool(self & getattr(self, name.upper()))
        if name in self._member_map_:
            return bool(self & getattr(self, name))

        attr = getattr(self, name.lower(), getattr(self, f"_{name.lower()}", name))
        if callable(attr):
            try:
                attr = attr()
            except Exception:
                raise ValueError(
                    f"{flag!r} is a callable that cannot be evaluated for {type(self).__name__}"
                )

        if isinstance(attr, bool):
            return attr

        if isinstance(attr, str):
            text = attr.replace("&", " and ").replace("|", " or ").replace("~", " not ")
            # Build a dict of variable → boolean
            env = {}
            for part in _SPLIT_RE.split(text):
                token = part.strip()
                if not token or token in env or token in ("and", "or", "not"):
                    continue
                try:
                    env[token] = bool(self.is_(token))
                except ValueError:
                    raise ValueError(
                        f"{part!r} is neither a flag nor a pseudo-flag of {type(self).__name__}"
                    )

            if len(env) == 1:
                return next(iter(env.values()))
            if len(env) > 1:
                return eval(text, {"__builtins__": None}, env)

        raise ValueError(f"{flag!r} is neither a flag nor a pseudo-flag of {type(self).__name__}")

    @property
    def signature(self):
        """Return a list of active state names (lowercase)."""
        return [str(flag.name).lower() for flag in self]

    @classmethod
    def from_array(cls, array: ArrayLike | ArrayView):
        """
        Compute structural state from a NumPy array.

        This method should be implemented by subclasses to define how the structural signature is derived from the data.
        The `threshold` parameter can be used for any computations that require a tunable cutoff (e.g., near-monotonicity).
        """
        raise NotImplementedError(
            "Subclasses must implement from_array method to compute structural signature."
        )

    @classmethod
    def from_ndarray(cls, array: ArrayLike | ArrayView, axis: int = -1, **kwargs) -> np.ndarray:
        """Convenience method to compute signature directly from a NumPy array."""
        if axis < 0:
            return np.array([cls.from_array(array, **kwargs)], dtype=object)

        if not isinstance(array, ArrayView):
            return np.apply_along_axis(cls.from_array, axis, array)

        array.axis = axis
        if not array._cached:
            cls.from_array(array, **kwargs)

        shape = array.arr.shape
        out_shape = shape[:axis] + shape[axis + 1 :]
        out = np.empty(out_shape, dtype=object)
        # Iterate over all indices of the "other" axes
        for idx_out in np.ndindex(out_shape):
            # Build full index: insert slice(None) at the evaluation axis
            full_idx: list[int | slice] = list(idx_out)
            full_idx.insert(axis, slice(None))
            out[idx_out] = cls.from_array(array[tuple(full_idx)], **kwargs)

        # n = array.arr.shape[axis]
        # out = np.empty(n, dtype=object)

        # if axis == 0:
        #     for j in range(n):
        #         out[j] = cls.from_array(array[:, j])
        # else:
        #     for i in range(n):
        #         out[i] = cls.from_array(array[i, :])

        return out

        # apply_along_axis will call cls.from_array on each 1D slice
        # out = np.apply_along_axis(cls.from_array, axis, arr)

        # ensure the result is a 1D object array
        # return np.asarray(out, dtype=object)

        # @classmethod
        # def from_axis(cls, view: ArrayView, axis: int):
        # arr = view.arr
        # arr = ArrayView.asview(array)  # ensures reuse if already an analysis object


class ArraySignature(SigFlag):
    """Multi-state structural signature for frequency axes."""

    NONE = 0
    EMPTY = auto()
    SCALAR = auto()
    VECTOR = auto()
    NON_ZERO = auto()  # freq (no dc)
    POSITIVE = auto()  # freq
    NEGATIVE = auto()
    UNIMODAL = auto()
    SYMMETRIC = auto()
    MONOTONIC = auto()  # freq
    INCREASING = auto()  # freq
    DECREASING = auto()  # freq
    NEAR_MONOTONIC = auto()
    STRICTLY_MONOTONIC = auto()  # freq

    _flat = nonmember("monotonic and not increasing and not decreasing")

    @classmethod
    def from_array(
        cls, array: ArrayLike | ArrayView, *, threshold: float = 0.95
    ) -> "ArraySignature":
        """
        Compute structural state from a NumPy array using ArrayView
        for efficient, demand-driven computation.
        """
        s_arr = SigArrayView.asview(array)  # ensures reuse if already an analysis object

        if s_arr.n == 0:
            return cls.EMPTY

        arr = s_arr.arr
        n = s_arr.n

        state = cls.NONE

        # Sign / positivity structure
        if np.all(arr != 0):
            state |= cls.NON_ZERO
        if np.all(arr >= 0):
            state |= cls.POSITIVE
        elif np.all(arr <= 0):
            state |= cls.NEGATIVE

        # Scalar / vector structure
        if n == 1:
            return state | cls.SCALAR

        state |= cls.VECTOR

        # Monotonicity primitives
        nz = s_arr.diff_signs_nonzero()

        if nz.size == 0:
            return state | cls.MONOTONIC

        inc_deg = np.count_nonzero(nz > 0) / (n - 1)
        dec_deg = np.count_nonzero(nz < 0) / (n - 1)

        # Determine dominant direction
        deg = 0.5
        direction = cls.NONE
        if inc_deg > dec_deg:
            direction = cls.INCREASING
            deg = inc_deg
        elif dec_deg > inc_deg:
            direction = cls.DECREASING
            deg = dec_deg

        # Strict monotonicity
        if deg == 1.0:
            return state | direction | cls.MONOTONIC | cls.STRICTLY_MONOTONIC

        # Update deg to reflect flat segments as well (flat deg => value not included in inc/dec deg)
        deg += 1.0 - inc_deg - dec_deg
        if deg == 1.0:
            return state | direction | cls.MONOTONIC
        if deg >= threshold:
            state |= direction | cls.NEAR_MONOTONIC

        # Peak structure (unimodality + symmetry) for non-monotonic arrays
        # indices in `signs` where sign changes
        change_idx = np.flatnonzero(nz[1:] != nz[:-1])
        # map nz indices back to original diff indices
        nz_idx = np.flatnonzero(s_arr.diff_signs() != 0)  # peak locations in arr
        peaks = nz_idx[change_idx] + 1  # +1 because diff index k → arr[k+1]

        if peaks.size % 2 == 1:
            mid_peak = peaks[peaks.size // 2]
            center = n // 2

            if abs(mid_peak - center) <= 1:
                # Symmetry slice logic
                if np.isclose(arr[mid_peak], arr[mid_peak + 1]):
                    left = arr[mid_peak + 2 - center : mid_peak + 1]
                    right = arr[mid_peak + 1 : mid_peak + center][::-1]
                else:
                    left = arr[mid_peak + 1 - center : mid_peak + 1]
                    right = arr[mid_peak : mid_peak + center][::-1]

                if np.allclose(left, right):
                    state |= cls.SYMMETRIC

        if peaks.size == 1:
            state |= cls.UNIMODAL

        return state

    @classmethod
    def from_ndarray(cls, array: ArrayLike | ArrayView, axis: int = -1, **kwargs) -> np.ndarray:
        """Convenience method to compute signature directly from a NumPy array."""
        array = SigArrayView.asview(array)  # ensures reuse if already an analysis object
        array.axis = axis
        array.diff_signs()  # pre-cache monotonicity primitives for all slices along the axis
        return super().from_ndarray(array, axis=axis, **kwargs)


# --------------------------------------------------------------
# Complexer logics: Heuristic scoring for characteristic matching
# --------------------------------------------------------------

MAX_SCORE: dict[str | int, int] = {}
# Characteristic row indices
FREQ, REAL, MAG, IMAG, PHASE = range(0, 5)
COMP_NAMES = ["real", "imag", "mag", "phase", "alt_phase"]

PHASE_EXP_LIMIT = np.log10(180)


# 3: unique_unsorted, balance_inverse_scores, test_candidate_array
def col_compare(arr, rtol=1e-16, inverse=False):
    """
    Return a boolean matrix where entry (i,j) is True if col j is the duplicate (or inverse) of col i.
    """
    # Compare all pairs via broadcasting
    if inverse:
        diffs = np.abs(arr[:, :, None] + arr[:, None, :])  # (rows, cols, cols)
    else:
        diffs = np.abs(arr[:, :, None] - arr[:, None, :])  # (rows, cols, cols)

    mask = diffs.max(0) <= rtol * np.linalg.norm(arr, axis=0)[None, :]

    np.fill_diagonal(mask, False)  # Zero out diagonal (self-comparison)
    return mask


# 1: eval_imag_section
def unique_unsorted(arr, get_idx=False, rtol=1e-16):
    """
    Return unique columns of arr, preserving their first-seen order.
    Also return reverse indices: each original column maps to the index
    of its representative in the unique array.
    """
    mask = col_compare(arr, rtol=rtol)
    ncols = arr.shape[1]

    keep = np.ones(ncols, dtype=bool)
    for i in range(ncols):
        if keep[i]:
            # Any later column marked as duplicate of i should be dropped
            keep[i + 1 :] &= ~mask[i, i + 1 :]

    if get_idx:
        kmask = (keep & mask).nonzero()
        rev_idx = np.arange(ncols)  # start with identity mapping
        rev_idx[kmask[0]] = kmask[1]
        rep_to_pos = {rep: pos for pos, rep in enumerate(np.nonzero(keep)[0])}
        return arr[:, keep], np.array([rep_to_pos[rep] for rep in rev_idx])
    return arr[:, keep], np.array([])


# 1: parse_z_array
def eval_imag_section(arr0: np.ndarray, sign: int = 1) -> tuple[
    NDArray[np.floating],
    tuple[()] | tuple[NDArray[np.complexfloating], NDArray[np.floating] | None],
    tuple[int, ...],
]:
    """Evaluate an array for imaginary components and separate them."""
    arr: NDArray[np.floating] = np.real(arr0.copy()).astype(float)
    n_cols = arr0.shape[1]
    idx = tuple()
    parts: tuple[()] | tuple[np.ndarray, np.ndarray | None] = tuple()

    # classify columns
    has_imag = (np.imag(arr0) != 0).any(axis=0)

    if not has_imag.any():
        return unique_unsorted(arr)[0], parts, idx

    has_real = (np.real(arr0) != 0).any(axis=0)

    # --- handle mixed columns ---
    r_idx = 0  # 'real' index (assumed to be left of 'imag')
    if (has_real & has_imag).any():
        parts = (arr0[:, np.argmax(has_real & has_imag)], None)
        n_avail = (has_real & ~has_imag).sum()

        if n_avail == 0:
            return arr, parts, idx
        if n_avail == 1:
            col = arr[:, has_real & ~has_imag][:, 0]
            if not np.allclose(np.abs(parts[0].real), np.abs(col)) and not np.allclose(
                np.abs(parts[0].imag), np.abs(col)
            ):
                return arr, (parts[0], col), idx
            return arr, parts, idx

        arr[:, has_imag] = np.imag(arr0[:, has_imag])
        # insert separated complex (as real/imag), later "unique" will remove old col
        arr = np.hstack(
            [
                np.real(parts[0])[:, None],
                np.imag(parts[0])[:, None],
                arr,
            ]
        )
        idx = (0, 1)
        n_idx = 2
    else:  # --- handle pure imag columns ---
        arr[:, has_imag] = np.imag(arr0[:, has_imag])
        idx = (int(np.argmax(has_imag)),)
        r_idx = idx[0] - 1
        parts = (0 + 1j * sign * arr[:, idx[0]], None)
        n_idx = 1

    # --- remove deduplicates ---
    arr, idx_map = unique_unsorted(arr, True)
    n_cols = arr.shape[1]
    if n_idx == 1:
        r_idx = int(idx_map[r_idx]) if r_idx >= 0 else r_idx
        idx = (int(idx_map[idx[0]]),)

    # --- Evaluated exits ---
    if n_cols == n_idx:  # no avail col to get (n_cols = 1-2)
        return arr, parts, idx

    if n_cols - n_idx > 1:  # Return if still too many cols to parse (n_cols = 3-4 or more)
        if r_idx > idx[-1]:  # Make sure "imag" still follows the col it did.
            cols = np.r_[0 : idx[-1], idx[-1] + 1 : r_idx + 1, idx[-1], r_idx + 1 : n_cols]
            arr = arr[:, cols]
            idx = ((cols == idx[-1]).nonzero()[0].item(),)
        parts = tuple()
    elif n_cols == 3:  # Implies n_idx == 2 (mixed case) & other is 'freq' col
        parts = (parts[0], arr[:, -1])
    elif n_cols == 2:  # Implies n_idx == 1 (imag case) & other is 'real' col
        parts = ((arr[:, 1 - idx[0]] + parts[0]).astype(complex), None)
    else:
        parts = tuple()
    return arr, parts, idx


# --------------------------------------------------------------
# Complexer logics: Scoring functions
# --------------------------------------------------------------


# 1: check_pairing
def score_pair_ordering(
    scores: np.ndarray, masks: np.ndarray, char_1: int = 0, char_2: int = 1, offset: int = 1
):
    """Update a scoring matrix with ordering bonuses for paired characteristics."""
    offset = abs(offset)
    bonus = np.zeros_like(scores, dtype=float)

    expected = masks[char_1, :-offset] & masks[char_2, offset:]
    bonus[char_1, :-offset][expected] += 1.0
    bonus[char_2, offset:][expected] += 1.0

    if bonus.max() == 0:
        return scores
    return scores + bonus / bonus.max()


def check_pairing(scores, nulls, strict=True, **_):
    """Scores based on pairing of REAL/IMAG and MAG/PHASE characteristics."""
    order_mask = scores >= scores.max(axis=1, keepdims=True) * 0.9
    if strict:
        order_mask = order_mask & (nulls.astype(bool))
    scores = score_pair_ordering(scores, order_mask, 1, 3)
    scores = score_pair_ordering(scores, order_mask, 2, 4)
    return scores, nulls


def check_freq_score(scores, nulls, **kwargs):
    """
    Enforce decisive assignment if exactly one column reaches the max score
    for trait 0 (freq) only.
    """
    tol = kwargs.get("tol", 1e-12)
    idxs = np.where(scores[0] + tol >= MAX_SCORE[0])[0]

    if idxs.size == 1:
        f_idx = idxs.item()
        nulls[0] = 0  # null row
        nulls[:, f_idx] = 0  # null col
        nulls[0, f_idx] = 1  # keep peak

    return scores, nulls


def check_pos_frac(scores, nulls, arr, **_):
    """Scores based on fraction of non-negative values in each column."""
    non_neg_frac = (arr >= 0).mean(axis=0)

    nulls[0, non_neg_frac < 1.0] = 0  # freq should be all non-negative
    nulls[2, non_neg_frac < 1.0] = 0  # mag should be all non-negative

    scores[:3] += non_neg_frac  # freq, real, mag
    scores[3:] += np.maximum(non_neg_frac, 1 - non_neg_frac)
    return scores, nulls


def check_monotonicity(scores, nulls, arr, **_):
    """Scores based on monotonicity of each column."""
    diffs = np.diff(arr, axis=0)
    mono_frac = np.abs((diffs / np.maximum(np.ptp(arr, 0), 1e-12)).sum(0))
    noise_frac = 1 - abs((diffs > 0).mean(0) + (np.diff(diffs, axis=0) > 0).mean(0) - 1)
    mask = (mono_frac < 0.2) & (noise_frac > 0.8)
    mono_frac[mask] = noise_frac[mask]
    # noise_frac = np.maximum(np.ptp(diffs, 0), 2e-12) / np.maximum(np.ptp(np.abs(diffs), 0), 1e-12) - 1
    scores[:3] += mono_frac
    return scores, nulls


def check_ranges(scores, nulls, arr, log_arr, **_):
    """Scores based on expected ranges for imaginary and phase data."""
    a_arr = np.abs(arr)
    a_min, a_max = np.min(arr, axis=0), np.max(a_arr, axis=0)
    nulls[4, (a_max > 180) | (np.ptp(arr, axis=0) > 180)] = 0

    lim = np.where(a_max <= np.pi, np.pi, 180.0)
    range_min = np.minimum(np.maximum(a_min, -lim), 0)
    above_min = arr >= range_min

    # scores[3] += (a_arr > 180).mean(axis=0)
    scores[3] += (np.abs(log_arr) > PHASE_EXP_LIMIT).mean(axis=0) * 2  # above & below phase limit
    scores[4] += (above_min & (arr <= range_min + lim)).mean(axis=0) * 0.75  # w/in 180 deg
    scores[4] += (above_min & (arr <= range_min + lim / 2)).mean(axis=0) * 0.75  # w/in 90 deg
    scores[4] += (log_arr > -PHASE_EXP_LIMIT).mean(axis=0) * 0.5
    return scores, nulls


def check_spacing(scores, nulls, arr, log_arr, **_):
    """Scores based on even spacing in log-space (or normal space if scores are low)."""
    diffs = np.diff(log_arr, axis=0)  # spacing in log-space
    tol = 1e-12 * np.maximum(1, np.linalg.norm(log_arr, axis=0))  # tol scaled to data magnitude
    score = np.mean(np.abs(diffs - np.median(diffs, axis=0)) <= tol, axis=0)
    if not (score > 0.95).any():
        diffs = np.diff(arr, axis=0)
        tol = 1e-12 * np.maximum(1, np.linalg.norm(arr, axis=0))
        score = np.maximum(score, np.mean(np.abs(diffs - np.median(diffs, axis=0)) <= tol, axis=0))
    scores[0] += score
    return scores, nulls


HEURISTIC_MAP = {
    "freq": {check_pos_frac, check_monotonicity, check_spacing, check_freq_score},
    "real": {check_pos_frac, check_monotonicity, check_pairing},
    "mag": {check_pos_frac, check_monotonicity, check_pairing},
    "imag": {check_pos_frac, check_ranges, check_pairing},
    "phase": {check_pos_frac, check_ranges, check_pairing},
}
TARGETS = set().union(HEURISTIC_MAP.keys())
HEURISTICS = set().union(*HEURISTIC_MAP.values())
SCORE_HEURISTICS = {check_freq_score, check_pairing}

NEEDS_LOG = {check_ranges, check_spacing}

MAX_SCORE |= {k: len(v - {check_freq_score}) for k, v in HEURISTIC_MAP.items()}
MAX_SCORE["imag"] += 1
MAX_SCORE["phase"] += 1
MAX_SCORE |= {i: MAX_SCORE[t] for i, t in enumerate(HEURISTIC_MAP)}

MAX_SCORE_ARR = np.array([MAX_SCORE[n] for n in range(5)])

# --------------------------------------------------------------
# Complexer logics: Scoring director and it's utilities
# --------------------------------------------------------------


# 1: compute_scores
def balance_inverse_scores(scores, nulls, arr, **_):
    """
    Adjust scores so that inverse columns (same up to sign)
    get equalized in rows 3 and 4 by taking the maximum.
    """
    # Boolean matrix: mask[i,j] True if col j is inverse of col i
    mask = col_compare(arr, inverse=True)  # shape (n_cols, n_cols)
    if not mask.any():
        return scores, nulls

    z_mask = scores == 0
    mask_with_self = mask | np.eye(mask.shape[0], dtype=bool)

    scores[1:3][(~mask.any(1)) & (scores[1:3] != 0)] += 1
    scores[3] = (mask_with_self * scores[3]).max(axis=1)
    scores[4] = (mask_with_self * scores[4]).max(axis=1)
    scores[z_mask] = 0
    return scores, nulls


# 1: compute_scores
def set_knowns_to_max(scores, nulls, max_idxs=(), max_mod=1.25, **_):
    """
    Adjusts scores so that inverse columns (same up to sign)
    get equalized in rows 3 and 4 by taking the maximum.
    """
    val = scores.max() * max_mod
    for pair in max_idxs:
        if len(pair) == 2:
            scores[:, pair[1]] = 0
            scores[tuple(pair)] = val
            nulls[:, pair[1]] = 0
            nulls[tuple(pair)] = 1
    return scores, nulls


# 1: parse_z_array
def compute_scores(
    arr: np.ndarray, strict: bool = True, targets: set = TARGETS, **kwargs
) -> np.ndarray:
    """Generates a scoring matrix for complex characteristic matching."""
    nulls = np.ones((5, arr.shape[1]))
    scores = np.zeros((5, arr.shape[1]))

    targs = targets & TARGETS

    if not targs:
        targs = {t.lower()[:4] if t.lower()[0] != "m" else t.lower()[:3] for t in targets}
        targs &= TARGETS
        if not targs:
            return scores

    kwargs = {"arr": arr, "strict": strict} | kwargs.copy()

    if targs == TARGETS:
        heuristics = HEURISTICS

        with np.errstate(divide="ignore", invalid="ignore"):
            l_arr = np.zeros_like(arr)
            np.log10(np.abs(arr), out=l_arr, where=arr != 0)
            kwargs["log_arr"] = l_arr
    else:
        # Union of heuristics for requested targets
        heuristics = set().union(*(HEURISTIC_MAP[t] for t in targs))

        if not heuristics.isdisjoint(NEEDS_LOG):
            with np.errstate(divide="ignore", invalid="ignore"):
                l_arr = np.zeros_like(arr)
                np.log10(np.abs(arr), out=l_arr, where=arr != 0)
                kwargs["log_arr"] = l_arr

    # Run them with standardized contract
    for h in heuristics - SCORE_HEURISTICS:
        scores, nulls = h(scores, nulls, **kwargs)

    scores, nulls = set_knowns_to_max(scores, nulls, kwargs.get("max_idxs", ()), 1.1)

    if check_freq_score in heuristics:
        scores, nulls = check_freq_score(scores, nulls, **kwargs)

    if strict:
        scores *= nulls

    if check_pairing in heuristics:
        scores, nulls = balance_inverse_scores(scores, nulls, arr)
        scores, nulls = check_pairing(scores, nulls, **kwargs)
    else:
        scores, nulls = balance_inverse_scores(scores, nulls, arr)

    if strict:
        scores *= nulls

    return scores


# --------------------------------------------------------------
# Complexer logics: Heuristic scoring for characteristic matching
# --------------------------------------------------------------
# 1: check_max_scores
def normalize_scores(
    scores: np.ndarray, mode: int = 3, eps: float = 1e-12, fast=False
) -> np.ndarray:
    """
    Normalize a score matrix by its maxima with flexible direction control.

    Parameters
    ----------
    scores : ndarray of shape (n_rows, n_cols)
        The scoring matrix to normalize.

    mode : int, default=3
        Normalization mode:
        0 : column-wise only
        1 : row-wise only
        2 : always blend row- and column-wise
       <0 : column-wise, blend with row-wise if ambiguity detected
       >2 : row-wise, blend with column-wise if ambiguity detected

    Returns
    -------
    norm_scores : ndarray of shape (n_rows, n_cols)
        The normalized score matrix.
    """
    scores = scores.copy()
    nulls = scores == 0
    scores[nulls] = eps

    if 0 <= mode <= 1:
        scores = scores / scores.max(axis=mode, keepdims=True)  # * ~nulls
    else:
        row_norm = scores / scores.max(axis=1, keepdims=True)  # * ~nulls
        col_norm = scores / scores.max(axis=0, keepdims=True)  # * ~nulls

        if (
            mode == 2
            or (mode < 0 and np.any((col_norm == 1).sum(axis=1) > 1))
            or (mode > 2 and np.any((row_norm == 1).sum(axis=0) > 1))
        ):
            scores = (row_norm + col_norm) / 2  # blend both
        else:
            scores = col_norm if mode < 0 else row_norm

    if fast:
        scores[np.abs(scores) <= eps] = 0
        return scores * ~nulls

    return np.round(scores, decimals=12) * ~nulls


# 1: parse_z_array
def check_max_scores(scores, tol=1e-12, knowns=(), mode=3, eps=1e-12):
    """
    Max-count heuristic across all traits.
    - Uses MAX_SCORE[trait] as the valid ceiling for each trait.
    - Normalizes results internally.
    - Applies 0.75/0.25 redistribution if there are peak values.
    """
    max_scores = MAX_SCORE_ARR.copy()
    for kn in knowns:
        if len(kn) == 2:
            max_scores[kn[0]] += 1
    is_max = scores + tol >= max_scores[:, None]
    norm_scores = normalize_scores(scores, mode, eps)
    if is_max.any():
        norm_scores *= 0.75
        norm_scores[is_max] = 1
    return norm_scores


# 1: resolve_with_scores
def get_peaks_pairs(
    scores: np.ndarray,
    row_a: int,
    row_b: int,
    tol: float = 1e-12,
    in_order: bool = True,
    is_adjacent: bool = True,
) -> np.ndarray:
    """
    Return candidate (col_a, col_b) index pairs for two score rows.

    Parameters
    ----------
    scores : ndarray of shape (n_rows, n_cols)
        Scoring matrix.
    row_a, row_b : int
        Row indices to compare (e.g. REAL vs IMAG).
    tol : float, default=1e-12
        Tolerance for considering values equal to the maximum.
    in_order : bool, default=True
        If True, require col_a < col_b (ordering).
    is_adjacent : bool, default=True
        If True, require adjacency (col_a ± 1 == col_b).
        If False, allow any distinct indices.

    Returns
    -------
    candidates : ndarray of shape (n_pairs, 2)
        Array of (col_a, col_b) index pairs.
    """
    # Find peak indices for each row
    idx_a = np.asarray(scores[row_a] >= np.abs(scores[row_a].max() - tol)).nonzero()[0]
    idx_b = np.asarray(scores[row_b] >= np.abs(scores[row_b].max() - tol)).nonzero()[0]

    # Cartesian product of candidates
    if idx_a.size == 0 or idx_b.size == 0:
        return np.empty((0, 2), dtype=int)

    if idx_a.size == 1 and idx_b.size == 1:
        a, b = idx_a.item(), idx_b.item()
        if a == b or (in_order and a > b) or (is_adjacent and abs(a - b) != 1):
            return np.empty((0, 2), dtype=int)
        return np.array([[a, b]], dtype=int)

    aa, bb = np.meshgrid(idx_a, idx_b, indexing="ij")
    pairs = np.column_stack([aa.ravel(), bb.ravel()])

    # Distinctness
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]

    adj_mask = np.abs(pairs[:, 0] - pairs[:, 1]) == 1
    order_mask = pairs[:, 0] < pairs[:, 1]

    t1_mask = adj_mask & order_mask
    if in_order and is_adjacent:
        return pairs[t1_mask]
    t2_mask = adj_mask & ~t1_mask
    if is_adjacent:
        return np.vstack([pairs[t1_mask], pairs[t2_mask]])
    t3_mask = order_mask & ~t1_mask
    if in_order:
        return np.vstack([pairs[t1_mask], pairs[t3_mask]])
    t4_mask = ~(t1_mask | t2_mask | t3_mask)
    return np.vstack([pairs[t1_mask], pairs[t2_mask], pairs[t3_mask], pairs[t4_mask]])


# 1: resolve_with_scores
def test_candidate_array(data: np.ndarray, idxs: list[int], sign: int, mode: str) -> dict:
    """
    Validate a candidate complex reconstruction using np.unique.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_cols)
        Each column is a component vector (e.g., REAL, IMAG, MAG, PHASE).
    idxs : (int, int)
        Column indices used for reconstruction.
    sign : int
        Sign convention for imaginary/phase component.
    mode : {'rect', 'polar'}
        Reconstruction mode.

    Returns
    -------
    {
      'complex': ndarray of shape (n_samples,),
      'original_reproduced': bool,
      'explained_cols': list[int],
      'explained_forms': list[str]
    }
    """
    # Build candidate complex array
    n_orig = data.shape[1]
    deg = True
    if mode == "rect":
        comp = data[:, idxs[0]] + sign * 1j * data[:, idxs[1]]
        f_idxs = [n_orig, n_orig + 1]
    elif mode == "polar":
        f_idxs = [n_orig + 2, n_orig + 3]
        if abs(data[:, idxs[1]]).max() > np.pi / 2:
            comp = data[:, idxs[0]] * np.exp(sign * 1j * np.deg2rad(data[:, idxs[1]]))
        else:
            deg = False
            comp = data[:, idxs[0]] * np.exp(sign * 1j * data[:, idxs[1]])
    else:
        raise ValueError("mode must be 'rect' or 'polar'")

    # Derive real-valued forms from candidate
    derived = np.column_stack(
        [
            np.real(comp),
            np.imag(comp),
            np.abs(comp),
            np.angle(comp, deg),
            np.angle(comp, not deg),
        ]
    )  # shape: (n_samples, 5)

    # Stack original + derived
    mask = col_compare(np.abs(np.hstack([data, derived])), 1e-12)

    # Check that the chosen reconstruction columns reproduce the expected derived forms
    if not all(mask[idxs[i], f_idxs[i]] for i in range(2)):
        return {}

    submask = mask[:n_orig, n_orig:]

    submask[idxs] = False

    dupl_forms = {col.item(): COMP_NAMES[match_idx] for col, match_idx in zip(*submask.nonzero())}

    return {
        "array": comp,
        "explained": len(dupl_forms),
        "explained_forms": dupl_forms,
    }


# 1: parse_z_array
def resolve_with_scores(
    data, scores, eval_polar=True, sign=1, check_order=True, require_single=True, fast=True
) -> tuple[np.ndarray | None, np.ndarray]:
    """
    Resolve a complex-valued array from candidate component columns using a scoring matrix.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_cols)
        The dataset columns. Each row in `scores` corresponds to a characteristic
        (REAL, MAG, IMAG, PHASE), and each column in `scores` corresponds
        to a column in `data`.
    scores : ndarray of shape (4, n_cols)
        Scoring matrix indicating how well each dataset column matches a characteristic.
        It is recommended that the scores are normalized prior to passing.
    eval_polar : bool, default=True
        If True, allow magnitude/phase resolution. If False, MAG/PHASE are ignored.
    sign : int, default=1
        Sign convention for imaginary or phase component.
    check_order : bool, default=True
        Requires that the pairs be in expected order.
    require_single : bool, default=True
        Requires that the peak be the only option.
    fast : bool, default=True
        If True, stop at the first valid candidate. If False, iterate all candidates
        and choose the one with highest score (ties broken by explained count).

    Returns
    -------
    complex_array : ndarray of shape (n_samples,) or None
        The resolved complex-valued array, or None if resolution is not possible.
    """
    s_matrix = scores.copy()

    # Candidate pairs
    rect_pairs = get_peaks_pairs(s_matrix, REAL, IMAG, in_order=check_order, is_adjacent=True)
    pol_pairs = np.empty((0, 2), dtype=int)
    if eval_polar:
        pol_pairs = get_peaks_pairs(s_matrix, MAG, PHASE, in_order=check_order, is_adjacent=True)

    # Enforce require_single: if more than one candidate, discard
    if require_single:
        if rect_pairs.shape[0] > 1:
            rect_pairs = np.empty((0, 2), dtype=int)
        if pol_pairs.shape[0] > 1:
            pol_pairs = np.empty((0, 2), dtype=int)

    # Sort candidates by score
    rect_best, rect_info, rect_score = None, {}, -1
    polar_best, polar_info, pol_score = None, {}, -1

    # Iterate rect candidates
    for r, i in rect_pairs:
        info = test_candidate_array(data, [r, i], sign, "rect")
        if not info:
            continue
        score_val = scores[REAL, r] + scores[IMAG, i]
        if fast:
            rect_best, rect_info, rect_score = [r, i], info, score_val
            break
        if score_val + info["explained"] > rect_score + rect_info.get("explained", -1):
            rect_best, rect_info, rect_score = [r, i], info, score_val

    # Iterate polar candidates
    for m, p in pol_pairs:
        info = test_candidate_array(data, [m, p], sign, "polar")
        if not info:
            continue
        score_val = scores[MAG, m] + scores[PHASE, p]
        if fast:
            polar_best, polar_info, pol_score = [m, p], info, score_val
            break
        if score_val + info["explained"] > pol_score + polar_info.get("explained", -1):
            polar_best, polar_info, pol_score = [m, p], info, score_val

    # Compare rect vs polar if both exist
    if rect_best is not None and polar_best is not None:
        rect_score += rect_info["explained"]
        pol_score += polar_info["explained"]
        if rect_score == pol_score or (rect_score <= 1.75 and pol_score <= 1.75):
            # Tie-breaker 1: median subtraction
            rect_score -= np.median(scores[REAL]) + np.median(scores[IMAG])
            pol_score -= np.median(scores[MAG]) + np.median(scores[PHASE])
        if rect_score < pol_score:
            rect_best = None  # kill to bypass case 1

    # Case 1: rectangular best/only
    if rect_best is not None:
        r, i = rect_best
        s_matrix[:, [r, i]] = 0
        s_matrix[REAL, r] = s_matrix[IMAG, i] = 1
        return rect_info["array"], s_matrix

    # Case 2: polar best/only
    if polar_best is not None:
        m, p = polar_best
        s_matrix[:, [m, p]] = 0
        s_matrix[MAG, m] = s_matrix[PHASE, p] = 1
        return polar_info["array"], s_matrix

    # Case 3: neither valid
    return None, scores


# --------------------------------------------------------------
# Complexer logics: Heuristic scoring for characteristic matching
# --------------------------------------------------------------


def parse_z_array(
    value: Any,
    eval_polar: bool = True,
    sign: int = 1,
    strict: bool = True,
) -> tuple[NDArray[np.complexfloating], NDArray[np.floating] | None]:
    """
    Normalize input into (data, freq) if possible where data is a 1d complex array and
    freq is optional 1d real array (or None).

    Parameters
    ----------
    value : Any
        Input array-like data.
    eval_polar : bool
        Whether to consider polar representations (magnitude/phase).
    sign : int
        Sign convention for imaginary components.
    strict : bool
        Whether to strictly enforce common characteristics (see notes).

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None]
        Parsed complex data array and optional frequency array.

    """
    # Normalize value to np.array with shape (n, k), n>=k
    arr0 = np.array(value)
    arr0 = np.squeeze(arr0) if 1 in arr0.shape else arr0
    # arr0 = np.squeeze(np.array(value))
    if arr0.ndim == 1 and np.iscomplex(arr0).any():
        return arr0, None
    if arr0.size <= 1:
        return np.atleast_1d(arr0).astype(complex), None
    if arr0.ndim != 2:
        if arr0.ndim < 2:
            arr0 = arr0[:, None]
        else:
            raise ValueError("Expected <= 2D array-like input.")

    arr0 = arr0.T if arr0.shape[0] < arr0.shape[1] else arr0

    freq = f_idx = comp_arr = None
    sign = -1 if sign < 0 else 1

    # look for col with 1j component (ie complex)
    arr, parts, idx = eval_imag_section(arr0, sign)

    if parts:
        if strict and parts[1] is not None:
            scores = compute_scores(parts[1][:, None], strict, targets={"freq"})
            if scores[0, 0] < MAX_SCORE[0] - 1e-12:  # or np.any(scores[1:, 0] != 0):
                parts = (parts[0], None)
        return parts

    n_idx = len(idx)
    m_pair = tuple()
    if n_idx == 2:  # comp_arr found but freq needs parsing
        comp_arr = arr[:, 0] + 1j * arr[:, 1]
        arr = arr[:, 2:]
    elif n_idx == 1:
        m_pair = (3, idx[0])

    # --- Scoring ---
    scores = compute_scores(arr, strict, max_idxs=[m_pair], max_mod=1.0)

    # Make best guess at frequency column if still needed
    n_scores = check_max_scores(scores, 0.1, knowns=[m_pair])

    if comp_arr is None:  # Not (fully) found in eval imag
        comp_arr, n_scores = resolve_with_scores(arr, n_scores, eval_polar, sign)

    if comp_arr is not None:
        peak_idxs = (scores[0] >= MAX_SCORE[0] - 1e-12).nonzero()[0]
        f_idx = peak_idxs.item() if peak_idxs.size == 1 else -1
        # comp_arr cols removed from consideration in n_scores -> freq from remaining columns
        if f_idx >= 0 and n_scores[0, f_idx] == n_scores[:, f_idx].max():
            return comp_arr, arr[:, f_idx]
        return comp_arr, None

    row_max_mask = n_scores >= n_scores.max(axis=1, keepdims=True)
    row_max_mask[n_scores.max(axis=1) == 0] = False
    f_idx = int(row_max_mask[0].nonzero()[0][0])

    # Conflict if any other row has f_idx as a max and that max > 0
    if f_idx >= 0 and row_max_mask[1:, f_idx].nonzero()[0].size == 0:
        freq = arr[:, f_idx]
        n_scores[:, f_idx] = 0

    N = 2  # number of boolean flags
    i = (1 << N) - 1  # start at 2^N - 1 (binary countdown)
    while comp_arr is None and i >= 0:
        flags = [bool(i & (1 << bit)) for bit in reversed(range(N))]
        comp_arr, n_scores = resolve_with_scores(arr, n_scores, eval_polar, sign, *flags)
        i -= 1

    if comp_arr is None:
        comp_arr = arr[:, 0] + 1j * arr[:, 1] if arr0.shape[1] >= 2 else arr[:, 0] + 1j * 0

    return comp_arr, freq


# def unique_unsorted_old(arr, rtol=1e-16, inverse=False):
#     """
#     Return unique columns of arr, preserving their first-seen order.
#     """
#     mask = col_compare(arr, rtol=rtol, inverse=inverse)
#     ncols = arr.shape[1]

#     keep = np.ones(ncols, dtype=bool)
#     for i in range(ncols):
#         if keep[i]:
#             # Any later column marked as duplicate of i should be dropped
#             keep[i + 1 :] &= ~mask[i, i + 1 :]

#     return arr[:, keep]

if __name__ == "__main__":
    a0 = np.logspace(-3, 7)

    test_sig0 = ArraySignature.from_array(np.ones_like(a0) * 10)

    print(test_sig0.is_("flat"))
    # test_sig0.is_("a")

    test_sig1 = ArraySignature.from_array(a0)

    print(test_sig1.is_("flat"))

    b0 = np.hstack([a0, a0[1:][::-1]])
    test_sig2 = ArraySignature.from_array(b0)

    ass = test_sig0.VECTOR

    b1 = np.hstack([a0[1:], a0[::-1]])
    test_sig3 = ArraySignature.from_array(b1)

    # b0 = np.stack([a0, a0[1:][::-1]])

    c = np.cos(np.linspace(0, 4 * np.pi, 100))
    test_sig4 = ArraySignature.from_array(c[:-1])

    from testing.rc_ckt_sim import RCCircuit  # noqa: F401
    from eis_analysis.z_system.system import ComplexSystem
    from eis_analysis.impedance_supplement.ops import get_impedance  # noqa: F401

    ckt = RCCircuit(true_values=[24, 1e9, 1e-11], noise=0.01)
    rc_system = ComplexSystem(data=ckt.Z_noisy, frequency=ckt.freq, area=25, thickness=500e-4)

    test_arr = rc_system.get_df("freq", "Z", "Z.mag", "Z.phase").to_numpy()

    test_parsed0 = ArraySignature.from_ndarray(test_arr, 0)
