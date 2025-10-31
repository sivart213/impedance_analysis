from __future__ import annotations

import datetime as dt
from os import PathLike
from typing import (
    Any,
    Literal,
    TypeVar,
    TypeAlias,
)
from collections.abc import Callable, Hashable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

# --- TypeVars for generic typing ---
T = TypeVar("T")
K = TypeVar("K", bound=Hashable)
V = TypeVar("V")
XY = TypeVar("XY", bound=tuple[int, int] | tuple[str, str])
XYZ = TypeVar("XYZ", bound=tuple[int, int, int] | tuple[str, str, str])


# --- Literal types ---
IsFalse: TypeAlias = Literal[False]
IsTrue: TypeAlias = Literal[True]
IsNone: TypeAlias = Literal[None]
DefinedBool: TypeAlias = Literal[True, False]
Tristate: TypeAlias = Literal[True, False, None]

# --- Number-like and string-like ---
BaseTypes: TypeAlias = int | float | str
RealTypes: TypeAlias = float | int
NumberTypes: TypeAlias = int | float | complex
StringTypes: TypeAlias = str | bytes
DateTypes: TypeAlias = dt.date | dt.datetime | dt.timedelta | dt.time

# combined
ScalarTypes: TypeAlias = RealTypes | StringTypes
ValueTypes: TypeAlias = NumberTypes | StringTypes
DataTypes: TypeAlias = ValueTypes | DateTypes

# with numpy types
DateLike: TypeAlias = DateTypes | np.datetime64 | np.timedelta64
DataLike: TypeAlias = DataTypes | np.datetime64 | np.timedelta64

# --- Vector/Array types ---
# Specific array-like types
NPArray: TypeAlias = np.ndarray[tuple[int, ...], Any]
VectorLike: TypeAlias = NPArray | pd.Series

# Most generic array-like type
# VarArray: TypeAlias = list[Any] | tuple[Any, ...]

# Most generic array-like type
VarArray: TypeAlias = list[T] | tuple[T, ...]
# Generic array-like types that hold typical data types found in data analysis
DataArray: TypeAlias = list[DataLike] | tuple[DataLike, ...]
# Array-like types that hold values typical in raw data
ValueArray: TypeAlias = list[ValueTypes] | tuple[ValueTypes, ...]
# Array-like types that hold values typical in raw data
ScalarArray: TypeAlias = list[ScalarTypes] | tuple[ScalarTypes, ...]
# Array-like types that hold numbers
NumberArray: TypeAlias = list[NumberTypes] | tuple[NumberTypes, ...]
# Array-like types that hold strings
StringArray: TypeAlias = list[StringTypes] | tuple[StringTypes, ...]
# Array-like types that hold dates
DateArray: TypeAlias = list[DateLike] | tuple[DateLike, ...]


# --- dict/tree/map related types ---
# Strict mapping from type to type
BaseMap: TypeAlias = dict[str, str] | dict[int, int] | dict[float, float]
# Generic dict from any valid key to any value
VarDict: TypeAlias = dict[K, V]
# Generic dict from any valid key to any value
ItemDict: TypeAlias = dict[Hashable, Any]
# Generic dict with string keys to any value but with at least one level of nesting
ItemTree: TypeAlias = dict[Hashable, ItemDict]
# Dict with string keys to any value (common for JSON-like or config data)
CommonDict: TypeAlias = dict[str, Any]
# Dict with string keys and at least one level of nesting
CommonTree: TypeAlias = dict[str, CommonDict]
# Dict with valid keys to lists of values (e.g., for grouping)
DataBranch: TypeAlias = dict[Hashable, DataArray | VectorLike]
# Nested form of DataBranch
DataTree: TypeAlias = dict[Hashable, "DataBranch | DataTree"] | DataBranch
# Dict with valid keys to scalar values common in attributes or metadata
AttrDict: TypeAlias = dict[str, None | DataTypes | PathLike[Any]]
# More flexable dict for attributes might hold arrays
AttrLikeDict: TypeAlias = dict[str, None | DataTypes | PathLike[Any] | DataArray]
# Collection of types common found in datasets (that might have attributes)
DatasetLeaf: TypeAlias = DataArray | VectorLike | AttrDict
# Dict with valid keys to dataset type values (see DatasetLeaf)
DatasetBranch: TypeAlias = dict[Hashable, DatasetLeaf]
# Nested form of DatasetBranch
DatasetTree: TypeAlias = dict[Hashable, "DatasetBranch | DatasetTree"] | DatasetBranch


# --- Pandas related types ---
# basic pandas types
DF = TypeVar("DF", bound=pd.DataFrame)
PandasLike: TypeAlias = pd.DataFrame | pd.Series
DataFrameLike: TypeAlias = (
    ArrayLike
    | list[DataArray | VectorLike]
    | list[dict[Hashable, Any]]
    | dict[Hashable, DataArray | VectorLike]
)

# For functions that accept a DataFrame or dict of DataFrames
NestedDFDict: TypeAlias = dict[str, "pd.DataFrame | NestedDFDict"]

# Index and key types
MultiIndexKey: TypeAlias = tuple[BaseTypes | slice, ...]
PandasKey: TypeAlias = BaseTypes | tuple[BaseTypes | slice, ...]
IndexKey: TypeAlias = int | tuple[BaseTypes | slice, ...]
ColumnKey: TypeAlias = str | tuple[str, ...]
# MultiIndexKey = TypeVar("MultiIndexKey", bound=tuple[BaseTypes | slice, ...])
# PandasKey = TypeVar("PandasKey", bound=BaseTypes | tuple[BaseTypes | slice, ...])
# IndexKey = TypeVar("IndexKey", bound=int | tuple[BaseTypes | slice, ...])
# ColumnKey = TypeVar("ColumnKey", bound=str | tuple[str, ...])


# collections of index and key types
PandasKeys: TypeAlias = pd.Index | list[PandasKey] | tuple[PandasKey, ...]
IndexKeys: TypeAlias = pd.Index | list[IndexKey] | tuple[IndexKey, ...]
ColumnKeys: TypeAlias = pd.Index | list[ColumnKey] | tuple[ColumnKey, ...]
# collections of index and key types
IPandasKeys: TypeAlias = pd.Index | list[BaseTypes] | tuple[BaseTypes, ...]
MIPandasKeys: TypeAlias = pd.MultiIndex | list[MultiIndexKey] | tuple[MultiIndexKey, ...]


# For keys in MultiIndex operations
LevelIndex: TypeAlias = int | list[int] | None

# For mapping old column names to new column names
PandasMap: TypeAlias = dict[PandasKey, PandasKey]
IndexMap: TypeAlias = dict[IndexKey, IndexKey]
ColumnMap: TypeAlias = dict[ColumnKey, ColumnKey]

# Callable for DataFrame transformations
DFTransform: TypeAlias = Callable[[pd.DataFrame], pd.DataFrame]


# __all__ = [
#     "PathLike",
#     "Callable",
#     "Hashable",
#     "Mapping",
#     "Literal",
#     "overload",
#     "cast",
#     "Any",
#     "TypeAlias",
#     "TypeVar",
#     "Type",
#     "ParamSpec",
#     "Concatenate",
#     "Optional",
#     "Iterator",
#     "Generator",
#     "NamedTuple",
#     "ArrayLike",
#     "XY",
#     "XYZ",
#     "IsFalse",
#     "IsTrue",
#     "IsNone",
#     "DefinedBool",
#     "Tristate",
#     "BaseTypes",
#     "RealTypes",
#     "NumberTypes",
#     "StringTypes",
#     "DateTypes",
#     "ScalarTypes",
#     "ValueTypes",
#     "DataTypes",
#     "DateLike",
#     "DataLike",
#     "NPArray",
#     "VectorLike",
#     "VarArray",
#     "DataArray",
#     "ValueArray",
#     "ScalarArray",
#     "NumberArray",
#     "StringArray",
#     "DateArray",
#     "BaseMap",
#     "VarDict",
#     "ItemTree",
#     "CommonDict",
#     "CommonTree",
#     "DataBranch",
#     "DataTree",
#     "AttrDict",
#     "AttrLikeDict",
#     "DatasetLeaf",
#     "DatasetBranch",
#     "DatasetTree",
#     "DF",
#     "PandasLike",
#     "DataFrameLike",
#     "NestedDFDict",
#     "MultiIndexKey",
#     "PandasKey",
#     "IndexKey",
#     "ColumnKey",
#     "PandasKeys",
#     "IndexKeys",
#     "ColumnKeys",
#     "IPandasKeys",
#     "MIPandasKeys",
#     "LevelIndex",
#     "PandasMap",
#     "IndexMap",
#     "ColumnMap",
#     "DFTransform",
# ]
