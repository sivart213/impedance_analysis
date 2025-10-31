# -*- coding: utf-8 -*-
"""
Insert module description/summary.

@author: j2cle
Created on Thu Sep 19 11:17:44 2024
"""

from .data_ops import (
    PD_Ops,
    TypeList,
    range_maker,
    shift_space,
    apply_extend,
    ensure_unique,
    clean_key_list,
    moving_average,
    generate_labels,
    evaluate_1D_array,
    evaluate_nD_array,
)
from .value_ops import (
    convert_val,
    find_nearest,
    most_frequent,
    sanitize_types,
    convert_unix_time_array,
    convert_unix_time_value,
)
from .dataset_ops import (
    CachedColumnSelector,
    get_valid_keys,
    modify_sub_dfs,
    impedance_concat,
    dataframe_manager,
    simplify_multi_index,
    drop_common_index_key,
    rename_columns_with_mapping,
)

# from ..complex_data.complex_data import (
#     ComplexSystem,
# )
from .data_analysis import (
    Statistics,
    FittingMethods,
    # ImpedanceConfidence,
    calculate_rc_freq,
)

__all__ = [
    # "ComplexSystem",
    "most_frequent",
    "sanitize_types",
    "clean_key_list",
    "convert_unix_time_value",
    "convert_unix_time_array",
    "convert_val",
    "find_nearest",
    "ensure_unique",
    "TypeList",
    "moving_average",
    "generate_labels",
    "apply_extend",
    "shift_space",
    "range_maker",
    "CachedColumnSelector",
    "get_valid_keys",
    "modify_sub_dfs",
    "drop_common_index_key",
    "rename_columns_with_mapping",
    "dataframe_manager",
    "simplify_multi_index",
    "impedance_concat",
    "calculate_rc_freq",
    # "ImpedanceConfidence",
    "Statistics",
    "FittingMethods",
    "PD_Ops",
    "evaluate_1D_array",
    "evaluate_nD_array",
]
