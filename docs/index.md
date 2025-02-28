# eis_analysis

## config_management

## data_treatment
### dataset_ops.py
- **Function**: extendspace
- **Function**: range_maker
- **Function**: most_frequent
- **Function**: moving_average
- **Function**: insert_inverse_col
- **Function**: modify_sub_dfs
- **Class**: Complexer
- **Class**: Complex_Imp
### data_analysis.py
- **Function**: gen_bounds
- **Function**: ode_bounds
- **Class**: IS_Ckt
### data_ops.py
- **Function**: sig_figs_ceil
- **Function**: sanitize_types
- **Function**: convert_from_unix_time
### dict_ops.py
- **Function**: dict_level_ops
- **Function**: rename_from_subset
- **Function**: flip_dict_levels
- **Function**: dict_key_sep
- **Function**: dict_flat
- **Function**: recursive_concat
- **Function**: merge_unique_sub_dicts
- **Function**: dict_df
- **Function**: dict_to_df

## equipment
### mfia_interface.py
- **Function**: plot_measured_data
- **Class**: MFIA
- **Class**: MFIA_Freq_Sweep
### mfia_ops.py
- **Function**: parse_labone_hdf5
- **Function**: convert_mfia_data
- **Function**: convert_mfia_time
- **Function**: convert_mfia_df_for_fit
- **Function**: hz_label
- **Function**: time_eq

## string_operations
### string_evaluation.py
- **Function**: eval_string
- **Function**: common_substring
- **Function**: str_in_list
### string_manipulation.py
- **Function**: sci_note
- **Function**: re_not
- **Function**: slugify
- **Function**: eng_not

## system_utilities
### file_io.py
- **Function**: find_path
- **Function**: find_files
- **Function**: save
- **Function**: load_file
- **Function**: load_hdf
- **Class**: DataImport
- **Function**: overlap
- **Function**: get_ds_dictionaries
### file_parsers.py
- **Function**: parse_path_str
- **Function**: my_walk
- **Function**: my_filter
- **Function**: get_config
### system_info.py
- **Function**: find_drives
- **Function**: detect_windows_drives
- **Function**: detect_posix_drives

## utils
### common.py
### decorators.py
- **Function**: handle_collection
- **Function**: handle_pandas
- **Function**: handle_dicts
- **Function**: recursive
- **Function**: sanitized_input
- **Function**: sanitized_after_recursion
- **Function**: raise_error_on_invalid
- **Function**: sanitized_after_recursion_w_error
- **Function**: sanitize_types
- **Function**: is_valid_float
### plotters.py
- **Function**: measured_data_bode
- **Function**: measured_data_nyquist
- **Function**: plot_measured_data
- **Function**: add_colormap
- **Function**: get_colormap_data
- **Function**: get_style
- **Function**: map_plt
- **Function**: scatter
- **Function**: nyquist
- **Function**: bode
- **Function**: nyquist2
- **Function**: bode2
- **Function**: nyquist_combined
- **Function**: lineplot_slider
- **Function**: update
- **Function**: reset
