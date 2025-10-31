# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""


import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QMessageBox

from ..z_system.system import ComplexSystem
from ..impedance_supplement import get_impedance, parse_parameters
from ..impedance_supplement.linkk import linKK
from ..impedance_supplement.model_ops import parse_model_groups, get_valid_sub_model


class DataGenerator:
    """Class to store data for plotting graphs."""

    def __init__(self, options=None, **kwargs):
        self.model = kwargs.get("model", "R0-p(R1,C1)")
        # self.linkk = ComplexSystem()
        self.options = options if options is not None else {}
        self.generated = self.get([1e1, 1e5, 1e-6], "R0-p(R1,C1)")
        self.linkk_info = ("R0-L0-K0", {"R0": 1e1, "L0": 1e-3, "K0_0": 1e6, "K0_1": 1})

    def parse_parameters(self, model=None):
        """Get the parameters of the model."""
        if model is None:
            model = self.model
        if not isinstance(model, str):
            QMessageBox.warning(
                None,
                "Warning",
                f"Model must be a string. Current value is {model} of type {type(model)}",
            )
            model = self.model
        if model.lower() == "linkk":
            return ["M", "mu"]
        return parse_parameters(model)

    def get_freq(
        self,
        freq=None,
        **kwargs,
    ) -> np.ndarray:
        """Create the fit data based on the current parameter values."""
        freq_num = kwargs.get("freq_num", self.options.get("freq_num", 200))
        if freq is None:
            freq_start = kwargs.get("freq_start", self.options.get("freq_start", -4.5))
            freq_stop = kwargs.get("freq_stop", self.options.get("freq_stop", 7))
            return np.logspace(freq_start, freq_stop, freq_num)

        sim_param_freq = kwargs.get("sim_param_freq", self.options.get("sim_param_freq", True))
        interp = kwargs.get("interp", self.options.get("interp", True))
        if interp and not sim_param_freq and freq_num > 2 * (u_num := len(np.unique(freq)) - 1):
            # if interp true, and sim_param_freq false, and freq_num > 2 * unique freq - 1
            num = (freq_num // u_num) * u_num + 1
            return np.logspace(min(np.log10(freq)), max(np.log10(freq)), num)
        return freq

    def get(
        self,
        params_values,
        model=None,
        freq=None,
        **kwargs,
    ) -> ComplexSystem:
        """Create the fit data based on the current parameter values."""
        if model is None:
            model = self.model

        if model.lower() == "linkk":
            # return self.generated
            model = self.linkk_info[0]
            params_values = list(self.linkk_info[1].values())

        freq = self.get_freq(freq, **kwargs)

        area = kwargs.get("area", self.options.get("area", 25))
        thickness = kwargs.get("thickness", self.options.get("thickness", 450e-4))

        if not params_values:
            raise ValueError("No parameter values provided.")
        try:
            Z = get_impedance(freq, *params_values, model=model)
            # circuit_func = wrapCircuit(model, {})
            # Z = np.array(np.hsplit(circuit_func(freq, *params_values), 2)).T
        except (IndexError, AssertionError) as exc:
            raise IndexError("List index out of range") from exc
        except (TypeError, ValueError, UnboundLocalError) as exc:
            raise ValueError("Invalid input") from exc

        self.generated = ComplexSystem(
            Z,
            # Z[:, 0] + 1j * Z[:, 1],  # array
            frequency=freq,
            thickness=thickness,
            area=area,
            model=model,
            **kwargs,
        )
        return self.generated

    def get_sub_groups(
        self,
        params_values,
        model=None,
        shift=False,
        re_order_shift=False,
        use_numbers=False,
        sub_groups=(),
        **kwargs,
    ):
        """
        Create a sequence of generated curves based on top-level series groups.

        Top-level series groups are defined as series-connected groups not within a parallel block.
        """
        if model is None:
            model = self.model

        param_dict = {k: v for k, v in zip(self.parse_parameters(model), params_values)}

        # Parse the model into top-level series groups
        parse_method = "numbers" if use_numbers else "series"
        if sub_groups:
            groups = []
            for gr in sub_groups:
                grp, _ = get_valid_sub_model(model, gr)
                if not grp:
                    groups = parse_model_groups(model, parse_method)
                    break
                groups.append(grp)
        else:
            groups = parse_model_groups(model, parse_method)

        if not params_values:
            return

        results = []

        # Generate data for each sub-group using the `generate` method
        for group in groups:
            try:
                result = self.get(
                    params_values=[param_dict[k] for k in self.parse_parameters(group)],
                    model=group,
                    **kwargs,
                )
            except (IndexError, ValueError):
                pass
            else:
                results.append(result)

        if shift:
            ordered_res = list(results)
            if re_order_shift:
                # Sort by RC peak frequency (descending)
                ordered_res = sorted(
                    results,
                    key=lambda res: res.frequency[np.argmax(np.abs(res.Z.imag))],
                    reverse=True,
                )

            # Perform a rolling addition of max(Z.real)
            rolling_real_shift = 0
            for result in ordered_res:
                max_real = np.max(result.Z.real)
                result.complexer_obj += rolling_real_shift
                rolling_real_shift += max_real

        return results

    def get_linkk(
        self,
        data,
        c=0.5,
        max_M=200,
        add_cap=False,
        f_min=-4.5,
        f_max=7,
        fit_type="complex",
        **kwargs,
    ):
        """Run the Lin-KK fit based on the selected model."""
        # Filter the scatter_data based on f_min and f_max
        area = kwargs.get("area", self.options.get("area", 25))
        thickness = kwargs.get("thickness", self.options.get("thickness", 450e-4))

        if isinstance(data, ComplexSystem):
            data.area = area
            data.thickness = thickness
            df = data.get_df("freq", "real", "imag")
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            df = self.generated.get_df("freq", "real", "imag")

        df = df[(df["freq"] >= f_min) & (df["freq"] <= f_max)]
        f = df["freq"].to_numpy(copy=True)
        Z = ComplexSystem(df[["real", "imag"]]).Z.array  # array
        # Z = ComplexSystem(df[["real", "imag"]]).Z.array  # array
        # Z = ComplexSystem(df[["Z.real", "Z.imag"]]).Z
        # Direct Access to ComplexSystemDx
        M, mu, Z_linKK, model, params = linKK(
            f,
            Z,
            c=c,
            max_M=max_M,
            fit_type=fit_type,
            add_cap=add_cap,
            verbose=False,  # Manually added to package
        )

        self.linkk_info = (model, params)

        # Create a DataFrame from Z_linKK
        self.generated = ComplexSystem(
            Z_linKK,  # array
            frequency=f,
            thickness=thickness,
            area=area,
        )

        return self.generated, (M, mu)


class DataManager:
    """Class to store and manage raw data."""

    def __init__(self):
        self.raw: dict[str, ComplexSystem] = {}
        self.raw_archive: dict[str, ComplexSystem] = {}

    def __getitem__(self, key):
        return self.raw[key]

    def __setitem__(self, key, value):
        if isinstance(value, ComplexSystem):
            self.raw[key] = value
        else:
            raise ValueError("Value must be an instance of ComplexSystem.")

    def clear(self):
        self.raw_archive |= self.raw.copy()
        self.raw.clear()

    def reset(self):
        self.raw.clear()
        self.raw_archive.clear()

    def restore(self):
        self.raw |= self.raw_archive.copy()
        self.raw_archive.clear()

    def get(self, key, default=None):
        try:
            return self.raw[key]
        except KeyError:
            return default

    def get_df(self, key, *args, thickness=None, area=None, **_) -> pd.DataFrame:
        """Get the base DataFrame for the selected key via get_df."""
        if key in self.raw:
            self.raw[key].thickness = thickness
            self.raw[key].area = area
            return self.raw[key].get_df(*args)
        return pd.DataFrame()

    def update_system(self, key, data, form=None, **kwargs):
        """Update the system based on the selected key."""
        if form is None:
            form = (
                data.attrs.get("form", "impedance")
                if isinstance(data, pd.DataFrame)
                else "impedance"
            )

        # Makes sure passed area/thickness/frequency overrides the data attrs
        # if isinstance(data, (ComplexSystem, pd.DataFrame)):
        #     kwargs = data.attrs | kwargs

        # Makes sure that if not passed, the data attrs are used
        kwargs.setdefault("frequency")
        kwargs.setdefault("thickness")
        kwargs.setdefault("area")

        # Depends on data
        system = ComplexSystem(data, form=form, **kwargs)
        if isinstance(key, str):
            if key in self.raw:
                old_sys = self.raw[key]
                old_sys.update(system)
                system = old_sys
            self[key] = system
            self.raw_archive[key] = self.raw[key].copy()
        return


# class DataHandler(JSONSettings):
#     """Class to store data for plotting graphs."""

#     def __init__(self, **kwargs):
#         # Initialize JSONSettings
#         super().__init__(**kwargs)
#         self.model = "p(R1,C1)"
#         self.load_dir = Path().home()
#         self.save_dir = Path().home()
#         self.export_dir = Path().home()
#         self.error_var = "Mean Abs Err"
#         self.sel_error_var = "Z"
#         self.afit_inputs = {}
#         self.plot_var = {}
#         self.is_log = {}
#         self.var_units = {}
#         self.option_inits = {}

#         self.raw: dict[str, ComplexSystem] = {}
#         self.raw_archive = {}
#         # self.linkk = ComplexSystem()
#         self.defaults = {}

#         self.load_settings()

#         self.error_methods = {
#             key.replace("_", " ").title(): key for key in Statistics().single_method_list.copy()
#         }
#         self.error_methods_abbr = {
#             k1: k2
#             for k1, k2 in zip(self.error_methods.keys(), Statistics().single_method_abbr.keys())
#         }
#         if self.error_var not in self.error_methods:
#             self.error_var = list(self.error_methods.keys())[0]

#         if self.sel_error_var not in ["Nyquist", "Bode", "Both", "Z", "Y", "C", "M", "User"]:
#             self.sel_error_var = "Z"

#     def get(self, key, default=None):
#         try:
#             return self.raw[key]
#         except KeyError:
#             return default

#     def save_settings(self, **kwargs):
#         """Save the local modified values to JSON file."""
#         if not kwargs:
#             return
#         #     kwargs = self.option_inits
#         for key, value in kwargs.items():
#             if hasattr(self, key):
#                 setattr(self, key, value)
#         super().save_settings(**kwargs)

#     def load_settings(self, **kwargs):
#         """Load settings from JSON files and set attributes."""
#         settings = super().load_settings(**kwargs)  # Use JSONSettings method

#         # Set attributes directly from the loaded settings
#         for key, value in settings.items():
#             setattr(self, key, value)

#         return settings

#     def restore_defaults(self, **kwargs):
#         """Restore the default settings."""
#         settings = super().restore_defaults(**kwargs)  # Use JSONSettings method

#         # Set attributes directly from the restored settings
#         for key, value in settings.items():
#             setattr(self, key, value)

#         return settings

#     def parse_default(self, name, defaults, override=None):
#         """Parse the default value for the parameter."""
#         if override is not None:
#             if name in override.keys():
#                 return override[name]

#         if self.model.lower() == "linkk":
#             return 1

#         if name in defaults.keys():
#             return defaults[name]
#         split_name = re.findall(r"(^[a-zA-Z]+)_?([0-9_]+)", name)[0]
#         if split_name[0] not in defaults.keys():
#             return 1
#         if "_" in split_name[1]:
#             index = int(eval(split_name[1].split("_")[-1], {}, {"inf": np.inf}))
#             return defaults[split_name[0]][index]
#         return defaults[split_name[0]]

#     def parse_form(self, form: str, type_only: bool = False) -> str:
#         """
#         Parse the given form string to extract the type and mode.

#         Args:
#             form (str): The string representing the desired format (e.g., "Z'", "Z''", "|Z|", "Z θ", "Z tan(δ)").

#         Returns:
#             str: The parsed key in the format "<type>.<mode>" (e.g., "Z.real", "Z.imag").
#         """
#         if not isinstance(form, str):
#             try:
#                 form = form.currentText()
#             except AttributeError:
#                 form = str(form)
#         # Define the mapping of suffixes to modes
#         mode_mapping = {
#             "'": "real",
#             "''": "imag",
#             "||": "mag",
#             "-''": "neg_imag",
#             "θ": "phase",
#             "tan(δ)": "tan",
#         }
#         if form in ["f", "ω", "σ_dc"]:
#             return form

#         valid_keys = [re.escape(t) for t in self.var_units.keys() if t not in ["f", "ω", "σ_dc"]]
#         type_pattern = rf"\W?({'|'.join(valid_keys)})(\s.+|\W+)$"
#         t_match = re.match(type_pattern, form)
#         key = t_match.group(1) if t_match else ""
#         type_ = ComplexSystem.aliases[key.lower()]
#         if type_only:
#             return type_
#         mode = mode_mapping[form.replace(key, "").strip()]
#         return f"{type_}.{mode}"

#     def get_types(self, target: str = "complex") -> dict:
#         """
#         Generate a list of formatted strings by applying each mode to each type.

#         Returns:
#             dict: A dict of formatted strings (e.g., ["Z'", "Z''", "|Z|", "-Z''", "Z θ", "Z tan(δ)", ...]).
#         """

#         if not isinstance(target, str):
#             target = "complex"

#         arr_types = {
#             k: v
#             for k, v in zip(
#                 ["f", "ω", "σ_dc"], ["frequency", "angular_frequency", "dc_conductivity"]
#             )
#         }
#         if "freq" in target or "omega" in target or "single" in target or "arr" in target:
#             return arr_types

#         formatted_types = []
#         root_types = []
#         for type_ in self.var_units.keys():
#             if type_ in arr_types:
#                 continue
#             root_type = ComplexSystem.aliases[type_.lower()]
#             formatted_types += [
#                 f"{type_}'",
#                 f"{type_}''",
#                 f"|{type_}|",
#                 f"-{type_}''",
#                 f"{type_} θ",
#                 f"{type_} tan(δ)",
#             ]
#             root_types += [
#                 f"{root_type}.real",
#                 f"{root_type}.imag",
#                 f"{root_type}.mag",
#                 f"{root_type}.neg_imag",
#                 f"{root_type}.phase",
#                 f"{root_type}.tan",
#             ]
#         comp_types = {k: v for k, v in zip(formatted_types, root_types)}
#         if "all" in target:
#             return {**comp_types, **arr_types}
#         if "append" in target:
#             return {**comp_types, **arr_types}
#         if "insert" in target:
#             return {**arr_types, **comp_types}
#         return comp_types

#     def parse_label(self, form):
#         """Parse the label based on the selected option."""
#         if not isinstance(form, str):
#             try:
#                 form = form.currentText()
#             except AttributeError:
#                 form = str(form)

#         # Handle special cases dynamically based on keywords
#         if "θ" in form:
#             units = "°"
#         elif "tan" in form:
#             units = "1"
#         elif "freq" in form.lower():
#             units = "Hz"
#         elif "ω" in form:
#             units = "1/s"
#         elif "σ_dc" in form:
#             units = "S/cm"
#         else:
#             # Search for the variable within the form using regex
#             type_pattern = rf"({'|'.join(re.escape(t) for t in self.var_units.keys())})"
#             type_match = re.search(type_pattern, form)
#             if type_match is None:
#                 return f"{form} (1)"  # Default to unitless if no match is found

#             # Extract the matched variable and its units
#             type_ = type_match.group(1)
#             units = self.var_units.get(type_, "1")  # Default to "1" if no unit is defined

#         return f"{form} ({units})"

#     def update_system(self, key, data, form=None, **kwargs):
#         """Update the system based on the selected key."""
#         if form is None and isinstance(data, pd.DataFrame) and "form" in data.attrs:
#             form = data.attrs["form"]

#         if isinstance(data, (ComplexSystem, pd.DataFrame)):
#             kwargs = data.attrs | kwargs

#         kwargs.setdefault("frequency")
#         kwargs.setdefault("thickness")
#         kwargs.setdefault("area")

#         # Depends on data
#         data = ComplexSystem(data, form=form, **kwargs)  # type: ignore
#         if isinstance(key, str):
#             if key in self.raw:
#                 self.raw[key].update(data)
#             else:
#                 self.raw[key] = data
#             self.raw_archive[key] = self.raw.copy()[key]
#         return

#     def get_df(self, key, *args, thickness=None, area=None, **kwargs):
#         """Get the base DataFrame for the selected key via get_df."""
#         if key in self.raw:
#             self.raw[key].thickness = thickness
#             self.raw[key].area = area
#             return self.raw[key].get_df(*args)
#         return pd.DataFrame()
