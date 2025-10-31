# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
import ast
import operator
from typing import Any, Generic, TypeVar
from collections.abc import Callable

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline, PchipInterpolator

try:
    from ..data_treatment.value_ops import find_nearest
except ImportError:
    from eis_analysis.data_treatment.value_ops import find_nearest


T = TypeVar("T")  # The return type of parse()


class BaseParser(Generic[T]):
    """
    Base mix-class for parsing and applying transformations to indexed attributes.

    This class is designed to be inherited by other classes to provide a consistent interface for applying
    transformations to indexed attributes. It requires that all valid transforming methods share a common suffix.
    """

    BINARY_OPERATORS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.MatMult: operator.matmul,
        # etc.
    }
    UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
        ast.UAdd: operator.pos,  # unary +
        ast.USub: operator.neg,  # unary -
        ast.Invert: operator.invert,
        ast.Not: operator.not_,
    }

    ALLOWED_FUNCTIONS: dict[str, Callable[..., Any]] = {
        "max": max,
        "min": min,
        "abs": abs,
    }

    def __init_subclass__(
        cls, updating_dict=None, update_forms=True, form_str="", short_form_len=3, **_
    ):
        """
        Automatically called when a subclass is created.
        Ensures that `_add_valid_forms` is called for the subclass and initializes attributes.

        Parameters:
        updating_dict (dict): A dictionary containing configuration dictionaries for the subclass.
                             Keys can include 'valid_forms' and other configuration dicts.
                             Example: {'valid_forms': {...}, 'savgol_kwargs': {...}}
        update_forms (bool): If False, prevents automatic changes to `_valid_forms`. Defaults to True.
        kwargs (dict): Additional keyword arguments, including `form_str` and `short_form_len`.

        Notes:
        - `form_str` must be a string and must correspond to at least one method in the class. If no method matches
        the provided `form_str`, a `ValueError` will be raised.
        - `_valid_forms` should not be modified directly, instead use `_add_valid_forms` which includes checks and
        automatic discovery if desired. Call `cls._add_valid_forms` within the `__init_subclass__` method as needed.
        - If `update_forms` is set to False, `_valid_forms` will not be updated. This is useful if
        manual additions are desired without interference from automatic discovery.
        - If `valid_forms` is provided, it will be processed only if `update_forms` is True and will prevent
        automatic discovery of methods. This allows for manual control over the valid forms.
        - If both manual additions (`valid_forms`) and automatic discovery are desired, it is recommended to pass
        the manual additions first or set 'update_forms' to false. Calling `cls._add_valid_forms` without arguments
        from `__init_subclass__` will automatically discover methods ending with `cls._form_str`.
        - Automatic short forms will only be added if the full form is not already included in `_valid_forms`.
        If a short form is desired for a method that already has a full form defined, it must be added manually.
        """
        # Extract form_str and short_form_len from kwargs
        if not isinstance(form_str, str):
            raise TypeError("form_str must be a string")
        if not form_str:
            # print(
            #     f"Skipping BaseParser __init_subclass__ for {cls.__name__} due to empty form_str"
            # )
            return
        if not any(m.endswith(form_str) for m in dir(cls)):
            raise ValueError(f"form_str '{form_str}' does not match any method in the class")
        cls._form_str = form_str
        cls._valid_forms = {}

        # Call _add_valid_forms on the subclass
        if update_forms and hasattr(cls, "_add_valid_forms") and callable(cls._add_valid_forms):
            cls._add_valid_forms(updating_dict, short_form_len)

    @classmethod
    def _add_valid_forms(cls, forms_dict=None, short_form_len=3):
        """
        Add valid forms to the `_valid_forms` attribute.

        Parameters:
        forms_dict (dict): A dictionary mapping form names (keys) to method names (values).
                        Both keys and values must be strings.
                        If None, `_add_valid_forms` will automatically discover methods ending with `cls._form_str`.
        short_form_len (int): The length of short forms to generate for automatically discovered methods. Defaults to 3.

        Notes:
        - If `forms_dict` is None or empty, `_add_valid_forms` will attempt to automatically discover methods
        ending with `cls._form_str`. The `cls._form_str` must correspond to at least one method in the class.
        - If `forms_dict` is provided, it alone will be processed.  Automatic discovery will require a separate call.
        - Short forms will only be added automatically if the full form is not already included in `_valid_forms`.
        If a short form is desired for a method that already has a full form, it must be added manually.

        - Duplicate keys in `forms_dict` will raise a `ValueError`.
        - If a method name ends with `cls._form_str` and is callable, it will be added to `_valid_forms` with its
        full name and, if applicable, a short form (based on `short_form_len`).
        """
        if forms_dict is None or not forms_dict:
            # Automatically discover `_form` methods in the class
            for attr_name in dir(cls):
                if attr_name.startswith("_"):
                    continue
                if attr_name.endswith(cls._form_str) and callable(getattr(cls, attr_name)):
                    full_name = attr_name[: -len(cls._form_str)]  # Remove cls._form_str suffix
                    # Use the first cls.short_form_len letters
                    if full_name not in cls._valid_forms:
                        cls._valid_forms[full_name] = attr_name
                        if (
                            isinstance(short_form_len, int)
                            and short_form_len > 0
                            and (short_name := full_name[:short_form_len]) not in cls._valid_forms
                        ):
                            cls._valid_forms[short_name] = attr_name
        else:
            # Add forms from the provided dictionary
            for key, value in forms_dict.items():
                if not isinstance(key, str):
                    raise TypeError("Form names must be strings")
                if not isinstance(value, str):
                    raise TypeError("Method names must be strings")
                if key in cls._valid_forms:
                    raise ValueError(f"Duplicate form name detected: {key} (method: {value})")
                if not hasattr(cls, value) or not callable(getattr(cls, value)):
                    raise ValueError(f"Invalid method name for form: {value}")
                if not value.endswith(cls._form_str):
                    raise ValueError(f"Method name '{value}' does not end with '{cls._form_str}'")
                if value.startswith("_"):
                    raise ValueError(f"Method name '{value}' cannot start with an underscore")
                cls._valid_forms[key] = value

    def __getitem__(self, index):
        """
        Allow slicing and indexing using the valid forms.
        """
        if isinstance(index, str):
            # Parse the index string and apply transformations
            if hasattr(self, index):
                return getattr(self, index)
            return self._parse_and_transform(index)
        else:
            raise TypeError("Index must be a string")

    @property
    def ast_attr_types(self) -> tuple[type, ...]:
        """
        Return a tuple of acceptable attribute types.
        """
        if not hasattr(self, "_ast_attr_types"):
            self._ast_attr_types = ()
        return self._ast_attr_types

    @ast_attr_types.setter
    def ast_attr_types(self, value: tuple[type, ...] | list[type]) -> None:
        """
        Set the acceptable attribute types.

        Parameters:
        value (tuple): A tuple of types.
        """
        if not isinstance(value, (tuple, list)):
            raise TypeError("ast_attr_types must be a tuple/list of types")
        for v in value:
            if not isinstance(v, type):
                raise TypeError("ast_attr_types must be a tuple of types")
        self._ast_attr_types = tuple(value)

    def _evaluate_ast(self, node, **kwargs) -> Any:
        """
        Recursively evaluate an AST node.

        Parameters:
        node: The AST node to evaluate.

        Returns:
        np.ndarray or value: The result of the evaluated node.
        """
        if isinstance(node, ast.BinOp):  # Binary operations (e.g., +, -, *, /, @)
            left = self._evaluate_ast(node.left)
            right = self._evaluate_ast(node.right)
            if isinstance(node.op, ast.MatMult):
                # Handle the `@` operator (i.e. 'array @ value' to find index)
                if not isinstance(right, (int, float)):
                    raise ValueError(
                        f"Right operand of '@' must resolve to a number, got {type(right).__name__}."
                    )
                # Find the value/index in `left` closest to `right`
                return find_nearest(left, right, kwargs.get("is_index", False))
            else:
                operator_func = self.BINARY_OPERATORS[type(node.op)]
                return operator_func(left, right)
        elif isinstance(node, ast.UnaryOp):
            # Unary operations (e.g., -x)
            operand = self._evaluate_ast(node.operand)
            operator_func = self.UNARY_OPERATORS[type(node.op)]
            return operator_func(operand)
        elif isinstance(node, ast.Call):
            # Function calls (e.g., log10(Z'))
            if isinstance(node.func, ast.Name):
                # Safely extract function name
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            else:
                raise ValueError(f"Unsupported function node: {ast.dump(node.func)}")

            if func_name in self.ALLOWED_FUNCTIONS:
                # Handle built-in functions like max, min
                func = self.ALLOWED_FUNCTIONS[func_name]
                args = [self._evaluate_ast(arg) for arg in node.args]
                return func(*args)
            elif func_name in self._valid_forms:
                # Handle modifying functions (e.g., log10, derivative)
                method_name = self._valid_forms[func_name]
                method = getattr(self, method_name, None)
                if not callable(method):
                    raise ValueError(f"Invalid function: {func_name}")
                args = [self._evaluate_ast(arg) for arg in node.args]
                kwargs = {str(kw.arg): self._evaluate_ast(kw.value) for kw in node.keywords}
                return method(*args, **kwargs)
            else:
                raise ValueError(f"Unsupported function: {func_name}")
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.MatMult):
                return self._evaluate_ast(node.value, is_index="ind" in str(node.attr).lower())
            # Attribute access (e.g., impedance.real)
            base = self._evaluate_ast(node.value)
            attr = node.attr
            if isinstance(base, self.ast_attr_types):
                if hasattr(base, attr):
                    return getattr(base, attr)
                else:
                    raise AttributeError(f"'{base}' has no attribute '{attr}'")
            else:
                raise ValueError(f"Base '{base}' is not an allowed object for attr access.")
        elif isinstance(node, ast.Name) and hasattr(self, node.id):
            # Variables (e.g., impedance, frequency)
            return getattr(self, node.id)
        elif isinstance(node, ast.Subscript):
            # Array indexing (e.g., impedance.real[0])
            base = self._evaluate_ast(node.value)
            index = self._evaluate_ast(node.slice, is_index=True)
            return base[index]
        elif isinstance(node, ast.Constant):  # Constants (e.g., numbers)
            val = node.value
            if kwargs.get("is_index", False):
                if not isinstance(val, (int, float)):
                    raise ValueError(
                        f"Index must resolve to an integer, got {type(val).__name__}."
                    )
                return int(val)
            # Non-index context: allow numbers, strings, bool, None, Ellipsis
            if isinstance(val, (int, float, complex, str, bytes, bool, type(None))):
                return val
            if val is Ellipsis:
                return val
            raise ValueError(f"Unsupported constant type: {type(val).__name__} with value {val!r}")
        else:
            raise ValueError(f"Unsupported operation: {ast.dump(node)}")

    def _parse_and_transform(self, index) -> T:
        """
        Parse the index and apply the corresponding transformations.

        Parameters:
        index (str): The index string.

        Returns:
        np.ndarray: The resulting array after applying transformations.
        """
        if not isinstance(index, str):
            raise TypeError(f"Index must be a string, got {type(index).__name__}.")

        # Parse the index into an abstract syntax tree (AST)
        try:
            tree = ast.parse(index, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Invalid expression: {index}") from e

        # Evaluate the AST
        res = self._evaluate_ast(tree.body)
        return res


class ItemTransforms(BaseParser[T], Generic[T]):
    """
    Class for handling transformation methods (_form methods) for ComplexSystem.
    Inherits from BaseParser.
    """

    _form_kwargs_base = {
        "savgol": {
            "window_length": 5,
            "polyorder": 3,
            "deriv": 1,
            "axis": -1,
            "mode": "interp",
            "cval": 0.0,
            "pre_interp": False,
        },
        "interp": {
            "smooth_first": True,
            "axis": 0,
            "bc_type": "not-a-knot",
            "extrapolate": None,
            "spl_type": "cubic",
        },
        "norm": {
            "normalize_to": "min",
        },
    }
    # _form_kwargs: dict = {}
    # _default_x: str = ""
    # _active_x: list | np.ndarray = []

    def __init_subclass__(cls, updating_dict=None, update_forms=True, short_form_len=3, **_):
        """
        Automatically called when a subclass is created.
        Ensures that `_add_valid_forms` is called and initializes savgol_kwargs.

        Parameters:
        updating_dict (dict): A dictionary containing configuration dictionaries for the subclass.
                             Keys can include 'valid_forms' and 'savgol_kwargs'.
                             Example: {'valid_forms': {...}, 'savgol_kwargs': {...}}
        update_forms (bool): If False, prevents automatic changes to `_valid_forms`. Defaults to True.
        kwargs (dict): Additional keyword arguments, including `form_str` and `short_form_len`.
        """
        if not update_forms:
            super().__init_subclass__(updating_dict, update_forms, "_form", short_form_len)
        else:
            valid_forms = {
                "S": "smoothed_form",
                "ƒₛₘ": "smoothed_form",
                "sm": "smoothed_form",
                "smooth": "smoothed_form",
                "smoothed": "smoothed_form",
            }
            alt_forms = {
                "∂": "derivative_form",
                "d": "derivative_form",
                "deriv": "derivative_form",
                "B": "interpolated_form",
                "spl": "interpolated_form",
                "ƒₛₚₗ": "interpolated_form",
            }
            super().__init_subclass__(valid_forms, update_forms, "_form", short_form_len)
            cls._add_valid_forms()
            cls._add_valid_forms(alt_forms)

        if isinstance(updating_dict, dict):

            for key, base_dict in cls._form_kwargs_base.items():
                if key in updating_dict:
                    # Update the subgroup directly if it exists in updating_dict
                    sub_dict = updating_dict[key]
                    if isinstance(sub_dict, dict):
                        # base_dict.update({k: v for k, v in sub_dict.items() if k in base_dict})
                        base_dict.update(sub_dict)
                elif all(k in base_dict for k in updating_dict.keys()):
                    # Check if updating_dict contains keys for this subgroup
                    base_dict.update(updating_dict)
                    break

    def __init__(self, *_, default_x="", **__):
        # _form_kwargs: dict = {}
        self._default_x: str = ""
        self._active_x: list | np.ndarray = []
        self._form_kwargs = {
            key: sub_dict.copy() for key, sub_dict in self._form_kwargs_base.items()
        }

        if isinstance(default_x, str):
            self.default_x = default_x
        elif isinstance(default_x, (list, np.ndarray)):
            self.active_x = default_x

    @property
    def default_x(self) -> str:
        """
        Default x-axis values for interpolation.
        """
        return self._default_x

    @default_x.setter
    def default_x(self, value) -> None:
        if isinstance(value, str):
            try:
                test_arr = self[value]
                if isinstance(test_arr, (np.ndarray, list)):
                    self._default_x = value
            except Exception:
                pass

    @property
    def active_x(self) -> np.ndarray:
        """
        Default x-axis values for interpolation.
        """
        if isinstance(self._active_x, list) and self._active_x:
            return np.array(self._active_x)
        if isinstance(self._active_x, np.ndarray) and self._active_x.size:
            return self._active_x
        if self.default_x:
            return getattr(self, self.default_x)
        return np.array([])

    @active_x.setter
    def active_x(self, value) -> None:
        if isinstance(value, str):
            try:
                value = self[value]
            except Exception:
                pass
        if isinstance(value, (np.ndarray, list)):
            self._active_x = value

    @property
    def form_kwargs(self) -> dict:
        """
        Property to store and retrieve all form function parameters.
        """
        return self._form_kwargs

    @form_kwargs.setter
    def form_kwargs(self, value) -> None:
        """
        Setter for form_kwargs. Ensures only valid kwargs for all form functions are updated.
        """
        if not value:
            return
        if not isinstance(value, dict):
            raise TypeError("savgol_kwargs must be a dictionary")

        for key, base_dict in self._form_kwargs.items():
            # if key in value:
            if any(key in k for k in value.keys()):
                # Update the subgroup directly if it exists in updating_dict
                if key not in value:
                    sub_dict = value[[k for k in value.keys() if key in k][0]]
                else:
                    sub_dict = value[key]
                if isinstance(sub_dict, dict):
                    base_dict.update(sub_dict)
            elif all(k in self._form_kwargs_base[key] for k in value.keys()):
                # Check if updating_dict contains keys for this subgroup
                base_dict.update(value)

    @property
    def savgol_kwargs(self) -> dict:
        """
        Property to store and retrieve Savitzky-Golay filter parameters.
        """
        return self._form_kwargs["savgol"]

    @savgol_kwargs.setter
    def savgol_kwargs(self, value) -> None:
        """
        Setter for savgol_kwargs. Ensures only valid kwargs for savgol_filter are updated.
        """
        if not value:
            return
        if isinstance(value, dict) and ("savgol" in value or "savgol_kwargs" in value):
            value = value.get("savgol", value.get("savgol_kwargs"))
        if not isinstance(value, dict):
            raise TypeError("savgol_kwargs must be a dictionary")

        self._form_kwargs["savgol"].update(value)

    @property
    def interp_kwargs(self) -> dict:
        """
        Property to store and retrieve interpolation parameters.
        """
        return self._form_kwargs["interp"]

    @interp_kwargs.setter
    def interp_kwargs(self, value) -> None:
        """
        Setter for interp_kwargs. Ensures only valid kwargs for interp_kwargs are updated.
        """
        if not value:
            return
        if isinstance(value, dict) and (
            "interp" in value or "interpolate" in value or "interp_kwargs" in value
        ):
            value = value.get("interp", value.get("interpolate", value.get("interp_kwargs")))
        if not isinstance(value, dict):
            raise TypeError("interp_kwargs must be a dictionary")

        self._form_kwargs["interp"].update(value)

    @property
    def norm_kwargs(self) -> dict:
        """
        Property to store and retrieve interpolation parameters.
        """
        return self._form_kwargs["norm"]

    @norm_kwargs.setter
    def norm_kwargs(self, value) -> None:
        """
        Setter for norm_kwargs. Ensures only valid kwargs for norm_kwargs are updated.
        """
        if not value:
            return
        if isinstance(value, dict) and (
            "norm" in value or "normalize" in value or "norm_kwargs" in value
        ):
            value = value.get("norm", value.get("normalize", value.get("norm_kwargs")))

        if not isinstance(value, dict):
            raise TypeError("norm_kwargs must be a dictionary")

        self._form_kwargs["norm"].update(value)

    def ln_form(self, value) -> np.ndarray:
        """
        Perform the natural logarithm transformation on the input value.
        """
        array = self._ensure_array(value)
        if array.dtype.kind == "c":
            return self._complex_array_eval(array, self.ln_form)

        # Handle negative and zero values
        mask_negative = array < 0
        array[mask_negative] *= -1
        logged_array = np.zeros_like(array)  # Initialize with zeros for zero values
        logged_array[array != 0] = np.log(array[array != 0])
        logged_array[mask_negative] *= -1  # Restore negative sign for negative values

        return logged_array

    def log10_form(self, value) -> np.ndarray:
        """
        Perform the base-10 logarithm transformation on the input value.
        """
        array = self._ensure_array(value)
        if array.dtype.kind == "c":
            return self._complex_array_eval(array, self.log10_form)

        # Handle negative and zero values
        mask_negative = array < 0
        array[mask_negative] *= -1
        logged_array = np.zeros_like(array)  # Initialize with zeros for zero values
        logged_array[array != 0] = np.log10(array[array != 0])
        logged_array[mask_negative] *= -1  # Restore negative sign for negative values

        return logged_array

    def derivative_form(self, value, **kwargs) -> np.ndarray:
        """
        Perform the derivative transformation using Savitzky-Golay filter.

        Parameters:
        value: The input value to calculate the derivative for.
        **kwargs: Additional arguments to override default parameters.

        Returns:
        np.ndarray: The derivative of the input value.
        """
        # Ensure array is valid
        array = self._ensure_array(value)
        if array.dtype.kind == "c":
            return self._complex_array_eval(array, self.derivative_form, **kwargs)

        # Extract parameters, prioritizing kwargs, then self.savgol_kwargs, then hardcoded defaults
        window_length = kwargs.pop("window_length", self.savgol_kwargs.get("window_length", 5))
        polyorder = kwargs.pop("polyorder", self.savgol_kwargs.get("polyorder", 3))
        deriv = kwargs.pop("deriv", self.savgol_kwargs.get("deriv", 1))
        delta = kwargs.pop("delta", self.savgol_kwargs.get("delta", None))
        axis = kwargs.pop("axis", self.savgol_kwargs.get("axis", -1))
        mode = kwargs.pop("mode", self.savgol_kwargs.get("mode", "interp"))
        cval = kwargs.pop("cval", self.savgol_kwargs.get("cval", 0.0))
        pre_interp = kwargs.pop("pre_interp", self.savgol_kwargs.get("pre_interp", False))

        is_uniform = True
        # Handle delta specifically
        if isinstance(delta, str):
            delta = getattr(self, delta, None)  # Resolve delta if it's a string
        if delta is None and self.active_x.size:
            log_diffs = np.diff(np.log10(self.active_x))
            delta = np.mean(abs(log_diffs))
            is_uniform = np.allclose(log_diffs, delta, rtol=1e-2) if pre_interp else True
        if delta is None:
            delta = 1
        delta = float(delta)

        # Call savgol_filter with explicit arguments
        if is_uniform:
            return savgol_filter(
                array,
                window_length=window_length,
                polyorder=polyorder,
                deriv=deriv,
                delta=delta,
                axis=axis,
                mode=mode,
                cval=cval,
            )
        else:
            # Save the original x-array for later interpolation back
            original_x = self.active_x.copy()

            # First interpolate to uniform spacing without smoothing
            uniform_array = self.interpolated_form(array, smooth_first=False)
            delta = float(np.mean(abs(np.diff(np.log10(self.active_x)))))

            # Apply savgol filter on the uniform data
            uniform_filtered = savgol_filter(
                uniform_array,
                window_length=window_length,
                polyorder=polyorder,
                deriv=deriv,
                delta=delta,
                axis=axis,
                mode=mode,
                cval=cval,
            )

            # Interpolate back to the original x values
            return self.interpolated_form(uniform_filtered, new_x=original_x, smooth_first=False)

    def smoothed_form(self, value, **kwargs) -> np.ndarray:
        """
        Perform the smoothing transformation using Savitzky-Golay filter.

        Parameters:
        value: The input value to smooth.
        **kwargs: Additional arguments to update savgol_kwargs.

        Returns:
        np.ndarray: The smoothed input value.
        """
        return self.derivative_form(value, **{**kwargs, "deriv": 0})

    def interpolated_form(self, value, **kwargs) -> np.ndarray:
        """
        Perform the interpolation transformation on the input value.

        Parameters:
        value: The input value to interpolate.
        **kwargs: Additional arguments to override defaults for interpolation parameters.
                Supported keys:
                - old_x (np.ndarray): The original x-axis values.
                - new_x (np.ndarray): The new x-axis values. Defaults to a logspace array.
                - smooth_first (bool): Whether to smooth the data before interpolation.
                - axis (int): The axis along which to interpolate.
                - bc_type (str): Boundary condition type for CubicSpline.
                - extrapolate (bool): Whether to extrapolate beyond the data range.

        Returns:
        np.ndarray: The interpolated data.
        """
        array = self._ensure_array(value)
        if array.dtype.kind == "c":
            return self._complex_array_eval(array, self.interpolated_form, **kwargs)

        # Parse defaults from kwargs or self.interp_kwargs
        old_x = kwargs.pop("old_x", self.interp_kwargs.get("old_x", self.active_x))
        new_x = kwargs.pop("new_x", self.interp_kwargs.get("new_x", None))
        smooth_first = kwargs.pop("smooth_first", self.interp_kwargs.get("smooth_first", True))
        spl_type = kwargs.pop("spl_type", self.interp_kwargs.get("spl_type", "pchip"))
        axis = kwargs.pop("axis", self.interp_kwargs.get("axis", 0))
        bc_type = kwargs.pop("bc_type", self.interp_kwargs.get("bc_type", "not-a-knot"))
        extrapolate = kwargs.pop("extrapolate", self.interp_kwargs.get("extrapolate", None))

        old_x = self._ensure_array(old_x)

        if not old_x.size or old_x.size != array.shape[axis]:
            raise ValueError(
                f"old_x length {old_x.size} does not match data length {array.shape[axis]} along axis {axis}"
            )
        # Smooth the data first if required
        data = self.derivative_form(array, **{**kwargs, "deriv": 0}) if smooth_first else array

        if new_x is None and old_x is not None:
            new_x = np.logspace(
                np.log10(old_x.min()),
                np.log10(old_x.max()),
                num=len(old_x),
            )

        # Interpolate the smoothed data
        if "pchip" in spl_type:
            interpolator = PchipInterpolator(old_x, data, axis=axis, extrapolate=extrapolate)
        else:
            interpolator = CubicSpline(
                old_x, data, axis=axis, bc_type=bc_type, extrapolate=extrapolate
            )

        self.active_x = new_x
        return interpolator(new_x)

    def _ensure_array(self, value) -> np.ndarray:
        """
        Helper function to ensure the input is converted to a numpy array.

        Parameters:
        value: The input value to be converted.

        Returns:
        np.ndarray: The converted numpy array.
        """
        if hasattr(value, "array"):
            value = value.array
        return np.array(value)

    def _complex_array_eval(
        self, value: np.ndarray, func: Callable[..., np.ndarray], **kwargs
    ) -> np.ndarray:
        """
        Helper function to ensure the input is converted to a Complexer object.

        Parameters:
        value: The input value to be converted.
        func: The function to apply to the real and imaginary parts.
        **kwargs: Additional arguments to pass to the function.

        Returns:
        np.ndarray: The result of applying the function to the complex array.
        """
        if value.dtype.kind == "c":
            real = func(value.real, **kwargs.copy())
            imag = func(value.imag, **kwargs.copy())
            return real + 1j * imag
        return func(value, **kwargs)


# class BaseParser:
#     """
#     Base mix-class for parsing and applying transformations to indexed attributes.

#     This class is designed to be inherited by other classes to provide a consistent interface for applying
#     transformations to indexed attributes. It requires that all valid transforming methods share a common suffix.
#     """

#     BINARY_OPERATORS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
#         ast.Add: operator.add,
#         ast.Sub: operator.sub,
#         ast.Mult: operator.mul,
#         ast.Div: operator.truediv,
#         ast.MatMult: operator.matmul,
#         # etc.
#     }
#     UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
#         ast.UAdd: operator.pos,  # unary +
#         ast.USub: operator.neg,  # unary -
#         ast.Invert: operator.invert,
#         ast.Not: operator.not_,
#     }

#     ALLOWED_FUNCTIONS: dict[str, Callable[..., Any]] = {
#         "max": max,
#         "min": min,
#         "abs": abs,
#     }

#     def __init_subclass__(
#         cls, updating_dict=None, update_forms=True, form_str="", short_form_len=3, **_
#     ):
#         """
#         Automatically called when a subclass is created.
#         Ensures that `_add_valid_forms` is called for the subclass and initializes attributes.

#         Parameters:
#         updating_dict (dict): A dictionary containing configuration dictionaries for the subclass.
#                              Keys can include 'valid_forms' and other configuration dicts.
#                              Example: {'valid_forms': {...}, 'savgol_kwargs': {...}}
#         update_forms (bool): If False, prevents automatic changes to `_valid_forms`. Defaults to True.
#         kwargs (dict): Additional keyword arguments, including `form_str` and `short_form_len`.

#         Notes:
#         - `form_str` must be a string and must correspond to at least one method in the class. If no method matches
#         the provided `form_str`, a `ValueError` will be raised.
#         - `_valid_forms` should not be modified directly, instead use `_add_valid_forms` which includes checks and
#         automatic discovery if desired. Call `cls._add_valid_forms` within the `__init_subclass__` method as needed.
#         - If `update_forms` is set to False, `_valid_forms` will not be updated. This is useful if
#         manual additions are desired without interference from automatic discovery.
#         - If `valid_forms` is provided, it will be processed only if `update_forms` is True and will prevent
#         automatic discovery of methods. This allows for manual control over the valid forms.
#         - If both manual additions (`valid_forms`) and automatic discovery are desired, it is recommended to pass
#         the manual additions first or set 'update_forms' to false. Calling `cls._add_valid_forms` without arguments
#         from `__init_subclass__` will automatically discover methods ending with `cls._form_str`.
#         - Automatic short forms will only be added if the full form is not already included in `_valid_forms`.
#         If a short form is desired for a method that already has a full form defined, it must be added manually.
#         """
#         # Extract form_str and short_form_len from kwargs
#         if not isinstance(form_str, str):
#             raise TypeError("form_str must be a string")
#         if not form_str:
#             # print(
#             #     f"Skipping BaseParser __init_subclass__ for {cls.__name__} due to empty form_str"
#             # )
#             return
#         if not any(m.endswith(form_str) for m in dir(cls)):
#             raise ValueError(f"form_str '{form_str}' does not match any method in the class")
#         cls._form_str = form_str
#         cls._valid_forms = {}

#         # Call _add_valid_forms on the subclass
#         if update_forms and hasattr(cls, "_add_valid_forms") and callable(cls._add_valid_forms):
#             cls._add_valid_forms(updating_dict, short_form_len)

#     @classmethod
#     def _add_valid_forms(cls, forms_dict=None, short_form_len=3):
#         """
#         Add valid forms to the `_valid_forms` attribute.

#         Parameters:
#         forms_dict (dict): A dictionary mapping form names (keys) to method names (values).
#                         Both keys and values must be strings.
#                         If None, `_add_valid_forms` will automatically discover methods ending with `cls._form_str`.
#         short_form_len (int): The length of short forms to generate for automatically discovered methods. Defaults to 3.

#         Notes:
#         - If `forms_dict` is None or empty, `_add_valid_forms` will attempt to automatically discover methods
#         ending with `cls._form_str`. The `cls._form_str` must correspond to at least one method in the class.
#         - If `forms_dict` is provided, it alone will be processed.  Automatic discovery will require a separate call.
#         - Short forms will only be added automatically if the full form is not already included in `_valid_forms`.
#         If a short form is desired for a method that already has a full form, it must be added manually.

#         - Duplicate keys in `forms_dict` will raise a `ValueError`.
#         - If a method name ends with `cls._form_str` and is callable, it will be added to `_valid_forms` with its
#         full name and, if applicable, a short form (based on `short_form_len`).
#         """
#         if forms_dict is None or not forms_dict:
#             # Automatically discover `_form` methods in the class
#             for attr_name in dir(cls):
#                 if attr_name.startswith("_"):
#                     continue
#                 if attr_name.endswith(cls._form_str) and callable(getattr(cls, attr_name)):
#                     full_name = attr_name[: -len(cls._form_str)]  # Remove cls._form_str suffix
#                     # Use the first cls.short_form_len letters
#                     if full_name not in cls._valid_forms:
#                         cls._valid_forms[full_name] = attr_name
#                         if (
#                             isinstance(short_form_len, int)
#                             and short_form_len > 0
#                             and (short_name := full_name[:short_form_len]) not in cls._valid_forms
#                         ):
#                             cls._valid_forms[short_name] = attr_name
#         else:
#             # Add forms from the provided dictionary
#             for key, value in forms_dict.items():
#                 if not isinstance(key, str):
#                     raise TypeError("Form names must be strings")
#                 if not isinstance(value, str):
#                     raise TypeError("Method names must be strings")
#                 if key in cls._valid_forms:
#                     raise ValueError(f"Duplicate form name detected: {key} (method: {value})")
#                 if not hasattr(cls, value) or not callable(getattr(cls, value)):
#                     raise ValueError(f"Invalid method name for form: {value}")
#                 if not value.endswith(cls._form_str):
#                     raise ValueError(f"Method name '{value}' does not end with '{cls._form_str}'")
#                 if value.startswith("_"):
#                     raise ValueError(f"Method name '{value}' cannot start with an underscore")
#                 cls._valid_forms[key] = value

#     def __getitem__(self, index):
#         """
#         Allow slicing and indexing using the valid forms.
#         """
#         if isinstance(index, str):
#             # Parse the index string and apply transformations
#             if hasattr(self, index):
#                 return getattr(self, index)
#             return self._parse_and_transform(index)
#         else:
#             raise TypeError("Index must be a string")

#     def _evaluate_ast(self, node, **kwargs) -> Any:
#         """
#         Recursively evaluate an AST node.

#         Parameters:
#         node: The AST node to evaluate.

#         Returns:
#         np.ndarray or value: The result of the evaluated node.
#         """
#         if isinstance(node, ast.BinOp):  # Binary operations (e.g., +, -, *, /, @)
#             left = self._evaluate_ast(node.left)
#             right = self._evaluate_ast(node.right)
#             if isinstance(node.op, ast.MatMult):
#                 # Handle the `@` operator (i.e. 'array @ value' to find index)
#                 if not isinstance(right, (int, float)):
#                     raise ValueError(
#                         f"Right operand of '@' must resolve to a number, got {type(right).__name__}."
#                     )
#                 # Find the index in `left` closest to `right`
#                 return find_nearest(left, right, kwargs.get("is_index", False))
#             else:
#                 operator_func = self.BINARY_OPERATORS[type(node.op)]
#                 return operator_func(left, right)
#         elif isinstance(node, ast.UnaryOp):
#             # Unary operations (e.g., -x)
#             operand = self._evaluate_ast(node.operand)
#             operator_func = self.UNARY_OPERATORS[type(node.op)]
#             return operator_func(operand)
#         elif isinstance(node, ast.Call):
#             # Function calls (e.g., log10(Z'))
#             if isinstance(node.func, ast.Name):
#                 # Safely extract function name
#                 func_name = node.func.id
#             elif isinstance(node.func, ast.Attribute):
#                 func_name = node.func.attr
#             else:
#                 raise ValueError(f"Unsupported function node: {ast.dump(node.func)}")

#             if func_name in self.ALLOWED_FUNCTIONS:
#                 # Handle built-in functions like max, min
#                 func = self.ALLOWED_FUNCTIONS[func_name]
#                 args = [self._evaluate_ast(arg) for arg in node.args]
#                 return func(*args)
#             elif func_name in self._valid_forms:
#                 # Handle modifying functions (e.g., log10, derivative)
#                 method_name = self._valid_forms[func_name]
#                 method = getattr(self, method_name, None)
#                 if not callable(method):
#                     raise ValueError(f"Invalid function: {func_name}")
#                 args = [self._evaluate_ast(arg) for arg in node.args]
#                 kwargs = {str(kw.arg): self._evaluate_ast(kw.value) for kw in node.keywords}
#                 return method(*args, **kwargs)
#             else:
#                 raise ValueError(f"Unsupported function: {func_name}")
#         elif isinstance(node, ast.Attribute):
#             # Attribute access (e.g., impedance.real)
#             base = self._evaluate_ast(node.value)
#             attr = node.attr
#             if isinstance(base, Complexer):
#                 if hasattr(base, attr):
#                     return getattr(base, attr)
#                 else:
#                     raise AttributeError(f"'{base}' has no attribute '{attr}'")
#             else:
#                 raise ValueError(f"Base '{base}' is not a Complexer object.")
#         elif isinstance(node, ast.Name) and hasattr(self, node.id):
#             # Variables (e.g., impedance, frequency)
#             return getattr(self, node.id)
#         elif isinstance(node, ast.Subscript):
#             # Array indexing (e.g., impedance.real[0])
#             base = self._evaluate_ast(node.value)
#             index = self._evaluate_ast(node.slice, is_index=True)
#             return base[index]
#         elif isinstance(node, ast.Constant):  # Constants (e.g., numbers)
#             val = node.value
#             if kwargs.get("is_index", False):
#                 if not isinstance(val, (int, float)):
#                     raise ValueError(
#                         f"Index must resolve to an integer, got {type(val).__name__}."
#                     )
#                 return int(val)
#             # Non-index context: allow numbers, strings, bool, None, Ellipsis
#             if isinstance(val, (int, float, complex, str, bytes, bool, type(None))):
#                 return val
#             if val is Ellipsis:
#                 return val
#             raise ValueError(f"Unsupported constant type: {type(val).__name__} with value {val!r}")
#         else:
#             raise ValueError(f"Unsupported operation: {ast.dump(node)}")

#     def _parse_and_transform(self, index) -> Complexer | np.ndarray | float | int:
#         """
#         Parse the index and apply the corresponding transformations.

#         Parameters:
#         index (str): The index string.

#         Returns:
#         np.ndarray: The resulting array after applying transformations.
#         """
#         if not isinstance(index, str):
#             raise TypeError(f"Index must be a string, got {type(index).__name__}.")

#         # Parse the index into an abstract syntax tree (AST)
#         try:
#             tree = ast.parse(index, mode="eval")
#         except SyntaxError as e:
#             raise ValueError(f"Invalid expression: {index}") from e

#         # Evaluate the AST
#         res = self._evaluate_ast(tree.body)
#         return res


# class ItemTransforms(BaseParser):
#     """
#     Class for handling transformation methods (_form methods) for ComplexSystem.
#     Inherits from BaseParser.
#     """

#     _form_kwargs_base = {
#         "savgol": {
#             "window_length": 5,
#             "polyorder": 3,
#             "deriv": 1,
#             "axis": -1,
#             "mode": "interp",
#             "cval": 0.0,
#             "pre_interp": False,
#         },
#         "interp": {
#             "smooth_first": True,
#             "axis": 0,
#             "bc_type": "not-a-knot",
#             "extrapolate": None,
#             "spl_type": "cubic",
#         },
#         "norm": {
#             "normalize_to": "min",
#         },
#     }
#     _form_kwargs: dict = {}
#     _default_x: str = ""
#     _active_x: list | np.ndarray = []

#     def __init_subclass__(cls, updating_dict=None, update_forms=True, short_form_len=3, **_):
#         """
#         Automatically called when a subclass is created.
#         Ensures that `_add_valid_forms` is called and initializes savgol_kwargs.

#         Parameters:
#         updating_dict (dict): A dictionary containing configuration dictionaries for the subclass.
#                              Keys can include 'valid_forms' and 'savgol_kwargs'.
#                              Example: {'valid_forms': {...}, 'savgol_kwargs': {...}}
#         update_forms (bool): If False, prevents automatic changes to `_valid_forms`. Defaults to True.
#         kwargs (dict): Additional keyword arguments, including `form_str` and `short_form_len`.
#         """
#         if not update_forms:
#             super().__init_subclass__(updating_dict, update_forms, "_form", short_form_len)
#         else:
#             valid_forms = {
#                 "S": "smoothed_form",
#                 "ƒₛₘ": "smoothed_form",
#                 "sm": "smoothed_form",
#                 "smooth": "smoothed_form",
#                 "smoothed": "smoothed_form",
#             }
#             alt_forms = {
#                 "∂": "derivative_form",
#                 "d": "derivative_form",
#                 "deriv": "derivative_form",
#                 "B": "interpolated_form",
#                 "spl": "interpolated_form",
#                 "ƒₛₚₗ": "interpolated_form",
#             }
#             super().__init_subclass__(valid_forms, update_forms, "_form", short_form_len)
#             cls._add_valid_forms()
#             cls._add_valid_forms(alt_forms)

#         if isinstance(updating_dict, dict):

#             for key, base_dict in cls._form_kwargs_base.items():
#                 if key in updating_dict:
#                     # Update the subgroup directly if it exists in updating_dict
#                     sub_dict = updating_dict[key]
#                     if isinstance(sub_dict, dict):
#                         # base_dict.update({k: v for k, v in sub_dict.items() if k in base_dict})
#                         base_dict.update(sub_dict)
#                 elif all(k in base_dict for k in updating_dict.keys()):
#                     # Check if updating_dict contains keys for this subgroup
#                     base_dict.update(updating_dict)
#                     break

#     def __init__(self, *_, default_x="", **__):

#         self._form_kwargs = {
#             key: sub_dict.copy() for key, sub_dict in self._form_kwargs_base.items()
#         }

#         if isinstance(default_x, str):
#             self.default_x = default_x
#         elif isinstance(default_x, (list, np.ndarray)):
#             self.active_x = default_x

#     @property
#     def default_x(self) -> str:
#         """
#         Default x-axis values for interpolation.
#         """
#         return self._default_x

#     @default_x.setter
#     def default_x(self, value) -> None:
#         if isinstance(value, str):
#             try:
#                 test_arr = self[value]
#                 if isinstance(test_arr, (np.ndarray, list)):
#                     self._default_x = value
#             except Exception:
#                 pass

#     @property
#     def active_x(self) -> np.ndarray:
#         """
#         Default x-axis values for interpolation.
#         """
#         if isinstance(self._active_x, list) and self._active_x:
#             return np.array(self._active_x)
#         if isinstance(self._active_x, np.ndarray) and self._active_x.size:
#             return self._active_x
#         if self.default_x:
#             return getattr(self, self.default_x)
#         return np.array([])

#     @active_x.setter
#     def active_x(self, value) -> None:
#         if isinstance(value, str):
#             try:
#                 value = self[value]
#             except Exception:
#                 pass
#         if isinstance(value, (np.ndarray, list)):
#             self._active_x = value

#     @property
#     def form_kwargs(self) -> dict:
#         """
#         Property to store and retrieve all form function parameters.
#         """
#         return self._form_kwargs

#     @form_kwargs.setter
#     def form_kwargs(self, value) -> None:
#         """
#         Setter for form_kwargs. Ensures only valid kwargs for all form functions are updated.
#         """
#         if not value:
#             return
#         if not isinstance(value, dict):
#             raise TypeError("savgol_kwargs must be a dictionary")

#         for key, base_dict in self._form_kwargs.items():
#             # if key in value:
#             if any(key in k for k in value.keys()):
#                 # Update the subgroup directly if it exists in updating_dict
#                 if key not in value:
#                     sub_dict = value[[k for k in value.keys() if key in k][0]]
#                 else:
#                     sub_dict = value[key]
#                 if isinstance(sub_dict, dict):
#                     base_dict.update(sub_dict)
#             elif all(k in self._form_kwargs_base[key] for k in value.keys()):
#                 # Check if updating_dict contains keys for this subgroup
#                 base_dict.update(value)

#     @property
#     def savgol_kwargs(self) -> dict:
#         """
#         Property to store and retrieve Savitzky-Golay filter parameters.
#         """
#         return self._form_kwargs["savgol"]

#     @savgol_kwargs.setter
#     def savgol_kwargs(self, value) -> None:
#         """
#         Setter for savgol_kwargs. Ensures only valid kwargs for savgol_filter are updated.
#         """
#         if not value:
#             return
#         if isinstance(value, dict) and ("savgol" in value or "savgol_kwargs" in value):
#             value = value.get("savgol", value.get("savgol_kwargs"))
#         if not isinstance(value, dict):
#             raise TypeError("savgol_kwargs must be a dictionary")

#         self._form_kwargs["savgol"].update(value)

#     @property
#     def interp_kwargs(self) -> dict:
#         """
#         Property to store and retrieve interpolation parameters.
#         """
#         return self._form_kwargs["interp"]

#     @interp_kwargs.setter
#     def interp_kwargs(self, value) -> None:
#         """
#         Setter for interp_kwargs. Ensures only valid kwargs for interp_kwargs are updated.
#         """
#         if not value:
#             return
#         if isinstance(value, dict) and (
#             "interp" in value or "interpolate" in value or "interp_kwargs" in value
#         ):
#             value = value.get("interp", value.get("interpolate", value.get("interp_kwargs")))
#         if not isinstance(value, dict):
#             raise TypeError("interp_kwargs must be a dictionary")

#         self._form_kwargs["interp"].update(value)

#     @property
#     def norm_kwargs(self) -> dict:
#         """
#         Property to store and retrieve interpolation parameters.
#         """
#         return self._form_kwargs["norm"]

#     @norm_kwargs.setter
#     def norm_kwargs(self, value) -> None:
#         """
#         Setter for norm_kwargs. Ensures only valid kwargs for norm_kwargs are updated.
#         """
#         if not value:
#             return
#         if isinstance(value, dict) and (
#             "norm" in value or "normalize" in value or "norm_kwargs" in value
#         ):
#             value = value.get("norm", value.get("normalize", value.get("norm_kwargs")))

#         if not isinstance(value, dict):
#             raise TypeError("norm_kwargs must be a dictionary")

#         self._form_kwargs["norm"].update(value)

#     def ln_form(self, value) -> np.ndarray:
#         """
#         Perform the natural logarithm transformation on the input value.
#         """
#         array = self._ensure_array(value)
#         if array.dtype.kind == "c":
#             return self._complex_array_eval(array, self.ln_form)

#         # Handle negative and zero values
#         mask_negative = array < 0
#         array[mask_negative] *= -1
#         logged_array = np.zeros_like(array)  # Initialize with zeros for zero values
#         logged_array[array != 0] = np.log(array[array != 0])
#         logged_array[mask_negative] *= -1  # Restore negative sign for negative values

#         return logged_array

#     def log10_form(self, value) -> np.ndarray:
#         """
#         Perform the base-10 logarithm transformation on the input value.
#         """
#         array = self._ensure_array(value)
#         if array.dtype.kind == "c":
#             return self._complex_array_eval(array, self.log10_form)

#         # Handle negative and zero values
#         mask_negative = array < 0
#         array[mask_negative] *= -1
#         logged_array = np.zeros_like(array)  # Initialize with zeros for zero values
#         logged_array[array != 0] = np.log10(array[array != 0])
#         logged_array[mask_negative] *= -1  # Restore negative sign for negative values

#         return logged_array

#     def derivative_form(self, value, **kwargs) -> np.ndarray:
#         """
#         Perform the derivative transformation using Savitzky-Golay filter.

#         Parameters:
#         value: The input value to calculate the derivative for.
#         **kwargs: Additional arguments to override default parameters.

#         Returns:
#         np.ndarray: The derivative of the input value.
#         """
#         # Ensure array is valid
#         array = self._ensure_array(value)
#         if array.dtype.kind == "c":
#             return self._complex_array_eval(array, self.derivative_form, **kwargs)

#         # Extract parameters, prioritizing kwargs, then self.savgol_kwargs, then hardcoded defaults
#         window_length = kwargs.pop("window_length", self.savgol_kwargs.get("window_length", 5))
#         polyorder = kwargs.pop("polyorder", self.savgol_kwargs.get("polyorder", 3))
#         deriv = kwargs.pop("deriv", self.savgol_kwargs.get("deriv", 1))
#         delta = kwargs.pop("delta", self.savgol_kwargs.get("delta", None))
#         axis = kwargs.pop("axis", self.savgol_kwargs.get("axis", -1))
#         mode = kwargs.pop("mode", self.savgol_kwargs.get("mode", "interp"))
#         cval = kwargs.pop("cval", self.savgol_kwargs.get("cval", 0.0))
#         pre_interp = kwargs.pop("pre_interp", self.savgol_kwargs.get("pre_interp", False))

#         is_uniform = True
#         # Handle delta specifically
#         if isinstance(delta, str):
#             delta = getattr(self, delta, None)  # Resolve delta if it's a string
#         if delta is None and self.active_x.size:
#             log_diffs = np.diff(np.log10(self.active_x))
#             delta = np.mean(abs(log_diffs))
#             is_uniform = np.allclose(log_diffs, delta, rtol=1e-2) if pre_interp else True
#         if delta is None:
#             delta = 1
#         delta = float(delta)

#         # Call savgol_filter with explicit arguments
#         if is_uniform:
#             return savgol_filter(
#                 array,
#                 window_length=window_length,
#                 polyorder=polyorder,
#                 deriv=deriv,
#                 delta=delta,
#                 axis=axis,
#                 mode=mode,
#                 cval=cval,
#             )
#         else:
#             # Save the original x-array for later interpolation back
#             original_x = self.active_x.copy()

#             # First interpolate to uniform spacing without smoothing
#             uniform_array = self.interpolated_form(array, smooth_first=False)
#             delta = float(np.mean(abs(np.diff(np.log10(self.active_x)))))

#             # Apply savgol filter on the uniform data
#             uniform_filtered = savgol_filter(
#                 uniform_array,
#                 window_length=window_length,
#                 polyorder=polyorder,
#                 deriv=deriv,
#                 delta=delta,
#                 axis=axis,
#                 mode=mode,
#                 cval=cval,
#             )

#             # Interpolate back to the original x values
#             return self.interpolated_form(uniform_filtered, new_x=original_x, smooth_first=False)

#     def smoothed_form(self, value, **kwargs) -> np.ndarray:
#         """
#         Perform the smoothing transformation using Savitzky-Golay filter.

#         Parameters:
#         value: The input value to smooth.
#         **kwargs: Additional arguments to update savgol_kwargs.

#         Returns:
#         np.ndarray: The smoothed input value.
#         """
#         return self.derivative_form(value, **{**kwargs, "deriv": 0})

#     def interpolated_form(self, value, **kwargs) -> np.ndarray:
#         """
#         Perform the interpolation transformation on the input value.

#         Parameters:
#         value: The input value to interpolate.
#         **kwargs: Additional arguments to override defaults for interpolation parameters.
#                 Supported keys:
#                 - old_x (np.ndarray): The original x-axis values.
#                 - new_x (np.ndarray): The new x-axis values. Defaults to a logspace array.
#                 - smooth_first (bool): Whether to smooth the data before interpolation.
#                 - axis (int): The axis along which to interpolate.
#                 - bc_type (str): Boundary condition type for CubicSpline.
#                 - extrapolate (bool): Whether to extrapolate beyond the data range.

#         Returns:
#         np.ndarray: The interpolated data.
#         """
#         array = self._ensure_array(value)
#         if array.dtype.kind == "c":
#             return self._complex_array_eval(array, self.interpolated_form, **kwargs)

#         # Parse defaults from kwargs or self.interp_kwargs
#         old_x = kwargs.pop("old_x", self.interp_kwargs.get("old_x", self.active_x))
#         new_x = kwargs.pop("new_x", self.interp_kwargs.get("new_x", None))
#         smooth_first = kwargs.pop("smooth_first", self.interp_kwargs.get("smooth_first", True))
#         spl_type = kwargs.pop("spl_type", self.interp_kwargs.get("spl_type", "pchip"))
#         axis = kwargs.pop("axis", self.interp_kwargs.get("axis", 0))
#         bc_type = kwargs.pop("bc_type", self.interp_kwargs.get("bc_type", "not-a-knot"))
#         extrapolate = kwargs.pop("extrapolate", self.interp_kwargs.get("extrapolate", None))

#         old_x = self._ensure_array(old_x)

#         if not old_x.size or old_x.size != array.shape[axis]:
#             raise ValueError(
#                 f"old_x length {old_x.size} does not match data length {array.shape[axis]} along axis {axis}"
#             )
#         # Smooth the data first if required
#         data = self.derivative_form(array, **{**kwargs, "deriv": 0}) if smooth_first else array

#         if new_x is None and old_x is not None:
#             new_x = np.logspace(
#                 np.log10(old_x.min()),
#                 np.log10(old_x.max()),
#                 num=len(old_x),
#             )

#         # Interpolate the smoothed data
#         if "pchip" in spl_type:
#             interpolator = PchipInterpolator(old_x, data, axis=axis, extrapolate=extrapolate)
#         else:
#             interpolator = CubicSpline(
#                 old_x, data, axis=axis, bc_type=bc_type, extrapolate=extrapolate
#             )

#         self.active_x = new_x
#         return interpolator(new_x)

#     def _ensure_array(self, value) -> np.ndarray:
#         """
#         Helper function to ensure the input is converted to a numpy array.

#         Parameters:
#         value: The input value to be converted.

#         Returns:
#         np.ndarray: The converted numpy array.
#         """
#         if hasattr(value, "array"):
#             value = value.array
#         return np.array(value)

#     def _complex_array_eval(
#         self, value: np.ndarray, func: Callable[..., np.ndarray], **kwargs
#     ) -> np.ndarray:
#         """
#         Helper function to ensure the input is converted to a Complexer object.

#         Parameters:
#         value: The input value to be converted.
#         func: The function to apply to the real and imaginary parts.
#         **kwargs: Additional arguments to pass to the function.

#         Returns:
#         np.ndarray: The result of applying the function to the complex array.
#         """
#         if value.dtype.kind == "c":
#             real = func(value.real, **kwargs.copy())
#             imag = func(value.imag, **kwargs.copy())
#             return real + 1j * imag
#         return func(value, **kwargs)
