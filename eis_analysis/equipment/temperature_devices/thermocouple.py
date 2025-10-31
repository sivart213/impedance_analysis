# -*- coding: utf-8 -*-
"""
Thermocouple module for temperature conversions and measurements.

This module provides a Thermocouple class that can be used to convert
voltage (mV) to temperature (°C). It is designed to be inserted between a
serial (measurement) device and a logger but can be used to convert previously
measured values.

Original inspiration: `https://github.com/andygock/Thermocouple`
Source for numbers and equations: `https://its90.nist.gov/OverviewThermo`
"""
__all__ = ["Thermocouple"]

# import os
import re
import json
import warnings
import importlib.resources
from collections.abc import Callable, Sequence

import numpy as np

# # Load the JSON file with coefficients
# json_path = os.path.join(os.path.dirname(__file__), "its90nist_allcoeff.json")
# with open(json_path, "r", encoding="utf-8") as f:
#     COEFFICIENTS = json.load(f, parse_float=np.float64)["coefficients"]
json_path = importlib.resources.files("eis_analysis.equipment.temperature_devices").joinpath(
    "its90nist_allcoeff.json"
)
with json_path.open("r", encoding="utf-8") as f:
    COEFFICIENTS = json.load(f, parse_float=np.float64)["coefficients"]

BASIC_CONVERSIONS = {
    "f_to_c": lambda f: (f - 32.0) / (9.0 / 5.0),
    "k_to_c": lambda k: k - 273.15,
    "c_to_f": lambda c: c * (9.0 / 5.0) + 32.0,
    "c_to_k": lambda c: c + 273.15,
    "v_to_mv": lambda v: v * 1000.0,
    "uv_to_mv": lambda uv: uv / 1000.0,
    "mv_to_uv": lambda mv: mv * 1000.0,
    "mv_to_v": lambda mv: mv / 1000.0,
}

NumericArray = Sequence[float] | np.typing.NDArray[np.float64]


def scale_to_target(
    values: float | NumericArray,
    target_range: float | NumericArray,
    number_base: float = 10,
    return_multiplier: bool = False,
) -> float | np.ndarray:
    """
    Find (and optionally return) a multiplier to convert values to fit within an expected range.

    This function is useful for converting measured values which were saved as proportional values
    equivalent to the actual values. Using the expected values as the anchor points, this function
    can be used to restore the actual values from the proportional values.

    Parameters
    ----------
    values : float, sequence of floats, or NumPy array of floats.
        Input values to convert.
    target_range : float, sequence of floats, or NumPy array of floats.
        Target range or value for the values.
    number_base : int, optional
        Base number for limiting multipliers. Default is 10.
    return_multiplier : bool, optional
        Return the multiplier instead of the converted values. Default is False.

    Returns
    -------
    float or np.ndarray
        Resulting values or the multiplier.
    """
    values = np.asarray(values, dtype=np.float64)
    target_range = np.asarray(target_range, dtype=np.float64)
    number_base = float(number_base)

    best_multiplier = 1
    best_fit = 1

    range_min = target_range.min()
    range_max = target_range.max()

    if not np.all((values >= target_range.min()) & (values <= range_max)):
        best_fit = np.inf

        min_exp = int(np.floor(np.log(abs(range_min / values.min())) / np.log(number_base)))
        max_exp = int(np.ceil(np.log(abs(range_max / values.max())) / np.log(number_base)))

        exponents = np.arange(min_exp - 2, max_exp + 3)
        multipliers = np.concatenate((number_base**exponents, -(number_base**exponents)))

        for multiplier in multipliers:
            converted = values * multiplier

            delta = (range_max - range_min) / 2
            if delta == 0:
                normalized = converted - range_min
            else:
                normalized = (converted - range_min) / delta - 1
            fit = np.mean(np.abs(normalized))

            if fit < best_fit:
                best_fit = fit
                best_multiplier = multiplier

        if values.size > 1:
            in_range = np.mean(
                [
                    range_min < v < range_max
                    for v in np.asarray(values * best_multiplier, dtype=np.float64)
                ]
            ).item()
            if in_range < 1:
                warnings.warn(f"{int(in_range*100)}% of the values fit within the target range.")

    if return_multiplier:
        return best_multiplier
    res = values * best_multiplier
    return res if res.size > 1 else res.item()


def scale_array_to_target(
    values: NumericArray,
    target_range: NumericArray,
    number_base: float = 10,
    sequential: bool = True,
) -> np.ndarray:
    """
    Find multipliers to convert values to fit within an expected range, allowing for multiple scales.

    This function is useful for converting measured values which were saved as proportional values
    equivalent to the actual values. Using the expected values as the anchor points, this function
    can be used to restore the actual values from the proportional values.

    Parameters
    ----------
    values : sequence of floats or NumPy array of floats.
        Input values to convert.
    target_range : sequence of floats or NumPy array of floats.
        Target range or value for the values.
    number_base : int, optional
        Base number for limiting multipliers. Default is 10.
    sequential : bool, optional
        If True, ensures that valid scales are sequential. Default is True.

    Returns
    -------
    np.ndarray
        Resulting values.
    """
    values = np.asarray(values, dtype=np.float64)
    target_range = np.asarray(target_range, dtype=np.float64)
    number_base = float(number_base)

    if np.all((values >= target_range.min()) & (values <= target_range.max())):
        return scale_to_target(values, target_range, number_base, return_multiplier=False)  # type: ignore

    # Vectorize the scale_to_target function to apply it to each element in the array
    vectorized_scale_to_target = np.vectorize(
        lambda v: scale_to_target(v, target_range, number_base, return_multiplier=True)
    )

    best_multipliers = vectorized_scale_to_target(values)

    if sequential:
        # Find the most common multiplier
        unique, counts = np.unique(best_multipliers, return_counts=True)
        most_common_multiplier = unique[np.argmax(counts)]
        iter_start = np.argmax(best_multipliers == most_common_multiplier)

        if iter_start > 0:
            # Iterate backwards to 0
            for i in range(iter_start, -1, -1):
                if (
                    i > 0
                    and best_multipliers[i] != best_multipliers[i - 1]
                    and abs(
                        np.log(best_multipliers[i] / best_multipliers[i - 1]) / np.log(number_base)
                    )
                    > 1
                ):
                    best_multipliers[i - 1] = best_multipliers[i]

        # Iterate forwards to the end
        for i in range(iter_start, len(values)):
            if (
                i > 0
                and best_multipliers[i] != best_multipliers[i - 1]
                and abs(
                    np.log(best_multipliers[i] / best_multipliers[i - 1]) / np.log(number_base)
                )
                > 1
            ):
                best_multipliers[i] = best_multipliers[i - 1]

    res = values * best_multipliers
    return res


class Thermocouple:
    """
    Thermocouple class to handle temperature conversions and measurements.

    This class is primarily intended to convert a device's output voltage (mV)
    to temperature (°C). It is designed to be able to insert between the
    measureing device any any existing codebase which reads the source value.
    To achieve that, it is possible to pass a source function to this class and
    "rename" the get function to whatever is expected in the existing codebase.
    This way when the codebase calls to get the measurement output, the source
    value is read and converted to the desired unit which is then returned.
    It is important to note that the source function must return either a value
    in mV or °C. Static method 'wrap_source_function' is provided to simplify
    this process but is not required to be used as long as the input
    requirements are met.

    The class can alternatively be used to convert a set of values (i.e. saved
    outputs) via the 'convert' method. Note that calling 'convert' will not
    trigger the source function making calls to 'convert' independent of the
    source function. As a result, this class can be initialized without a
    source function if only conversion is required.  Passing any kwargs to
    'convert' (or 'get') will alter the output as directed but will not change
    the default behavior of the class which is set at initialization.

    Lastly, it is possible to get a standalone conversion function using the
    'get_conversion_function' method. The returned function will only accept a
    single argument which is the value or array of values to convert. The
    desired behavior (such as including error) must be set at the time of
    calling 'get_conversion_function'.

    Parameters
    ----------
    thermo_type : str, optional
        Type of the thermocouple (e.g., "K", "J"). Default is "K".
    unit : str, optional
        Desired output unit, typically a temperature unit (e.g., "C", "F").
        Default is "C".
    source : callable, optional
        Function to read the source value (e.g., voltage). Default is None.
        Function must return either a value in mV or °C.
    voltage_source : bool, optional
        Boolean indicating if the source value is in voltage (mV). Default is
        True.
    error : bool, optional
        Boolean indicating whether to include error calculations. Default is False.
    show_warnings : bool, optional
        Boolean indicating whether to show warnings. Default is True.
    call_name : str, optional
        Name of the method to call the get function. Default is "temp".

    Attributes
    ----------
    unit : str
        Desired output unit.
    source : callable
        Function to read the source value.
    voltage_source : bool
        Boolean indicating if the source value is in voltage (mV).
    error : bool
        Boolean indicating whether to include error calculations.
    show_warnings : bool
        Boolean indicating whether to show warnings.
    thermo_type : str
        Type of the thermocouple.
    call_name : str
        Name of the method to call the get function.

    Methods
    -------
    get_conversion_function(thermo_type=None, voltage_source=None, error=None, show_warnings=None)
        Generates a conversion function based on the given parameters.
    get(unit=None, error=None)
        Read the value from the source if available and convert.
    convert(value, unit=None, **kwargs)
        Convert given thermocouple output value(s).
    wrap_source_function(func, conversion)
        Wrap the output of a function in a secondary function that converts to mV or °C as required.
    get_thermo_information()
        Get the available thermocouple information.

    Examples
    --------
    **Example 1** Create a Thermocouple instance and get temperature readings
    in °C and °F from a source object in communication with a device:
    Assumes the source object has a method to read the temperature (e.g., read)
    in V and the device is a type K thermocouple.

    >>> source_func = device_communication_class.read
    >>> wrapped_source_func = Thermocouple.wrap_source_function(
                                        source_func, "v_to_mv",
                                    )
    >>> tc = Thermocouple(source=wrapped_source_func, thermo_type="K", unit="C", call_name="read")
    >>> tempC = tc.get()
    >>> tempF = tc.get("F")
    >>> print(f"Temperature in °C: {tempC} and in °F: {tempF}")
    Temperature in °C: 51.89608545106927 and in °F: 125.41295381192468 #random
    >>> tempCr = tc.read()
    >>> tempFr = tc.read("F")
    >>> print(f"Temperature in °C: {tempCr} and in °F: {tempFr}")
    Temperature in °C: 51.89608545106927 and in °F: 125.41295381192468 #random

    **Example 2** Get the same temperature readings as above but with error
    calculations:

    >>> tempCe = tc.get(error=True)
    >>> tempCe
    [51.89608545106927, 51.84608545106927, 51.93608545106927] °C #random

    **Example 3** Convert a set of values from a device to temperature values:
    Assumes the device values are stored in a CSV file and the values are in mV.

    >>> import pandas as pd
    >>> values = pd.read_csv("device_values.csv")["values"]
    >>> converted_values = tc.convert(values)
    >>> converted_values
    array([
        49.87654321098765, 50.12345678901234, 50.23456789012345, 49.98765432109876,
        50.34567890123456, 49.87654321098765, 50.45678901234567, 50.56789012345678,
        49.76543210987654, 50.67890123456789
    ]) #random
    """

    def __init__(
        self,
        thermo_type: str = "K",
        unit: str = "C",
        source: Callable[[], int | float] | None = None,
        voltage_source: bool = True,
        error: bool = False,
        show_warnings: bool = True,
        call_name: str = "temp",
    ):
        self.unit = unit if isinstance(unit, str) else "C"
        self.source = source
        self.voltage_source = voltage_source
        self.error = error
        self.show_warnings = show_warnings
        self.thermo_type = thermo_type
        self.call_name = call_name

    def __getattr__(self, name: str):
        if name == self.call_name:
            return self.get
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __getstate__(self):
        state = self.__dict__.copy()
        # Exclude the un-pickleable attribute
        state[self.call_name] = (
            "Dynamically added method which renames 'get' for inter-object compatibility"
        )
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    @property
    def source(self) -> Callable[[], int | float]:
        """Get the source function."""

        def return_zero() -> int:
            return 0

        return self._source or return_zero

    @source.setter
    def source(self, func: Callable[[], int | float] | None):
        """Set the source function and validate it."""
        if func is None:
            self._source = None
        elif callable(func):
            func_res = func()
            if not isinstance(func_res, (int, float)):
                raise ValueError("Source must return an int or float.")
            self._source = func
        else:
            raise ValueError("Source must be a callable or None.")

    @property
    def voltage_source(self) -> bool:
        """Get the voltage source flag."""
        return self._voltage_source

    @voltage_source.setter
    def voltage_source(self, value: bool):
        """Set the voltage source flag and validate it."""
        if not isinstance(value, bool):
            raise ValueError("voltage_source must be a boolean.")
        self._voltage_source = value

    @property
    def thermo_type(self) -> str:
        """Get the thermocouple type."""
        return self._thermo_type

    @thermo_type.setter
    def thermo_type(self, value: str):
        """Set the thermocouple type and update the conversion function."""
        self._thermo_type = self._validate_thermo_type(value)
        self._conversion_function = self.get_conversion_function()

    @property
    def min_mV(self) -> float:
        """Get the minimum voltage value for the thermocouple type."""
        return COEFFICIENTS[self._thermo_type]["mV_range"][0]

    @property
    def max_mV(self) -> float:
        """Get the maximum voltage value for the thermocouple type."""
        return COEFFICIENTS[self._thermo_type]["mV_range"][1]

    @property
    def min_C(self) -> float:
        """Get the minimum temperature value for the thermocouple type."""
        return COEFFICIENTS[self._thermo_type]["c_range"][0]

    @property
    def max_C(self) -> float:
        """Get the maximum temperature value for the thermocouple type."""
        return COEFFICIENTS[self._thermo_type]["c_range"][1]

    def _validate_thermo_type(self, value: str) -> str:
        """Sanitize the thermocouple type by removing any non-alphabetic characters."""
        value = re.sub(r"\s*type\s*", "", value, flags=re.IGNORECASE).upper()
        if value not in ["B", "E", "J", "K", "N", "R", "S", "T"]:
            raise ValueError("Invalid thermocouple type")
        return value

    def get_conversion_function(
        self,
        thermo_type: str | None = None,
        voltage_source: bool | None = None,
        error: bool | None = None,
        show_warnings: bool | None = None,
    ) -> Callable[[float | np.ndarray], float | np.ndarray]:
        """
        Generates a conversion function based on the given parameters.

        Parameters
        ----------
        thermo_type : str, optional
            Type of the thermocouple (e.g., "K", "J"). Defaults to the class variable value.
        voltage_source : bool, optional
            Boolean indicating if the source value is in voltage (mV). Defaults to the class variable value.
        error : bool, optional
            Boolean indicating whether to include error calculations. Defaults to the class variable value.
        show_warnings : bool, optional
            Boolean indicating whether to show warnings. Defaults to the class variable value.

        Returns
        -------
        Callable[[float | np.ndarray], float | np.ndarray]
            Conversion function to convert between mV and °C.
        """
        thermo_type = self._validate_thermo_type(thermo_type or self._thermo_type)
        voltage_source = voltage_source if voltage_source is not None else self._voltage_source
        error = error if error is not None else self.error
        show_warnings = show_warnings if show_warnings is not None else self.show_warnings

        if voltage_source:
            coeffs = COEFFICIENTS[thermo_type.upper()]["mV_to_degC"]
            type_range = COEFFICIENTS[thermo_type.upper()]["mV_range"]
        else:
            coeffs = COEFFICIENTS[thermo_type.upper()]["degC_to_mV"]
            type_range = COEFFICIENTS[thermo_type.upper()]["c_range"]

        def equation_compiler(values: float | np.ndarray) -> float | np.ndarray:
            """Converts mV to °C or vice versa. Values can be scalar or array."""
            values = np.asarray(values, dtype=np.float64)

            if values.size == 0:
                return np.array([])

            # Apply range limits
            below_min = values < type_range[0]
            above_max = values > type_range[1]

            if np.any(below_min):
                values[below_min] = type_range[0]
                if show_warnings:
                    msg_start = f"{sum(below_min)} values are" if values.size > 1 else "Value is"
                    warnings.warn(
                        f"{msg_start} below the minimum range ({type_range[0]}) for type {thermo_type.upper()} thermocouples."
                    )
            if np.any(above_max):
                values[above_max] = type_range[1]
                if show_warnings:
                    msg_start = f"{sum(above_max)} values are" if values.size > 1 else "Value is"
                    warnings.warn(
                        f"{msg_start} above the maximum range ({type_range[1]}) for type {thermo_type.upper()} thermocouples."
                    )

            results = np.zeros_like(values, dtype=np.float64)
            errors = np.zeros((values.size, 2), dtype=np.float64)

            for range_ in coeffs:
                if range_["max"] == type_range[1]:
                    mask = (values >= range_["min"]) & (values <= range_["max"])
                else:
                    mask = (values >= range_["min"]) & (values < range_["max"])

                if "exponential" in range_:
                    a0 = range_["exponential"]["a0"]
                    a1 = range_["exponential"]["a1"]
                    a2 = range_["exponential"]["a2"]
                    results[mask] = sum(
                        c * values[mask] ** p for p, c in enumerate(range_["values"])
                    ) + a0 * np.exp(a1 * (values[mask] - a2) ** 2)
                else:

                    results[mask] = sum(
                        c * values[mask] ** p for p, c in enumerate(range_["values"])
                    )
                if error:
                    errors[mask] = range_["error"]

            if error:
                results = np.atleast_2d(results.T).T
                results = np.concatenate((results, results + errors), axis=1)
                return results if results.size > 3 else results.flatten().tolist()  # type: ignore

            return results if results.size > 1 else results.item()

        return equation_compiler

    def revert(
        self,
        value: float | np.ndarray,
        **kwargs,
    ) -> float | np.ndarray:
        """
        Convert value(s) in the reverse direction of the current settings.

        If the current instance converts mV to °C, this will convert °C to mV, and vice versa.

        Parameters
        ----------
        value : float or np.ndarray
            Value(s) to convert in the reverse direction.
        **kwargs : dict, optional
            Additional arguments for conversion function.

        Returns
        -------
        float or np.ndarray
            Converted value(s) in the reverse direction.
        """
        # Reverse the voltage_source flag for the conversion direction
        reverse_kwargs = dict(kwargs)
        reverse_kwargs["voltage_source"] = not self.voltage_source
        conversion = self.get_conversion_function(**reverse_kwargs)
        # print("Converted")
        return conversion(value)

    def convert(
        self,
        value: float | np.ndarray,
        unit: str | None = None,
        ambient: float | None = None,
        **kwargs,
    ) -> float | np.ndarray:
        """
        Convert given thermocouple output value(s). Typically used to convert mV to °C but can also convert in reverse.

        Parameters
        ----------
        value : float or np.ndarray
            Temperature value(s) to convert.
        unit : str, optional
            Desired output unit, typically a temperature unit c or f. Default is None.
        ambient : float, optional
            Ambient temperature (in °C) or voltage (in mV) to adjust for cold junction compensation.
            If provided, will be converted to the appropriate units and applied to the input value(s)
            before conversion (added for mV to °C, subtracted for °C to mV).
        **kwargs : dict, optional
            Additional arguments for conversion function (e.g., voltage_source, error, show_warnings, etc.).

        Returns
        -------
        float or np.ndarray
            Converted value(s).
        """
        unit = unit or self.unit

        # Handle ambient adjustment if provided
        if ambient is not None:
            if kwargs.get("voltage_source", self.voltage_source):
                # mV to °C: convert ambient (°C) to mV and add to value
                ambient_mv = self.revert(ambient, **kwargs)
                value = np.asarray(value, dtype=np.float64) + ambient_mv
            else:
                # °C to mV: convert ambient (mV) to °C and subtract from value
                ambient_c = self.revert(ambient, **kwargs)
                value = np.asarray(value, dtype=np.float64) - ambient_c

        if kwargs:
            conversion = self.get_conversion_function(**kwargs)
            value = conversion(value)
        else:
            value = self._conversion_function(value)

        if "v" in unit.lower() and "mv_to_" + unit.lower() in BASIC_CONVERSIONS:
            value = BASIC_CONVERSIONS["mv_to_" + unit.lower()](value)
        elif "c_to_" + unit.lower() in BASIC_CONVERSIONS:
            value = BASIC_CONVERSIONS["c_to_" + unit.lower()](value)
        return value

    def get(self, unit: str | None = None, error: bool | None = None) -> float | np.ndarray:
        """
        Read the value from the source if available and convert.

        Parameters
        ----------
        unit : str, optional
            Desired output unit, typically a temperature unit c or f. Default is None.
        error : bool, optional
            Boolean indicating whether to include error calculations. Default is None.

        Returns
        -------
        float or np.ndarray
            Converted temperature value.
        """
        unit = unit or self.unit
        if self._source is None:
            return 0
        if isinstance(error, bool):
            value = self.convert(self.source(), unit, error=error)
        else:
            value = self.convert(self.source(), unit)

        return value

    def find_pre_conversion(
        self,
        values: float | NumericArray,
        target_range: float | NumericArray,
        number_base: float = 10,
        convert_range: bool = True,
    ) -> Callable[[float | NumericArray], float | np.ndarray]:
        """
        Find a conversion function to convert values to fit within a target range before the primary conversion.

        Parameters
        ----------
        values : float, Sequence[float], or np.ndarray
            Input values to convert.
        target_range : float, Sequence[float], or np.ndarray
            Target range for the values.
        number_base : float, optional
            Base number for limiting multipliers. Default is 10.
        convert_range : bool, optional
            Whether to convert the target range using the primary conversion function. Default is True.

        Returns
        -------
        Callable[[float | Sequence[float, np.ndarray]], float | np.ndarray]
            Conversion function to convert values to fit within the target range.

        Notes
        -----
        The class typically converts mV to °C so this function is similarly expecting voltages which need to
        be converted to mV before the primary conversion and since a temperature range is often the easier to define
        for expected values, 'convert_range' is defaulted to True (i.e. it assumes the 'target_range' is in temperature).
        However, it would be more accurate to state that the function assumes 'target_range' is in the same units as
        the output of the primary conversion function. This is important because if 'self.voltage_source' is False,
        (i.e. the primary conversion function converts from °C to mV) then the default assumption is that 'target_range'
        is in mV. For this reason it is important to be aware of the class settings when defining the input arguments
        for this function.
        """
        values = np.asarray(values, dtype=np.float64)
        target_range = np.asarray(target_range, dtype=np.float64)
        number_base = float(number_base)

        if convert_range:
            target_range = self.convert(target_range, voltage_source=not self.voltage_source)

        mult = scale_to_target(values, target_range, number_base, return_multiplier=True)

        def conversion_func(val: float | NumericArray) -> float | np.ndarray:
            return np.asarray(val, dtype=np.float64) * mult

        return conversion_func

    @staticmethod
    def wrap_source_function(
        func: Callable[[], int | float],
        conversion: str | Callable[[int | float], int | float],
    ) -> Callable[[], int | float]:
        """
        Wrap the output of a function in a secondary function that converts to mV or °C as required.

        Parameters
        ----------
        func : callable
            The original function to wrap.
        conversion : str or callable
            The conversion function or a string key from BASIC_CONVERSIONS.

        Returns
        -------
        callable
            A wrapped function that converts the output to the desired unit.
        """
        if isinstance(conversion, str):
            if conversion.lower() in BASIC_CONVERSIONS:
                conversion_func = BASIC_CONVERSIONS[conversion.lower()]
            elif conversion.lower() + "_to_mv" in BASIC_CONVERSIONS:
                conversion_func = BASIC_CONVERSIONS[conversion.lower() + "_to_mv"]
            elif conversion.lower() + "_to_c" in BASIC_CONVERSIONS:
                conversion_func = BASIC_CONVERSIONS[conversion.lower() + "_to_c"]
            else:
                raise ValueError(
                    f"'{conversion}' is not an available conversion. Conversions from units other than F, K, V, and uV should be specified as a function."
                )
        elif callable(conversion):
            conversion_func = conversion
        else:
            raise ValueError(
                "Conversion must be a callable or one of the available conversion types(F, K, V, and uV)."
            )

        def wrapped_func() -> int | float:
            value = func()
            return conversion_func(value)

        return wrapped_func

    @staticmethod
    def get_thermo_information() -> dict:
        """
        Get all available thermocouple information.

        Returns
        -------
        dict
            Available thermocouple information.
        """
        # with open(json_path, "r", encoding="utf-8") as file:
        #     data = json.load(file)
        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data


if __name__ == "__main__":
    # Example usage of the Thermocouple class
    # def example_source() -> float:
    #     """Example source function returning 2.1e-3 V."""
    #     return 2.1e-3  # Example source function returning 2.1e-3 V

    # # Wrap the example source function to convert from kV to mV
    # wrapped_source = Thermocouple.wrap_source_function(example_source, "v_to_mv")

    # # Create a Thermocouple instance with the wrapped source function
    # tc = Thermocouple(source=wrapped_source, thermo_type="K", unit="C")

    # # Get the temperature reading
    # tempC = tc.temp()
    # tempCe = tc.temp(error=True)
    # tempF = tc.temp("F")

    # print(f"Temperature in °C: {tempC} and in °F: {tempF}")
    # print(f"Temperature in °C with error: {tempCe} °C")

    # meas_vals = np.linspace(1, 3, 10) + np.random.normal(0, 0.1, 10)
    # print(f"Generated values: {meas_vals}")

    # # Test the convert function
    # converted_values = tc.convert(meas_vals)
    # print(f"Converted values in °C: {converted_values}")

    # converted_values = tc.convert(meas_vals, "F", error=True)
    # print(f"Converted values in °F: {converted_values}")

    tc = Thermocouple(thermo_type="K", unit="C")
    tc_rev = Thermocouple(thermo_type="K", unit="C", voltage_source=False)

    # double_T = tc.convert(tc_rev.convert(100))
    # print(double_T)
