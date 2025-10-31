# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
import warnings
from typing import Any, Self, overload

import numpy as np
from numpy.typing import ArrayLike

try:
    from .complexer import Complexer
    from .definitions import COMP_ALIASES
    from ..data_treatment.value_ops import convert_val
except ImportError:
    from eis_analysis.z_system.complexer import Complexer
    from eis_analysis.z_system.definitions import COMP_ALIASES
    from eis_analysis.data_treatment.value_ops import convert_val


class UnitFloat:
    def __init__(
        self, unit: str = "cm", exponent: int = 1, default: float = 1.0, strict: bool = False
    ):
        self.unit = unit
        self.exp = exponent
        self.default = float(default)
        self.strict = strict
        unit_str = f"{unit}^{exponent}" if exponent < 0 or 1 < exponent else unit
        self.__doc__ = f"float value in {unit_str}"

    def __set_name__(self, owner, name: str) -> None:
        # Python calls this automatically when the class is created
        self.private_name = "_" + name
        self.__doc__ = f"{name} {self.__doc__}"

    @overload
    def __get__(self, instance: None, owner: type) -> "UnitFloat": ...
    @overload
    def __get__(self, instance: object, owner: type) -> float: ...

    def __get__(self, instance: Any, owner: type) -> Any:
        if instance is None:
            return self  # type: ignore
        return getattr(instance, self.private_name, self.default)
        # return instance.__dict__.get(self.private_name, self.default)

    def __set__(self, instance: Any, value: Any) -> None:
        val = None
        if isinstance(value, (int, float)) and value:
            val = float(value)
        elif isinstance(value, str) and value:
            val = convert_val(value, f_unit=self.unit, exponent=self.exp)
        elif isinstance(value, (tuple, list)) and value:
            exp = self.exp if len(value) < 3 else value[2]
            val = convert_val(*value[:2], f_unit=self.unit, exponent=exp)
        elif isinstance(value, dict) and value:
            value["f_unit"] = self.unit
            value.setdefault("exponent", self.exp)
            val = convert_val(**value)
        elif self.strict:
            raise TypeError(f"Unsupported type for {self.private_name}: {type(value)}")
        if val is not None and val:
            setattr(instance, self.private_name, float(val))
            # instance.__dict__[self.private_name] = float(val)


class ZDataOps:
    """
    Mixin class for operations on complex data.
    """

    complexer_obj: Complexer
    _frequency: np.ndarray

    thickness: UnitFloat = UnitFloat(unit="cm", exponent=1, default=1.0, strict=False)
    area: UnitFloat = UnitFloat(unit="cm", exponent=2, default=1.0, strict=False)

    def _set_data(self, data: ArrayLike | None = None, frequency: ArrayLike | None = None):
        """
        Initialize or reset impedance-related data.
        Ensures complexer_obj exists before frequency is set.
        """
        if data is not None:
            self.complexer_obj = Complexer(data)
        elif not hasattr(self, "complexer_obj"):
            self.complexer_obj = Complexer()

        # Frequency: default or provided
        if not hasattr(self, "_frequency"):
            self._frequency = np.array([1.0])
        if frequency is not None:
            frequency = np.asarray(frequency)
            if len(frequency) == len(self.complexer_obj.array):
                self._frequency = frequency

        if len(self._frequency) != len(self.complexer_obj.array):
            self._frequency = np.ones(len(self.complexer_obj.array))

    @property
    def array(self) -> np.ndarray:
        """Return the complex array which should also be the impedance by default."""
        return self.complexer_obj.array

    @property
    def frequency(self) -> np.ndarray:
        """Return the frequency array."""
        return self._frequency

    @property
    def angular_frequency(self) -> np.ndarray:
        """Calculate the angular frequency :math: \\omega = 2 * \\pi * f."""
        return 2 * np.pi * self.frequency

    @property
    def _mu(self) -> np.ndarray:
        """Calculate the common conversion factor :math: \\mu = j * \\omega * C_0."""
        return 1j * self.angular_frequency * self.C_0

    @property
    def area_over_thickness(self) -> float:
        """Calculate the area over thickness ratio. :math: \\frac{A}{d}."""
        return self.area / self.thickness

    a_d = area_over_thickness

    @property
    def permittivity_constant(self) -> float:
        """Return the permittivity of free space (:math: \\epsilon_0 = 8.85418782 \\times 10^{-14} F/cm)."""
        return 8.85418782e-14  # F/cm

    e_0 = permittivity_constant

    @property
    def characteristic_capacitance(self) -> float:
        """Calculate the characteristic capacitance. :math: C_0 = \\epsilon_0 * \\frac{A}{d}."""
        return self.e_0 * self.a_d

    C_0 = characteristic_capacitance

    @property
    def impedance(self) -> Complexer:
        """Return the complex impedance."""
        return Complexer._from_valid(self.complexer_obj.array, copy=True)

    Z = impedance

    @property
    def admittance(self) -> Complexer:
        """Calculate the complex admittance. :math: Y = \\frac{1}{Z}."""
        return Complexer._from_valid(1 / self.complexer_obj.array)

    Y = admittance

    @property
    def modulus(self) -> Complexer:
        """Calculate the complex modulus. :math: M = \\mu * Z = j * \\omega * C_0 * Z."""
        return Complexer._from_valid(self._mu * self.complexer_obj.array)

    M = modulus

    @property
    def capacitance(self) -> Complexer:
        """Calculate the complex capacitance. :math: C = \\frac{1}{j * \\omega * Z}."""
        return Complexer._from_valid(1 / (1j * self.angular_frequency * self.complexer_obj.array))

    C = capacitance

    @property
    def permittivity(self) -> Complexer:
        """Calculate the complex permittivity. :math: \\epsilon = \\frac{C}{C_0} = \\frac{1}{j * \\omega * C_0 * Z}."""
        return Complexer._from_valid(self.e_0 / (self._mu * self.complexer_obj.array), sign=-1)

    @property
    def relative_permittivity(self) -> Complexer:
        """Calculate the relative complex permittivity. :math: \\epsilon_r = \\frac{1}{\\mu * Z} = \\frac{1}{j * \\omega * C_0 * Z}."""
        return Complexer._from_valid(1 / (self._mu * self.complexer_obj.array), sign=-1)

    e_r = relative_permittivity

    @property
    def relative_permittivity_corrected(self) -> Complexer:
        """Calculate the corrected relative complex permittivity. (e-je) generic discription."""
        arr = 1 / (self._mu * self.complexer_obj.array)
        return Complexer._from_valid(arr.real + (arr.imag + self.dc_conductivity) * 1j, sign=-1)
        # return Complexer._from_valid(arr.real - arr.imag - self.dc_conductivity * 1j, sign=-1)
        # return Complexer._from_valid(self.e_r.real - (self.e_r.imag + self.dc_conductivity) * 1j, sign=-1)

    @property
    def conductivity(self) -> Complexer:
        """Calculate complex conductivity. (sigma + jsigma) generic discription."""
        return Complexer._from_valid((1 / self.complexer_obj.array) / self.a_d)

    @property
    def resistivity(self) -> Complexer:
        """Calculate complex resistivity. (rho - jrho) generic discription."""
        return Complexer._from_valid(self.complexer_obj.array * self.a_d)

    @property
    def susceptibility(self) -> Complexer:
        """Calculate complex susceptibility. (rho - jrho) generic discription."""
        # arr = 1 / self.M - self.val_towards_infinity(self.relative_permittivity.real)
        arr = 1 / (self._mu * self.complexer_obj.array)
        return Complexer._from_valid(arr - self.val_towards_infinity(arr.real), sign=-1)

    @property
    def dc_conductivity(self) -> np.ndarray:
        """Calculate the DC conductivity."""
        dc_val = self.val_towards_zero(self.conductivity.real)
        return dc_val / (self.permittivity_constant * self.angular_frequency)

    @property
    def dissipation_factor(self) -> np.ndarray:
        """Calculate complex resistivity. (rho - jrho) generic discription."""
        return self.relative_permittivity.imag / self.relative_permittivity.real

    @property
    def loss_index(self) -> np.ndarray:
        """Calculate complex resistivity. (rho - jrho) generic discription."""
        return self.relative_permittivity.imag

    @property
    def loss_angle(self) -> np.ndarray:
        """Calculate complex resistivity. (rho - jrho) generic discription."""
        return self.relative_permittivity.phase

    def val_towards_zero(self, array: np.ndarray, perc: int | float = 5) -> np.number:
        """
        Returns the average value of the array close to the lower end of self.frequency.
        """
        # Determine which end of the array corresponds to the lower end of self.frequency
        if self.frequency[0] < self.frequency[-1]:
            try:
                # Determine the number of elements to consider based on the percentage
                num_elements = max(1, int(len(array) * perc / 100))
                # Lower end is at the beginning of the array
                return np.mean(array[:num_elements])
            except FloatingPointError:
                return array[0]
        else:
            try:
                # Determine the number of elements to consider based on the percentage
                num_elements = max(1, int(len(array) * perc / 100))
                # Lower end is at the end of the array
                return np.mean(array[-num_elements:])
            except FloatingPointError:
                return array[-1]

    def val_towards_infinity(self, array: np.ndarray, perc: int | float = 5) -> np.number:
        """
        Returns the average value of the array close to the higher end of self.frequency.
        """
        # Determine which end of the array corresponds to the higher end of self.frequency
        if self.frequency[0] > self.frequency[-1]:
            try:
                # Determine the number of elements to consider based on the percentage
                num_elements = max(1, int(len(array) * perc / 100))
                # Higher end is at the beginning of the array
                return np.mean(array[:num_elements])
            except FloatingPointError:
                return array[0]
        else:
            try:
                # Determine the number of elements to consider based on the percentage
                num_elements = max(1, int(len(array) * perc / 100))
                # Higher end is at the end of the array
                return np.mean(array[-num_elements:])
            except FloatingPointError:
                return array[-1]


class ImpedanceHelper(ZDataOps):

    def __init__(
        self,
        data: ArrayLike | object | Self,
        frequency: ArrayLike | None = None,
        thickness: int | float = 1.0,
        area: int | float = 1.0,
        **_,
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
        self._set_data()
        self.base_forms = set(
            "impedance, admittance, modulus, capacitance, resistivity, conductivity, permittivity, "
            "relative_permittivity, relative_permittivity_corrected, susceptibility".split(", ")
        )
        self.area = area if area != 1.0 else getattr(data, "area", 1.0)
        self.thickness = thickness if thickness != 1.0 else getattr(data, "thickness", 1.0)
        freq = getattr(data, "frequency", frequency) if frequency is None else frequency
        self._set_data(getattr(data, "complexer_obj", data), freq)  # type: ignore

    def __getitem__(self, name) -> Complexer:
        """Allow slicing and indexing."""
        if name in self.base_forms:
            return getattr(self, name)
        else:
            raise TypeError("Index must be a valid property name")

    def valid_phase(self) -> bool:
        """Check if the data has a valid phase (between -90 and 90 degrees)."""
        for name in self.base_forms:
            data = getattr(self, name)
            if np.any((abs(data.phase) > 91)):
                return False
        return True

    def clone(self, data: ArrayLike | object | Self, **kwargs) -> Self:
        """
        Create a new ImpedanceHelper instance with new data, while preserving
        this instance's thickness and area. Kwargs override current values during construction.
        """
        # Step 1: create a fresh instance with the same class
        kwargs.setdefault("frequency", self.frequency)
        kwargs.setdefault("thickness", self.thickness)
        kwargs.setdefault("area", self.area)
        new = self.__class__(data, **kwargs)

        # Step 2: copy over thickness/area from *this* instance
        new.thickness = self.thickness
        new.area = self.area

        return new


def _convert_once(
    system: ImpedanceHelper,
    form: str,
    phys_const: float | int = 0.0,
) -> Complexer:
    """
    Perform a single conversion from `form` to impedance,
    returning a Complexer result (does not mutate system).
    """
    if form == "modulus":
        return Complexer(system.array / system._mu)
    elif form == "resistivity":
        return Complexer(system.array / system.a_d)
    elif form == "susceptibility":
        return Complexer(1 / (system._mu * (system.array + phys_const)), sign=-1)
    elif form == "relative_permittivity_corrected":
        return Complexer(1 / (system._mu * system.array + phys_const * system.a_d), sign=-1)
    else:
        return system[form]


def convert(
    data: ArrayLike | Complexer | ImpedanceHelper,
    from_form: str = "impedance",
    to_form: str = "impedance",
    system: ImpedanceHelper | object | None = None,
    phys_const: int | float = 0.0,
    strict: bool = True,
    **kwargs,
) -> Complexer:
    """
    Convert a stored property to impedance (Z) as a Complexer.

    This method applies the inverse of the forward definitions (see the table below)
    so that Z is recovered from the given form. For most properties, where the
    relationship can be expressed as ƒ(ω) = A·Z(ω)⁻¹ with A representing a multiplier
    (e.g. 1 or μ (see Notes)), the reversion follows directly. Certain forms
    (M, ρ, ε_r(corr), χ) require explicit handling; ε_r(corr) and χ additionally
    depend on a physical constant. If that constant is omitted, the result will
    only be partially correct.

    Parameters
    ----------
    form : str, default="impedance"
        Property name or alias to convert from (e.g. Z, M, ρ, Y, C, σ, ε, χ).
    data : array-like, Complexer, ComplexSystem, or None
        Optional replacement data; if given, a copy of self is updated.
    phys_const : float or int, default=0.0
        Physical constant(s) required for special cases:
        - χ (susceptibility): phys_const = ε∞
        - ε_r(corr) (dc-corrected permittivity): phys_const = σ_dc

    Returns
    -------
    Complexer
        Impedance representation of the property.

    Notes
    -----
    - All valid form names and aliases are defined in ``COMP_ALIASES``.
    - Quick reference of forward (from Z) and reverse (to Z) mappings:
        - μ is defined as μ = j·ω·C₀, with C₀ = ε₀·A/d.

    Z (impedance): root form
    Y (admittance): Y = 1/Z → Z = 1/Y
    M (modulus): M = μ·Z → Z = M/μ
    C (capacitance): C = 1/(j·ω·Z) → Z = 1/(j·ω·C)
    ρ (resistivity): ρ = Z·(A/d) → Z = ρ/(A/d)
    σ (conductivity): σ = 1/(Z·(A/d)) → Z = 1/(σ·(A/d))
    ε (permittivity): ε = ε₀/(μ·Z) → Z = ε₀/(μ·ε_r)
    ε_r (relative_permittivity): ε_r = 1/(μ·Z) → Z = 1/(μ·ε_r)
    ε_r(corr) (relative_permittivity_corrected): ε_r(corr) = ε_r + j·σ_dc/(ε₀·ω) → Z = 1/(μ·ε_r(corr) + σ_dc·A/d)
    χ (susceptibility): χ = ε_r − ε∞ → Z = 1/(μ·(χ + ε∞))
    """

    # Create a copy and update with new data if provided
    kwargs["form"] = "impedance"
    if isinstance(system, ImpedanceHelper):
        system = ImpedanceHelper(
            data, kwargs.get("frequency", system.frequency), system.thickness, system.area
        )
    else:
        system = ImpedanceHelper(data, **kwargs)

    if strict:
        form = COMP_ALIASES[str(from_form).lower()]
        target = COMP_ALIASES[str(to_form).lower()]
    else:
        form = COMP_ALIASES.get(str(from_form).lower(), "impedance")
        target = COMP_ALIASES.get(str(to_form).lower(), "impedance")

    if form == target:
        return system.complexer_obj

    # --- Primary conversion (local variable, not committed yet) ---
    initial_ = system.array
    system.complexer_obj = _convert_once(system, form, phys_const=phys_const)

    # --- Verification step ---
    if not system.valid_phase():
        # Retry once with inverted imaginary part
        system = system.clone(initial_.real - 1j * initial_.imag)
        system.complexer_obj = _convert_once(system, form, phys_const=phys_const)
        if not system.valid_phase():
            message = f"Conversion check failed when converting from '{form}'. "
            if strict:
                raise ValueError(message)
            else:
                warnings.warn(message, RuntimeWarning, stacklevel=2)

    if target == "impedance":
        return system.impedance
    return system[target]
