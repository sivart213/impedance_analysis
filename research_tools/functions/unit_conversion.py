# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""


from dataclasses import astuple, dataclass
import numpy as np
import pandas as pd
import re

units_dict = {}
units_dict["length"] = {
    "m": 1,
    "Ang": 1e-10,
    "inch": 0.0254,
    "ft": 0.3048,
    "yd": 0.9144,
    "mi": 1609.34,
    "NM": 1852,
    "ftm": 1.8288,
}
units_dict["mass"] = {
    "g": 1,
    "lb": 453.59,
    "oz": 48.35,
    "ton": 0.907e6,
    "ton_uk": 1.016e6,
    "t": 1e6,
    "Mt": 1e12,
    "Gt": 1e15,
}
units_dict["time"] = {
    "s": 1,
    "sec": 1,
    "second": 1,
    "m": 60,
    "mn": 60,
    "min": 60,
    "minute": 60,
    "h": 3600,
    "hr": 3600,
    "hour": 3600,
    "d": 86400,
    "day": 86400,
    "wk": 604800,
    "week": 604800,
    "mo": 2.628e6,
    "month": 2.628e6,
    "yr": 3.154e7,
    "year": 3.154e7,
    "dec": 3.154e8,
    "decade": 3.154e8,
    "c": 3.154e9,
    "century": 3.154e9,
}
units_dict["temp"] = {"K": 1}
units_dict["curr"] = {"A": 1}
units_dict["mole"] = {"mol": 1}
units_dict["energy"] = {"J": 1, "eV": 1.60217662e-19}
units_dict["volt"] = {"V": 1}
units_dict["res"] = {"Ohm": 1}


units_df = pd.DataFrame(
    [(m, n, o) for m in list(units_dict.keys()) for n, o in units_dict[m].items()],
    columns=["Type", "unit", "value"],
)


def ConvComplex(obj, target):
    """Return sum of squared errors (pred vs actual)."""
    base = list(obj.parcer(obj.unit).items())
    goal = list(obj.parcer(target).items())
    val = obj.value
    for n in range(len(base)):
        val = BaseUnits(val, base[n][0], base[n][1])[goal[n][0]]
    return BaseUnits(val, target)


class Search:
    """Return sum of squared errors (pred vs actual)."""

    def __init__(self, unit="", base_unit="", user_list=None, user_dict={}):
        if user_list is None:
            user_list = ["T", "G", "M", "k", "d", "c", "m", "u", "n", "p", "f"]

        _pre_base = {
            "T": 1e12,
            "G": 1e9,
            "M": 1e6,
            "k": 1e3,
            "d": 1e-1,
            "c": 1e-2,
            "m": 1e-3,
            "u": 1e-6,
            "n": 1e-9,
            "p": 1e-12,
            "f": 1e-15,
        }
        try:
            _pre_base = {x: _pre_base[x] for x in user_list}
        except KeyError:
            _pre_base = {
                x: _pre_base[x]
                for x in ["T", "G", "M", "k", "d", "c", "m", "u", "n", "p", "f"]
            }

        if list(units_df["Type"][units_df["unit"] == base_unit]) != []:
            value = list(units_df["Type"][units_df["unit"] == base_unit])[0]
            self.si_dict = {k + base_unit: v for k, v in _pre_base.items()}
            self.add_dict = units_dict[value]
        elif list(units_df["Type"][units_df["unit"] == unit]) != []:
            value = list(units_df["Type"][units_df["unit"] == unit])[0]
            self.si_dict = {k + base_unit: v for k, v in _pre_base.items()}
            self.add_dict = units_dict[value]
        elif list(units_df["Type"][units_df["unit"] == unit[1:]]) != []:
            value = list(units_df["Type"][units_df["unit"] == unit[1:]])[0]
            self.si_dict = {k + base_unit: v for k, v in _pre_base.items()}
            self.add_dict = units_dict[value]
        else:
            self.si_dict = {k + base_unit: v for k, v in _pre_base.items()}
            self.add_dict = {unit: 1}

        if "." in base_unit:
            right = base_unit[base_unit.find(".") :]
            self.add_dict = {m + right: n for m, n in self.add_dict.items()}
        elif "/" in base_unit:
            right = base_unit[base_unit.find("/") :]
            self.add_dict = {m + right: n for m, n in self.add_dict.items()}

        self.user_dict = user_dict

    @property
    def class_units(self):
        """Return sum of squared errors (pred vs actual)."""
        return {**self.si_dict, **self.add_dict, **self.user_dict}

    @class_units.setter
    def class_units(self, _):
        pass

    """Return sum of squared errors (pred vs actual)."""


@dataclass(frozen=True)
class BaseUnits(object):
    """Return sum of squared errors (pred vs actual)."""

    value: float = 1.0
    unit: str = ""
    exp: int = 1
    print_unit: str = "units"
    base_unit: str = "units"
    _fig_set: int = 8

    def __post_init__(self):
        """Return sum of squared errors (pred vs actual)."""
        if self.print_unit == "units" and self.unit != "":
            object.__setattr__(self, "print_unit", self.unit)
        if self.base_unit == "units" and self.unit != "":
            object.__setattr__(self, "base_unit", self.unit)
        object.__setattr__(
            self,
            "class_units",
            Search(
                unit=list(self.parcer(self.unit).keys())[0],
                base_unit=self.base_unit,
            ).class_units,
        )
        self.set_units()

    def __repr__(self):
        """Return the desired output if called."""
        # self.set_units()
        return repr(getattr(self, self.print_unit))

    def __add__(self, other):
        """Return sum of squared errors (pred vs actual)."""
        if isinstance(other, BaseUnits):
            if other.unit == self.unit:
                other = other.value
            else:
                other = 0
        return self.__class__(
            self.value + other,
            self.unit,
            self.exp,
            self.print_unit,
            self.base_unit,
            self._fig_set,
        ).set_units(self.class_units)

    def __radd__(self, other):
        """Return sum of squared errors (pred vs actual)."""
        if isinstance(other, BaseUnits):
            if other.unit == self.unit:
                other = other.value
            else:
                other = 0
        return self.__add__(other)

    def __sub__(self, other):
        """Return sum of squared errors (pred vs actual)."""
        if isinstance(other, BaseUnits):
            if other.unit == self.unit:
                other = other.value
            else:
                other = 0
        return self.__class__(
            self.value - other,
            self.unit,
            self.exp,
            self.print_unit,
            self.base_unit,
            self._fig_set,
        ).set_units(self.class_units)

    def __rsub__(self, other):
        """Return sum of squared errors (pred vs actual)."""
        if isinstance(other, BaseUnits):
            if other.unit == self.unit:
                other = other.value
            else:
                other = 0
        return self.__class__(
            other - self.value,
            self.unit,
            self.exp,
            self.print_unit,
            self.base_unit,
            self._fig_set,
        ).set_units(self.class_units)

    def __mul__(self, other):
        """Return sum of squared errors (pred vs actual)."""
        if isinstance(other, BaseUnits):
            units1 = self.parcer(self.base_unit)
            units2 = other.parcer(other.base_unit)
            unit, exp = self.joiner(units1, units2)
            if len(units1.keys()) == 1 and units1.keys() == units2.keys():
                return self.__class__(
                    self._value * other._value,
                    self.base_unit,
                    exp,
                    self.print_unit,
                    self.base_unit,
                    self._fig_set,
                )
            else:
                return BaseUnits(self._value * other._value, unit, exp, unit, unit)
        else:
            return self.__class__(
                self.value * other,
                self.unit,
                self.exp,
                self.print_unit,
                self.base_unit,
                self._fig_set,
            )

    def __rmul__(self, other):
        """Return sum of squared errors (pred vs actual)."""
        # if isinstance(other, self.__class__):
        # other = other.value
        return self * other

    def __truediv__(self, other):
        """Return sum of squared errors (pred vs actual)."""
        if isinstance(other, BaseUnits):
            units1 = self.parcer(self.base_unit)
            units2 = other.parcer(other.base_unit)
            units2 = {m: -1 * n for m, n in units2.items()}
            unit, exp = self.joiner(units1, units2)
            if len(units1.keys()) == 1 and units1.keys() == units2.keys():
                return self.__class__(
                    self._value / other._value,
                    self.base_unit,
                    exp,
                    self.print_unit,
                    self.base_unit,
                    self._fig_set,
                )
            else:
                return BaseUnits(self._value / other._value, unit, exp, unit, unit)
        else:
            return self.__class__(
                self.value / other,
                self.unit,
                self.exp,
                self.print_unit,
                self.base_unit,
                self._fig_set,
            )

    def __rtruediv__(self, other):
        """Return sum of squared errors (pred vs actual)."""
        if isinstance(other, BaseUnits):
            units1 = self.parcer(self.base_unit)
            units1 = {m: -1 * n for m, n in units1.items()}
            units2 = other.parcer(other.base_unit)
            unit, exp = self.joiner(units1, units2)
            if len(units1.keys()) == 1 and units1.keys() == units2.keys():
                return self.__class__(
                    other._value / self._value,
                    self.base_unit,
                    exp,
                    self.print_unit,
                    self.base_unit,
                    self._fig_set,
                )
            else:
                return BaseUnits(other._value / self._value, unit, exp, unit, unit)
        else:
            return self.__class__(
                other / self.value,
                self.unit,
                self.exp,
                self.print_unit,
                self.base_unit,
                self._fig_set,
            )

    def __iter__(self):
        """Return sum of squared errors (pred vs actual)."""
        return iter(astuple(self))

    def __len__(self):
        """Return sum of squared errors (pred vs actual)."""
        try:
            return len(self.value)
        except TypeError:
            return 1

    def __getitem__(self, item):
        """Return sum of squared errors (pred vs actual)."""
        return getattr(self, item)

    def figs(self, num):
        """Return sum of squared errors (pred vs actual)."""
        if isinstance(num, np.ndarray):
            for m, n in enumerate(num):
                num[m] = round(
                    n, -(int("{:e}".format(n).split("e")[1]) - (self._fig_set - 1))
                )
        else:
            num = round(
                num, -(int("{:e}".format(num).split("e")[1]) - (self._fig_set - 1))
            )
        return num

    def set_units(self, new_dict={}):
        """Return sum of squared errors (pred vs actual)."""
        object.__setattr__(self, "class_units", {**self.class_units, **new_dict})
        for term, val in self.class_units.items():
            object.__setattr__(self, term, self.figs(self._value / val**self.exp))
        # object.__setattr__(self, self.base_unit, self.figs(self._value))
        return self

    def parcer(self, units):
        """Return list of units parced from resuls."""
        unit_list = re.split("([^a-zA-Z0-9])", units)
        if len(unit_list) == 1:
            return {unit_list[0]: self.exp}
        sign = 1
        units = []
        exp = []
        for pars in unit_list:
            if pars == "/":
                sign *= -1
                continue
            if pars == ".":
                continue
            if pars[-1].isnumeric():
                pars, pars_exp = pars[:-1], int(pars[-1])
            else:
                pars_exp = 1
            units = units + [pars]
            exp = exp + [pars_exp * sign]

        return {units[i]: exp[i] for i in range(len(units))}

    def joiner(self, units1, units2):
        """Return sum of squared errors (pred vs actual)."""
        units = {**units1, **units2}
        units = {
            m: units1[m] + units2[m] if m in units1 and m in units2 else n
            for m, n in units.items()
        }
        num = {m: n for m, n in units.items() if n > 0}
        nums = [m + str(n) if n > 1 else m for m, n in num.items()]
        den = {m: n for m, n in units.items() if n < 0}
        dens = [m + str(n) if n > 1 else m for m, n in den.items()]
        unit = "/".join([".".join(nums), ".".join(dens)])
        if unit[-1] == "/":
            unit = unit[:-1]
        if len(units) == 1:
            exp = list(units.values())[0]
        else:
            exp = 1
        return unit, exp

    @property
    def _value(self) -> float:
        if self.unit in self.class_units.keys():
            return self.value * self.class_units[self.unit] ** self.exp
        else:
            return self.value

    @_value.setter
    def _value(self, _):
        pass

    def class_values(self):
        """Return sum of squared errors (pred vs actual)."""
        dict_s = pd.Series(self.__dict__)
        return dict_s[self.class_units.keys()].astype(float)


@dataclass(frozen=True)
class Length(BaseUnits):
    """Return sum of squared errors (pred vs actual)."""

    value: float = 1.0
    unit: str = "m"
    exp: int = 1
    print_unit: str = "units"
    base_unit: str = "m"
    _fig_set: int = 8

    def __post_init__(self):
        """Return sum of squared errors (pred vs actual)."""
        if self.print_unit == "units":
            object.__setattr__(self, "print_unit", self.unit)
        object.__setattr__(
            self,
            "class_units",
            Search(
                unit=list(self.parcer(self.unit).keys())[0],
                base_unit=self.base_unit,
            ).class_units,
        )
        self.set_units()

    def __repr__(self):
        """Return the result of the assumed unit."""
        return repr(getattr(self, self.print_unit))


@dataclass(frozen=True)
class Mass(BaseUnits):
    """Return sum of squared errors (pred vs actual)."""

    value: float = 1.0
    unit: str = "g"
    exp: int = 1
    print_unit: str = "units"
    base_unit: str = "g"
    _fig_set: int = 8

    def __post_init__(self):
        """Return sum of squared errors (pred vs actual)."""
        if self.print_unit == "units":
            object.__setattr__(self, "print_unit", self.unit)
        object.__setattr__(
            self,
            "class_units",
            Search(
                unit=list(self.parcer(self.unit).keys())[0],
                base_unit=self.base_unit,
            ).class_units,
        )
        self.set_units()

    def __repr__(self):
        """Return sum of squared errors (pred vs actual)."""
        return repr(getattr(self, self.print_unit))


@dataclass(frozen=True)
class Temp(BaseUnits):
    """Return sum of squared errors (pred vs actual)."""

    value: float = 1.0
    unit: str = "K"
    exp: int = 1
    print_unit: str = "units"
    base_unit: str = "K"
    _fig_set: int = 8

    def __post_init__(self):
        """Return sum of squared errors (pred vs actual)."""
        if self.print_unit == "units":
            object.__setattr__(self, "print_unit", self.unit)
        object.__setattr__(
            self,
            "class_units",
            Search(
                unit=list(self.parcer(self.unit).keys())[0],
                base_unit=self.base_unit,
                user_list=["k", "m", "u", "n", "p"],
            ).class_units,
        )
        self.set_units()
        object.__setattr__(self, "inv_kK", self.figs(1000 / self._value))
        object.__setattr__(self, "R", self.figs(self._value * 9.0 / 5.0))
        object.__setattr__(self, "C", self.figs(self._value - 273.15))
        object.__setattr__(self, "F", self.figs(self._value * 9.0 / 5.0 - 459.67))

    def __repr__(self):
        """Return sum of squared errors (pred vs actual)."""
        return repr(getattr(self, self.print_unit))

    @property
    def _value(self) -> float:
        if self.unit.upper() == "R":
            return self.value * 5.0 / 9.0
        elif self.unit == "inv_kK":
            return 1000 / self.value
        elif self.unit.upper() == "C":
            return self.value + 273.15
        elif self.unit.upper() == "F":
            return (self.value + 459.67) / 9.0 * 5.0
        else:
            return self.value

    @_value.setter
    def _value(self, _):
        pass


@dataclass(frozen=True)
class Time(BaseUnits):
    """Return sum of squared errors (pred vs actual)."""

    value: float = 1.0
    unit: str = "s"
    exp: int = 1
    print_unit: str = "units"
    base_unit: str = "s"
    _fig_set: int = 8

    def __post_init__(self):
        """Return sum of squared errors (pred vs actual)."""
        if self.print_unit == "units":
            object.__setattr__(self, "print_unit", self.unit)
        object.__setattr__(
            self,
            "class_units",
            Search(
                unit=list(self.parcer(self.unit).keys())[0],
                base_unit=self.base_unit,
                user_list=["m", "u", "n", "p", "f"],
            ).class_units,
        )
        self.set_units()

    def __repr__(self):
        """Return sum of squared errors (pred vs actual)."""
        return repr(getattr(self, self.print_unit))


@dataclass(frozen=True)
class Mole(BaseUnits):
    """Return sum of squared errors (pred vs actual)."""

    value: float = 1.0
    unit: str = "mol"
    exp: int = 1
    print_unit: str = "units"
    base_unit: str = "mol"
    _fig_set: int = 8

    def __post_init__(self):
        """Return sum of squared errors (pred vs actual)."""
        if self.print_unit == "units":
            object.__setattr__(self, "print_unit", self.unit)
        object.__setattr__(
            self,
            "class_units",
            Search(
                unit=list(self.parcer(self.unit).keys())[0],
                base_unit=self.base_unit,
            ).class_units,
        )
        self.set_units()

    def __repr__(self):
        """Return sum of squared errors (pred vs actual)."""
        return repr(getattr(self, self.print_unit))


@dataclass(frozen=True)
class Energy(BaseUnits):
    """Return sum of squared errors (pred vs actual)."""

    value: float = 1.0
    unit: str = "J"
    exp: int = 1
    print_unit: str = "units"
    base_unit: str = "J"
    _fig_set: int = 8

    def __post_init__(self):
        """Return sum of squared errors (pred vs actual)."""
        if self.print_unit == "units":
            object.__setattr__(self, "print_unit", self.unit)
        object.__setattr__(
            self,
            "class_units",
            Search(
                unit=list(self.parcer(self.unit).keys())[0],
                base_unit=self.base_unit,
            ).class_units,
        )
        self.set_units()

    def __repr__(self):
        """Return sum of squared errors (pred vs actual)."""
        return repr(getattr(self, self.print_unit))


@dataclass(frozen=True)
class Curr(BaseUnits):
    """Return sum of squared errors (pred vs actual)."""

    value: float = 1.0
    unit: str = "A"
    exp: int = 1
    print_unit: str = "units"
    base_unit: str = "A"
    _fig_set: int = 8

    def __post_init__(self):
        """Return sum of squared errors (pred vs actual)."""
        if self.print_unit == "units":
            object.__setattr__(self, "print_unit", self.unit)
        object.__setattr__(
            self,
            "class_units",
            Search(
                unit=list(self.parcer(self.unit).keys())[0],
                base_unit=self.base_unit,
            ).class_units,
        )
        self.set_units()

    def __repr__(self):
        """Return sum of squared errors (pred vs actual)."""
        return repr(getattr(self, self.print_unit))


@dataclass(frozen=True)
class Volt(BaseUnits):
    """Return sum of squared errors (pred vs actual)."""

    value: float = 1.0
    unit: str = "V"
    exp: int = 1
    print_unit: str = "units"
    base_unit: str = "V"
    _fig_set: int = 8

    def __post_init__(self):
        """Return sum of squared errors (pred vs actual)."""
        if self.print_unit == "units":
            object.__setattr__(self, "print_unit", self.unit)
        object.__setattr__(
            self,
            "class_units",
            Search(
                unit=list(self.parcer(self.unit).keys())[0],
                base_unit=self.base_unit,
            ).class_units,
        )
        self.set_units()

    def __repr__(self):
        """Return sum of squared errors (pred vs actual)."""
        return repr(getattr(self, self.print_unit))


@dataclass(frozen=True)
class Res(BaseUnits):
    """Return sum of squared errors (pred vs actual)."""

    value: float = 1.0
    unit: str = "Ohm"
    exp: int = 1
    print_unit: str = "units"
    base_unit: str = "Ohm"
    _fig_set: int = 8

    def __post_init__(self):
        """Return sum of squared errors (pred vs actual)."""
        if self.print_unit == "units":
            object.__setattr__(self, "print_unit", self.unit)
        object.__setattr__(
            self,
            "class_units",
            Search(
                unit=list(self.parcer(self.unit).keys())[0],
                base_unit=self.base_unit,
            ).class_units,
        )
        self.set_units()

    def __repr__(self):
        """Return sum of squared errors (pred vs actual)."""
        return repr(getattr(self, self.print_unit))


# @dataclass(frozen=True)
# class CompUnits():
#     """Return sum of squared errors (pred vs actual)."""

#     value: float = 1.
#     unit: str = ''
#     exp: int = 1
#     print_unit: str = 'units'
#     base_unit: str = 'unit'
#     _fig_set: int = 8

#     def __post_init__(self):
#         """Return sum of squared errors (pred vs actual)."""
#         si_list = [Length, Mass, Time, Temp, Curr, Mole, Energy, Volt]

#         if self.print_unit == 'units':
#             object.__setattr__(self, 'print_unit', self.unit)

#         object.__setattr__(self, 'class_units', Search(unit=list(self.parcer(self.unit).keys())[0],
#                                                        base_unit=self.base_unit,
#                                                        ).class_units)
#         self.set_units()

#     def __repr__(self):
#         """Return sum of squared errors (pred vs actual)."""
#         return repr(getattr(self, self.print_unit))

#     def parcer(self, units):
#         """Return list of units parced from results"""
#         unit_list = re.split('([^a-zA-Z0-9])', units)
#         if len(unit_list) == 1:
#             return {unit_list[0]: self.exp}
#         sign = 1
#         units = []
#         exp = []
#         for pars in unit_list:
#             if pars == '/':
#                 sign *= -1
#                 continue
#             if pars == '.':
#                 continue
#             if pars[-1].isnumeric():
#                 pars, pars_exp = pars[:-1], int(pars[-1])
#             else:
#                 pars_exp = 1
#             units = units + [pars]
#             exp = exp + [pars_exp*sign]

#         return {units[i]: exp[i] for i in range(len(units))}

#     def joiner(self, units1, units2):
#         """Return sum of squared errors (pred vs actual)."""
#         units = {**units1, **units2}
#         units = {m: units1[m] + units2[m] if m in units1 and m in units2
#                  else n for m, n in units.items()}
#         num = {m: n for m, n in units.items() if n > 0}
#         nums = [m+str(n) if n > 1 else m for m, n in num.items()]
#         den = {m: n for m, n in units.items() if n < 0}
#         dens = [m+str(n) if n > 1 else m for m, n in den.items()]
#         unit = '/'.join(['.'.join(nums), '.'.join(dens)])
#         if unit[-1] == '/':
#             unit = unit[:-1]
#         if len(units) == 1:
#             exp = list(units.values())[0]
#         else:
#             exp = 1
#         return unit, exp

# base_charge = BaseUnits(1.60217662e-19, 'C')  # Element Charge: Coulombs
# boltz = BaseUnits(1.380649e-23, 'J/K')  # Boltz: J/k
# avag_num = BaseUnits(6.02214076e23, 'mol', -1)  # Avag: atoms/mol
# planks = BaseUnits(6.62607015e-34, 'J.s')  # Planks: Js
# planks_bar = planks/(2*np.pi)  # Planks Red: Js
# speed_of_light = BaseUnits(299792458, 'm/s')  # Sped of Light: m/s
# elem_mass = Mass(9.10938356e-31, 'kg')  # Mass of Elect: kg
# perm_of_vac = BaseUnits(8.8541878128e-12, 'F/m')  # Perm of Free: F/m

# b=Length(np.array([5,2]),'um')
# Na=pt.formula('Na[23]')
# Na_r=Length2(190,'pm')

# Na_v=Length(4/3*np.pi*(Na_r.cm**2),'cm')
# EVA=pt.formula('28%wt C[12]4H[1]6O[16]2 //  C[12]2H[1]4', natural_density=0.92)

# Na_conc = 1e20/pt.constants.avogadro_number*Na.mass
# Na_mass = Na_conc

# EVA_mass = (1-Na_mass/Na.density)*EVA.density
