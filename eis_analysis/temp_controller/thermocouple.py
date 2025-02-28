#!/usr/bin/env python
#
# Python script to convert thermocouple voltages to temperatures and vice versa
# Supports all major thermocouple types. All temperatures are in units of degrees Celcius
# "tc.py -h" for help
#

import math


# import argparse
# import sys
# import re
# import traceback
def f_to_c(f):
    """Convert Fahrenheit to Celsius."""
    return (f - 32.0) / 1.8


def c_to_f(c):
    """Convert Celsius to Fahrenheit."""
    return c * 1.8 + 32.0


class Thermocouple:
    """
    Thermocouple class to handle temperature conversions and measurements.
    """

    def __init__(self, thermoType="K", unit="C", source=None, source_unit="mV"):
        self._thermoType = thermoType
        self.unit = unit
        self.source = source
        self.source_unit = source_unit

    @property
    def thermoType(self):
        return self._thermoType

    @thermoType.setter
    def thermoType(self, value):
        if len(value) > 1:
            value = value[-1]
        if value.lower() not in ["b", "e", "j", "k", "n", "r", "s", "t"]:
            raise ValueError("Invalid thermocouple type")
        self._thermoType = value
        self._temp_to_mv = getattr(self, f"type{value.lower()}_to_mv")
        self._mv_to_temp = getattr(self, f"mv_to_type{value.lower()}")

    def temp_to_mv(self, temp):
        if self.unit.lower() == "f":
            temp = (temp - 32) * 5.0 / 9.0
        return self._temp_to_mv(temp)

    def mv_to_temp(self, mv):
        temp = self._mv_to_temp(mv)
        if self.unit.lower() == "f":
            return temp * 9.0 / 5.0 + 32
        return temp

    @property
    def source(self):
        def return_zero():
            return 0

        return self._source or return_zero

    @source.setter
    def source(self, func):
        if isinstance(func, (tuple, list)):
            if isinstance(func[1], str):
                self.source_unit = func[1]
                func = func[0]
            else:
                raise ValueError("Invalid source and/or source unit")
        if func is None:
            self._source = None
        elif callable(func):
            func_res = func()
            if not isinstance(func_res, (int, float)):
                raise ValueError("Source must return an int or float.")
            self._source = func
        else:
            raise ValueError("Source must be a callable or None.")

    def temp(self, temp=None):
        if self._source is None and temp is None:
            return 0
        elif self._source is not None:
            # temp is None,
            temp = self.source()
        # else temp is not none (and source is) and/or neither are none (but temp has still been set to self.source)
        if self.source_unit.lower() == "mv":
            return self.mv_to_temp(temp)
        elif self.source_unit.lower() == "v":
            return self.mv_to_temp(temp * 1000)
        elif self.source_unit.lower() == "uv":
            return self.mv_to_temp(temp / 1000)
        elif self.source_unit.lower() == "c" and self.unit.lower() == "f":
            return c_to_f(temp)
        elif self.source_unit.lower() == "f" and self.unit.lower() == "c":
            return f_to_c(temp)
        return temp
        # if "v" in self.source_unit.lower():
        #     return self.mv_to_temp(temp)
        # if self.source_unit == "F":
        #     temp = (temp - 32) * 5.0 / 9.0
        # if self.unit == "F":
        #     return temp * 9.0 / 5.0 + 32
        # return temp

    @staticmethod
    def typeb_to_mv(degc):
        """ """
        tab1 = [
            0.000000000000e00,
            -0.246508183460e-03,
            0.590404211710e-05,
            -0.132579316360e-08,
            0.156682919010e-11,
            -0.169445292400e-14,
            0.629903470940e-18,
        ]
        tab2 = [
            -0.389381686210e01,
            0.285717474700e-01,
            -0.848851047850e-04,
            0.157852801640e-06,
            -0.168353448640e-09,
            0.111097940130e-12,
            -0.445154310330e-16,
            0.989756408210e-20,
            -0.937913302890e-24,
        ]
        if 0 <= degc <= 630.615:
            val = tab1
        elif 630.615 < degc <= 1820:
            val = tab2
        else:
            raise ValueError(
                "Temperature specified is out of range for Type B thermocouple"
            )

        e = 0
        for p, c in enumerate(val):
            e += c * degc**p

        return e

    @staticmethod
    def mv_to_typeb(mv):
        """ """
        tab1 = [
            9.8423321e01,
            6.9971500e02,
            -8.4765304e02,
            1.0052644e03,
            -8.3345952e02,
            4.5508542e02,
            -1.5523037e02,
            2.9886750e01,
            -2.4742860e00,
        ]
        tab2 = [
            2.1315071e02,
            2.8510504e02,
            -5.2742887e01,
            9.9160804e00,
            -1.2965303e00,
            1.1195870e-01,
            -6.0625199e-03,
            1.8661696e-04,
            -2.4878585e-06,
        ]
        if 0.291 <= mv <= 2.431:
            val = tab1
        elif 2.431 < mv <= 13.820:
            val = tab2
        else:
            raise ValueError(
                "Voltage specified is out of range for Type B thermocouple"
            )

        t = 0.0
        for p, c in enumerate(val):
            t += c * mv**p

        return t

    @staticmethod
    def typee_to_mv(degc):
        """ """
        tab1 = [
            0.000000000000e00,
            0.586655087080e-01,
            0.454109771240e-04,
            -0.779980486860e-06,
            -0.258001608430e-07,
            -0.594525830570e-09,
            -0.932140586670e-11,
            -0.102876055340e-12,
            -0.803701236210e-15,
            -0.439794973910e-17,
            -0.164147763550e-19,
            -0.396736195160e-22,
            -0.558273287210e-25,
            -0.346578420130e-28,
        ]
        tab2 = [
            0.000000000000e00,
            0.586655087100e-01,
            0.450322755820e-04,
            0.289084072120e-07,
            -0.330568966520e-09,
            0.650244032700e-12,
            -0.191974955040e-15,
            -0.125366004970e-17,
            0.214892175690e-20,
            -0.143880417820e-23,
            0.359608994810e-27,
        ]
        if -270 <= degc <= 0:
            val = tab1
        elif 0 < degc <= 1000:
            val = tab2
        else:
            raise ValueError(
                "Temperature specified is out of range for Type E thermocouple"
            )

        e = 0
        for p, c in enumerate(val):
            e += c * degc**p

        return e

    @staticmethod
    def mv_to_typee(mv):
        """ """
        tab1 = [
            0.0000000e00,
            1.6977288e01,
            -4.3514970e-01,
            -1.5859697e-01,
            -9.2502871e-02,
            -2.6084314e-02,
            -4.1360199e-03,
            -3.4034030e-04,
            -1.1564890e-05,
            0.0000000e00,
        ]
        tab2 = [
            0.0000000e00,
            1.7057035e01,
            -2.3301759e-01,
            6.5435585e-03,
            -7.3562749e-05,
            -1.7896001e-06,
            8.4036165e-08,
            -1.3735879e-09,
            1.0629823e-11,
            -3.2447087e-14,
        ]
        if -8.825 <= mv <= 0.0:
            val = tab1
        elif 0.0 < mv <= 76.373:
            val = tab2
        else:
            raise ValueError(
                "Voltage specified is out of range for Type E thermocouple"
            )

        t = 0.0
        for p, c in enumerate(val):
            t += c * mv**p

        return t

    @staticmethod
    def typej_to_mv(degc):
        """ """
        tab1 = [
            0.000000000000e00,
            0.503811878150e-01,
            0.304758369300e-04,
            -0.856810657200e-07,
            0.132281952950e-09,
            -0.170529583370e-12,
            0.209480906970e-15,
            -0.125383953360e-18,
            0.156317256970e-22,
        ]
        tab2 = [
            0.296456256810e03,
            -0.149761277860e01,
            0.317871039240e-02,
            -0.318476867010e-05,
            0.157208190040e-08,
            -0.306913690560e-12,
        ]

        if -210 <= degc <= 760:
            val = tab1
        elif 760 < degc <= 1200:
            val = tab2
        else:
            raise ValueError(
                "Temperature specified is out of range for Type J thermocouple"
            )

        e = 0
        for p, c in enumerate(val):
            e += c * degc**p

        return e

    @staticmethod
    def mv_to_typej(mv):
        """ """
        tab1 = [
            0.0000000e00,
            1.9528268e01,
            -1.2286185e00,
            -1.0752178e00,
            -5.9086933e-01,
            -1.7256713e-01,
            -2.8131513e-02,
            -2.3963370e-03,
            -8.3823321e-05,
        ]
        tab2 = [
            0.000000e00,
            1.978425e01,
            -2.001204e-01,
            1.036969e-02,
            -2.549687e-04,
            3.585153e-06,
            -5.344285e-08,
            5.099890e-10,
            0.000000e00,
        ]
        tab3 = [
            -3.11358187e03,
            3.00543684e02,
            -9.94773230e00,
            1.70276630e-01,
            -1.43033468e-03,
            4.73886084e-06,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
        ]

        if -8.095 <= mv <= 0.0:
            val = tab1
        elif 0.0 < mv <= 42.914:
            val = tab2
        elif 42.914 < mv <= 69.553:
            val = tab3
        else:
            raise ValueError(
                "Voltage specified is out of range for Type J thermocouple"
            )

        t = 0.0
        for p, c in enumerate(val):
            t += c * mv**p

        return t

    @staticmethod
    def typek_to_mv(degc):
        """ """
        tab1 = [
            0.000000000000e00,
            0.394501280250e-01,
            0.236223735980e-04,
            -0.328589067840e-06,
            -0.499048287770e-08,
            -0.675090591730e-10,
            -0.574103274280e-12,
            -0.310888728940e-14,
            -0.104516093650e-16,
            -0.198892668780e-19,
            -0.163226974860e-22,
        ]

        tab2 = [
            -0.176004136860e-01,
            0.389212049750e-01,
            0.185587700320e-04,
            -0.994575928740e-07,
            0.318409457190e-09,
            -0.560728448890e-12,
            0.560750590590e-15,
            -0.320207200030e-18,
            0.971511471520e-22,
            -0.121047212750e-25,
        ]

        a0 = 0.118597600000e00
        a1 = -0.118343200000e-03
        a2 = 0.126968600000e03

        if -270 <= degc <= 0:
            val = tab1
        elif 0 < degc <= 1372:
            val = tab2
        else:
            raise ValueError(
                "Temperature specified is out of range for Type K thermocouple"
            )

        e = 0
        for p, c in enumerate(val):
            e += c * degc**p

        if degc > 0:
            e += a0 * math.exp(a1 * math.pow(degc - a2, 2))

        return e

    @staticmethod
    def mv_to_typek(mv):
        """ """
        tab1 = [
            0.0000000e00,
            2.5173462e01,
            -1.1662878e00,
            -1.0833638e00,
            -8.9773540e-01,
            -3.7342377e-01,
            -8.6632643e-02,
            -1.0450598e-02,
            -5.1920577e-04,
            0.0000000e00,
        ]

        tab2 = [
            0.000000e00,
            2.508355e01,
            7.860106e-02,
            -2.503131e-01,
            8.315270e-02,
            -1.228034e-02,
            9.804036e-04,
            -4.413030e-05,
            1.057734e-06,
            -1.052755e-08,
        ]

        tab3 = [
            -1.318058e02,
            4.830222e01,
            -1.646031e00,
            5.464731e-02,
            -9.650715e-04,
            8.802193e-06,
            -3.110810e-08,
            0.000000e00,
            0.000000e00,
            0.000000e00,
        ]

        if -5.891 <= mv <= 0.0:
            val = tab1
        elif 0.0 < mv <= 20.644:
            val = tab2
        elif 20.644 < mv <= 54.886:
            val = tab3
        else:
            raise ValueError(
                "Voltage specified is out of range for Type K thermocouple"
            )

        t = 0.0
        for p, c in enumerate(val):
            t += c * mv**p

        return t

    @staticmethod
    def typen_to_mv(degc):
        """ """
        tab1 = [
            0.000000000000e00,
            0.261591059620e-01,
            0.109574842280e-04,
            -0.938411115540e-07,
            -0.464120397590e-10,
            -0.263033577160e-11,
            -0.226534380030e-13,
            -0.760893007910e-16,
            -0.934196678350e-19,
        ]
        tab2 = [
            0.000000000000e00,
            0.259293946010e-01,
            0.157101418800e-04,
            0.438256272370e-07,
            -0.252611697940e-09,
            0.643118193390e-12,
            -0.100634715190e-14,
            0.997453389920e-18,
            -0.608632456070e-21,
            0.208492293390e-24,
            -0.306821961510e-28,
        ]

        if -270 <= degc <= 0:
            val = tab1
        elif 0 < degc <= 1300:
            val = tab2
        else:
            raise ValueError(
                "Temperature specified is out of range for Type N thermocouple"
            )

        e = 0
        for p, c in enumerate(val):
            e += c * degc**p

        return e

    @staticmethod
    def mv_to_typen(mv):
        """ """
        tab1 = [
            0.0000000e00,
            3.8436847e01,
            1.1010485e00,
            5.2229312e00,
            7.2060525e00,
            5.8488586e00,
            2.7754916e00,
            7.7075166e-01,
            1.1582665e-01,
            7.3138868e-03,
        ]
        tab2 = [
            0.00000e00,
            3.86896e01,
            -1.08267e00,
            4.70205e-02,
            -2.12169e-06,
            -1.17272e-04,
            5.39280e-06,
            -7.98156e-08,
            0.00000e00,
            0.00000e00,
        ]
        tab3 = [
            1.972485e01,
            3.300943e01,
            -3.915159e-01,
            9.855391e-03,
            -1.274371e-04,
            7.767022e-07,
            0.000000e00,
            0.000000e00,
            0.000000e00,
            0.000000e00,
        ]

        if -3.99 <= mv <= 0.0:
            val = tab1
        elif 0.0 < mv <= 20.613:
            val = tab2
        elif 20.613 < mv <= 47.513:
            val = tab3
        else:
            raise ValueError(
                "Voltage specified is out of range for Type N thermocouple"
            )

        t = 0.0
        for p, c in enumerate(val):
            t += c * mv**p

        return t

    @staticmethod
    def typer_to_mv(degc):
        """ """
        tab1 = [
            0.000000000000e00,
            0.528961729765e-02,
            0.139166589782e-04,
            -0.238855693017e-07,
            0.356916001063e-10,
            -0.462347666298e-13,
            0.500777441034e-16,
            -0.373105886191e-19,
            0.157716482367e-22,
            -0.281038625251e-26,
        ]
        tab2 = [
            0.295157925316e01,
            -0.252061251332e-02,
            0.159564501865e-04,
            -0.764085947576e-08,
            0.205305291024e-11,
            -0.293359668173e-15,
        ]
        tab3 = [
            0.152232118209e03,
            -0.268819888545e00,
            0.171280280471e-03,
            -0.345895706453e-07,
            -0.934633971046e-14,
        ]
        if -50 <= degc <= 1064.18:
            val = tab1
        elif 1064.18 < degc <= 1664.5:
            val = tab2
        elif 1664.5 < degc <= 1768.1:
            val = tab3
        else:
            raise ValueError(
                "Temperature specified is out of range for Type R thermocouple"
            )

        e = 0
        for p, c in enumerate(val):
            e += c * degc**p

        return e

    @staticmethod
    def mv_to_typer(mv):
        """ """
        tab1 = [
            0.0000000e00,
            1.8891380e02,
            -9.3835290e01,
            1.3068619e02,
            -2.2703580e02,
            3.5145659e02,
            -3.8953900e02,
            2.8239471e02,
            -1.2607281e02,
            3.1353611e01,
            -3.3187769e00,
        ]
        tab2 = [
            1.334584505e01,
            1.472644573e02,
            -1.844024844e01,
            4.031129726e00,
            -6.249428360e-01,
            6.468412046e-02,
            -4.458750426e-03,
            1.994710149e-04,
            -5.313401790e-06,
            6.481976217e-08,
            0.000000000e00,
        ]
        tab3 = [
            -8.199599416e01,
            1.553962042e02,
            -8.342197663e00,
            4.279433549e-01,
            -1.191577910e-02,
            1.492290091e-04,
            0.000000000e00,
            0.000000000e00,
            0.000000000e00,
            0.000000000e00,
            0.000000000e00,
        ]
        tab4 = [
            3.406177836e04,
            -7.023729171e03,
            5.582903813e02,
            -1.952394635e01,
            2.560740231e-01,
            0.000000000e00,
            0.000000000e00,
            0.000000000e00,
            0.000000000e00,
            0.000000000e00,
            0.000000000e00,
        ]

        if -0.226 <= mv <= 1.923:
            val = tab1
        elif 1.923 < mv <= 11.361:
            val = tab2
        elif 11.361 < mv <= 19.739:
            val = tab3
        elif 19.739 < mv <= 21.103:
            val = tab4
        else:
            raise ValueError(
                "Voltage specified is out of range for Type R thermocouple"
            )

        t = 0.0
        for p, c in enumerate(val):
            t += c * mv**p

        return t

    @staticmethod
    def types_to_mv(degc):
        """ """
        tab1 = [
            0.000000000000e00,
            0.540313308631e-02,
            0.125934289740e-04,
            -0.232477968689e-07,
            0.322028823036e-10,
            -0.331465196389e-13,
            0.255744251786e-16,
            -0.125068871393e-19,
            0.271443176145e-23,
        ]
        tab2 = [
            0.132900444085e01,
            0.334509311344e-02,
            0.654805192818e-05,
            -0.164856259209e-08,
            0.129989605174e-13,
        ]
        tab3 = [
            0.146628232636e03,
            -0.258430516752e00,
            0.163693574641e-03,
            -0.330439046987e-07,
            -0.943223690612e-14,
        ]

        if -50 <= degc <= 1064.18:
            val = tab1
        elif 1064.18 < degc <= 1664.5:
            val = tab2
        elif 1664.5 < degc <= 1768.1:
            val = tab3
        else:
            raise ValueError(
                "Temperature specified is out of range for Type S thermocouple"
            )

        e = 0
        for p, c in enumerate(val):
            e += c * degc**p

        return e

    @staticmethod
    def mv_to_types(mv):
        """ """
        tab1 = [
            0.00000000e00,
            1.84949460e02,
            -8.00504062e01,
            1.02237430e02,
            -1.52248592e02,
            1.88821343e02,
            -1.59085941e02,
            8.23027880e01,
            -2.34181944e01,
            2.79786260e00,
        ]
        tab2 = [
            1.291507177e01,
            1.466298863e02,
            -1.534713402e01,
            3.145945973e00,
            -4.163257839e-01,
            3.187963771e-02,
            -1.291637500e-03,
            2.183475087e-05,
            -1.447379511e-07,
            8.211272125e-09,
        ]
        tab3 = [
            -8.087801117e01,
            1.621573104e02,
            -8.536869453e00,
            4.719686976e-01,
            -1.441693666e-02,
            2.081618890e-04,
            0.000000000e00,
            0.000000000e00,
            0.000000000e00,
            0.000000000e00,
        ]
        tab4 = [
            5.333875126e04,
            -1.235892298e04,
            1.092657613e03,
            -4.265693686e01,
            6.247205420e-01,
            0.000000000e00,
            0.000000000e00,
            0.000000000e00,
            0.000000000e00,
            0.000000000e00,
        ]

        if -0.235 <= mv <= 1.874:
            val = tab1
        elif 1.874 < mv <= 10.332:
            val = tab2
        elif 10.332 < mv <= 17.536:
            val = tab3
        elif 17.536 < mv <= 18.693:
            val = tab4
        else:
            raise ValueError(
                "Voltage specified is out of range for Type S thermocouple"
            )

        t = 0.0
        for p, c in enumerate(val):
            t += c * mv**p

        return t

    @staticmethod
    def typet_to_mv(degc):
        """ """
        tab1 = [
            0.000000000000e00,
            0.387481063640e-01,
            0.441944343470e-04,
            0.118443231050e-06,
            0.200329735540e-07,
            0.901380195590e-09,
            0.226511565930e-10,
            0.360711542050e-12,
            0.384939398830e-14,
            0.282135219250e-16,
            0.142515947790e-18,
            0.487686622860e-21,
            0.107955392700e-23,
            0.139450270620e-26,
            0.797951539270e-30,
        ]
        tab2 = [
            0.000000000000e00,
            0.387481063640e-01,
            0.332922278800e-04,
            0.206182434040e-06,
            -0.218822568460e-08,
            0.109968809280e-10,
            -0.308157587720e-13,
            0.454791352900e-16,
            -0.275129016730e-19,
        ]

        if -270 <= degc <= 0:
            val = tab1
        elif 0 < degc <= 400:
            val = tab2
        else:
            raise ValueError(
                "Temperature specified is out of range for Type T thermocouple"
            )

        e = 0
        for p, c in enumerate(val):
            e += c * degc**p

        return e

    @staticmethod
    def mv_to_typet(mv):
        """ """
        tab1 = [
            0.0000000e00,
            2.5949192e01,
            -2.1316967e-01,
            7.9018692e-01,
            4.2527777e-01,
            1.3304473e-01,
            2.0241446e-02,
            1.2668171e-03,
        ]
        tab2 = [
            0.000000e00,
            2.592800e01,
            -7.602961e-01,
            4.637791e-02,
            -2.165394e-03,
            6.048144e-05,
            -7.293422e-07,
            0.000000e00,
        ]

        if -5.603 <= mv <= 0:
            val = tab1
        elif 0 < mv <= 20.872:
            val = tab2
        else:
            raise ValueError(
                "Voltage specified is out of range for Type T thermocouple"
            )

        t = 0.0
        for p, c in enumerate(val):
            t += c * mv**p

        return t


# def meter(meter, mv):

#     if meter == 'u1272a' or meter == 'u1271a':
#         # Accordance with Agilent U1271A, U1272A 30,000 count DMM, DC specifications
#         if mv < 30:
#             # 30mV range, 0.001 mV resolution
#             upper = mv * 1.0005 + 0.020
#             lower = mv * 0.9995 - 0.020
#         elif 30 <= mv < 300:
#             # 300mV range, 0.01 mV resolution
#             upper = mv * 1.0005 + 0.05
#             lower = mv * 0.9995 - 0.05
#         elif 300 <= mv < 3000:
#             # 3V range, 0.1 mV resolution
#             upper = mv * 1.0005 + 0.5
#             lower = mv * 0.9995 - 0.5
#         else:
#             raise ValueError("Voltage out of range for selected meter.")

#     elif meter == '187' or meter == '189':
#         # Fluke 187 or 189, 50,000 count DMM, DC specifications
#         if mv < 50:
#             # 50mV range, 0.001 mV resolution
#             upper = mv * 1.001 + 0.020
#             lower = mv * 0.999 - 0.020
#         elif 50 <= mv < 500:
#             # 500mV range, 0.01 mV resolution
#             upper = mv * 1.0003 + 0.02
#             lower = mv * 0.9997 - 0.02
#         elif 500 <= mv < 3000:
#             # 3V range, 0.1 mV resolution
#             upper = mv * 1.00025 + 0.5
#             lower = mv * 0.99975 - 0.5
#         else:
#             raise ValueError("Voltage out of range for selected meter.")

#     elif meter == '83v':
#         # Fluke 83 Series V
#         if mv < 600:
#             # 600 mV range, 0.1 mV resolution
#             upper = mv * 1.003 + 0.1
#             lower = mv * 0.997 - 0.1
#         else:
#             raise ValueError("Voltage out of range for selected meter.")

#     elif meter == '87v':
#         # Fluke 87 Series V
#         if mv < 600:
#             # 600 mV range, 0.1 mV resolution
#             upper = mv * 1.001 + 0.1
#             lower = mv * 0.991 - 0.1
#         else:
#             raise ValueError("Voltage out of range for selected meter.")

#     else:
#         # Shouldn't reach here anyway! ArgumentParser.add_argument() takes care of this.
#         raise ValueError("Invalid meter type for the --meter option. Use the --help option for more information.")

#     return [lower, mv, upper]


# if __name__ == "__main__":

#     tctypes = ['b', 'e', 'j', 'k', 'n', 'r', 's', 't']
#     choices = []

#     for tc in tctypes:
#         choices.append('mv2' + tc)
#         choices.append(tc + '2mv')

#     parser = argparse.ArgumentParser()
#     parser.add_argument('values', nargs='+', type=float)
#     parser.add_argument('--offset-input', nargs=1, default=[0.0], type=float)
#     parser.add_argument('--offset-output', nargs=1, default=[0.0], type=float)
#     parser.add_argument('--mode', nargs=1, default=['mv2k'], choices=choices)
#     parser.add_argument('--meter', nargs=1, type=str, choices=['u1271a', 'u1272a', '187', '189', '83v', '87v'])
#     args = parser.parse_args()
#     # print(args); sys.exit(1)

#     offset_input = args.offset_input[0]
#     offset_output = args.offset_output[0]

#     convert_function = None

#     for tc in tctypes:
#         if args.mode:
#             if args.mode[0] == 'mv2' + tc:
#                 convert_function = 'mv_to_type' + tc
#             elif args.mode[0] == tc + '2mv':
#                 convert_function = 'type' + tc + '_to_mv'
#         else:
#             # should never reach here, as we set ['v2k'] as default for args.mode in parser
#             pass
#     convert_function = getattr(Thermocouple, convert_function)

#     if re.match(r"^mv2", args.mode[0]):
#         # voltage to temperature mode
#         output_decimal_places = 1
#     else:
#         # temperature to voltage mode
#         output_decimal_places = 3

#     error_count = 0
#     for v in args.values:

#         try:
#             if args.meter and args.meter[0] and re.match(r"^mv2", args.mode[0]):
#                 # voltage to temperature in meter mode
#                 v_range = meter(args.meter[0], v + offset_input)
#                 lower = convert_function(v_range[0] + offset_input) + offset_output
#                 mid = convert_function(v_range[1]  + offset_input) + offset_output
#                 upper = convert_function(v_range[2] + offset_input) + offset_output
#                 print("{0:.1f} (uncertainty: {1:.1f} to {2:.1f})".format(mid, lower, upper))

#             else:

#                 if output_decimal_places == 1:
#                     # voltage to temperature mode
#                     output = convert_function(v + offset_input) + offset_output
#                     print("{0:.1f}".format(output))

#                 elif output_decimal_places == 3:
#                     # temperature to voltage mode
#                     output = convert_function(v + offset_input) + offset_output
#                     print("{0:.3f}".format(output))

#         except Exception as ex:
#             print("Error with {}: {}".format(v + offset_input, ex.message))
#             # print(traceback.format_exc()) # debugging mode only
#             # print ex.message
#             error_count += 1

#     if error_count > 0:
#         sys.exit(1)
#     else:
#         sys.exit(0)
