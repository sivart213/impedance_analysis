# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
# from abc import ABC, abstractmethod
from functools import partial

import numpy as np


def randles_gen(
    freq: tuple[int | float, ...] | list[int | float] | np.ndarray = (-3, 6, 100),
    R0: int | float = 0,
    R1: int | float = 1e6,
    C1: int | float = 1e-6,
    *,
    as_stack: bool = False,
) -> np.ndarray:
    """
    Generate impedance data for a Randles circuit.  Replacement for the func returned by wrapCircuit.

    Parameters
    ----------
    freq : tuple, list, or np.ndarray, optional
        Frequency data. If tuple or list of length 3, interpreted as (start_exp, stop_exp, num_points)
        to generate logspace data from 10**start_exp to 10**stop_exp with num_points points.
        Default is (-3, 6, 100).
    R0 : int or float, optional
        Ohmic resistance value. Default is 0.
    R1 : int or float, optional
        Charge transfer resistance value. Default is 1e6.
    C1 : int or float, optional
        Double layer capacitance value. Default is 1e-6.
    as_stack : bool, optional
        If True, return real and imaginary parts stacked horizontally. Default is False.

    Returns
    -------
    np.ndarray
        Calculated impedance data as a complex numpy array or stacked real/imaginary parts.

    Raises
    ------
    ValueError
        If R1, C1, or frequency data are zero or negative
        If freq is empty.

    """
    f = np.asarray(freq)
    if isinstance(freq, (list, tuple)) and len(freq) == 3:
        f = np.logspace(freq[0], freq[1], num=int(freq[2]))
    if any(val == 0 for val in (R1, C1, f.size)) or np.any(f <= 0):
        raise ValueError("R1, C1, and frequencies must be non-zero and positive.")

    Z = R0 + R1 / (1 + 1j * 2 * np.pi * f * C1 * R1)
    if as_stack:
        return np.hstack([Z.real, Z.imag])
    return Z


class RCCircuit:
    """Class to create a test dataset for RC circuit fitting."""

    def __init__(
        self,
        freq=(-3, 6, 100),
        true_values=(101.56e3, 10.210e4, 142.453e-7),
        noise=0.01,
        guess_range=0.9,
        bounds_range=2,
    ):
        self._freq = np.ones(1)
        self._Z = np.array([])
        self._Z_noisy = np.array([])
        self._true_values = [0.0, 1.0, 0.1]
        self._initial_guess = [0.0, 1.01, 0.09]
        self._noise = np.clip(abs(noise) or 0.01, 0.0, 1.0)
        self._bnd_mult = abs(bounds_range) or 2.0

        self.freq = freq
        self.true_values = true_values
        self.initial_guess = guess_range  # Default initial guess multiplier
        self.bounds = bounds_range  # Default bounds multiplier
        self.circuit_func = partial(randles_gen, as_stack=True)

    def get_guess(self, mult: float) -> list[float]:
        """Return randomized initial guess list using numpy."""
        if not 0 <= mult <= 1:
            print(f"Warning: modifier {mult} out of range, clamping to boundary [0, 1]")
            mult = 0.0 if mult < 0 else 1.0

        r_vals = np.random.uniform(*self.bounds)
        return [(1 - mult) * v + mult * rv for v, rv in zip(self._true_values, r_vals)]

    def get_noisy_z(self, noise=0.0) -> np.ndarray:
        """Generate noisy impedance data based on current true values and frequency."""
        self._noise = abs(noise) or self._noise
        Z = self.Z
        np.random.seed(0)
        noise_real = np.random.normal(0, self._noise * abs(Z), size=Z.real.shape)
        noise_imag = np.random.normal(0, self._noise * abs(Z), size=Z.imag.shape)
        return Z + noise_real + 1j * noise_imag

    @property
    def model(self):
        return "R1-p(R2,C2)"

    @property
    def true_values(self):
        return self._true_values

    @true_values.setter
    def true_values(self, value):
        value = tuple(value)
        if len(value) == 2:
            value = (0, value[0], value[1])
        if len(value) != 3:
            raise ValueError("true_values must be of length 3")

        self._true_values = [float(v) for v in value]
        self._initial_guess = self.get_guess(self._noise)
        self._Z = randles_gen(self._freq, *self._true_values)
        self._Z_noisy = self.get_noisy_z(self._noise)

    @property
    def freq(self):
        return self._freq

    @freq.setter
    def freq(self, value):
        if isinstance(value, (list, tuple)) and len(value) == 3:
            self._freq = np.logspace(value[0], value[1], num=int(value[2]))
        else:
            self._freq = np.asarray(value)
        self._Z = randles_gen(self._freq, *self._true_values)
        self._Z_noisy = self.get_noisy_z(self._noise)
        # self.Z

    @property
    def Z(self) -> np.ndarray:
        if self._Z.size != self._freq.size:
            self._Z = randles_gen(self._freq, *self._true_values)
        return self._Z

    @property
    def Z_noisy(self) -> np.ndarray:
        if self._Z_noisy.size != self._freq.size:
            self._Z_noisy = self.get_noisy_z(self._noise)
        return self._Z_noisy

    @Z_noisy.setter
    def Z_noisy(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("Z_noisy multiplier must be an int or float")
        self._noise = value

    @property
    def bounds(self):
        lower_bounds = [v / self._bnd_mult for v in self._true_values]
        upper_bounds = [v * self._bnd_mult for v in self._true_values]
        return (lower_bounds, upper_bounds)

    @bounds.setter
    def bounds(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("Bounds multiplier must be an int or float")
        if value <= 1:
            raise ValueError("Bounds multiplier must be greater than 1")
        self._bnd_mult = float(value)

    @property
    def pair_bounds(self):
        return list(zip(*self.bounds))

    @property
    def initial_guess(self) -> list[float]:
        """Return the initial guess values."""
        return self._initial_guess

    @initial_guess.setter
    def initial_guess(self, value):
        if isinstance(value, (int, float)):
            self._initial_guess = self.get_guess(float(value))
        elif isinstance(value, (list, tuple)):
            if len(value) == 2:
                value = (0, value[0], value[1])
            if len(value) != 3:
                raise ValueError("initial_guess must be of length 3")
            self._initial_guess = [float(v) for v in value]
        else:
            raise TypeError("initial_guess must be a list, tuple, int, or float")

    @property
    def Z_hstack(self):
        if self._Z is None:
            return None
        return np.hstack([self.Z.real, self.Z.imag])

    @property
    def Z_noisy_hstack(self):
        if self._Z is None:
            return None
        return np.hstack([self.Z_noisy.real, self.Z_noisy.imag])

    @property
    def lsq_kwargs(self):
        return {
            # "absolute_sigma": False,
            # "check_finite": None,
            "method": "trf",
            "jac": "3-point",
            "x_scale": "jac",
            "ftol": 1e-14,
            "xtol": 1e-8,
            "gtol": 1e-8,
            "loss": "cauchy",
            "diff_step": None,
            "tr_solver": None,
            "tr_options": {},
            "jac_sparsity": None,
            "verbose": 1,
            "max_nfev": 1e6,
        }

    @property
    def objective(self):
        circuit_func = self.circuit_func

        def minimizer(params, f, Z_data):
            Z0 = np.array(np.hsplit(circuit_func(f, *params), 2)).T
            Z_fit = np.hstack([Z0[:, 0], Z0[:, 1]])
            if len(Z_data) == len(Z_fit) / 2:
                Z_data = np.hstack([Z_data.real, Z_data.imag])
            return Z_data - Z_fit

        return minimizer

    @property
    def objective_complex(self):
        circuit_func = self.circuit_func

        def minimizer(params, freq, Z_data):
            Z0 = np.array(np.hsplit(circuit_func(freq, *params), 2)).T
            Z_fit = Z0[:, 0] + 1j * Z0[:, 1]
            Z2 = np.array(np.hsplit(Z_data, 2)).T
            Z_noisy = Z2[:, 0] + 1j * Z2[:, 1]
            return Z_noisy - Z_fit

        return minimizer

    @property
    def objective_sq(self):
        circuit_func = self.circuit_func

        def minimizer(params, f, Z_data):
            Z0 = np.array(np.hsplit(circuit_func(f, *params), 2)).T
            Z_fit = np.hstack([Z0[:, 0], Z0[:, 1]])
            if len(Z_data) == len(Z_fit) / 2:
                Z_data = np.hstack([Z_data.real, Z_data.imag])
            return (Z_data - Z_fit) ** 2

        return minimizer


if __name__ == "__main__":
    rc = RCCircuit()
    print("True values:", rc.true_values)
    print("Initial guess:", rc.initial_guess)
    print("Bounds:", rc.bounds)
    print("Frequencies:", rc.freq)
    print("Impedance (noisy):", rc.Z_noisy)
