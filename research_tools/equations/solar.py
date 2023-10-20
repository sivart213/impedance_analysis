# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""


import numpy as np
import sympy as sp

from research_tools.functions import get_const, has_units, all_symbols

from research_tools.equations.physics import ni_Si, mobility_generic


# %% Quantum Efficiency
def IQE_emitter(ab, We, Le, De, Se, z=1):
    """Return the internal quantum efficiency of a solar cell emitter
    Where:
    ab - absorption coefficient (/cm)
    We - thickness of the emitter (cm)
    De - diffusivty of carriers in the emitter (cm²/s)
    Se - recombination at the front surface (cm/s)
    Hovel, I think."""
    GF = (
        (Se * Le / De)
        + ab * Le
        - (sp.exp(-ab * We * z))
        * ((Se * Le / De) * sp.cosh(We / Le) + sp.sinh(We / Le))
    ) / ((Se * Le / De) * sp.sinh(We / Le) + sp.cosh(We / Le)) - Le * ab * sp.exp(
        -ab * We * z
    )
    res = (Le * ab / (ab * ab * Le * Le - 1)) * GF
    if isinstance(res, sp.Number):
        return float(res)
    return res


def IQE_base(ab, We_Wd, Wb, Lb, Db, Sb, z=1):
    """Return quantum efficiency of the base of a solar cell
    where:
    ab -  absorption coefficient (cm)
    We_Wd - junction depth (cm)
    Sb - surface recombination velocity (cm/s)
    Lb - diffusion length of minority carrier in the base (cm)
    Db - diffusivity of minority carriers in the base (cm²/Vs)
    """
    GF = ab * Lb - (
        (Sb * Lb / Db) * (sp.cosh(Wb / Lb) - sp.exp(-ab * Wb * z))
        + sp.sinh(Wb / Lb)
        + Lb * ab * sp.exp(-ab * Wb * z)
    ) / ((Sb * Lb / Db) * sp.sinh(Wb / Lb) + sp.cosh(Wb / Lb))
    res = (sp.exp(-ab * We_Wd * z)) * (Lb * ab / (ab**2 * Lb**2 - 1)) * GF
    if isinstance(res, sp.Number):
        return float(res)
    return res


def IQE_IBC_emitter1(ab, We_Wd, We, Le, De, Se, z=1):
    """Return the internal quantum efficiency of a solar cell emitter
    Where:
    ab - absorption coefficient (/cm)
    We_Wd - junction depth (cm)
    We - thickness of the emitter (cm)
    De - diffusivty of carriers in the emitter (cm²/s)
    Se - recombination at the front surface (cm/s)
    Hovel, I think."""
    GF = (
        (Se * Le / De)
        + ab * Le
        - abs(sp.exp(-ab * We * z))
        * ((Se * Le / De) * sp.cosh(We / Le) + sp.sinh(We / Le))
    ) / ((Se * Le / De) * sp.sinh(We / Le) + sp.cosh(We / Le)) - Le * ab * abs(
        sp.exp(-ab * We * z)
    )
    res = abs(sp.exp(-ab * We_Wd * z)) * (Le * ab / (ab * ab * Le * Le - 1)) * GF
    if isinstance(res, sp.Number):
        return float(res)
    return res


def IQE_IBC_emitter2(ab, We_Wd, Wb, Lb, Db, Sb, z=1):
    """Return the internal quantum efficiency of a solar cell emitter
    Where:
    ab - absorption coefficient (/cm)
    We_Wd - junction depth (cm)
    We - thickness of the emitter (cm)
    De - diffusivty of carriers in the emitter (cm²/s)
    Se - recombination at the front surface (cm/s)
    Hovel, I think."""
    GF = ab * Lb - (
        (Sb * Lb / Db) * (sp.cosh(Wb / Lb) - sp.exp(-ab * Wb * z))
        + sp.sinh(Wb / Lb)
        + Lb * ab * sp.exp(-ab * Wb * z)
    ) / ((Sb * Lb / Db) * sp.sinh(Wb / Lb) + sp.cosh(Wb / Lb))
    res = (sp.exp(-ab * We_Wd * z)) * (Lb * ab / (ab**2 * Lb**2 - 1)) * GF
    if isinstance(res, sp.Number):
        return float(res)
    return res


def IQE_bulk(ab, We_Wd, Wb, Lb, Db, Sb, z=1):
    """Return quantum efficiency of the base of a solar cell
    where:
    ab -  absorption coefficient (cm)
    We_Wd - junction depth (cm)
    Sb - surface recombination velocity (cm/s)
    Lb - diffusion length of minority carrier in the base (cm)
    Db - diffusivity of minority carriers in the base (cm²/Vs)
    """
    GF = ab * Lb - (
        (Sb * Lb / Db) * (sp.cosh(Wb / Lb) - abs(sp.exp(-ab * Wb * z)))
        + sp.sinh(Wb / Lb)
        + Lb * ab * abs(sp.exp(-ab * Wb * z))
    ) / ((Sb * Lb / Db) * sp.sinh(Wb / Lb) + sp.cosh(Wb / Lb))
    res = abs(sp.exp(-ab * We_Wd * z)) * (Lb * ab / (ab**2 * Lb**2 - 1)) * GF
    if isinstance(res, sp.Number):
        return float(res)
    return res


def IQE_bulk2(ab, We_Wd, We, Le, De, Se, z=1):
    """Return the internal quantum efficiency of a solar cell emitter
    Where:
    ab - absorption coefficient (/cm)
    We_Wd - junction depth (cm)
    We - thickness of the emitter (cm)
    De - diffusivty of carriers in the emitter (cm²/s)
    Se - recombination at the front surface (cm/s)
    Hovel, I think."""
    GF = (
        (Se * Le / De)
        + ab * Le
        - sp.exp(-ab * We * z) * ((Se * Le / De) * sp.cosh(We / Le) + sp.sinh(We / Le))
    ) / ((Se * Le / De) * sp.sinh(We / Le) + sp.cosh(We / Le)) - Le * ab * sp.exp(
        -ab * We * z
    )
    res = (sp.exp(-ab * We_Wd * z)) * (Le * ab / (ab * ab * Le * Le - 1)) * GF
    if isinstance(res, sp.Number):
        return float(res)
    return res


def IQE_depletion(ab, We, Wd):
    res = sp.exp(-ab * We) * (1 - sp.exp(-ab * Wd))
    if isinstance(res, sp.Number):
        return float(res)
    return res


def IQE(ab, Wd, Se, Le, De, We, Sb, Wb, Lb, Db):
    """We is the thickness of emitter and start of the junction"""
    QEE = IQE_emitter(ab, We, Le, De, Se)
    QEB = IQE_base(ab, We + Wd, Wb, Lb, Db, Sb)
    QED = IQE_depletion(ab, We, Wd)
    IQEt = QEE + QEB + QED
    return QEE, QEB, QED, IQEt


# def QE2SR(wavelength, QE, R=0):
#     """'converts a QE in units to spectral response
#     given the wavelength (nm)"""
#     spectral_response = QE * wavelength * (1 - R) / 1239.8
#     return spectral_response


# def SR2QE(wavelength, spectral_response):
#     """convert SR (A/W) to QE (unit 0 to 1)
#     assumes that the wavelegth is in  nm"""
#     QE = spectral_response * wavelength / 1239.8
#     return QE


def implied_carrier(V, N, ni=8697277437.298948, T=298.15, n=1):
    """Return excess carrier concentration (cm-3).
    Given: voltage and doping determine"""
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)

    k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["eV", "K"]]))
    res = (-N + sp.sqrt(N**2 + 4 * ni**2 * sp.exp(V / (n * k_B * T)))) / 2
    if isinstance(res, sp.Number):
        return float(res)
    return res


# %% Current
def J0_layer(W, N, D, L, S, ni=8697277437.298948):
    """Return the saturation current density (A/cm2) for the narrow case.
    Where:
    W - layer thickness (cm)
    N - doping (cm-3)
    L - diffusion length (cm)
    S - surface recombination velocity (cm/s)
    Optional:
    ni - intrinsic carrier concentration (cm-3)
    """
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)

    q = get_const("elementary_charge", *([True] if symbolic else [w_units, ["C"]]))

    F = (S * sp.cosh(W / L) + D / L * sp.sinh(W * L)) / (
        D / L * sp.cosh(W * L) + S * sp.sinh(W / L)
    )
    res = q * ni**2 * F * D / (L * N)
    if isinstance(res, sp.Number):
        return float(res)
    return res


def J0_factor(W, N, D, L, S, ni=8697277437.298948):
    """Return the saturation current density (A/cm2) for the narrow case.
    Where:
    W - layer thickness (cm)
    N - doping (cm-3)
    L - diffusion length (cm)
    S - surface recombination velocity (cm/s)
    Optional:
    ni - intrinsic carrier concentration (cm-3)
    """
    res = (S * sp.cosh(W / L) + D / L * sp.sinh(W * L)) / (
        D / L * sp.cosh(W * L) + S * sp.sinh(W / L)
    )
    if isinstance(res, sp.Number):
        return float(res)
    return res


def J0(ni, We, Ne, De, Le, Se, Nb, Wb, Db, Lb, Sb):
    """determines J0, the dark saturation current, under the narrow base diode
    condition.L > W."""
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)

    q = get_const("elementary_charge", *([True] if symbolic else [w_units, ["C"]]))
    Fe = (Se * sp.cosh(We / Le) + De / Le * sp.sinh(We * Le)) / (
        De / Le * sp.cosh(We * Le) + Se * sp.sinh(We / Le)
    )
    Fb = (Sb * sp.cosh(Wb / Lb) + Db / Lb * sp.sinh(Wb * Lb)) / (
        Db / Lb * sp.cosh(Wb * Lb) + Sb * sp.sinh(Wb / Lb)
    )
    res = q * ni**2 * (Fe * De / (Le * Ne) + Fb * Db / (Lb * Nb))
    if isinstance(res, sp.Number):
        return float(res)
    return res


def current2gen(curr):
    """Return generation (eh pairs/s) given current (amps)."""
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)

    q = get_const("elementary_charge", *([True] if symbolic else [w_units, ["C"]]))
    res = curr / q
    if isinstance(res, sp.Number):
        return float(res)
    return res


def I_diode(V, I0, T=298.15, n=1):
    """Return the current (A) in an ideal diode.
    I0 is the saturation current (A),
    V is the voltage across the junction (volts), T is the temperature (K),
    and n is the ideallity factor (units).
    For current density. I0 is in A/cm² and current density is returned"""
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)
    k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["eV", "K"]]))
    res = I0 * sp.exp(V / (n * k_B * T) - 1)
    if isinstance(res, sp.Number):
        return float(res)
    return res


def I_cell(V, IL, I0, T=298.15, n=1):
    """Return current (amps) of a solar cell
    given voltage, light generated current, I0
    also works for J0
    """
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)
    k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["eV", "K"]]))
    res = IL - I0 * sp.exp(V / (n * k_B * T))
    if isinstance(res, sp.Number):
        return float(res)
    return res


def I_cell_DD(V, IL, I01, n1, I02, n2, T=298.15):
    """Return current (amps) of a solar cell
    given voltage, light generated current, I0
    also works for J0
    """
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)
    k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["eV", "K"]]))

    res = (
        IL
        - I01 * (sp.exp(V / (n1 * k_B * T)) - 1)
        - I02 * (sp.exp(V / (n2 * k_B * T)) - 1)
    )
    if isinstance(res, sp.Number):
        return float(res)
    return res


def I_cell_Rseries(V, Voc, Vmp, IL, I0, Imp):
    """Return current (amps) of a solar cell
    given voltage, light generated current, I0
    also works for J0
    """
    C1 = IL
    C2 = (Vmp - Voc) / (sp.log(1 - Imp / IL))

    res = IL - C1 * sp.exp(-1 * Voc / C2) * (sp.exp(V / C2) - 1)
    if isinstance(res, sp.Number):
        return float(res)
    return res


def I_cell_Rshunt(V, IL, I0, Rshunt, T=298.15, n=1):
    """Return current (A) of a solar cell from"""
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)
    k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["eV", "K"]]))
    res = IL - I0 * sp.exp(V / (n * k_B * T)) - V / Rshunt
    if isinstance(res, sp.Number):
        return float(res)
    return res


# %% Voltage
def impliedV(Δn, N, T=298.15, n=1):
    """Return voltage (V).
    Δn is the excess carrier concentration (cm-3),
    N is the doping (cm-3),
    T is the temperature (K).
    Implied voltage is often used to convert the carrier concentration in a lifetime
    tester to voltage."""
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)
    k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["eV", "K"]]))
    res = (n * k_B * T) * sp.log((Δn + N) * Δn / ni_Si(T) ** 2)
    if isinstance(res, sp.Number):
        return float(res)
    return res


def V_Rseries(voltage, curr, Rs):
    """Returns the voltage of a solar cells under the effect of series resistance"""
    return voltage - curr * Rs


def Voc(IL, I0, T=298.15, n=1):
    """Return the open circuit voltage, Voc, (volts) from IL(A) and I0(A).
    IL and Io must be in the same units, Eg, (A), (mA) etc
    Using (mA/cm**2) uses J0 and JL instead.
    """
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)
    k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["eV", "K"]]))
    res = (n * k_B * T) * sp.log(IL / I0 + 1)
    if isinstance(res, sp.Number):
        return float(res)
    return res


def V_cell(curr, IL, I0, T=298.15, n=1):
    """Return the voltage (V) in an ideal solar cell.
    I0 is the saturation current (A),
    curr is the current (A), T is the temperature (K) and n is the ideallity factor (units).
    For current density. I0 is in A/cm² and current density is returned"""
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)
    k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["eV", "K"]]))
    res = (n * k_B * T) * sp.log((IL - curr) / I0 + 1)
    if isinstance(res, sp.Number):
        return float(res)
    return res


# %% Cell Resistances
def emitter_resistance(Rsheet, Sf):
    """return the contribution of the emitter to cell series resistance (ohm cm²)
    given the spacing of the fingers (cm) and the emitter sheet resistivty (ohm/ sqr)"""
    return Rsheet * (Sf**2) / 12


def base_resistance(H, Nb, dopant="B"):
    """return the contribution of the emitter to cell series resistance (ohm cm²)
    given the spacing of the fingers (cm) and the emitter sheet resistivty (ohm/ sqr)"""
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)

    q = get_const("elementary_charge", *([True] if symbolic else [w_units, ["C"]]))
    res = (1 / (q * mobility_generic(Nb, dopant) * Nb)) * H
    if isinstance(res, sp.Number):
        return float(res)
    return res


def finger_resistance(Rfinger, Sf, L, wf, df):
    """return the contribution of the emitter to cell series resistance (ohm cm²)
    given the spacing of the fingers (cm) and the emitter sheet resistivty (ohm/ sqr)"""
    return Rfinger * Sf * L**2 / (3 * wf * df)


def finger_resistivity(L, Jmp, Sf, resistivity, wf, df, Vmp):
    """Return the fractional resistivity power loss in a finger (0 to 1)
    Given:
        L: finger length (cm)
        Jmp: currrent density at the max power point in A/cm2
        Sf: finger spacing (cm)
    """
    return (L**2 * Jmp * Sf * resistivity) / (3 * wf * df * Vmp) * 100.0


def finger_sheet(Sf, Jmp, Rsheet, Vmp):
    return (Sf**2 * Jmp * Rsheet) / (12 * Vmp) * 100.0


def busbar_resistance(Rbus, W, Z, wb, db, m):
    """return the contribution of the emitter to cell series resistance (ohm cm²)
    given the spacing of the fingers (cm) and the emitter sheet resistivty (ohm/ sqr)"""
    return Rbus * W * Z**2 / (3 * wb * db * m)


def IBC_metal_resistance(Rmetal, W, Z, wfn, wfp, df, Sf):
    """return the metal resistance of the metal contacts on the back of a IBC cell"""
    unit = np.floor(Z / (wfn + Sf + wfp + Sf))
    #    center = unit*wfn*(W-.0125*2) + unit*wfp*(W-.0125*2)
    #    edge = Z*.01*2 + unit*wfn*.0025 + unit*wfp*.0025
    return Rmetal * W * Z**2 / (3 * wfn * df * unit) + Rmetal * W * Z**2 / (
        3 * wfp * df * unit
    )


# %% Cell Evaluations
def cell_params(V, curr):
    """Return key parameters of a solar cell IV curve.
    V is a voltage array and
    curr is a current array, both with type numpy.array.
    Voc (V), Isc (A), FF, Vmp(V), Imp(A) given voltage vector in (volts)
    current vector in (amps) or (A/cm²)
    If curr is in (A/cm²) then Isc will be Jsc and Imp will be Jmp.
    No attempt is made to fit the fill factor.
    """
    Voc = np.interp(0, -curr, V)
    Isc = np.interp(0, V, curr)
    idx = np.argmax(V * curr)
    Vmp = V[idx]
    Imp = curr[idx]
    FF = Vmp * Imp / (Voc * Isc)
    return Voc, Isc, FF, Vmp, Imp, Vmp * Imp


def efficiency(Voc, Isc, FF, A=1):
    """Return the efficiency of a solar cell (units not percentage).
    given: Voc (volts), Isc in (amps) and  FF (units).
    also works for Jsc since area of 1 is assumed
    """
    return 1000 * Voc * Isc * FF / A


def finger_shading(wf, Sf):
    """Return the fractional power loss due to finger shading (0 to 1) where wf is the wideth of the
    finger and Sf is the finger spacing."""
    return (wf / Sf) * 100.0


def finger_total_loss(L, Jmp, Sf, resistivity, Rsheet, wf, df, Vmp):
    """Return the fractional power loss in a finger
    Given:
        L: finger length (cm)
        Jmp: currrent density at the max power point in A/cm2
        Sf: finger spacing (cm)
    """
    Presistivity = finger_resistivity(L, Jmp, Sf, resistivity, wf, df, Vmp)
    Pshading = finger_shading(wf, Sf)
    Psheet = finger_sheet(Sf, Jmp, Rsheet, Vmp)
    return Presistivity + Pshading + Psheet, Presistivity, Pshading, Psheet


def FF(Vmp, Imp, Voc, Isc):
    """Return FFv the fill factor of a solar cell.
    given Voc - open circuit voltage (volts)"""
    return (Vmp * Imp) / (Voc * Isc)


def FF_ideal(Voc, ideality=1, T=298.15):
    """Return the FF (units)
    given Voc - open circuit voltage (volts), ideality factor, defaults to 1 (units)"""
    voc = normalised_Voc(Voc, ideality, T)
    res = (voc - sp.log(voc + 0.72)) / (voc + 1)
    if isinstance(res, sp.Number):
        return float(res)
    return res


def normalised_Voc(Voc, ideality, T=298.15, n=1):
    """Return the normalised voc of a solar cell.Voc is the open-circuit voltage,
    'ideality' is the ideality factor and T is the temperature (K)"""
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)
    k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["eV", "K"]]))
    res = Voc / (ideality * (n * k_B * T))
    if isinstance(res, sp.Number):
        return float(res)
    return res


def FF_Rs(Voc, Isc, Rseries, ideality=1, T=298.15):
    """Return the FF (units)
    Given:
        Voc - open circuit voltage (volts)
        Isc - short circuit current (amps)
        Rseries - series resistance (ohms)
        ideality factor (units)
        T - temperature (K)
    """
    # voc = normalised_Voc(Voc, ideality, T)
    RCH = Voc / Isc
    rs = Rseries / RCH
    FF0 = FF_ideal(Voc, ideality, T)
    FF = FF0 * (1 - 1.1 * rs) + (rs**2 / 5.4)
    return FF


def FF_Rsh(Voc, Isc, Rshunt, ideality=1, T=298.15):
    """Return the FF (units)
    Given:
        Voc - open circuit voltage (volts)
    """
    voc = normalised_Voc(Voc, ideality, T)
    RCH = Voc / Isc
    rsh = Rshunt / RCH
    FF0 = FF_ideal(Voc, ideality, T)
    FF = FF0 * (1 - ((voc + 0.7) * FF0) / (voc * rsh))
    return FF


def FF_RsRsh(Voc, Isc, Rseries, Rshunt, ideality=1, T=298.15):
    voc = normalised_Voc(Voc, ideality, T)
    RCH = Voc / Isc
    rsh = Rshunt / RCH
    # FF0 = FF_ideal(Voc, ideality, T)
    FFRs = FF_Rs(Voc, Isc, Rseries, ideality=1, T=298.15)
    FFRsRsh = FFRs * (1 - ((voc + 0.7) * FFRs) / (voc * rsh))
    return FFRsRsh


# %% silicon material properties
# silicon material properties
def optical_properties(fname):
    """Returns an array with the optical properties of a material
    column 0 - wavelngth (nm)
    column 1 - absorption coefficient (/cm)
    column 2 - real refractive index
    column 3 - imaginary refractive index
    if so file is given then silicon is used
    Eg: wavelength, abs_coeff, n, KB_J = optical_properties()
    """
    # if fname is None:
    #     package_path = os.path.dirname(os.path.abspath(__file__))
    #     fname = os.path.join(package_path, "silicon_optical_properties.txt")
    wavelength, abs_coeff, nd, kd = np.loadtxt(fname, skiprows=1, unpack=True)
    return wavelength, abs_coeff, nd, kd


# processing
def phos_active(T):
    """Return the active limit of phosphorous in silicon
    given temperature (K)"""
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)

    k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["eV", "K"]]))

    res = 1.3e22 * sp.exp(-0.37 / (k_B * T))
    if isinstance(res, sp.Number):
        return float(res)
    return res


def phos_solubility(T):
    """Return the solubility limit of phosphorous in silicon
    given the temperature (K)"""
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)

    k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["eV", "K"]]))

    res = 2.45e23 * sp.exp(-0.62 / (k_B * T))
    if isinstance(res, sp.Number):
        return float(res)
    return res

    # # modules(Pedro)
    # def read_cell_info(selected, path=None, file=None):
    #     # TODO Rework with p_find
    #     if path is None:
    #         path = os.sep.join(("work", "Data"))
    #     package_path = pathify(path)
    #     if file is None:
    #         fname = os.path.join(package_path, "cell_info.txt")
    #     else:
    #         fname = os.path.join(package_path, file)

    #     with open(fname, "r") as f:
    #         for line in f:
    #             col1, col2, col3, col4 = line.split()
    #             if col1 == selected:
    #                 semicondutor = col1
    #                 J_SC = float(col2)
    #                 V_OC = float(col3)
    #                 J_0 = float(col4)
    #     return semicondutor, J_SC, V_OC, J_0

    # def module_current(M, N, T, material):
    #     arg_in = vars().copy()
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)


#     q = get_const("elementary_charge", *([True] if symbolic else [w_units, ["C"]]))
#     k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["eV", "K"]]))

#     semicondutor, J_SC, V_OC, J_0 = read_cell_info(material)
#     I_0 = J_0 * 15.6 * 15.6
#     I_L = J_SC * 15.6 * 15.6
#     V_T = V_OC * N
#     res = M * (I_L - I_0 * sp.exp((q * V_T / N) / (k_B * T) - 1))

#     if isinstance(res, sp.Number):
#         return float(res)
#     return res
