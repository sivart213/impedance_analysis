# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""


import numpy as np
import sympy as sp

from research_tools.functions import get_const, has_units, has_arrays, pick_math_module

from research_tools.equations.general import inv_sum_invs, erfc


# %% Electrical
def capacitance(er, A, L, **kwargs):
    """Calculate. generic discription."""
    arg_in = vars().copy()
    e0 = kwargs.get("e0", None)
    if e0 is None:
        w_units = has_units(arg_in)
        e0 = get_const("e0", w_units, ["farad", "cm"])

    res = er * e0 * A / L

    # if isinstance(res, nsp.Number):
    #     return float(res)
    return res


def resistance(rho, A, L):
    """Calculate resistance from the resistivity and system dimensions."""
    res = rho * L / A
    return res


def ohms_law(V, R):
    """Calculate current from Ohms Law."""
    res = V / R
    return res


def voltage_divider(R, V=1, R0=1):
    """Calculate the component voltage from the voltage devider."""
    if isinstance(R, dict):
        Rt = sum(R.values())
        R0 = R.get(R0, R0)
    elif isinstance(R, (list, tuple, np.ndarray)):
        Rt = sum(R)
        try:
            R0 = R[int(R0)]
        except (ValueError, IndexError):
            if R0 not in R:
                R0 = 1
    else:
        return None
    res = V * R0 / Rt
    return res


def sheet_resistivity(doping, thickness, dopant=None):
    """Calculate the sheet resistivity from doping."""
    arg_in = vars().copy()
    if dopant is not None:
        mob = mobility_masetti(doping, dopant)
    else:
        mob = 300  # assume constant mobility

    w_units = has_units(arg_in)
    q = get_const("elementary_charge", w_units, ["C"])

    res = 1 / (q * doping * mob * thickness)

    # if isinstance(res, nsp.Number):
    #     return float(res)
    return res


def conductivity(n, p, ue, uh):
    """Return the conductivity of a material(siemens)
    Where:
    n - concentration of electrons (cm-3)
    p - concentration of holes (cm-3)
    ue - electron mobility (cm²/Vs)
    uh - hole mobility (cm²/Vs)"""
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    q = get_const("elementary_charge", w_units, ["C"])

    res = q * ue * n + q * uh * p

    # if isinstance(res, nsp.Number):
    #     return float(res)
    return res


def resistivity_Si_n(Ndonor):
    """Return the resistivity of n-type silicon (ohm cm)
    given the doping of donors(cm-3)"""
    arg_in = vars().copy()
    n_minority = ni_Si() ** 2 / Ndonor

    w_units = has_units(arg_in)
    q = get_const("elementary_charge", w_units, ["C"])

    res = 1 / (
        (q * mobility_thurber(Ndonor, False) * Ndonor)
        + (q * mobility_thurber(n_minority, False, False) * n_minority)
    )

    # if isinstance(res, nsp.Number):
    #     return float(res)
    return res


def resistivity_Si_p(Nacceptor):
    """Return the resistivity of p-type silicon (ohm cm)
    given the doping of acceptors(cm-3)"""
    arg_in = vars().copy()
    n_minority = ni_Si() ** 2 / Nacceptor

    w_units = has_units(arg_in)
    q = get_const("elementary_charge", w_units, ["C"])

    res = 1 / (
        (q * mobility_thurber(Nacceptor) * Nacceptor)
        + (q * mobility_thurber(n_minority, True, False) * n_minority)
    )
    # if isinstance(res, nsp.Number):
    #     return float(res)
    return res


def resistivity(N, dopant, W):
    """Return the resistivity of p-type silicon (ohm cm)
    given the doping of acceptors(cm-3)"""
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    q = get_const("elementary_charge", w_units, ["C"])
    res = 1 / (q * mobility_generic(N, dopant) * N * W)
    # if isinstance(res, nsp.Number):
    #     return float(res)
    return res


# %% Semiconductors
def v_thermal(T=298.15):
    """Return thermal voltage (volts) at given temperature, T(Kelvin).
    The default temperature is 298.15 K, which is equal to 25 °C"""
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    k_B = get_const("boltzmann", w_units, ["eV", "K"])
    res = k_B * T
    # if isinstance(res, nsp.Number):
    #     return float(res)
    return res


def depletion_region(Na, Nd, T=298.15):
    """Find thickness of doped layers based on doping"""
    arg_in = vars().copy()
    lam_res, symb_var = has_arrays(arg_in, depletion_region)
    if lam_res:
        return symb_var

    nsp = pick_math_module(arg_in)
    w_units = has_units(arg_in)

    q = get_const("elementary_charge", w_units, ["C"])
    e0 = get_const("e0", w_units, ["farad", "cm"])
    k_B = get_const("boltzmann", w_units, ["eV", "K"])

    Vbi = k_B * T * nsp.log(Na * Nd / ni_Si(T) ** 2)

    pre = 2 * 11.8 * e0 / q * Vbi * (1 / (Na + Nd))
    xp = nsp.sqrt(pre * Nd / Na)
    xn = nsp.sqrt(pre * Na / Nd)
    # if isinstance(xp, nsp.Number):
    #     xp = float(xp)
    # if isinstance(xn, nsp.Number):
    #     xn = float(xn)
    return xn, xp


def probability_fermi_dirac(E, Ef, T):
    """Return the fermi dirac function (units) where E is the energy (),
    Ef is the fermi given the energies in electron volts"""
    arg_in = vars().copy()
    nsp = pick_math_module(arg_in)
    w_units = has_units(arg_in)
    k_B = get_const("boltzmann", w_units, ["eV", "K"])

    res = 1 / (nsp.exp((E - Ef) / (k_B * T)) + 1.0)
    # if isinstance(res, nsp.Number):
    #     return float(res)
    return res


def probability_maxwell_boltzmann(E, Ef, T):
    """Given the energies in electron volts return the fermi dirac function"""
    arg_in = vars().copy()
    nsp = pick_math_module(arg_in)
    w_units = has_units(arg_in)
    k_B = get_const("boltzmann", w_units, ["eV", "K"])
    res = 1 / (nsp.exp((E - Ef) / (k_B * T)))
    # if isinstance(res, nsp.Number):
    #     return float(res)
    return res


def probability_bose_einstein(E, Ef, T):
    """Given the energies in electron volts return the fermi dirac function"""
    arg_in = vars().copy()
    nsp = pick_math_module(arg_in)
    w_units = has_units(arg_in)

    k_B = get_const("boltzmann", w_units, ["eV", "K"])
    res = 1 / (nsp.exp((E - Ef) / (k_B * T)) - 1.0)
    # if isinstance(res, nsp.Number):
    #     return float(res)
    return res


def equilibrium_carrier(doping, **kwargs):
    """
    Return the majority and minority carrier concentrations (cm-3) of a semiconductor at equilibrium
    where N is the doping level (cm-3) and ni is the intrinsic carrier concentratoin (cm-3)
    Strictly N and ni just have to be in the same units but (cm-3 is almost always used.
    """
    ni = kwargs.get("ni", ni_Si(kwargs.get("temp", 298.15)))
    carrier = doping / (ni**2)
    return max(doping, carrier), min(doping, carrier)


def ni_Si(temp=298.15, narrowing=True):
    """Return the intrinsic carrier concentration of silicon (cm**-3) according to Sproul94.
    where temp is the temperature (K)
    http://dx.doi.org/10.1063/1.357521
    Return the intrinsic carrier concentration (cm-3) without band gap narrowing
    according to Misiakos, where temp is the temperature (K).
    DOI http://dx.doi.org/10.1063/1.354551
    """
    arg_in = vars().copy()
    nsp = pick_math_module(arg_in)
    if narrowing:
        res = 9.38e19 * (temp / 300) * (temp / 300) * nsp.exp(-6884 / temp)
    else:
        res = 5.29e19 * (temp / 300) ** 2.54 * nsp.exp(-6726 / temp)
    # if isinstance(res, nsp.Number):
    #     return float(res)
    return res


def ni_eff(N_D, N_A, Δn, T=298.15):
    """Return effective ni (cm-3)
    given
    donor concentration N_D=n0 (1/cm³)      only one dopant type possible
    acceptor concentration N_A=p0 (1/cm³)    only one dopant type possible
    excess carrier density (1/cm³)
    temperature (K)
    calculation of the effective intrinsic concentration n_ieff including BGN
    according to Altermatt JAP 2003
    """
    arg_in = vars().copy()
    lam_res, symb_var = has_arrays(arg_in, ni_eff)
    if lam_res:
        return symb_var

    w_units = has_units(arg_in)
    k_B = get_const("boltzmann", w_units, ["eV", "K"])
    # n_i without BGN according to Misiakos93, parameterization fits very well
    # to value of Altermatt2003 at 300K
    ni0 = ni_Si(T, False)

    ni = ni0  # ni0 as starting value for n_ieff for calculation of n0 & p0

    n0 = np.where(N_D > N_A, N_D, N_A / ni**2)
    p0 = np.where(N_D > N_A, N_D / ni**2, N_A)

    # self-conistent iterative calculation of n_ieff

    for i in range(5):  # lazy programmer as it converges pretty fast anyway
        n = n0 + Δn
        p = p0 + Δn
        dEc, dEv = bandgap_schenk(n, p, N_A, N_D, Δn, T)
        ni = ni0 * np.exp(
            (dEc + dEv) / (2 * (k_B * T))
        )  # there is something wrong here as the units don't match up.
        n0 = np.where(N_D > N_A, N_D, N_A / ni**2)
        p0 = np.where(N_D > N_A, N_D / ni**2, N_A)

    # print('iterations',ni)
    # if isinstance(ni, nsp.Number):
    #     return float(ni)
    return ni


def bandgap_paessler(temp=298.15):
    """Return the bandgap of silicon (eV) according to Paessler2002,
    where temp is the temperature (K).
    Code adapted from Richter Fraunhofer ISE
    https://doi.org/10.1103/PhysRevB.66.085201
    """
    arg_in = vars().copy()
    nsp = pick_math_module(arg_in)
    lam_res, symb_var = has_arrays(arg_in, bandgap_paessler)
    if lam_res:
        return symb_var

    # constants from Table I on page 085201-7
    α = 3.23 * 0.0001  # (eV/K)
    Θ = 446  # (K)
    Δ = 0.51
    Eg0_T0 = 1.17  # eV     band gap of Si at 0 K

    Tdelta = 2 * temp / Θ
    wurzel = (
        1
        + nsp.pi**2 / (3 * (1 + Δ**2)) * Tdelta**2
        + (3 * Δ**2 - 1) / 4 * Tdelta**3
        + 8 / 3 * Tdelta**4
        + Tdelta**6
    ) ** (1 / 6)
    Eg0 = Eg0_T0 - α * Θ * (
        (1 - 3 * Δ**2) / (nsp.exp(Θ / temp) - 1) + 3 / 2 * Δ**2 * (wurzel - 1)
    )
    # if isinstance(Eg0, nsp.Number):
    #     return float(Eg0)
    return Eg0


def bandgap_schenk(n_e, n_h, N_D, N_A, Δn, T=298.15):
    """
    returns the band gap narowing in silicon
    delta conduction band, delta valence band in eV
    given:

    n_e => total electron density with Δn (1/cm³)
    n_h => total hole density with Δn (1/cm³)
    N_A => acceptor concentration (1/cm³)
    N_D => donor concentration (1/cm³)
    Δn  => excess carrier density (1/cm³)
    temp   => temperature (K)

    Band-gap narrowing after Schenk 1998, JAP 84(3689))
    model descriped very well in K. McIntosh IEEE PVSC 2010
    model confirmed by Glunz2001 & Altermatt2003
    nomenclatur and formula no. according to McIntosh2010, table no. according to Schenk1998
    ==========================================================================
    Input parameters:

    ==========================================================================
    Code adapted from Richter at Fraunhofer ISE
    http://dx.doi.org/10.1063%2F1.368545
    """
    arg_in = vars().copy()
    lam_res, symb_var = has_arrays(arg_in, bandgap_schenk)
    if lam_res:
        return symb_var

    nsp = pick_math_module(arg_in)
    w_units = has_units(arg_in)
    k_B = get_const("boltzmann", w_units, ["eV", "K"])

    # Silicon material parameters (table 1)
    g_e = 12  # degeneracy factor for electrons
    g_h = 4  # degeneracy factor for holes
    alfa_e = 0.5187  # µ*/m_e
    alfa_h = 0.4813  # µ*/m_h
    Ry_ex = 0.01655  # eV    excitonic Rydberg constant
    alfa_ex = 0.0000003719  # cm     excitonic Bohr radius

    # Parameters for Pade-Approximation (tab. 2 & 3)
    b_e = 8
    b_h = 1
    c_e = 1.3346
    c_h = 1.2365
    d_e = 0.893
    d_h = 1.153
    p_e = 7 / 30
    p_h = 7 / 30
    h_e = 3.91
    h_h = 4.2
    j_e = 2.8585
    j_h = 2.9307
    k_e = 0.012
    k_h = 0.19
    q_e = 3 / 4
    q_h = 1 / 4

    # ==========================================================================
    # pre-calculations:
    F = ((k_B * T)) / Ry_ex  # eq. 29
    a3 = alfa_ex**3

    # Normalizing of the densities
    n_e *= a3
    n_h *= a3
    N_D *= a3
    N_A *= a3

    # for eq. 33 (normalized)
    n_sum_xc = n_e + n_h
    n_p_xc = alfa_e * n_e + alfa_h * n_h

    # for eq. 37 (normalized)
    n_sum_i = N_D + N_A  # eq.39 bzw. eq. 29
    n_p_i = alfa_e * N_D + alfa_h * N_A  # eq.39 bzw. eq. 29

    Ui = n_sum_i**2 / F**2  # eq. 38
    n_ionic = n_sum_i  # McIntosh2010

    # exchange quasi-partical shift Eq33:
    delta_xc_h = -(
        (4 * nsp.pi) ** 3
        * n_sum_xc**2
        * (
            (48 * n_h / (nsp.pi * g_h)) ** (1 / 3)
            + c_h * nsp.log(1 + d_h * n_p_xc**p_h)
        )
        + (8 * nsp.pi * alfa_h / g_h) * n_h * F**2
        + nsp.sqrt(8 * nsp.pi * n_sum_xc) * F ** (5 / 2)
    ) / (
        (4 * nsp.pi) ** 3 * n_sum_xc**2
        + F**3
        + b_h * nsp.sqrt(n_sum_xc) * F**2
        + 40 * n_sum_xc ** (3 / 2) * F
    )
    delta_xc_e = -(
        (4 * nsp.pi) ** 3
        * n_sum_xc**2
        * (
            (48 * n_e / (nsp.pi * g_e)) ** (1 / 3)
            + c_e * nsp.log(1 + d_e * n_p_xc**p_e)
        )
        + (8 * nsp.pi * alfa_e / g_e) * n_e * F**2
        + nsp.sqrt(8 * nsp.pi * n_sum_xc) * F ** (5 / 2)
    ) / (
        (4 * nsp.pi) ** 3 * n_sum_xc**2
        + F**3
        + b_e * nsp.sqrt(n_sum_xc) * F**2
        + 40 * n_sum_xc ** (3 / 2) * F
    )

    # ionic quasi-partical shift Eq37:
    delta_i_h = (
        -n_ionic
        * (1 + Ui)
        / (
            nsp.sqrt(0.5 * F * n_sum_i / nsp.pi)
            * (1 + h_h * nsp.log(1 + nsp.sqrt(n_sum_i) / F))
            + j_h * Ui * n_p_i**0.75 * (1 + k_h * n_p_i**q_h)
        )
    )
    delta_i_e = (
        -n_ionic
        * (1 + Ui)
        / (
            nsp.sqrt(0.5 * F * n_sum_i / nsp.pi)
            * (1 + h_e * nsp.log(1 + nsp.sqrt(n_sum_i) / F))
            + j_e * Ui * n_p_i**0.75 * (1 + k_e * n_p_i**q_e)
        )
    )

    # rescale BGN
    dE_gap_h = -Ry_ex * (delta_xc_h + delta_i_h)
    dE_gap_e = -Ry_ex * (delta_xc_e + delta_i_e)
    # if isinstance(dE_gap_h, nsp.Number):
    #     dE_gap_h = float(dE_gap_h)
    # if isinstance(dE_gap_e, nsp.Number):
    #     dE_gap_e = float(dE_gap_e)
    return dE_gap_e, dE_gap_h


# %%  Mobilities
def mobility_generic(N, dopant):
    """Return the mobility of carriers in silicon according to
    the model of Thurbur as a function of doping
    Where:
    N - doping level (cm-3)
    Data is included for specific dopant values as given in mini-project 3
    https://archive.org/details/relationshipbetw4006thur"""
    if "A" in dopant:
        umin = 52.2
        umax = 1417
        Nref = 9.68e16
        a = 0.68
    if "P" in dopant:
        umin = 68.5
        umax = 1414
        Nref = 9.20e16
        a = 0.711
    if "B" in dopant:
        umin = 44.9
        umax = 470.5
        Nref = 2.23e17
        a = 0.719
    return umin + (umax - umin) / (1 + ((N / Nref) ** a))


def mobility_thurber(N, p_type=True, majority=True):
    """Return the mobility of carriers in silicon according to the model of Thurbur
    as a function of doping
    Where:
    N - doping level (cm-3)
    p_type is True or 1 for p doped material and False or 0 for n-type.
    majority is True or 1 for majority carriers and False or 0 for minority carriers.
    https://archive.org/details/relationshipbetw4006thur"""
    i = 2 * p_type + majority
    # n-type minority, n-type majority, p-type minority, p-type majority
    umax = [1417, 1417, 470, 470][i]
    umin = [160, 60, 155, 37.4][i]
    Nref = [5.6e16, 9.64e16, 1e17, 2.82e17][i]
    a = [0.647, 0.664, 0.9, 0.642][i]
    return umin + (umax - umin) / (1 + ((N / Nref) ** a))


def mobility_masetti(N, dopant=0):
    """mobility model from Masetti
    DOI: 10.1109/T-ED.1983.21207"""
    if dopant == 0:
        µmax = 1414
        µmin = 68.5
        u1 = 56.1
        Nref1 = 9.20e16
        Nref2 = 3.41e20
        a = 0.711
        b = 1.98
    if dopant == 1:
        µmax = 470.5
        µmin = 44.9
        u1 = 29.0
        Nref1 = 2.23e17
        Nref2 = 6.1e20
        a = 0.719
        b = 1.98
    return (
        µmin + (µmax - µmin) / (1 + ((N / Nref1) ** a)) - u1 / (1 + ((Nref2 / N) ** b))
    )


def mobility_klassen(Nd, Na, Δn=1, temp=298.16):
    """Return the mobility (cm2/Vs)
    given the doping etc."""
    s1 = 0.89233
    s2 = 0.41372
    s3 = 0.19778
    s4 = 0.28227
    s5 = 0.005978
    s6 = 1.80618
    s7 = 0.72169
    r1 = 0.7643
    r2 = 2.2999
    r3 = 6.5502
    r4 = 2.367
    r5 = -0.01552
    r6 = 0.6478
    fCW = 2.459
    fBH = 3.828
    mh_me = 1.258
    me_m0 = 1

    temp = 298.16
    n0, p0 = equilibrium_carrier(Nd)

    cA = 0.5
    cD = 0.21
    Nref_A = 7.20e20
    Nref_D = 4.00e20

    p = p0 + Δn
    n = n0 + Δn
    cc = p + n

    Za_Na = 1 + 1 / (cA + (Nref_A / Na) ** 2)
    Zd_Nd = 1 + 1 / (cD + (Nref_D / Nd) ** 2)

    Na_h = Za_Na * Na
    Nd_h = Zd_Nd * Nd

    boron_µmax = 470.5
    boron_µmin = 44.9
    boron_Nref_1 = 2.23e17
    boron_α = 0.719
    boron_θ = 2.247

    phosphorus_µmax = 1414
    phosphorus_µmin = 68.5
    phosphorus_Nref_1 = 9.20e16
    phosphorus_α = 0.711
    phosphorus_θ = 2.285

    µ_eN = (
        phosphorus_µmax**2
        / (phosphorus_µmax - phosphorus_µmin)
        * (temp / 300) ** (3 * phosphorus_α - 1.5)
    )
    µ_hN = (
        boron_µmax**2
        / (boron_µmax - boron_µmin)
        * (temp / 300) ** (3 * boron_α - 1.5)
    )

    µ_ec = (
        phosphorus_µmax
        * phosphorus_µmin
        / (phosphorus_µmax - phosphorus_µmin)
        * (300 / temp) ** 0.5
    )
    µ_hc = boron_µmax * boron_µmin / (boron_µmax - boron_µmin) * (300 / temp) ** 0.5

    Ne_sc = Na_h + Nd_h + p
    Nh_sc = Na_h + Nd_h + n

    PBHe = 1.36e20 / cc * me_m0 * (temp / 300) ** 2
    PBHh = 1.36e20 / cc * mh_me * (temp / 300) ** 2

    PCWe = 3.97e13 * (1 / (Zd_Nd**3 * (Nd_h + Na_h + p)) * ((temp / 300) ** 3)) ** (
        2 / 3
    )
    PCWh = 3.97e13 * (1 / (Za_Na**3 * (Nd_h + Na_h + n)) * ((temp / 300) ** 3)) ** (
        2 / 3
    )

    Pe = 1 / (fCW / PCWe + fBH / PBHe)
    Ph = 1 / (fCW / PCWh + fBH / PBHh)

    G_Pe = (
        1
        - s1 / ((s2 + (1 / me_m0 * 300 / temp) ** s4 * Pe) ** s3)
        + s5 / (((me_m0 * 300 / temp) ** s7 * Pe) ** s6)
    )
    G_Ph = (
        1
        - s1 / ((s2 + (1 / (me_m0 * mh_me) * temp / 300) ** s4 * Ph) ** s3)
        + s5 / (((me_m0 * mh_me * 300 / temp) ** s7 * Ph) ** s6)
    )

    F_Pe = (r1 * Pe**r6 + r2 + r3 / mh_me) / (Pe**r6 + r4 + r5 / mh_me)
    F_Ph = (r1 * Ph**r6 + r2 + r3 * mh_me) / (Ph**r6 + r4 + r5 * mh_me)

    Ne_sc_eff = Nd_h + G_Pe * Na_h + p / F_Pe
    Nh_sc_eff = Na_h + G_Ph * Nd_h + n / F_Ph

    # Lattice Scattering
    µ_eL = phosphorus_µmax * (300 / temp) ** phosphorus_θ
    µ_hL = boron_µmax * (300 / temp) ** boron_θ

    µe_Dah = µ_eN * Ne_sc / Ne_sc_eff * (
        phosphorus_Nref_1 / Ne_sc
    ) ** phosphorus_α + µ_ec * ((p + n) / Ne_sc_eff)
    µh_Dae = µ_hN * Nh_sc / Nh_sc_eff * (boron_Nref_1 / Nh_sc) ** boron_α + µ_hc * (
        (p + n) / Nh_sc_eff
    )

    µe = 1 / (1 / µ_eL + 1 / µe_Dah)
    µh = 1 / (1 / µ_hL + 1 / µh_Dae)

    return µe, µh


# %% Diffusion
def mobility_diffusion(D=1e-15, T=298.15, z=1, **kwargs):
    """Return the mobility (cm²/Vs) or Diffusivity (cm²/s) given the other value.
    This is also known as the Einstein relation"""
    arg_in = vars().copy()
    lam_res, symb_var = has_arrays(arg_in, mobility_diffusion)
    if lam_res:
        return symb_var

    w_units = has_units(arg_in)

    k_B = get_const("boltzmann", w_units, ["eV", "K"])
    res = D * z / (k_B * T)

    # if isinstance(res, nsp.Number):
    #     return float(res)
    return res


def characteristic_length(D, t, z, T, E):
    """Calculate via the characteristic t."""
    arg_in = vars().copy()
    lam_res, symb_var = has_arrays(arg_in, characteristic_length)
    if lam_res:
        return symb_var

    nsp = pick_math_module(arg_in)
    w_units = has_units(arg_in)

    k_B = get_const("boltzmann", w_units, ["eV", "K"])
    mob = D * z / (k_B * T)

    res = 2 * nsp.sqrt(D * t) + mob * E * t

    # if isinstance(res, nsp.Number):
    #     return float(res)
    return res


def diffusion_length(t, D):
    """Return carrier Diffusion length (cm)
    given carrier t(s) and D (units)
    """
    arg_in = vars().copy()
    lam_res, symb_var = has_arrays(arg_in, diffusion_length)
    if lam_res:
        return symb_var
    nsp = pick_math_module(arg_in)
    res = nsp.sqrt(t * D)
    # if isinstance(res, nsp.Number):
    #     return float(res)
    return res


def nernst_planck_analytic_sol(
    x=5e-2, t=3600, L=5e-2, E=0, D=1e-15, T=298.15, z=1, conc0=1
):
    """Calculate the ratio of C/C0 for arithmatic solution to np"""
    arg_in = vars().copy()
    lam_res, symb_var = has_arrays(arg_in, nernst_planck_analytic_sol)
    if lam_res:
        return symb_var

    if x is None:
        x = L
    elif L is None:
        L = x

    nsp = pick_math_module(arg_in)
    w_units = has_units(arg_in)

    k_B = get_const("boltzmann", w_units, ["eV", "K"])
    mob = D * z / (k_B * T)

    term_A1 = erfc((x - mob * E * t) / (2 * nsp.sqrt(D * t)))
    term_A2 = erfc(-(x - 2 * L + mob * E * t) / (2 * nsp.sqrt(D * t)))
    term_B = erfc(-mob * E * t / (2 * nsp.sqrt(D * t)))
    res = (conc0 / (2 * term_B)) * (term_A1 + term_A2)

    # if isinstance(res, nsp.Number):
    #     return float(res)
    return res


def nernst_planck_fundamental_sol(x=5e-2, t=3600, D=1e-15, conc0=1):
    """Calculate the ratio of C/C0 for arithmatic solution to np"""
    arg_in = vars().copy()
    lam_res, symb_var = has_arrays(arg_in, nernst_planck_fundamental_sol)
    if lam_res:
        return symb_var
    nsp = pick_math_module(arg_in)

    res = conc0 * erfc((x) / (2 * nsp.sqrt(D * t)))
    # if isinstance(res, nsp.Number):
    #     return float(res)
    return res


def debye_length(z, er, C, T):
    arg_in = vars().copy()
    lam_res, symb_var = has_arrays(arg_in, debye_length)
    if lam_res:
        return symb_var

    nsp = pick_math_module(arg_in)
    w_units = has_units(arg_in)

    q = get_const("elementary_charge", w_units, ["C"])
    e0 = get_const("e0", w_units, ["farad", "cm"])
    k_B = get_const("boltzmann", w_units, ["joule", "K"])

    if isinstance(C, (tuple, list)):
        if not isinstance(z, (tuple, list)):
            z = [z]*len(C)
        charges = sum([C[n]*(q*z[n])**2 for n in range(len(C))])
    else:
        charges = C*(q*z)**2
    res = nsp.sqrt(er*e0*k_B*T/(charges))

    return res

def bjerrum_length(er, T):
    arg_in = vars().copy()
    w_units = has_units(arg_in)

    q = get_const("elementary_charge", w_units, ["C"])
    e0 = get_const("e0", w_units, ["farad", "cm"])
    k_B = get_const("boltzmann", w_units, ["joule", "K"])

    res = q**2/(er*e0*k_B*T)

    # if isinstance(res, nsp.Number):
    #     return float(res)
    return res

def screened_permitivity(er, k_D, x=1):
    arg_in = vars().copy()
    nsp = pick_math_module(arg_in)
    w_units = has_units(arg_in)

    e0 = get_const("e0", w_units, ["farad", "cm"])

    res = er*e0*nsp.exp(k_D*x)

    # if isinstance(res, nsp.Number):
    #     return float(res)
    return res

# %% Recombination & Lifetime
def U_radiative(n, p):
    B_rad = 4.73e-15
    U_radiative = n * p * B_rad
    return U_radiative


def U_radiative_alt(n0, p0, Δn, temp=298.15):
    n_p = n0 + p0 + 2 * Δn
    n = n0 + Δn
    p = p0 + Δn
    B_low = 4.73e-15
    b_min = 0.2 + (0 - 0.2) / (1 + (temp / 320) ** 2.5)
    b1 = 1.5e18 + (10000000 - 1.5e18) / (1 + (temp / 550) ** 3)
    b3 = 4e18 + (1000000000 - 4e18) / (1 + (temp / 365) ** 3.54)
    B_rel = b_min + (1 - b_min) / (
        1 + (0.5 * n_p / b1) ** 0.54 + (0.5 * n_p / b3) ** 1.25
    )
    B_rad = B_low * B_rel
    U_radiative_alt = n * p * B_rad
    return U_radiative_alt


def U_SRH(n, p, Et, τ_n, τ_p, ni_eff=8.5e9, T=298.15):
    """Return the shockley read hall recombination cm-3
    given Et (eV) trap level from intrinsic"""
    arg_in = vars().copy()
    nsp = pick_math_module(arg_in)
    w_units = has_units(arg_in)
    k_B = get_const("boltzmann", w_units, ["eV", "K"])
    n1 = ni_eff * nsp.exp(Et / (k_B * T))
    p1 = ni_eff * nsp.exp(-Et / (k_B * T))
    res = (n * p - ni_eff**2) / (τ_p * (n + n1) + τ_n * (p + p1))
    # if isinstance(res, nsp.Number):
    #     return float(res)
    return res


def U_auger_richter(n0, p0, Δn, ni_eff):
    """Return the auger recombination
    18 and 19
    https://doi.org/10.1016/j.egypro.2012.07.034"""
    arg_in = vars().copy()
    lam_res, symb_var = has_arrays(arg_in, U_auger_richter)
    if lam_res:
        return symb_var
    nsp = pick_math_module(arg_in)

    B_n0 = 2.5e-31
    C_n0 = 13
    D_n0 = 3.3e17
    exp_n0 = 0.66
    B_p0 = 8.5e-32
    C_p0 = 7.5
    D_p0 = 7e17
    exp_p0 = 0.63
    C_dn = 3e-29
    D_dn = 0.92
    g_eeh = 1 + C_n0 * (1 - nsp.tanh((n0 / D_n0) ** exp_n0))
    g_ehh = 1 + C_p0 * (1 - nsp.tanh((p0 / D_p0) ** exp_p0))
    np_ni2 = (n0 + Δn) * (p0 + Δn) - ni_eff**2
    res = np_ni2 * (B_n0 * n0 * g_eeh + B_p0 * p0 * g_ehh + C_dn * Δn**D_dn)
    # if isinstance(res, nsp.Number):
    #     return float(res)
    return res


def U_low_doping(n0, p0, Δn):
    """recombination due to Auger and radiative
    equation 21 in DOI: 10.1103/PhysRevB.86.165202"""
    B_low = 4.73e-15
    n = n0 + Δn
    p = p0 + Δn
    U = Δn / (
        n
        * p
        * (8.7e-29 * n0**0.91 + 6.0e-30 * p0**0.94 + 3.0e-29 * Δn**0.92 + B_low)
    )
    return U


def U_surface(n, p, Sn, Sp, n1=8.3e9, p1=8.3e9, **kwargs):
    """Return the carrier recombination (/s) at a surface.
    Where.
    Sn, Sp: surface recombination for electrons and holes
    n1, p1 XXX
    ni - intrinsice carrier concentratoin (cm-3)"""
    ni = kwargs.get("ni", ni_Si(kwargs.get("temp", 298.15)))
    U_surface = Sn * Sp * (n * p - ni**2) / (Sn * (n + n1) + Sp * (p + p1))
    return U_surface


def lifetime(U, Δn):
    """Return the lifetime (seconds).
    U is the recombination  and Δn is the excess minority carrier density.
    This is the definition of lifetime"""
    return Δn / U


def lifetime_eff(*lifetimes):
    """Return the lifetime (seconds).
    U is the recombination  and Δn is the excess minority carrier density.
    This is the definition of lifetime"""
    return inv_sum_invs(*lifetimes)


def lifetime_bulk(tau_eff, S, thickness):
    """Return the bulk lifetime (s)
    Given tau_eff (s)
    surface recombination (cm/s)
    thickness (cm)
    """
    return tau_eff - thickness / (2 * S)


def lifetime_minority(N, tao_0=0.001, N_ref=1e17):
    """Return the miority carrier lifetime for a given doping level"""
    return tao_0 / (1 + N / N_ref)


# not sure if I should keep these
def lifetime_auger(Δn, Ca=1.66e-30):
    """Returns the Auger lifetime (s) at high level injection
    given the injection level (cm-3)"""
    return 1 / (Ca * Δn**2)


def lifetime_SRH(N, Nt, Et, σ_n, σ_p, Δn, temp=298.15):
    # TODO needs correction
    # p0 = N
    # n0 = (ni_Si(temp) ** 2) / N
    # τ_n0 = 1 / (Nt * σ_n * vth)
    # τ_p0 = 1 / (Nt * σ_p * vth)
    # n1 = Nc * np.exp(-Et / Vt())
    # p1 = Nv * np.exp((-Et - Eg) / Vt())
    # k_ratio = σ_n / σ_p
    # τ_SRH = (τ_p0 * (n0 + n1 + Δn) + τ_n0 * (p0 + p1 + Δn)) / (n0 + p0 + Δn)
    # return τ_SRH
    return print("non-functional")
