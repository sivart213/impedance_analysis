import numpy as np
from impedance.validation import fit_linKK, eval_linKK, get_tc_distribution


def linKK(
    f: np.ndarray,
    Z: np.ndarray,
    c: float = 0.85,
    max_M: int | float = 50,
    fit_type: str = "real",
    add_cap: bool = False,
    verbose=False,
):
    """A method for implementing the Lin-KK test for validating linearity [1]

    Parameters
    ----------
    f: np.ndarray
        measured frequencies
    Z: np.ndarray of complex numbers
        measured impedances
    c: np.float
        cutoff for mu
    max_M: int
        the maximum number of RC elements
    fit_type: str
        selects which components of data are fit ('real', 'imag', or
        'complex')
    add_cap: bool
        option to add a serial capacitance that helps validate data with no
        low-frequency intercept

    Returns
    -------
    M: int
        number of RC elements used
    mu: np.float
        under- or over-fitting measure
    Z_fit: np.ndarray of complex numbers
        impedance of fit at input frequencies
    resids_real: np.ndarray
        real component of the residuals of the fit at input frequencies
    resids_imag: np.ndarray
        imaginary component of the residuals of the fit at input frequencies


    Notes
    -----

    The lin-KK method from Schönleber et al. [1] is a quick test for checking
    the
    validity of EIS data. The validity of an impedance spectrum is analyzed by
    its reproducibility by a Kramers-Kronig (KK) compliant equivalent circuit.
    In particular, the model used in the lin-KK test is an ohmic resistor,
    :math:`R_{Ohm}`, and :math:`M` RC elements.

    .. math::

        \\hat Z = R_{Ohm} + \\sum_{k=1}^{M} \\frac{R_k}{1 + j \\omega \\tau_k}

    The :math:`M` time constants, :math:`\\tau_k`, are distributed
    logarithmically,

    .. math::
        \\tau_1 = \\frac{1}{\\omega_{max}} ; \\tau_M = \\frac{1}{\\omega_{min}}
        ; \\tau_k = 10^{\\log{(\\tau_{min}) + \\frac{k-1}{M-1}\\log{{(
            \\frac{\\tau_{max}}{\\tau_{min}}}})}}

    and are not fit during the test (only :math:`R_{Ohm}` and :math:`R_{k}`
    are free parameters).

    In order to prevent under- or over-fitting, Schönleber et al. propose using
    the ratio of positive resistor mass to negative resistor mass as a metric
    for finding the optimal number of RC elements.

    .. math::

        \\mu = 1 - \\frac{\\sum_{R_k \\ge 0} |R_k|}{\\sum_{R_k < 0} |R_k|}

    The argument :code:`c` defines the cutoff value for :math:`\\mu`. The
    algorithm starts at :code:`M = 3` and iterates up to :code:`max_M` until a
    :math:`\\mu < c` is reached. The default of 0.85 is simply a heuristic
    value based off of the experience of Schönleber et al., but a lower value
    may give better results.

    If the argument :code:`c` is :code:`None`, then the automatic determination
    of RC elements is turned off and the solution is calculated for
    :code:`max_M` RC elements. This manual mode should be used with caution as
    under- and over-fitting should be avoided.

    [1] Schönleber, M. et al. A Method for Improving the Robustness of
    linear Kramers-Kronig Validity Tests. Electrochimica Acta 131, 20–27 (2014)
    `doi: 10.1016/j.electacta.2014.01.034
    <https://doi.org/10.1016/j.electacta.2014.01.034>`_.

    """
    M = max_M
    ts = get_tc_distribution(f, M)
    elements, mu0 = fit_linKK(f, ts, M, Z, fit_type, add_cap)

    if c is not None:
        M = 0
        mu = 1
        while mu > c and M < max_M:
            M += 1
            ts = get_tc_distribution(f, M)
            elements, mu = fit_linKK(f, ts, M, Z, fit_type, add_cap)

            if verbose and M % 10 == 0:
                Z_fit = eval_linKK(elements, ts, f)
                print(M, mu, np.linalg.norm(Z_fit - Z) / np.sqrt(len(Z_fit)))
    Z_fit = eval_linKK(elements, ts, f)
    model, params = parse_linKK(elements, ts)

    return M, mu, Z_fit, model, params


def get_kk_dist(f, Z, M, fit_type="real", add_cap=False):
    """
    A method for extracting the KK-compliant distribution from EIS data

    Parameters
    ----------
    f: np.ndarray
        measured frequencies
    Z: np.ndarray of complex numbers
        measured impedances
    M: int
        number of RC elements to use
    fit_type: str
        selects which components of data are fit ('real', 'imag', or 'complex')
    add_cap: bool
        option to add a serial capacitance that helps validate data with no
        low-frequency intercept

    Returns
    -------
    model: str
        circuit model string of the fitted distribution
    params: dict
        dictionary of fitted parameters
    Z_fit: np.ndarray of complex numbers
        impedance of fit at input frequencies
    """
    ts = get_tc_distribution(f, M)
    elements, _ = fit_linKK(f, ts, M, Z, fit_type, add_cap)

    model, params = parse_linKK(elements, ts)
    Z_fit = eval_linKK(elements, ts, f)
    return model, params, Z_fit


def parse_linKK(elements, ts):
    """Builds a circuit of RC elements to be used in LinKK"""
    model = "R0-L0-C0" if elements.size == (ts.size + 3) else "R0-L0"
    params = {}
    params["R0"] = elements[0]
    params["L0"] = elements[-1]
    if "C0" in model:
        params["C0"] = 1 / elements[-2]
    for i, tau in enumerate(ts):
        params[f"K{i}_0"] = elements[i + 1]
        params[f"K{i}_1"] = tau
        model += f"-K{i}"
    return model, params


# # # %% Main
# # if __name__ == "__main__":
# #     try:
# #         from ..utils.common import RCCircuit
# #     except ImportError:
# #         from testing.rc_ckt_sim import RCCircuit
# #     ckt = RCCircuit()
# #     res = linKK(ckt.freq, ckt.Z, c=0.5, max_M=200, fit_type="complex", add_cap=False)

# # shift_elem_num("R1-C1", 0, "R0")
# # validate_model("R0-p(p(R1,C1)-p(R2,C2),R3-C3)")
# # test = voight_to_max(1e9, 1e-10, 1e9, 1e-10)
# # max_to_voight(2000000000.0, 5e-11, 2000000000.0, 0.0)
# # parse_model_groups("p(C4,p(R4),p(,p(C3,R3-p(C5,R5))-R2))-p(R1,C1)", False)
# # parse_model_groups("p(C4,p(R4),p(,p(C3,R3-p(C5,R5))-R2))", False)
