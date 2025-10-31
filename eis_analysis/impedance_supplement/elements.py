import numpy as np
from impedance.models.circuits.elements import element, circuit_elements


# %% impedance.py additions/overrides
@element(num_params=2, units=["Ohm sec^-a", ""], overwrite=True)
def ICPE(p, f):
    """defines a inverse constant phase element

    Notes
    -----
    .. math::

        Z = Q \\times (j 2 \\pi f)^-\\alpha

    where :math:`Q` = p[0] and :math:`\\alpha` = p[1].
    """
    omega = 2 * np.pi * np.array(f)
    Q, alpha = p[0], p[1]
    Z = Q * (1j * omega) ** -alpha
    return Z


# %% Constants
ELEMENTS = set(k for k in circuit_elements.keys() if k not in ["np", "p", "s"])
ELEMENT_MAP = {e.lower(): e for e in sorted(ELEMENTS, key=len)}
ELEMENT_PAIR_MAP = {
    "R": "ICPE",
    "C": "CPE",
    "ICPE": "R",
    "CPE": "C",
}
