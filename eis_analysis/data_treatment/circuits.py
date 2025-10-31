# -*- coding: utf-8 -*-
"""
Insert module description/summary.

@author: j2cle
Created on Thu Sep 19 11:17:44 2024
"""
import numpy as np

# %% Circuit manupulation utilities
CKTS = {
    "sRC": "R0-C0",
    "pRC": "p(R0,C0)",
    "Randles": "R0-p(R1,C1)",
    "Voight": "p(R0,C0)-p(R1,C1)",
    "Maxwell": "p(R0,C0,R1-C1)",
    "cVoight": "p(C0,p(R1,C1)-p(R2,C2))",
}


def rc_to_k(r, c):
    """
    Convert a parallel RC circuit to a K element.

    Parameters
    ----------
    r : float
        Resistance of RC circuit
    c : float
        Capacitance of RC circuit

    Returns
    -------
    r : float
        Resistance of K element
    t : float
        Time constant of K element
    """
    return r, r * c


def k_to_rc(r, t):
    """
    Convert a K element to a parallel RC circuit.

    Parameters
    ----------
    r : float
        Resistance of K element
    t : float
        Time constant of K element

    Returns
    -------
    r : float
        Resistance of RC circuit
    c : float
        Capacitance of RC circuit
    """
    return r, t / r


def voight_to_max(r1, c1, r2, c2):
    """
    Convert a Voight circuit to a Maxwell circuit.

    Parameters
    ----------
    r1, c1 : float
        First RC branch of Voight circuit (R and C in parallel)
    r2, c2 : float
        Second RC branch of Voight circuit (R and C in parallel)
    Returns
    -------
    ra, ca : float
        First RC branch (R and C in parallel)
    rb, cb : float
        Second RC branch (R and C in series)
    """
    sq_delta = (r1 * c1 - r2 * c2) ** 2

    ra = r1 + r2
    rb = r1 * r2 * ra * (c1 + c2) ** 2 / sq_delta if sq_delta else 0
    ca = c1 * c2 / (c1 + c2)
    cb = sq_delta / ((c1 + c2) * ra**2) if sq_delta else ca
    return ra, ca, rb, cb


def max_to_voight(ra, ca, rb, cb):
    """
    Convert a Maxwell circuit to a Voight circuit.

    Parameters
    ----------
    ra, ca : float
        First RC branch (R and C in parallel)
    rb, cb : float
        Second RC branch (R and C in series)
    Returns
    -------
    r1, c1 : float
        First RC branch of Voight circuit (R and C in parallel)
    r2, c2 : float
        Second RC branch of Voight circuit (R and C in parallel)
    """
    c_ratio = ca / cb if cb else 0
    r_ratio = rb / ra if ra else 0

    k = (c_ratio + r_ratio + 1) ** 2 - 4 * c_ratio * r_ratio

    p1 = (r_ratio - c_ratio + 1) / k ** (1 / 2) if k else 0
    if p1 == 1:
        c1, c2 = [2 * ca] * 2
    else:
        c1 = 2 * ca / (1 - p1) if (1 - p1) else 0
        c2 = 2 * ca / (1 + p1) if (1 + p1) else 0
    c12 = [c1, c2]

    p2 = (c_ratio - r_ratio + 1) / k ** (1 / 2) if k else 0
    if p2 == 1:
        r1, r2 = [ra / 2] * 2
    else:
        r1 = ra / 2 * (1 + p2) if (1 + p2) else 0
        r2 = ra / 2 * (1 - p2) if (1 - p2) else 0
    r12 = [r1, r2]

    if r12[0] * c12[0] == 0 and r12[1] * c12[1] == 0:
        return max(r12), max(c12), min(r12), min(c12)
    elif r12[0] * c12[0] >= r12[1] * c12[1]:
        return r12[0], c12[0], r12[1], c12[1]
    elif r12[0] * c12[0] <= r12[1] * c12[1]:
        return r12[1], c12[1], r12[0], c12[0]
    elif r12[0] * c12[1] >= r12[1] * c12[0]:
        return r12[0], c12[1], r12[1], c12[0]
    elif r12[1] * c12[0] >= r12[0] * c12[1]:
        return r12[1], c12[0], r12[0], c12[1]
    return max(r12), max(c12), min(r12), min(c12)


def absorb_voight_ext_cap(c0, r1, c1, r2, c2):
    """
    Absorb an external capacitance C0 into a Voight circuit,
    adjusting the other capacitances accordingly.

    Parameters
    ----------
    c0 : float
        External capacitance to absorb
    r1, c1 : float
        First RC branch
    r2, c2 : float
        Second RC branch
    Returns
    -------
    r1, c1 : float
        Adjusted first RC branch
    r2, c2 : float
        Adjusted second RC branch
    """
    ra, ca, rb, cb = voight_to_max(r1, c1, r2, c2)
    ca += c0
    return max_to_voight(ra, ca, rb, cb)


def create_voight_ext_cap(r1, c1, r2, c2, c0=0):
    """
    Insert an external capacitance C0 into a Voight circuit,
    adjusting the other capacitances accordingly.

    Parameters
    ----------
    r1, c1 : float
        First RC branch
    r2, c2 : float
        Second RC branch
    c0 : float, optional
        External capacitance to insert (default 0)
    Returns
    -------
    c0 : float
        Adjusted external capacitance
    r1, c1 : float
        Adjusted first RC branch
    r2, c2 : float
        Adjusted second RC branch
    """
    ra, ca, rb, cb = voight_to_max(r1, c1, r2, c2)
    if c0 >= ca or not c0:
        c0 = ca / 2
    ca -= c0
    return c0, *max_to_voight(ra, ca, rb, cb)


def split_rc(R, C, n=2, distribution=()):
    """
    Split a parallel RC into n parallel RC branches in series.

    Parameters
    ----------
    R : float
        Original resistance
    C : float
        Original capacitance
    n : int, optional
        Number of branches (default 2)
    distribution : sequence, optional
        Flexible specification of resistor allocation:
        - Empty (default): uses [1,2,...,n] normalized as fractions.
        - If sum(distribution) <= 1: treated as fractions of R.
          Zeros are filled with equal shares of the remaining fraction.
        - If sum(distribution) > 1: treated as explicit resistor values.
          Zeros are filled with equal shares of the remaining resistance.

    Returns
    -------
    list of (R_i, C_i)
        Parameters of each branch
    """
    n = max(2, int(n))
    dist = abs(np.array(distribution, dtype=float))

    if dist.size < n:
        dist = np.pad(dist, (0, n - dist.size), constant_values=0)
    elif dist.size > n:
        dist = dist[:n]

    # --- Explicit R interpretation ---
    if dist.sum() > 1:
        if any(d > R for d in dist) or dist.sum() > R:
            raise ValueError("Distribution values cannot exceed total R.")
        dist = dist / R

    # --- Fraction interpretation ---
    missing = np.count_nonzero(dist == 0)
    if missing > 0:
        dist_0 = 10 ** np.arange(missing - 1, -1, -1)  # e.g. [1,2,...,n]
        dist[dist == 0] = dist_0 / dist_0.sum() * (1.0 - dist.sum())

    dist = dist / dist.sum()

    # Compute R_i and C_i
    R_vals = R * dist
    C_vals = (R / R_vals) * C

    flat = np.column_stack((R_vals, C_vals)).ravel().tolist()

    return flat


def join_rc(*branches):
    """
    Join multiple RC branches in series into a single equivalent RC.

    Parameters
    ----------
    *branches : sequence
        Can be:
        - flat list/array [R1, C1, R2, C2, ...]
        - list of tuples [(R1, C1), (R2, C2), ...]
        - unpacked directly as join_rc(R1, C1, R2, C2, ...)

    Returns
    -------
    (R_eq, C_eq)
        Equivalent RC parameters
    """
    # Flatten input into a numpy array
    values = np.array(branches, dtype=float).squeeze()

    # If flat, reshape into pairs
    if values.ndim == 1:
        if len(values) % 2 != 0:
            raise ValueError("Flat input must contain an even number of elements (R,C pairs).")
        values = values.reshape(-1, 2)
    # If 2D, validate orientation
    elif values.ndim == 2 and 2 in values.shape:
        if values.shape[0] == 2 and values.shape[1] != 2:
            values = values.T  # convert (2,n) -> (n,2)
    else:
        raise ValueError(f"Unsupported array shape {values.shape}")

    values[values == 0] = 1e-32  # avoid div by zero

    R_tot = float(np.sum(values[:, 0]))
    taus = values[:, 0] * values[:, 1]

    tau_eff = np.exp(np.sum(values[:, 0] / R_tot * np.log(taus)))

    C_eq = tau_eff / R_tot
    return R_tot, C_eq


# {conv to "key": [func, existing model, new model]}
CKT_FUNCS = {
    "To K": [rc_to_k, "p(R0,C0)", "K0", ""],
    "To pRC": [k_to_rc, "K0", "p(R0,C0)", ""],
    "To Voight": [max_to_voight, "p(R0,C0,R1-C1)", "p(R0,C0)-p(R1,C1)", ""],
    "To Maxwell": [voight_to_max, "p(R0,C0)-p(R1,C1)", "p(R0,C0,R1-C1)", ""],
    "Join pRC": [join_rc, "p(R0,C0)-p(R1,C1)", "p(R0,C0)", ""],
    "Split pRC": [split_rc, "p(R0,C0)", "p(R0,C0)-p(R1,C1)", ""],
    "Voight+C0": [create_voight_ext_cap, "p(R0,C0)-p(R1,C1)", "p(C0,p(R1,C1)-p(R2,C2))", "C0"],
    "cVoight-C0": [absorb_voight_ext_cap, "p(C0,p(R1,C1)-p(R2,C2))", "p(R0,C0)-p(R1,C1)", ""],
}
