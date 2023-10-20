# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 19:15:02 2023

@author: j2cle
"""

# import logging
# import numpy as np
# import pandas as pd
# import sympy as sp
# import sympy.physics.units as su

import periodictable as pt



def g_to_atoms(element, atoms=None, grams=None):
    chemical = pt.formula(element)
    res_dict = chemical.atoms
    res_dict["res"] = 1
    if atoms is not None:
        result = atoms / pt.constants.avogadro_number * chemical.mass
    else:
        result = grams / chemical.mass * pt.constants.avogadro_number
    res_dict = {key: val * result for key, val in res_dict.items()}

    return res_dict

# %% Testing
if __name__ == "__main__":
    # examples
    # x=g_to_atoms("Na")
    Na = pt.formula('Na[23]')
    NaCl = pt.formula('NaCl')
    H2O = pt.formula('H[2]O', natural_density=0.9957)
    EVA = pt.formula('28%wt C[12]4H[1]6O[16]2 //  C[12]2H[1]4', natural_density=0.92)

    Na_mol_mass = g_to_atoms('Na[23]',1e16)["res"] / Na.density
    H2O_mol_mass = g_to_atoms('H2O',grams=2e-3)["res"] / H2O.density

    Na_EVA_wt = Na.molecular_mass / EVA.molecular_mass
    H2O_EVA_wt = H2O.molecular_mass / EVA.molecular_mass

    Si = pt.formula('Si')
    P = pt.formula('P')
