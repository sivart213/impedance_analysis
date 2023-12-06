# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 22:12:37 2022.

@author: j2cle
"""

if __name__ == "__main__":
    from research_tools.functions import f_find, p_find, save, Complex_Imp
    from eis_analysis.data_analysis import DataImport, IS_Ckt

    """
    Import data using by first getting the appropriate filename.  f_find and
    p_find search for the desired files given a list of folder names.
    DataImport handles the actual importing of data
    """

    my_folder_path = p_find(
        "impedance_analysis", "testing", "Data", base="cwd"
    )
    files = f_find(my_folder_path / "Raw", re_filter="mfia_rc")

    data_in = DataImport(files[0], tool="MFIA", read_type="full")

    """
    The impedance class wraps the complex class with terms common to impedance.
    Used internally by several of the eis modules/classes.
    """
    imp_data = Complex_Imp(data_in[data_in.keys()[0]])

    """
    Begin fitting of impedance data by first declaring the initial conditions
    needed by impedance.py
    """
    model = "R_0-p(R_1,C_1)"
    guess = [1e4, 1e8, 1e-12]
    constants = {}

    """
    Establish the ckt object. Data is retained in the object for fitting and
    refitting as well as exporting.
    """
    ckt = IS_Ckt(data_in[data_in.keys()[0]], guess, constants, model)

    """
    Call base fit (which uses impedance.py fit, which in turn uses least
    squares) on the data contained within the object.
    """
    ckt.base_fit()

    save(ckt.data, my_folder_path / "Treated", "Test_single_fit")
