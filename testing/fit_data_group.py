# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 22:12:37 2022.

@author: j2cle
"""

if __name__ == "__main__":
    from scipy.optimize import Bounds
    from research_tools.functions import f_find, find_path
    from eis_analysis.data_analysis import IS_Data

    ckt_model = "R_0-p(R_1,C_1)"

    init_position = [1, 50, 1e-4]

    uni_bands = Bounds(
        [.01, 1, 1e-8],
        [100, 1e6, 1e-0],
        keep_feasible=True,
    )

    ls_kwargs = dict(
        ftol=1e-14,
        xtol=1e-6,
        maxfev=1e6,
        jac="3-point",
        x_scale="jac",
        loss="soft_l1",
        bounds=uni_bands,
    )

    """
    Import the data from the raw files exported from the meeasurement tool
    This provides the 1st filter, "patterns" which will only get the filenames
    which match the filter name.
    """
    my_hdf_path = find_path(
        "impedance_analysis", "testing", "Data", "Databases", base="cwd"
    )

    my_folder_path = find_path(
        "impedance_analysis", "testing", "Data", base="cwd"
    )
    files = find_files(my_folder_path / "Raw", patterns="(mfia_rc).*(seq)")

    """
    Create an object to operate on all of the available data.  This will also
    save the data into an hdf for persistant storage
    """

    test_group = IS_Data(
        "mfia_rc",
        my_hdf_path,
        model=ckt_model,
        init_pos=init_position,
        raw_files=files,
        tool="MFIA",
    )

    """
    Data can also be loaded after initialization directly via get_raw
    """

    test_group.get_raw(
        find_files(my_folder_path, patterns="(mfia_rc).*(sweep)"),
        tool="MFIA",
    )
    # %% Fitting Examples
    """
    This will run the fitting function
    """

    test_group.base_fitter(
        "mfia_rc_sequence",
        thresh=1,
        refit=False,
        bounds_by_conf=False,
        **ls_kwargs
    )
    """
    Fit a single dataset, verbose prints the values, nyquist and bode plot the
    fits. With a threshold of 1, this skips the fitting process
    """
    test_group.base_fitter(
        "mfia_rc_sequence/0",
        thresh=1,
        refit=False,
        bounds_by_conf=False,
        verbose=True,
        nyquist=True,
        bode=True,
        **ls_kwargs
    )

    test_group.base_fitter(
        "mfia_rc_sequence/0",
        thresh=0.1,
        refit=False,
        bounds_by_conf=False,
        verbose=True,
        nyquist=True,
        bode=True,
        **ls_kwargs
    )

    # test_func = lambda obj: obj.data["phase"] > 10
    test_group.base_fitter(
        "mfia_rc_sequence/0",
        thresh=0.1,
        refit=False,
        bounds_by_conf=False,
        verbose=True,
        nyquist=True,
        bode=True,
        mask=lambda obj: obj.data["phase"] < 10,
        **ls_kwargs,
    )

    test_group.base_fitter(
        "mfia_rc_sequence/0",
        thresh=0.1,
        refit=False,
        bounds_by_conf=False,
        verbose=True,
        nyquist=True,
        bode=True,
        mask=lambda obj: obj.data["freq"] < 1e4,
        **ls_kwargs,
    )
    """
    TODO: Add export examples, one for origin import, one for EISyFit
    """
