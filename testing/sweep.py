# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 19:38:34 2023

@author: j2cle

taken from the API commands log

"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def plot_measured_data(sweep_data: dict, **kwargs):
    """Plot the sweep data in bode plot."""
    _, (ax1, ax2) = plt.subplots(2, 1)

    frequency = sweep_data[kwargs.get("xkey", "frequency")]
    ax1.plot(frequency, sweep_data[kwargs.get("y1key", "realz")])
    ax2.plot(frequency, sweep_data[kwargs.get("y2key", "imagz")])

    ax1.set_title(kwargs.get("title", "Current Bode"))
    ax1.grid()
    ax1.set_ylabel(kwargs.get("y1label", kwargs.get("y1key", "Real (Ohm)")))
    ax1.set_xscale("log")

    ax2.grid()
    ax2.set_xlabel(kwargs.get("xlabel", kwargs.get("xkey", "Frequency ($Hz$)")))
    ax2.set_ylabel(kwargs.get("y2label", kwargs.get("y2key", "Imaginary (Ohm)")))
    ax2.set_xscale("log")
    ax2.autoscale()

    plt.draw()
    plt.show()

if __name__ == "__main__":
    from research_tools.functions import save, p_find
    from impedance_analysis.tool_interface import MFIA_Freq_Sweep

    config_path = p_find("impedance_analysis", "impedance_analysis", "tool_interface", base="cwd")
    save_path = p_find("Dropbox (ASU)", "Work Docs", "Data", "Raw", "MFIA", base="home")

    # sweep_obj = MFIA_Freq_Sweep(
    #     "dev6037", config_path/"config_mfia.ini", sections=["base_sweep_settings", "fast_sweep"],
    # )

    # results = sweep_obj.sweep(plot=plot_measured_data)

    # save(results, save_path, "otc_postPID_r1")