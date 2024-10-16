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
    from research_tools.functions import save, find_path
    from impedance_analysis.device_control import MFIA_Freq_Sweep

    config_path = find_path("impedance_analysis", "impedance_analysis", "device_control", base="cwd")
    
    save_path = find_path("Work Docs", "Data", "Raw", "MFIA",  base=find_path("ASU Dropbox", base="drive"))

    sweep_obj = MFIA_Freq_Sweep(
        "dev6037", config_path/"config_mfia.ini", sections=["base_sweep_settings", "fast_sweep"], start=500
    )

    # biases = [-0.1, 0.1, 0]
    biases = [-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    results = sweep_obj.biased_sweep([-0.2], plot=plot_measured_data)

    # save(results, save_path, "otc_postPID_r1")