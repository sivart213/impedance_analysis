# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
import os
import re

from pathlib import Path

import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as style

from matplotlib import ticker
from matplotlib.widgets import Slider, Button

# from ..system_utilities import (
#     find_path,
#     find_files,
# )

# from ..data_treatment import (
#     sig_figs_ceil,
# )

# from ..string_operations import (
#     slugify,
#     sci_note,
# )


def sig_figs_ceil(number, digits=3):
    """Round based on desired number of digits."""
    digits = digits - 1
    power = "{:e}".format(number).split("e")[1]
    root = 10 ** (int(power) - digits)
    return np.ceil(number / root) * root

def sci_note(num, prec=2):
    """Return formatted text version of value."""
    fmt = "{:.%dE}" % int(prec)
    return fmt.format(num)


def measured_data_bode(sweep_data: dict, **kwargs):
    """Plot the sweep data in bode plot."""
    _, (ax1, ax2) = plt.subplots(2, 1)

    frequency = sweep_data[kwargs.get("xkey", "frequency")]
    decades = [10**int(np.log10(f)) for f in frequency[~np.isnan(frequency)]]
    ax1.scatter(
        x=frequency[~np.isnan(frequency)],
        y=sweep_data[kwargs.get("y1key", "realz")][~np.isnan(frequency)],
        c=decades,
        )
    ax2.scatter(
        x=frequency[~np.isnan(frequency)], 
        y=sweep_data[kwargs.get("y2key", "imagz")][~np.isnan(frequency)],
        c=decades,
        )

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
    

def measured_data_nyquist(sweep_data: dict, **kwargs):
    """Plot the sweep data in bode plot."""
    frequency = sweep_data[kwargs.get("xkey", "frequency")]
    decades = [10**int(np.log10(f)) for f in frequency[~np.isnan(frequency)]]
    
    fig, ax = plt.subplots()
    plt.scatter(
        x=sweep_data[kwargs.get("y1key", "realz")][~np.isnan(frequency)],
        y=-1*sweep_data[kwargs.get("y2key", "imagz")][~np.isnan(frequency)],
        c=[10**int(np.log10(f)) for f in frequency[~np.isnan(frequency)]],
        norm=plt.matplotlib.colors.LogNorm(),
        # legend=False,
        edgecolor="none",
        # ax=ax,
    )

    max_val=max([max(abs(sweep_data[kwargs.get("y1key", "realz")])),max(abs(sweep_data[kwargs.get("y2key", "imagz")]))])

    ax.set_title(kwargs.get("title", "Current Nyquist"))
    ax.set_aspect("equal", adjustable="box", anchor="SW", share=True)
    ax.set_xlim(
        0, sig_figs_ceil(max_val * 1.25, 2)
    )
    ax.set_ylim(
        0, sig_figs_ceil(max_val * 1.25, 2)
    )
    ax.grid()

    plt.draw()
    plt.show()

def plot_measured_data(sweep_data: dict, **kwargs):
    """Plot the sweep data in bode plot."""
    measured_data_bode(sweep_data, **kwargs)
    measured_data_nyquist(sweep_data, **kwargs)


def get_style(styl_str):
    if styl_str in style.available:
        style.use(styl_str)
    else:
        has_any = [
            sum([st in av for st in re.split("[-_]", styl_str)])
            for av in style.available
        ]
        res = np.array(style.available)[[n == max(has_any) for n in has_any]]
        style.use(res[0]) if len(res) >= 1 else None
    return


def map_plt(
    x,
    y,
    z,
    xscale="linear",
    yscale="linear",
    zscale="log",
    xlimit=[0, 0],
    ylimit=[0, 0],
    zlimit=[0, 0],
    levels=50,
    name="",
    xname="X",
    yname="Y",
    zname="Z",
    ztick=10,
    save=None,
    show=True,
    **kwargs,
):
    """
    Create a map plot using contourf of matplotlib.

    Parameters:
    x (array-like): Data for the x-axis.
    y (array-like): Data for the y-axis.
    z (array-like): Data for the z-axis (contour levels).
    xscale (str, optional): Scale for the x-axis. Default is "linear".
    yscale (str, optional): Scale for the y-axis. Default is "linear".
    zscale (str, optional): Scale for the z-axis. Default is "log".
    xlimit (list, optional): Limits for the x-axis. Default is [0, 0], which auto-scales based on data.
    ylimit (list, optional): Limits for the y-axis. Default is [0, 0], which auto-scales based on data.
    zlimit (list, optional): Limits for the z-axis. Default is [0, 0], which auto-scales based on data.
    levels (int, optional): Number of contour levels. Default is 50.
    name (str, optional): Title of the plot. Default is an empty string.
    xname (str, optional): Label for the x-axis. Default is "X".
    yname (str, optional): Label for the y-axis. Default is "Y".
    zname (str, optional): Label for the z-axis. Default is "Z".
    ztick (int, optional): Interval for z-axis ticks. Default is 10.
    save (str or None, optional): File path to save the plot. Default is None.
    show (bool, optional): Whether to display the plot. Default is True.
    **kwargs: Additional keyword arguments for the plotting function.

    Returns:
    None
    """
    get_style("seaborn-colorblind")

    if xlimit == [0, 0]:
        xlimit = [min(x), max(x)]
    if ylimit == [0, 0]:
        ylimit = [min(y), max(y)]
    if zlimit[0] <= 0:
        zlimit = [z[z > 0].min(), z[z > 0].max()]

    if "log" in zscale:
        lvls = np.logspace(np.log10(zlimit[0]), np.log10(zlimit[1]), levels)
        tick_loc = ticker.LogLocator()
    else:
        lvls = np.linspace(zlimit[0], zlimit[1], levels)
        tick_loc = ticker.MaxNLocator()

    fig, ax = plt.subplots()
    csa = ax.contourf(
        x,
        y,
        z,
        lvls,
        locator=tick_loc,
        **kwargs,
    )

    ax.set_xlabel(xname, fontname="Arial", fontsize=18, fontweight="bold")
    ax.set_xlim(xlimit[0], xlimit[1])
    ax.set_ylabel(yname, fontname="Arial", fontsize=18, fontweight="bold")
    ax.set_ylim(ylimit[0], ylimit[1])
    # if "both" in logs.lower() or "x" in logs.lower():
    ax.set_xscale(xscale)
    # if "both" in logs.lower() or "y" in logs.lower():
    ax.set_yscale(yscale)
    ax.set_title(name, fontname="Arial", fontsize=20, fontweight="bold")

    for tick in ax.get_xticklabels():
        tick.set_fontname("Arial")
        tick.set_fontweight("bold")
        tick.set_fontsize(12)
    for tick in ax.get_yticklabels():
        tick.set_fontname("Arial")
        tick.set_fontweight("bold")
        tick.set_fontsize(12)

    if zscale == "log":
        cbar = fig.colorbar(csa)
        cbar.locator = ticker.LogLocator(**ztick)
        cbar.set_ticks(cbar.locator.tick_values(zlimit[0], zlimit[1]))
    elif zscale == "linlog":
        cbar = fig.colorbar(csa, format=ticker.LogFormatter(**ztick))
    else:
        cbar = fig.colorbar(csa)
    cbar.minorticks_off()
    cbar.set_label(zname, fontname="Arial", fontsize=18, fontweight="bold")
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontname("Arial")
        tick.set_fontweight("bold")
        tick.set_fontsize(12)
    plt.tight_layout()

    if save is not None:
        if isinstance(save, Path):
            save = str(save)
        if not os.path.exists(save):
            os.makedirs(save)
        new_name = re.sub(r"[<>:\"/\\|?*\x00-\x1F]", "_", name)
        plt.savefig(save / f"{new_name}.png")

    if not show:
        plt.close()

    return


def scatter(
    data,
    x="index",
    y=None,
    xscale="linear",
    yscale="linear",
    xlimit=None,
    ylimit=None,
    name="Residual Plot",
    xname=None,
    yname=None,
    zname=None,
    hline=None,
    colorbar=False,
    save=None,
    show=True,
    fig=None,
    ax=None,
    grid=False,
    **kwargs,
):
    """
    Create a scatter plot using seaborn.

    Parameters:
    data (pd.DataFrame or array-like): Input data for the plot.
    x (str or array-like): Data for the x-axis.
    y (str or array-like, optional): Data for the y-axis. Default is None, which uses the first column of data.
    xscale (str, optional): Scale for the x-axis. Default is "linear".
    yscale (str, optional): Scale for the y-axis. Default is "linear".
    xlimit (list or None, optional): Limits for the x-axis. Default is None, which auto-scales based on data.
    ylimit (list or None, optional): Limits for the y-axis. Default is None, which auto-scales based on data.
    title (str, optional): Title of the plot. Default is None.
    xlabel (str, optional): Label for the x-axis. Default is None.
    ylabel (str, optional): Label for the y-axis. Default is None.
    yname (str, optional): Name for the y-axis. Default is None.
    zname (str, optional): Name for the z-axis. Default is None.
    hline (float or None, optional): Position of a horizontal line to add to the plot. Default is None.
    colorbar (bool, optional): Whether to include a colorbar. Default is False.
    save (str or None, optional): File path to save the plot. Default is None.
    show (bool, optional): Whether to display the plot. Default is True.
    fig (matplotlib.figure.Figure or None, optional): Figure object to use for the plot. Default is None.
    ax (matplotlib.axes.Axes or None, optional): Axes object to use for the plot. Default is None.
    grid (bool, optional): Whether to include a grid in the plot. Default is False.
    **kwargs: Additional keyword arguments for the plotting function.

    Returns:
    None
    """
    try:
        import seaborn as sns
        seaborn_installed = True
    except ImportError:
        seaborn_installed = False
    
    get_style("seaborn-colorblind")

    sns.set_theme(context="talk", style="dark")

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    if x == "index":
        data["index"] = data.index
    if y is None:
        y = data.columns[0]

    if xlimit is None and all(x != data.index):
        xlimit = [data[x].min(), data[x].max()]
    elif xlimit is None:
        xlimit = [data.index.min(), data.index.max()]

    if ylimit is None:
        ylimit = [data[y].min(), data[y].max()]

    if xname is None:
        xname = x
    if yname is None:
        yname = y

    if fig is None and ax is None:
        fig, ax = plt.subplots()
    # Loading the dataset into the variable 'dataset'
    # Graph is created and stored in the variable 'graph' *added ax to g
    g = sns.scatterplot(x=x, y=y, data=data, ax=ax, **kwargs)
    # Drawing a horizontal line at point 1.25
    g.set(xlim=xlimit, ylim=ylimit, xscale=xscale, yscale=yscale)
    try:
        if hline is not None and (yscale != "log" or any(np.array(hline) > 0)):
            for h in hline:
                ax.axhline(h, color="k", linestyle=":")
    except TypeError:
        if hline is not None and (yscale != "log" or hline > 0):
            ax.axhline(hline, color="k", linestyle=":")

    # Labels and their fonts
    g.set_xlabel(xname, fontname="serif", fontsize=18, fontweight="bold")
    g.set_ylabel(yname, fontname="serif", fontsize=18, fontweight="bold")
    ax.set_title(name, fontname="serif", fontsize=20, fontweight="bold")

    # Adjust the font of the ticks
    for tick in ax.get_xticklabels():
        tick.set_fontname("serif")
        tick.set_fontweight("bold")
        tick.set_fontsize(12)
    for tick in ax.get_yticklabels():
        tick.set_fontname("serif")
        tick.set_fontweight("bold")
        tick.set_fontsize(12)

    if colorbar:
        if zname is None:
            zname = kwargs["hue"]
        norm = plt.Normalize(data[kwargs["hue"]].min(), data[kwargs["hue"]].max())
        sm = plt.cm.ScalarMappable(cmap=kwargs["palette"], norm=norm)
        sm.set_array([])

        # Remove the legend and add a colorbar
        ax.get_legend().remove()
        # ax.figure.colorbar(sm)
        cbar = ax.figure.colorbar(sm)
        cbar.set_label(zname, fontname="serif", fontsize=18, fontweight="bold")
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontname("serif")
            tick.set_fontweight("bold")
            tick.set_fontsize(12)

    ax.grid(grid)
    # The plot is shown
    plt.tight_layout()
    if save is not None:
        if isinstance(save, Path):
            save = str(save)
        if not os.path.exists(save):
            os.makedirs(save)
        new_name = re.sub(r"[<>:\"/\\|?*\x00-\x1F]", "_", name)
        plt.savefig(save / f"{new_name}.png")

    if not show:
        plt.close()


def bode(
    freq="freq",
    top="mag",
    bot="phase",
    scatter_data=None,
    line_data=None,
    band=None,
    bmin="min",
    bmax="max",
    title="bode",
    pad=1.25,
    return_fig=False,
    return_plt=False,
    fig=None,
    ax1=None,
    ax2=None,
    labels=None,
    label1="_none",
    label2="_none",
):
    """
    Generate a Bode plot. If freq, top, and bot are array-like, defaults to scatter plot. 

    Parameters:
    - freq (str or array-like, optional): Frequency data (x data).
    - top (str or array-like): Column name or array for top subplot data (y1).
    - bot (str or array-like): Column name or array for bottom subplot data (y2).
    - scatter_data (DataFrame, optional): Data for scatter plot.
    - line_data (DataFrame, optional): Data for line plot.
    - band (DataFrame, optional): Data for confidence band.
    - bmin (str): Column name for lower bound of band.
    - bmax (str): Column name for upper bound of band.
    - title (str): Title of the plot.
    - pad (float): Padding for axis limits.
    - return_plt (bool): If True, return the plt object.
    - return_fig (bool): If True, return the fig and ax objects.

    Returns:
    - plt or (fig, ax1, ax2) or None: Depending on return_plt and return_fig.
    """
    if scatter_data is None and line_data is None and not isinstance(freq, (np.ndarray, list)):
        return None

    get_style("seaborn-colorblind")

    if labels is None:
        labels = {
            "freq": "Frequency Hz",
            "real": "Z' [$\Omega$]",
            "imag": "Z'' [$\Omega$]",
            "inv_imag": "-Z'' [$\Omega$]",
            "mag": "|Z| [$\Omega$]",
            "phase": r"$\varphi$ [deg]",
            "inv_phase": r"-$\varphi$  [deg]",
        }
    data = None
    # Begin plot generation
    if ax1 is None or ax2 is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    if scatter_data is not None: # freq, top, and bot are column names in scatter_data
        ax1.scatter(x=freq, y=top, data=scatter_data, edgecolor="none", label=label1)
        ax2.scatter(x=freq, y=bot, data=scatter_data, edgecolor="none", label=label1)
        data = scatter_data
    elif line_data is None: # freq, top, and bot are arrays
        ax1.scatter(x=freq, y=top, edgecolor="none", label=label1)
        ax2.scatter(x=freq, y=bot, edgecolor="none", label=label1)
        data = pd.DataFrame({"freq": freq, "top": top, "bot": bot})
        top = "y1"
        bot = "y2"

    if line_data is not None: # freq, top, and bot are column names in line_data
        ax1.plot(freq, top, "r", data=line_data, label=label2)
        ax2.plot(freq, bot, "r", data=line_data, label=label2)
        if scatter_data is None:
            data = line_data

    if band is not None:
        try:
            ax1.fill_between(
                data["freq"],
                band[top][bmin],
                band[top][bmax],
                color="r",
                alpha=0.4,
            )
            ax2.fill_between(
                data["freq"],
                band[bot][bmin],
                band[bot][bmax],
                color="r",
                alpha=0.4,
            )
        except KeyError:
            pass
    
    # function to set y scale
    def set_y_scale(ax, scale_type, df, key, pad=1.25, inv=1, label_check="Z"):
        if scale_type == "log":
            ax.set_yscale("log")
            ax.set_ylim([
                10 ** np.floor(np.log10(df[key].min())-abs(pad-1)),
                10 ** np.ceil(np.log10(df[key].max())+abs(pad-1)),
            ])
        else:
            if "Z" not in label_check or "Y" not in label_check:
                ax.set_yscale("linear")
                ax.set_ylim([df[key].min(), df[key].max()])
            else:
                ax.set_yscale("linear")
                ax.set_ylim([
                    min(0, inv * sig_figs_ceil((inv * df[key]).max() * pad, 2)),
                    max(0, inv * sig_figs_ceil((inv * df[key]).max() * pad, 2)),
                ])

    # Begin formatting plot
    ax1.set_xscale("log")
    ax1.set_xlim([data["freq"].min(), data["freq"].max()])
    set_y_scale(ax1, "log" if "mag" in top.lower() else "linear", data, top, pad, 1, labels[top])
    ax2.set_xscale("log")
    ax2.set_xlim([data["freq"].min(), data["freq"].max()])
    # set_y_scale(ax1, "log" if "mag" in top.lower() else "linear", data, bot, pad)

    if "phase" in bot.lower():
        ax2.yaxis.set_ticks(np.arange(-90, 120, 30))
        ax2.set_ylim(-100, 100)
    elif "imag" in bot.lower():
        inv = 1 if "inv" in bot.lower() else -1
        set_y_scale(ax2, "linear", data, bot, pad, inv, labels[bot])
    else:
        set_y_scale(ax2, "log", data, bot, pad)

    
    font_props = {"fontname": "Arial", "fontsize": 18, "fontweight": "bold"}
    ax2.set_xlabel(labels[freq], **font_props)
    ax1.set_ylabel(labels[top] if top in labels.keys() else str(top), **font_props)
    ax2.set_ylabel(labels[bot] if bot in labels.keys() else str(bot), **font_props)
    ax1.set_title(title, **font_props)
    ax1.tick_params(axis='both', which='major', labelsize=16, labelfontfamily="Arial")
    ax2.tick_params(axis='both', which='major', labelsize=16, labelfontfamily="Arial")
    ax1.grid(True)
    ax2.grid(True)
    plt.tight_layout()

    if return_plt:
        return plt
    elif return_fig:
        return fig, (ax1, ax2)
    else:
        plt.show()


def nyquist(
    x="real",
    y="imag",
    c="freq",
    scatter_data=None,
    line_data=None,
    band=None,
    bmin="min",
    bmax="max",
    title="Nyquist",
    pad=1.25,
    return_fig=False,
    return_plt=False,
    fig=None,
    ax=None,
    labels=None,
):
    """
    Generate a Nyquist plot.

    Parameters:
    - x (str or array-like, optional): X-axis data.
    - y (str or array-like): Y-axis data.
    - c (str or array-like): Color data.
    - scatter_data (DataFrame, optional): Data for scatter plot.
    - line_data (DataFrame, optional): Data for line plot.
    - band (DataFrame, optional): Data for confidence band.
    - bmin (str): Column name for lower bound of band.
    - bmax (str): Column name for upper bound of band.
    - title (str): Title of the plot.
    - pad (float): Padding for axis limits.
    - return_plt (bool): If True, return the plt object.
    - return_fig (bool): If True, return the fig and ax objects.

    Returns:
    - plt or (fig, ax) or None: Depending on return_plt and return_fig.
    """
    if scatter_data is None and line_data is None and not isinstance(x, (np.ndarray, list)):
        return None

    get_style("seaborn-colorblind")
    
    if labels is None:
        labels = {
            "freq": "Frequency Hz",
            "real": "Z' [$\Omega$]",
            "imag": "Z'' [$\Omega$]",
            "inv_imag": "-Z'' [$\Omega$]",
            "mag": "|Z| [$\Omega$]",
            "phase": r"$\varphi$ [deg]",
            "inv_phase": r"-$\varphi$  [deg]",
        }

    # Begin plot generation
    if ax is None:
        fig, ax = plt.subplots()

    data_both = False
    edge_c = "black"
    
    data = None
    if scatter_data is not None: # x and y are column names in scatter_data
        scatter_data = scatter_data.copy().reset_index(drop=True)
        # if "inv" not in y or (scatter_data[y] < 0).mean() > 0.5:
        if (scatter_data[y] < 0).mean() > 0.5:
            scatter_data[y] = -scatter_data[y]

        if c not in scatter_data.columns:
            scatter_data[c] = np.arange(len(scatter_data))
        
        ax.scatter(
            x=scatter_data[x],
            y=scatter_data[y],
            c=scatter_data[c],
            cmap="plasma",
            norm=plt.matplotlib.colors.LogNorm(),
            edgecolor="none",
        )
        data = scatter_data
        
        edge_c = "grey"
    
    elif line_data is None: # x and y are arrays
        if (y < 0).mean() > 0.5:
            y = -1*y

        if c is not None and not isinstance(c, type(x)):
            c = np.arange(len(x))
        
        ax.scatter(
            x=x,
            y=y,
            c=c,
            cmap="plasma",
            norm=plt.matplotlib.colors.LogNorm(),
            edgecolor="none",
        )
        data = pd.DataFrame({"x": x, "y": y, "c": c})
        x = "x"
        y = "y"
        c = "c"

    if line_data is not None: # x and y are column names in line_data
        line_data = line_data.copy().reset_index(drop=True)

        # if "inv" not in y or (line_data[y] < 0).mean() > 0.5:
        if (line_data[y] < 0).mean() > 0.5:
            line_data[y] = -line_data[y]
        
        ax.plot(line_data[x], line_data[y], "r")
        
        data_both = True
        if scatter_data is None:
            data = line_data
            data_both = False

    if band is not None:
        ax.fill_between(band[x], band[bmin], band[bmax], color="r", alpha=0.5)
    # ax.set_xticks(ax.get_yticks())
    # begin formatting plot
    upper_lim = sig_figs_ceil(data[[x, y]].abs().max().max() * pad, 2)
    ax.set_xlim(0, upper_lim)
    ax.set_ylim(0, upper_lim)
    ax.set_aspect("equal", adjustable="box", anchor="C", share=True)
    ax.set_xticks(ax.get_yticks())
    ax.set_yticks(ax.get_xticks())

    font_props = {"fontname": "Arial", "fontsize": 18, "fontweight": "bold"}
    ax.set_xlabel(labels[x], **font_props)
    ax.set_ylabel(labels[y], **font_props)
    ax.set_title(title, **font_props)
    ax.tick_params(axis='both', which='major', labelsize=16, labelfontfamily="Arial")
    ax.grid(True)
    plt.tight_layout()


    d_labels = np.unique(np.floor(np.log10(data[c])), return_index=True)[1]
    d_labels = sorted(d_labels, key=lambda i: data.loc[i, c], reverse=True)  # Sort d_labels by frequency in descending order
    # xy_positions = []
    # label_positions = []
    # min_distance = upper_lim * 0.05  # Minimum distance between d_labels
    
    label_positions = add_labels(ax, data[[x,y,c]], d_labels, upper_lim * 0.05 , upper_lim * 0.1, 10, arrowprops={"edgecolor": edge_c})
    
    if data_both:
        d_labels = np.unique(np.floor(np.log10(line_data[c])), return_index=True)[1]
        d_labels = sorted(d_labels, key=lambda i: line_data.loc[i, c], reverse=True)  
        add_labels(ax, line_data[[x,y,c]], d_labels, upper_lim * 0.05 , upper_lim * 0.1, 30, label_positions=label_positions, color="red", arrowprops={"edgecolor": "red"})


    if return_plt:
        return plt
    elif return_fig:
        return fig, ax
    else:
        plt.show()

def add_labels(ax, data, labels, min_distance, mag_off, base_angle, **kwargs):
    """
    Add labels to a plot.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes to add labels to.
    - data (pd.DataFrame or np.ndarray): A 2-column DataFrame or NumPy array with x and y data.
    - labels (list): Indices of the data points to label.
    - min_distance (float): Minimum distance between labels.
    - mag_off (float): Magnitude offset for label positioning.
    - base_angle (float): Base angle for label positioning.
    - **kwargs: Additional keyword arguments for annotation.

    Returns:
    - label_positions (list): List of label positions.
    """
    label_positions = kwargs.pop('label_positions', [])
    xy_positions = []

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    # Get window from kwargs or derive from data
    window = kwargs.pop('window', None)
    if window is None:
        window = ((data.iloc[:, 0].min(), data.iloc[:, 0].max()), (data.iloc[:, 1].min(), data.iloc[:, 1].max()))

    arrowprops = kwargs.pop("arrowprops", {})
    bbox = kwargs.pop("bbox", {})
    
    for label in labels:
        x, y = data.iloc[label, :2]
        
        # Drop label if the data position has effectively already been labeled
        if any(np.linalg.norm(np.array([x, y]) - np.array(pos)) <= min_distance for pos in xy_positions):
            continue

        angle = base_angle
        iteration_count = 0
        max_iterations = 100  # Set a reasonable iteration limit
        while iteration_count < max_iterations:
            dx = mag_off * np.cos(np.radians(angle))
            dy = mag_off * np.sin(np.radians(angle))
            new_position = (x + dx, y + dy)
            
            if all(np.linalg.norm(np.array(new_position) - np.array(pos)) > min_distance for pos in label_positions):
                # Ensure the label position is inside the window
                if (window[0][0] <= new_position[0] <= window[0][1]) and (window[1][0] <= new_position[1] <= window[1][1]):
                    # Perturb the label away from the data points
                    if not any(np.linalg.norm(np.array(new_position) - np.array([dx, dy])) < min_distance for dx, dy in data.iloc[:, :2].values):
                        break
            
            angle += 10
            if angle >= 360:
                angle = 0
                mag_off *= 1.1
            
            iteration_count += 1

        # if iteration_count >= max_iterations:
        #     print(f"Warning: Maximum iterations reached for label at ({x}, {y}).")

        ax.annotate(
            int(np.floor(np.log10(data.iloc[label, 2]))),
            xy=(x, y),
            xytext=new_position,
            textcoords="data",
            ha='center',
            arrowprops={**dict(facecolor='none', edgecolor="black", arrowstyle="->", linewidth=2.5), **arrowprops},
            bbox={**dict(boxstyle="circle,pad=0.2", edgecolor="none", facecolor="white", alpha=0.7), **bbox},
            **kwargs,
        )
        xy_positions.append((x, y))
        label_positions.append(new_position)

    return label_positions
  
# def add_labels(ax, data, labels, min_distance, mag_off, base_angle, **kwargs):
#     """
#     Add labels to a plot.

#     Parameters:
#     - ax (matplotlib.axes.Axes): The axes to add labels to.
#     - data (pd.DataFrame or np.ndarray): A 2-column DataFrame or NumPy array with x and y data.
#     - labels (list): Indices of the data points to label.
#     - min_distance (float): Minimum distance between labels.
#     - mag_off (float): Magnitude offset for label positioning.
#     - base_angle (float): Base angle for label positioning.
#     - **kwargs: Additional keyword arguments for annotation.

#     Returns:
#     - label_positions (list): List of label positions.
#     """
#     label_positions = kwargs.pop('label_positions', [])
#     xy_positions = []

#     if not isinstance(data, pd.DataFrame):
#         data = pd.DataFrame(data)

#     arrowprops = kwargs.pop("arrowprops", {})
#     bbox = kwargs.pop("bbox", {})
    
#     for label in labels:
#         x, y = data.iloc[label, :2]
#         label_position = (x, y)
        
#         # Ensure labels are not too close to each other
#         if all(np.linalg.norm(np.array(label_position) - np.array(pos)) > min_distance for pos in xy_positions):
#             angle = base_angle
#             while True:
#                 dx = mag_off * np.cos(np.radians(angle))
#                 dy = mag_off * np.sin(np.radians(angle))
#                 new_position = (x + dx, y + dy)
                
#                 if all(np.linalg.norm(np.array(new_position) - np.array(pos)) > min_distance for pos in label_positions):
#                     break
                
#                 angle += 10
#                 if angle >= 360:
#                     angle = 0
#                     mag_off *= 1.1
            
#             # Perturb the label away from the data points
#             while any(np.linalg.norm(np.array(new_position) - np.array([dx, dy])) < min_distance for dx, dy in data.iloc[:, :2].values):
#                 new_position = (new_position[0] + mag_off * 0.1, new_position[1] + mag_off * 0.1)

#             ax.annotate(
#                 int(np.floor(np.log10(data.iloc[label, 2]))),
#                 xy=(x, y),
#                 xytext=new_position,
#                 textcoords="data",
#                 ha='center',
#                 arrowprops={**dict(facecolor='none', edgecolor="black", arrowstyle="->", linewidth=2.5), **arrowprops},
#                 bbox={**dict(boxstyle="circle,pad=0.2", edgecolor="none", facecolor="white", alpha=0.7), **bbox},
#                 **kwargs,
#             )
#             xy_positions.append((x, y))
#             label_positions.append(new_position)

#     return label_positions

# def add_labels(ax, data, labels, min_distance, mag_off, base_angle, **kwargs):
#     """
#     Add labels to a plot.

#     Parameters:
#     - ax (matplotlib.axes.Axes): The axes to add labels to.
#     - data (pd.DataFrame or np.ndarray): A 2-column DataFrame or NumPy array with x and y data.
#     - labels (list): Indices of the data points to label.
#     - min_distance (float): Minimum distance between labels.
#     - mag_off (float): Magnitude offset for label positioning.
#     - base_angle (float): Base angle for label positioning.
#     - label_positions (list, optional): List of existing label positions to avoid overlap.
#     - **kwargs: Additional keyword arguments for annotation.

#     Returns:
#     - label_positions (list): List of label positions.
#     """
#     label_positions = kwargs.pop('label_positions', [])
#     xy_positions = []

#     if not isinstance(data, (pd.DataFrame)):
#         data = pd.DataFrame(data)

#     arrowprops = kwargs.pop("arrowprops", {})
#     bbox = kwargs.pop("bbox", {})
    
#     for label in labels:
#         x, y = data.iloc[label, :2]
#         label_position = (x, y)
        
#         # Ensure labels are not too close to each other
#         if all(np.linalg.norm(np.array(label_position) - np.array(pos)) > min_distance for pos in label_positions):
#             angle = base_angle
#             while True:
#                 dx = mag_off * np.cos(np.radians(angle))
#                 dy = mag_off * np.sin(np.radians(angle))
#                 new_position = (x + dx, y + dy)
                
#                 if all(np.linalg.norm(np.array(new_position) - np.array(pos)) > min_distance for pos in label_positions):
#                     break
                
#                 angle += 10
#                 if angle >= 360:
#                     angle = 0
#                     mag_off *= 1.1
            
#             ax.annotate(
#                 int(np.floor(np.log10(data.iloc[label, 2]))),
#                 xy=(x, y),
#                 xytext=new_position,
#                 textcoords="data",
#                 ha='center',
#                 arrowprops={**dict(facecolor='none', edgecolor="black", arrowstyle="->", linewidth=2.5), **arrowprops},
#                 bbox={**dict(boxstyle="circle,pad=0.2", edgecolor="none", facecolor="white", alpha=0.7), **bbox},
#                 **kwargs,
#             )
#             xy_positions.append((x, y))
#             label_positions.append(new_position)


#         #     ax.annotate(
#         #     int(np.floor(np.log10(data.loc[label, c]))),
#         #     (x_pos, y_pos),
#         #     (lx_pos, ly_pos),
#         #     textcoords = "data",
#         #     arrowprops = {**dict(facecolor='none', edgecolor="black", arrowstyle="->", linewidth=2.5), **arrowprops},
#         #     fontsize = 12,
#         #     bbox = {**dict(boxstyle="circle,pad=0.2", edgecolor="none", facecolor="white", alpha=0.7), **bbox},
#         #     **kwargs,
#         # )
        
#         # xy_positions.append((x_pos, y_pos))
#         # label_positions.append((x_pos + x_off, y_pos + y_off))
    
#     return label_positions

def add_labels3(ax, x, y, c, data, labels, min_distance, mag_off, base_angle, label_positions=None, **kwargs):
    # labels = np.unique(np.floor(np.log10(data[c])), return_index=True)[1]
    # labels = sorted(labels, key=lambda i: data.loc[i, c], reverse=True)  # Sort labels by frequency in descending order
    xy_positions = []
    if label_positions is None:
        label_positions = []
    # min_distance = upper_lim * 0.05  # Minimum distance between labels
    
    arrowprops = kwargs.pop("arrowprops", {})
    bbox = kwargs.pop("bbox", {})
    
    for label in labels:
        x_pos = data.loc[label, x]
        y_pos = data.loc[label, y]

        # Check if the new label is sufficiently far from existing labels
        if all(np.sqrt((x_pos - lx)**2 + (y_pos - ly)**2) > min_distance for lx, ly in xy_positions):
            # mag_off = upper_lim * 0.075  # Magnitude of the offset vector
            # base_angle = 10
            x_off = mag_off * np.cos(np.radians(base_angle))  # Initial x offset (45 degrees)
            y_off = mag_off * np.sin(np.radians(base_angle))  # Initial y offset (45 degrees)

            # Adjust y_off dynamically based on surrounding data points
            data_distance = np.sqrt((data[x] - x_pos)**2 + (data[y] - y_pos)**2)
            nearby_points = data[(data_distance >= mag_off * 0.5) & (data_distance <= mag_off * 1.5)]
            if not nearby_points.empty:
                angles = np.degrees(np.arctan2((nearby_points[y] - y_pos), (nearby_points[x] - x_pos)))
                angles = angles[(angles <= 90) & (angles > 0)]
                if angles.empty:
                    pass
                elif (abs(angles.min() - base_angle) <= 30) or (abs(angles.max() - base_angle) <= 30):
                    y_off = mag_off * np.sin(np.radians(90 - base_angle))
                    x_off = mag_off * np.cos(np.radians(90 - base_angle))
                else:
                    y_off = mag_off * np.sin(np.radians(base_angle))
                    x_off = mag_off * np.cos(np.radians(base_angle))
            
            lx_pos = x_pos + x_off
            ly_pos = y_pos + y_off
            
            # Get a list of all label positions which are within the minimum distance
            close_labels = [(lx_pos - lx, ly_pos - ly) for lx, ly in label_positions if np.sqrt((lx_pos - lx)**2 + (ly_pos - ly)**2) <= min_distance]
            if close_labels != []:
                for n, l in enumerate(close_labels):
                    lx_pos += 1.5 * l[0]
                    ly_pos += 1.5 * l[1]
            
            ax.annotate(
                int(np.floor(np.log10(data.loc[label, c]))),
                (x_pos, y_pos),
                (lx_pos, ly_pos),
                textcoords = "data",
                arrowprops = {**dict(facecolor='none', edgecolor="black", arrowstyle="->", linewidth=2.5), **arrowprops},
                fontsize = 12,
                bbox = {**dict(boxstyle="circle,pad=0.2", edgecolor="none", facecolor="white", alpha=0.7), **bbox},
                **kwargs,
            )
            
            xy_positions.append((x_pos, y_pos))
            label_positions.append((x_pos + x_off, y_pos + y_off))
            
    return label_positions