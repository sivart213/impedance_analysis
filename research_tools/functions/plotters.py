# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns

from pathlib import Path
from matplotlib import ticker
from matplotlib.widgets import Slider, Button

# from research_tools.functions import slugify, p_find, sig_figs_ceil, sci_note
from research_tools.functions.system_utilities import slugify, p_find
from research_tools.functions.data_treatment import sig_figs_ceil, sci_note

warnings.simplefilter("ignore", np.RankWarning)
warnings.filterwarnings("ignore")


# %% Std Package wrappers
if "kindlmann" not in mpl.colormaps() or "kindlmann_r" not in mpl.colormaps():
    # TODO Convert to p_find and move doc
    Path()
    csv = pd.read_csv(
        p_find(
            "research_tools","functions","kindlmann-tables","kindlmann-table-float-1024.csv",
            base="cwd",
        )
    )
    col_arr = csv.iloc[:, 1:].to_numpy()

    if "kindlmann" not in mpl.colormaps():
        mpl.cm.register_cmap(
            "kindlmann",
            mpl.colors.LinearSegmentedColormap.from_list("kindlmann", col_arr),
        )
    if "kindlmann_r" not in mpl.colormaps():
        mpl.cm.register_cmap(
            "kindlmann_r",
            mpl.colors.LinearSegmentedColormap.from_list(
                "kindlmann", col_arr
            ).reversed(),
        )


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
    Create a map plot using contourf.
    """
    style.use("seaborn-colorblind")

    if xlimit == [0, 0]:
        xlimit = [min(x), max(x)]
    if ylimit == [0, 0]:
        ylimit = [min(y), max(y)]
    if zlimit[0] <= 0:
        zlimit = [z[z > 0].min(), z[z > 0].max()]
        # zlimit = [max(0, z.min()), z.max()]

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
    # xx,yy,zz = limitcontour(x,y,z,xlim=xlimit)

    # visual_levels = [1, 4, 7, 30, 365, 365*10]
    # lv_lbls = ['1 d', '4 d', '1 w', '1 mo', '1 yr', '10 yr']
    # ax = plt.gca()
    # csb = ax.contour(xx,yy,zz,visual_levels, colors='w',locator=ticker.LogLocator(),linestyles='--',norm=LogNorm(),linewidths=1.25)
    # csb.levels = lv_lbls

    # ax.clabel(csb, csb.levels, inline=True, fontsize=14, manual=False)
    if save is not None:
        if isinstance(save, Path):
            save = str(save)
        if not os.path.exists(save):
            os.makedirs(save)
        plt.savefig(os.sep.join((save, f"{slugify(name)}.png")))

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
    """Calculate. generic discription."""
    style.use("seaborn-colorblind")

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

    g.set_xlabel(xname, fontname="serif", fontsize=18, fontweight="bold")
    g.set_ylabel(yname, fontname="serif", fontsize=18, fontweight="bold")
    ax.set_title(name, fontname="serif", fontsize=20, fontweight="bold")

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
        plt.savefig(os.sep.join((save, f"{slugify(name)}.png")))

    if not show:
        plt.close()


def nyquist(
    data,
    freq=None,
    fit=None,
    band=None,
    bmin="min",
    bmax="max",
    title="Nyquist",
    pad=1.25,
    return_fig=False,
):
    """Calculate. generic discription."""
    style.use("seaborn-colorblind")

    data = data.copy()
    # if freq is not None:
    #     data["freq"] = np.trunc(np.log10(freq))
    # else:
    #     data["freq"] = np.trunc(np.log10(data["freq"]))
    if freq is not None:
        data["freq"] = freq

    fig, ax = plt.subplots()
    sns.scatterplot(
        x="real",
        y="inv_imag",
        data=data,
        hue="freq",
        palette="kindlmann",
        hue_norm=plt.matplotlib.colors.LogNorm(),
        legend=False,
        edgecolor="none",
        ax=ax,
    )
    if fit is not None:
        ax.plot(fit["real"], fit["inv_imag"])
        if band is not None:
            ax.fill_between(band["real"], band[bmin], band[bmax], color="r", alpha=0.5)

    norms = plt.matplotlib.colors.LogNorm(data["freq"].min(), data["freq"].max())
    sm = plt.cm.ScalarMappable(cmap="kindlmann", norm=norms)
    cbar = ax.figure.colorbar(sm)
    cbar.set_label("Freq", fontname="Arial", fontsize=18, fontweight="bold")

    ax.set_xlabel("Z' [Ohms]", fontname="Arial", fontsize=18, fontweight="bold")
    ax.set_ylabel("-Z'' [Ohms]", fontname="Arial", fontsize=18, fontweight="bold")
    ax.set_aspect("equal", adjustable="box", anchor="SW", share=True)
    ax.set_xlim(0, sig_figs_ceil(data[["real", "inv_imag"]].max().max() * pad, 2))
    ax.set_ylim(0, sig_figs_ceil(data[["real", "inv_imag"]].max().max() * pad, 2))
    ax.grid(True)
    ax.set_title(title, fontname="Arial", fontsize=18, fontweight="bold")

    labels = np.unique(np.floor(np.log10(data["freq"])), return_index=True)[1]
    for label in labels:
        ax.annotate(
            sci_note(data.loc[label, "freq"], prec=0),
            (data.loc[label, "real"] + 0.2, data.loc[label, "inv_imag"] + 0.2),
        )

    if return_fig:
        return fig
    else:
        plt.show()


def bode(
    data,
    freq=None,
    top="mag",
    bot="phase",
    fit=None,
    band=None,
    bmin="min",
    bmax="max",
    title="bode",
    return_fig=False,
):
    """Calculate. generic discription."""
    style.use("seaborn-colorblind")

    if freq is not None:
        data["freq"] = freq
    labels = {
        "real": "Z' [Ohms]",
        "imag": "Z'' [Ohms]",
        "inv_imag": "-Z'' [Ohms]",
        "mag": "|Z| [Ohms]",
        "phase": "Phase [deg]",
        "inv_phase": "Inv Phase [deg]",
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    sns.scatterplot(
        x="freq",
        y=top,
        data=data,
        edgecolor="none",
        ax=ax1,
    )
    sns.scatterplot(
        x="freq",
        y=bot,
        data=data,
        legend=False,
        edgecolor="none",
        ax=ax2,
    )
    if fit is not None:
        # sns.lineplot(x=data["freq"], y=top, data=fit, ax=ax1)
        # sns.lineplot(x=data["freq"], y=bot, data=fit, ax=ax2)
        ax1.plot(fit["freq"], fit[top])
        ax2.plot(fit["freq"], fit[bot])

        if band is not None:
            try:
                ax1.fill_between(
                    data["freq"], band[top][bmin], band[top][bmax], color="r", alpha=0.4
                )
                ax2.fill_between(
                    data["freq"], band[bot][bmin], band[bot][bmax], color="r", alpha=0.4
                )
            except KeyError:
                pass

    ax1.set(
        xscale="log",
        xlim=[data["freq"].min(), data["freq"].max()],
        yscale="log",
        ylim=[data[top].min(), data[top].max()],
    )
    # ax2.set(ylim=[-90, 90])
    if "phase" in bot.lower():
        ax2.yaxis.set_ticks(np.arange(-90, 120, 30))
        ax2.set_ylim(-100, 100)
    else:
        ax2.set(yscale="log", ylim=[data[bot].min(), data[bot].max()])

    ax2.set_xlabel("Frequency Hz", fontname="Arial", fontsize=18, fontweight="bold")
    ax1.set_ylabel(labels[top], fontname="Arial", fontsize=18, fontweight="bold")
    ax2.set_ylabel(labels[bot], fontname="Arial", fontsize=18, fontweight="bold")
    ax1.set_title(title, fontname="Arial", fontsize=18, fontweight="bold")
    ax1.grid(True)
    ax2.grid(True)
    plt.tight_layout()

    if return_fig:
        return fig, ax1, ax2
    else:
        plt.show()


def nyquist2(
    data,
    freq=None,
    fit=None,
    band=None,
    bmin="min",
    bmax="max",
    title="Nyquist",
    pad=1.25,
    return_fig=False,
):
    """Calculate. generic discription."""
    style.use("seaborn-colorblind")

    data = data.copy()
    # if freq is not None:
    #     data["freq"] = np.trunc(np.log10(freq))
    # else:
    #     data["freq"] = np.trunc(np.log10(data["freq"]))
    if freq is not None:
        data["freq"] = freq

    fig, ax = plt.subplots()
    ny = plt.scatter(
        x=data["real"],
        y=data["inv_imag"],
        c=data["freq"],
        cmap="kindlmann",
        norm=plt.matplotlib.colors.LogNorm(),
        # legend=False,
        edgecolor="none",
        # ax=ax,
    )
    if fit is not None:
        nyln = ax.plot(fit["real"], fit["inv_imag"], "r")
        if band is not None:
            ax.fill_between(band["real"], band[bmin], band[bmax], color="r", alpha=0.5)
    else:
        nyln = None

    norms = plt.matplotlib.colors.LogNorm(data["freq"].min(), data["freq"].max())
    sm = plt.cm.ScalarMappable(cmap="kindlmann", norm=norms)
    cbar = ax.figure.colorbar(sm)
    cbar.set_label("Freq", fontname="Arial", fontsize=18, fontweight="bold")

    ax.set_xlabel("Z' [Ohms]", fontname="Arial", fontsize=18, fontweight="bold")
    ax.set_ylabel("-Z'' [Ohms]", fontname="Arial", fontsize=18, fontweight="bold")
    ax.set_aspect("equal", adjustable="box", anchor="SW", share=True)
    ax.set_xlim(0, sig_figs_ceil(data[["real", "inv_imag"]].max().max() * pad, 2))
    ax.set_ylim(0, sig_figs_ceil(data[["real", "inv_imag"]].max().max() * pad, 2))
    ax.grid(True)
    ax.set_title(title, fontname="Arial", fontsize=18, fontweight="bold")

    labels = np.unique(np.floor(np.log10(data["freq"])), return_index=True)[1]
    for label in labels:
        ax.annotate(
            sci_note(data.loc[label, "freq"], prec=0),
            (data.loc[label, "real"] + 0.2, data.loc[label, "inv_imag"] + 0.2),
        )

    if return_fig:
        return fig, ny, nyln
    else:
        plt.show()


def bode2(
    data,
    freq=None,
    top="mag",
    bot="phase",
    fit=None,
    band=None,
    bmin="min",
    bmax="max",
    title="bode",
    return_fig=False,
):
    """Calculate. generic discription."""
    style.use("seaborn-colorblind")

    if freq is not None:
        data["freq"] = freq
    labels = {
        "real": "Z' [Ohms]",
        "imag": "Z'' [Ohms]",
        "inv_imag": "-Z'' [Ohms]",
        "mag": "|Z| [Ohms]",
        "phase": "Phase [deg]",
        "inv_phase": "Inv Phase [deg]",
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    bdtop = ax1.scatter(
        x="freq",
        y=top,
        data=data,
        edgecolor="none",
    )
    bdbot = ax2.scatter(
        x="freq",
        y=bot,
        data=data,
        edgecolor="none",
    )
    if fit is not None:
        # sns.lineplot(x=data["freq"], y=top, data=fit, ax=ax1)
        # sns.lineplot(x=data["freq"], y=bot, data=fit, ax=ax2)
        lntop = ax1.plot(fit["freq"], fit[top], "r")
        lnbot = ax2.plot(fit["freq"], fit[bot], "r")

        if band is not None:
            try:
                ax1.fill_between(
                    data["freq"], band[top][bmin], band[top][bmax], color="r", alpha=0.4
                )
                ax2.fill_between(
                    data["freq"], band[bot][bmin], band[bot][bmax], color="r", alpha=0.4
                )
            except KeyError:
                pass
    else:
        lntop = None
        lnbot = None

    ax1.set(
        xscale="log",
        xlim=[data["freq"].min(), data["freq"].max()],
        yscale="log",
        ylim=[
            10 ** np.floor(np.log10(data[top].min())),
            10 ** np.ceil(np.log10(data[top].max())),
        ],
    )
    # ax2.set(ylim=[-90, 90])
    if "phase" in bot.lower():
        ax2.yaxis.set_ticks(np.arange(-90, 120, 30))
        ax2.set_ylim(-100, 100)
    else:
        ax2.set(
            yscale="log",
            ylim=[
                10 ** np.floor(np.log10(data[bot].min())),
                10 ** np.ceil(np.log10(data[bot].max())),
            ],
        )

    ax2.set_xlabel("Frequency Hz", fontname="Arial", fontsize=18, fontweight="bold")
    ax1.set_ylabel(labels[top], fontname="Arial", fontsize=18, fontweight="bold")
    ax2.set_ylabel(labels[bot], fontname="Arial", fontsize=18, fontweight="bold")
    ax1.set_title(title, fontname="Arial", fontsize=18, fontweight="bold")
    ax1.grid(True)
    ax2.grid(True)
    plt.tight_layout()

    if return_fig:
        return fig, bdtop, bdbot, lntop, lnbot
    else:
        plt.show()


def lineplot_slider(
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
    save=None,
    return_fig=True,
    fig=None,
    ax=None,
    data2=None,
    **kwargs,
):
    """Calculate. generic discription."""
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    z = data.columns.to_numpy()
    if y is None:
        y = data[z[0]].to_numpy()
    if x == "index":
        x = data.index.to_numpy()

    if xlimit is None:
        xlimit = [x.min(), x.max()]

    if ylimit is None:
        ylimit = [data.min().min(), data.max().max()]

    if xname is None:
        xname = x
    if yname is None:
        yname = y

    if fig is None and ax is None:
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.3)
    # Loading the dataset into the variable 'dataset'
    # Graph is created and stored in the variable 'graph' *added ax to g
    (g1,) = ax.plot(x, y, "k", lw=4)
    if data2 is not None:
        y2 = data2[z[0]].to_numpy()
        x2 = data2.index.to_numpy()
        (g2,) = ax.plot(x2, y2, "r-.", lw=2)
    ax_z = fig.add_axes([0.15, 0.10, 0.65, 0.06])
    # g1 = ax.plot(x=x, y=y, data=data, ax=ax, **kwargs)
    # Drawing a horizontal line at point 1.25
    ax.set(xlim=xlimit, ylim=ylimit, xscale=xscale, yscale=yscale)
    ax.grid(True)

    ax.set_xlabel(xname, fontname="Arial", fontsize=18, fontweight="bold")
    ax.set_ylabel(yname, fontname="Arial", fontsize=18, fontweight="bold")
    ax.set_title(name, fontname="Arial", fontsize=20, fontweight="bold")

    for tick in ax.get_xticklabels():
        tick.set_fontname("Arial")
        tick.set_fontweight("bold")
        tick.set_fontsize(12)
    for tick in ax.get_yticklabels():
        tick.set_fontname("Arial")
        tick.set_fontweight("bold")
        tick.set_fontsize(12)

    # define the values to use for snapping
    allowed_amplitudes = z

    # create the sliders
    s_z = Slider(
        ax_z,
        zname,
        z.min(),
        z.max(),
        valinit=z[0],
        valstep=allowed_amplitudes,
        color="green",
    )

    def update(val):
        if data2 is not None:
            g2.set_ydata(data2[val].to_numpy())
        g1.set_ydata(data[val].to_numpy())
        fig.canvas.draw_idle()

    s_z.on_changed(update)

    ax_reset = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    button = Button(ax_reset, "Reset", hovercolor="0.975")

    def reset(event):
        s_z.reset()

    button.on_clicked(reset)

    # The plot is shown
    # plt.tight_layout()
    if save is not None:
        if isinstance(save, Path):
            save = str(save)
        if not os.path.exists(save):
            os.makedirs(save)
        plt.savefig(os.sep.join((save, f"{slugify(name)}.png")))

    if return_fig and data2 is not None:
        return fig, g1, g2, s_z, button
    elif return_fig:
        return fig, g1, s_z, button
    else:
        plt.show()
