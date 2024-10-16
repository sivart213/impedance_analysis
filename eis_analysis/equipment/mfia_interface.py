# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 10:41:50 2023

@author: j2cle

taken from the API commands log

"""
from functools import partial
import time
import numpy as np
import pandas as pd

import zhinst.core

import matplotlib.pyplot as plt

from IPython import get_ipython


from research_tools.functions import get_config

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


class MFIA(object):
    """
    Initializes and operates the MFIA tool.

    ...

    Attributes
    ----------
    device : str, optional
        The device name of the tool (device specific).  Most easily found from interfacing via
        the Zurich Instruments program.
        default : None
    config_file : Union[str, Path], optional
        The name or path to the configuration .ini file. Provide full path if not located near .py
        program location. Can alternately set initial configuration settings by defining as None.
        **Important** If defined as none, config_kwargs must be difined with all necessary settings.
        default : "config.ini"
    config_kwargs : dict
        Dictionary containing any alternate configurations desired for running the tool.
    kwargs : dict
        Any additional configurations to be added or changed manually during initiation.
        Pass "configurations" or "config_kwargs" to specify configurations.
        config_kwargs : dict, optional
    Properties
    ----------
    init_config : dict
        Calls get_config to get initial configuration settings from .ini file.  Initial conditions
        can be altrnately set by defining config_file as None and specifying all necessary settings
        via config_kwargs
    daq : zhinst obj
        The the obj containing the core daq module. Necessary for most MFIA operations

    """

    def __init__(self, device=None, config_file="config.ini", **kwargs):
        self.device = device
        # Pull configuration data into file
        self.config_file = config_file
        self.config_sections = kwargs.pop("sections", ["base"])
        if "config_kwargs" in kwargs.keys():
            self.config_kwargs = kwargs.get("config_kwargs", {})
        else:
            self.config_kwargs = kwargs

    @property
    def init_config(self):
        """Calls get_config to get initial configuration settings from .ini file."""
        if self.config_file is None:
            return self.config_kwargs
        if not hasattr(self, "_init_config"):
            self._init_config = get_config(
                self.config_file, self.config_sections, **self.config_kwargs
            )
        if self.device is None:
            self.device = self._init_config.pop("device", "dev6037")
        else:
            self._init_config.pop("device", "dev6037")
        return self._init_config

    @property
    def daq(self):
        """The the obj containing the core daq module."""
        if not hasattr(self, "device"):
            return None
        if not hasattr(self, "_daq"):
            d = zhinst.core.ziDiscovery()
            d.find(self.device)
            devProp = d.get(self.device)
            self._daq = zhinst.core.ziDAQServer(
                devProp["serveraddress"], devProp["serverport"], 6
            )
            self._daq.setInt(f"/{self.device}/imps/0/bias/enable", 0)
        return self._daq


class MFIA_Freq_Sweep(MFIA):
    """
    Initializes and operates the MFIA tool and sets up for a sweep operation.

    ...

    Attributes
    ----------
    sample_key : str
        The sample key location for subscription necessary for a sweep
    result : pd.Dataframe, dict
        A dataframe or dict of dataframes returned by all sweep functions saved
        within the class in case of error.
        Note: Any new sweep functions should update and return this attr.

    Properties
    ----------
    sweeper: object
        The daq.sweeper object from zhinst which contains and controls the sweep function of the
        MFIA tool. Sweeper settings can be adjusted by passing (key, value) pairs or dict.
    sweeper_conditions : dict
        A dict containing the conditions/settings of the current sweeper object


    Methods
    -------
    sweep(delay, verbose, plot)
        Performs a simple sweep without bias
    biased_sweep(biases, delay, verbose, plot)
        Performs single sweep or sequence of sweeps with bias

    """

    def __init__(self, device=None, config_file="config.ini", **kwargs):
        super().__init__(device, config_file, **kwargs)

        self.sweeper = self.init_config
        self.sample_key = kwargs.get("sample_key", f"/{self.device}/imps/0/sample")

    @property
    def sweeper(self):
        """The daq.sweeper object from zhinst which controls the sweep function of the MFIA."""
        if not hasattr(self, "_sweeper"):
            self._sweeper = self.daq.sweep()
            self._sweeper.set("device", self.device)
        return self._sweeper

    @sweeper.setter
    def sweeper(self, args):
        if not hasattr(self, "_sweeper"):
            self._sweeper = self.daq.sweep()
            self._sweeper.set("device", self.device)
        if (
            isinstance(args, (tuple, list))
            and len(args) == 2
            and "/"+args[0] in self._sweeper.listNodes("*",recursive=True)
            # and hasattr(self._sweeper, args[0])
        ):
            self._sweeper.set(args[0], args[1])
        elif isinstance(args, dict):
            for k, v in args.items():
                self.sweeper = (k, v)

    @property
    def sweeper_conditions(self):
        """A dict containing the conditions/settings of the current sweeper object."""
        return self.sweeper.get("*")

    def sweep(self, delay=2, verbose=True, plot=None):
        """
        Perform a simple sweep.

        Parameters
        ----------
        delay : Union[int, float]
            Specify the delay in reading time for verbosity or plotting.
            Higher value -> fewer updates
            Default : 2
        verbose : bool
            Tells system whether to print progress updates
        plot : function, optional
            Pass a function for plotting the data. Function must be limited to single data input
            of type dict or dataframe with keys appropriate to the native output of the sweeper obj.
            i.e. "frequency", "realz", "imagz", "absz", "phasez", "param0", or "param1"
            Default : None

        Returns
        -------
        result : pd.DataFrame
            Returns a DataFrame of the result of the dict
        """
        self.sweeper.subscribe(self.sample_key)
        self.sweeper.execute()

        data = 0
        if verbose:
            print("Sweep progress: 0.00%; ", end="\r")
        while not self.sweeper.finished():
            time.sleep(delay)
            if verbose:
                print(f"{self.sweeper.progress()[0]:.2%}; ", end="\r")
            data = self.sweeper.read(True)
            try:
                samples = data[self.sample_key][0][0]
                if callable(plot):
                    plot(samples)
            except KeyError:
                continue
        print("")

        self.sweeper.finish()
        result = self.sweeper.read(True)
        if result == {}:
            result = data
        self.sweeper.unsubscribe(self.sample_key)

        res_df = pd.DataFrame(
            {
                k: v
                for k, v in data[self.sample_key][0][0].items()
                if isinstance(v, np.ndarray)
            },
            dtype=np.float64,
        )
        if hasattr(self, "result") and isinstance(self.result, dict):
            return res_df
        
        self.result = res_df
        return self.result

    def biased_sweep(self, biases, delay=2, verbose=True, plot=None):
        """
        Perform a sweep with applied biasing, iterating If a list of biases is provided.

        Parameters
        ----------
        biases : [list, int, float]
            The bias or list of biases to be applied during a sweep operation.
        delay : [int, float], optional
            Specify the delay in reading time for verbosity or plotting.
            Higher value -> fewer updates
            Default : 2
        verbose : bool, optional
            Tells system whether to print progress updates
        plot : function, optional
            Function to be passed to the sweep method. Modifies title to specify the current bias.
            Function must have a "title" parameter to modify title

        Returns
        -------
        result : dict
            Returns a dict of DataFrames of the results.
        """
        if isinstance(biases, (int, float)):
            biases = [biases]
        
        # Enable bias after ensuring bias is 0 V
        self.daq.setDouble(f"/{self.device}/imps/0/bias/value", 0.0)
        self.daq.setInt(f"/{self.device}/imps/0/bias/enable", 1)
        
        self.result = {}
        
        pplot = None
        for bv in biases:
            print(f"Bias: {bv}")
            self.daq.setDouble(f"/{self.device}/imps/0/bias/value", bv)
            if callable(plot):
                pplot = partial(plot, title=f"Bias: {bv}")
            self.result[str(bv)] = self.sweep(delay, verbose, pplot)
        
        self.daq.setDouble(f"/{self.device}/imps/0/bias/value", 0.0)
        return self.result


if __name__ == "__main__":
    from research_tools.functions import save, find_path

    config_path = find_path("impedance_analysis", "eis_analysis", "device_control", base="cwd")
    save_path = find_path("impedance_analysis", "testing", "Data", "Raw", base="cwd")
    
    get_ipython().run_line_magic("matplotlib", "inline")
    # sweep_obj = MFIA_Freq_Sweep(
    #     "dev6037", config_path/"config_mfia.ini", sections=["base_sweep_settings", "fast_sweep"],
    # )
    
    # single_sweep = sweep_obj.sweep(plot=plot_measured_data)
    # save(single_sweep, save_path, "mfia_test_single")
    
    # biases = [-0.1, 0, 0.1]
    # sweep_sequence = sweep_obj.biased_sweep(biases, plot=plot_measured_data)

    # save(sweep_sequence, save_path, "mfia_test_sequence")
