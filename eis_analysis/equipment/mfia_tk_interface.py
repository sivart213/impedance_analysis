import time
import threading
import xml.etree.ElementTree as ET
from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd
from zhinst.toolkit import Session

try:
    from ..string_ops import safe_eval
except ImportError:
    from eis_analysis.string_ops.string_mod import safe_eval

# %% Functions
# def plot_measured_data(sweep_data: dict, **kwargs):
#     """Plot the sweep data in bode plot."""
#     _, (ax1, ax2) = plt.subplots(2, 1)

#     frequency = sweep_data[kwargs.get("xkey", "frequency")]
#     ax1.plot(frequency, sweep_data[kwargs.get("y1key", "realz")])
#     ax2.plot(frequency, sweep_data[kwargs.get("y2key", "imagz")])

#     ax1.set_title(kwargs.get("title", "Current Bode"))
#     ax1.grid()
#     ax1.set_ylabel(kwargs.get("y1label", kwargs.get("y1key", "Real (Ohm)")))
#     ax1.set_xscale("log")

#     ax2.grid()
#     ax2.set_xlabel(kwargs.get("xlabel", kwargs.get("xkey", "Frequency ($Hz$)")))
#     ax2.set_ylabel(kwargs.get("y2label", kwargs.get("y2key", "Imaginary (Ohm)")))
#     ax2.set_xscale("log")
#     ax2.autoscale()

#     plt.draw()
#     plt.show()

#

special_xml_dict = {
    "log": "xmapping",
    "fileformat": "savefileformat",
    "savemode": "savesave",
    "bandwidthMode": ("bandwidthcontrol", lambda x: abs(x - 2)),
    "filterorder": "order",
    "mask": "bitmask",
    "absoluteFrequency": "fftabsolute",
    "holdoff": "holdofftime",
}


class tkMFIA:
    """Class to initialize and operate the MFIA tool using zhinst toolkit."""

    def __init__(
        self,
        server_host: str,
        device_name: str,
        interface: str = "1GbE",
        settings_file: str | Path | None = None,
    ):
        self.server_host = server_host
        self.device_name = device_name
        self.interface = interface
        self.settings_file = settings_file

        self.session = Session(server_host)
        self.device = self.session.connect_device(device_name, interface=interface)

        if settings_file is not None:
            self.load_device_settings(settings_file)

        self._sweepers = []
        self._daqs = []

    def load_device_settings(
        self, settings_file: str | Path | None = None, modules=None, asynchronous=False
    ):
        settings_file = settings_file or self.settings_file
        if isinstance(settings_file, str):
            settings_file = Path(settings_file)
        if settings_file:
            if asynchronous:
                device_settings: Any = self.session.modules.device_settings
                device_settings.device(self.device)
                device_settings.filename(settings_file.stem)
                device_settings.path(settings_file.parent)
                device_settings.command("load")

                device_settings.execute()

                device_settings.finished.wait_for_state_change(1)
            else:
                self.session.modules.device_settings.load_from_file(settings_file, self.device)  # type: ignore
        if modules:
            n_swp = 0
            n_daq = 0
            for mod in modules:
                if isinstance(mod, str):
                    if "sweep" in mod:
                        module = self.sweeper
                        self.load_module_settings(module, mod, settings_file, n_swp)
                        n_swp += 1
                    elif "daq" in mod:
                        module = self.daq
                        self.load_module_settings(module, mod, settings_file, n_daq)
                        n_daq += 1

    @property
    def sweepers(self):
        return self._sweepers

    @property
    def daqs(self):
        return self._daqs

    @property
    def sweeper(self):
        return self._sweepers[0] if self._sweepers else self.gen_sweeper()

    @property
    def daq(self):
        return self._daqs[0] if self._daqs else self.gen_daq()

    def parse_module_settings(
        self, in_file: str | Path | None = None, node_name: str = "", instance=0, **kwargs
    ):
        in_file = in_file or self.settings_file
        if isinstance(in_file, str):
            in_file = Path(in_file)
        if not in_file or not in_file.exists():
            raise FileNotFoundError(f"Settings file {in_file} does not exist.")
        tree = ET.parse(in_file)
        root = tree.getroot()

        if not node_name or "sweep" in node_name:
            node_name = "sweeper"
        elif "daq" in node_name:
            node_name = "data_acquisition"

        settings_dict = {}
        nodes = root.findall(f".//tab[@html-name='{node_name}.html']/pages/page/figures/figure")
        if nodes:
            if instance < len(nodes):
                node = nodes[instance]
            else:
                node = nodes[-1]
        else:
            return settings_dict

        kwargs = {**special_xml_dict, **kwargs}

        if node is not None:
            for child in node:
                settings_dict[child.tag] = safe_eval(child.text)  # type: ignore

                if isinstance(settings_dict[child.tag], bool):
                    settings_dict[child.tag] = int(settings_dict[child.tag])

        for key, value in kwargs.items():
            if key in settings_dict.keys():
                if isinstance(value, tuple):
                    settings_dict[value[0]] = value[1](settings_dict.pop(key))
                else:
                    settings_dict[value] = settings_dict.pop(key)

        return settings_dict

    # def update_module_dict(self, ref_dict, targ_dict):
    #     for xkey, xvalue in ref_dict.items():
    #         for mkey in targ_dict.keys():
    #             if mkey.lower().replace("/", "") == xkey.lower():
    #                 targ_dict[mkey] = xvalue
    #                 break
    #     return targ_dict

    def cofigure_module(self, module, ref_dict):
        targ_dict = module.raw_module.get("*", flat=True)
        update_list = []
        mkeys = [mkey.lower().replace("/", "") for mkey in targ_dict.keys()]
        for xkey, xvalue in ref_dict.items():
            if xkey.lower() in mkeys:
                nkey = list(targ_dict.keys())[mkeys.index(xkey.lower())]
                update_list.append((nkey, xvalue))
        module.raw_module.set(update_list)
        # return targ_dict

    def get_children(self, objs):
        flat_dict = {}
        ign = ["node_info", "raw_tree", "root"]

        def process_obj(obj, parent=""):
            if obj.node_info.is_partial:
                for attr in dir(obj):
                    if not attr.startswith("_") and attr not in ign:
                        process_obj(obj[attr], f"{parent}/{attr}")
            else:
                try:
                    flat_dict[parent] = obj(enum=False)
                except ValueError:
                    flat_dict[parent] = obj(parse=False)

        process_obj(objs)
        return flat_dict

    def load_module_settings(self, module, module_name, settings_file=None, instance=0):
        settings_file = settings_file or self.settings_file

        # module_settings = self.get_children(module)
        # module_settings = module.raw_module.get("*", flat=True)
        xml_dict = self.parse_module_settings(settings_file, module_name, instance)
        # update_dict = self.update_module_dict(xml_dict, module_settings)
        self.cofigure_module(module, xml_dict)

    # def load_module_settings(self, module, module_name, settings_file=None, instance=0):
    #     settings_file = settings_file or self.settings_file

    #     # module_settings = self.get_children(module)
    #     module_settings = module.raw_module.get("*", flat=True)
    #     xml_dict = self.parse_module_settings(settings_file, module_name, instance)
    #     update_dict = self.update_module_dict(xml_dict, module)
    #     self.cofigure_module(module, **update_dict)

    # def cofigure_module(self, module, **nodes):
    #     if len(nodes) == 1 and isinstance(list(nodes.values())[0], dict):
    #         nodes = list(nodes.values())[0]
    #     for key, value in nodes.items():
    #         try:
    #             if module[key].node_info.writable:
    #                 module[key](value)
    #         except AttributeError as e:
    #             print(f"AttributeError: {e}")
    #             continue
    #         except KeyError as e:
    #             print(f"KeyError setting {key}: {e}")
    #             continue

    def gen_sweeper(self, sweeper: Any = None, configure=True, settings_file=None):
        if sweeper is None:
            sweeper = self.session.modules.sweeper
        sweeper.device(self.device)

        if configure:
            self.load_module_settings(sweeper, "sweeper", settings_file)
        self._sweepers.append(sweeper)
        return sweeper

    def gen_daq(self, daq: Any = None, configure=True, settings_file=None):
        if daq is None:
            daq = self.session.modules.daq
        daq.device(self.device)
        if not configure:
            self.load_module_settings(daq, "data_acquisition", settings_file)
        self._daqs.append(daq)
        return daq


class tkMFIA_Sweep:
    """Class to perform frequency sweeps using tkMFIA."""

    def __init__(self, tk_mfia, sample_nodes=None):
        self.tk_mfia = tk_mfia
        self.sample_nodes = sample_nodes or [
            f"/{tk_mfia.device_name}/imps/0/sample",
            f"/{tk_mfia.device_name}/demods/0/sample",
            f"/{tk_mfia.device_name}/demods/1/sample",
        ]
        self.result = None
        self._thread = None

    def _sweep_thread(self, delay, verbose, plot):
        """Thread function to perform the sweep."""
        for sample_node in self.sample_nodes:
            self.tk_mfia.sweeper.subscribe(sample_node)
        self.tk_mfia.sweeper.execute()

        self.data = {}
        if verbose:
            print("Sweep progress: 0.00%; ", end="\r")
        while not self.tk_mfia.sweeper.raw_module.finished():
            time.sleep(delay)
            if verbose:
                print(f"{self.tk_mfia.sweeper.raw_module.progress()[0]:.2%}; ", end="\r")
            self.data = self.tk_mfia.sweeper.read().to_dict()
            try:
                samples = self.data[self.sample_nodes[0]][0][0]
                if callable(plot):
                    plot(samples)
            except KeyError:
                continue
        print("")

        self.tk_mfia.sweeper.raw_module.finish()
        result = self.tk_mfia.sweeper.read().to_dict()
        if result:
            self.data = result
        for sample_node in self.sample_nodes:
            self.tk_mfia.sweeper.unsubscribe(sample_node)

        res_df = pd.concat(
            {
                sk: pd.DataFrame(
                    {k: v for k, v in sv[0][0].items() if isinstance(v, np.ndarray)},
                    dtype=np.float64,
                )
                for sk, sv in self.data.items()
                if isinstance(sv, list)
            },
            axis=1,
        )

        self.result = res_df
        self.stop()

    def sweep(self, delay=1, verbose=True, plot=None):
        """Perform a simple sweep."""
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._sweep_thread, args=(delay, verbose, plot))
            self._thread.start()

    def stop(self):
        """Stop the sweep if it is running."""
        if self._thread is not None and self._thread.is_alive():
            # self.tk_mfia.sweeper.raw_module.finish()
            self._thread.join()


if __name__ == "__main__":
    # Example usage
    host = "10.155.7.96"
    settings_path = Path(r"D:\Online\ASU Dropbox\Jacob Clenney\Work Docs\Data\Raw\MFIA\setting")
    swpfile = "JC_all_settings_v10swp.xml"
    daqfile = "JC_all_settings_v10daq.xml"
    filename = settings_path / swpfile
    dev_name = "dev6037"

    mfia = tkMFIA(host, dev_name, "1GbE", filename)
    mfia.gen_sweeper()
    mfia.gen_daq()

    mfia.sweeper.start(4e3)
    mfia.sweeper.stop(500e3)  # 500e3 for MF devices, 50e6 for others

    kt_sweep = tkMFIA_Sweep(mfia)
    kt_sweep.sweep()
    # kt_sweep._thread.join()  # Wait for the sweep to complete
    # print(kt_sweep.result)

    # sample_nodes = [
    #     mfia.device.imps[0].sample,
    #     mfia.device.demods[0].sample,
    #     mfia.device.demods[1].sample,
    #     ]
    # for sample_node in sample_nodes:
    #     mfia.sweeper.subscribe(sample_node)
    #     mfia.daq.subscribe(sample_node)
    # mfia.sweeper.execute()
    # mfia.daq.execute()
    # while not mfia.sweeper.raw_module.finished():
    #     time.sleep(1)
    #     data = mfia.sweeper.read()
    #     datadaq = mfia.daq.read(raw=False, clk_rate=mfia.device.clockbase())
    # result = mfia.sweeper.read()
    # resultdaq = mfia.daq.read(raw=False, clk_rate=mfia.device.clockbase())

    # for sample_node in sample_nodes:
    #     mfia.sweeper.unsubscribe(sample_node)
    #     mfia.daq.unsubscribe(sample_node)

# def parse_module_settings(in_file, node_name):
#     tree = ET.parse(in_file)
#     root = tree.getroot()

#     settings_dict = {}
#     node = root.find(f".//tab[@html-name='{node_name}.html']/pages/page/figures/figure")

#     if node is not None:
#         for child in node:
#             settings_dict[child.tag] = safe_eval(child.text)
#             if isinstance(settings_dict[child.tag], bool):
#                 settings_dict[child.tag] = int(settings_dict[child.tag])

#     return settings_dict

# def update_module_dict(ref_dict, targ_dict):
#     for xkey, xvalue in ref_dict.items():
#         # Attempt to find the closest matching key in the module_dict
#         for mkey in targ_dict.keys():
#             if mkey.lower().replace("/", "") == xkey.lower():
#                 targ_dict[mkey] = xvalue
#                 break
#     return targ_dict

# def get_children(objs):
#     flat_dict = {}
#     ign = ['node_info', 'raw_tree', 'root']
#     def process_obj(obj, parent=''):
#         if obj.node_info.is_partial:
#             for attr in dir(obj):
#                 if not attr.startswith('_') and attr not in ign:
#                     process_obj(obj[attr], f"{parent}/{attr}")
#         else:
#             try:
#                 flat_dict[parent] = obj(enum=False)
#             except ValueError:
#                 flat_dict[parent] = obj(parse=False)

#     process_obj(objs)
#     return flat_dict


# #%% Arguments
# # IP address of the host computer where the Data Servers run
# # server_host = 'localhost'
# # server_host = '127.0.0.1'
# server_host = '10.155.7.96'
# settings_path = Path(r"D:\Online\ASU Dropbox\Jacob Clenney\Work Docs\Data\Raw\MFIA\setting")
# # sfile1 = "JC_all_settings_v10_3.xml"
# # sfile2 = "JC_all_settings_v9.xml"
# sfile = "JC_all_settings_v10swp.xml"
# filename = settings_path/sfile
# device_name = "dev6037"
# interface = "1GbE"


# #%% Object items
# # A session opened to LabOne Data Server
# session = Session(server_host)
# # # A session opened to HF2 Data Server
# # hf2_session = Session(server_host, hf2=True)
# device = session.connect_device(device_name, interface=interface)

# # synchronous
# session.modules.device_settings.load_from_file(filename, device)
# # Asynchronous
# device_settings = session.modules.device_settings
# device_settings.device(device)
# device_settings.filename(filename.stem)
# device_settings.path(filename.parent)
# device_settings.command("load")


# sweeper = session.modules.sweeper
# sweeper.device(device)
# sweeper_settings = get_children(sweeper)
# xml_swp_dict = parse_module_settings(filename, "sweeper")
# update_dict = update_module_dict(xml_swp_dict, sweeper_settings)
# for key, value in update_dict.items():
#     sweeper[key](value)

# daq = session.modules.daq
# daq.device(device)
# daq_settings = get_children(daq)
# xml_daq_dict = parse_module_settings(filename, "data_acquisition")
# update_dict = update_module_dict(xml_daq_dict, daq_settings)
# for key, value in update_dict.items():
#     daq[key](value)
