# -*- coding: utf-8 -*-
"""
Temperature Logger and Incrementor classes for logging temperature data and incrementing temperature setpoints.
"""

import time
import threading
from datetime import datetime
from typing import List, Optional, Union
# from collections import namedtuple
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler

try:
    from .temperature_devices.watlow import Watlow
    from .temperature_devices.uwtc import UWTC
    from .utilities.signal_manager import SignalManager
    from .utilities.thread_tools import SThread
except ImportError:
    from equipment.temperature_devices.watlow import Watlow
    from equipment.temperature_devices.uwtc import UWTC
    from equipment.utilities.signal_manager import SignalManager
    from equipment.utilities.thread_tools import SThread

# import threading
# from typing import Callable, List, NamedTuple, Optional, Tuple, Union
# import numpy as np

class TempLogger:
    """
    Generic temperature logger that logs readings from multiple objects every n seconds.

    Parameters
    ----------
    interval : int, optional
        Interval in seconds between logging data points. Default is 30.
    auto_save : bool, optional
        Whether to automatically save the data to a CSV file. Default is False.
    save_path : str, optional
        Path to save the CSV file. Default is 'data_log.csv'.
    """

    def __init__(
        self, 
        signal_manager: SignalManager,
        signal_names: Union[str, List[str]] = "all",
        interval: int = 30, 
        ignore_below: Optional[float] = None,
        ignore_above: Optional[float] = None,
        auto_save: bool = False, 
        save_path: str = "data_log.csv"
    ):
        self.signal_manager = signal_manager
        self.signal_names = signal_names
        self.interval = interval
        self.ignore_below = ignore_below
        self.ignore_above = ignore_above
        self.auto_save = auto_save
        self.save_path = save_path
        self.start_time = datetime.now()
        self.df = pd.DataFrame(columns=["timestamp", "time"])
        self._thread = None
        self._running = False
        self._paused = False
        self._plotting = False
        self._lock = threading.Lock()
        self._plot_kwargs = {}

    # def add_device(self, obj: object, *args: Union[str, Tuple[Callable, str, bool]]):
    #     """
    #     Add a device to the logger.

    #     Parameters
    #     ----------
    #     obj : object
    #         The object to log data from.
    #     *args : tuple
    #         Each argument can be a string, a callable pair, or just a string.
    #     """
    #     self.signal_manager.add_device(obj, *args)
    #     for arg in args:
    #         if isinstance(arg, str):
    #             name = arg
    #         elif isinstance(arg, tuple) and len(arg) >= 2:
    #             name = arg[1]
    #         else:
    #             continue
    #         self.df[name] = None

    def _log_data(self):
        start_time = time.time()

        while self._running:
            if self._thread.thread_killed():
                break
            if not self._paused:
                timestamp = datetime.now()
                elapsed_time = time.time() - start_time
                data = self.signal_manager.get(self.signal_names)
                if self.ignore_below is not None and any(
                    d < self.ignore_below for d in data.values()
                ):
                    continue
                if self.ignore_above is not None and any(
                    d > self.ignore_above for d in data.values()
                ):
                    continue
                new_data = pd.DataFrame(
                    [{"timestamp": timestamp, "time": elapsed_time, **data}]
                )

                if self.df.empty:
                    self.df = new_data
                elif not new_data.empty:
                    with self._lock:
                        self.df = pd.concat([self.df, new_data], ignore_index=True)
                if self.auto_save:
                    self.save_data()
            # Calculate the time taken for the computation
            computation_time = time.time() - elapsed_time - start_time
            # Adjust the sleep time to account for computation time
            adjusted_sleep_time = self.interval - computation_time
            if adjusted_sleep_time > 0:
                time.sleep(adjusted_sleep_time)

    def _plot_data(self):
        cc = cycler(
            marker=[".", "o", "v", "^", "<", ">", "1", "2", "3", "4"]
        ) * cycler(ls=["-"]) + cycler(
            color=plt.rcParams["axes.prop_cycle"].by_key()["color"]
        )
        plt.rcParams['date.converter'] = 'concise'
        while self._plotting:
            with self._lock:
                if (
                    self._thread.thread_killed()
                    or self._plot_thread.thread_killed()
                ):
                    break
                if len(self.df) >= 1:

                    x_data = self.df["timestamp"].to_numpy()
                    x_label = "Time"
                    # x_data = self.df["time"].to_numpy()
                    # x_label = "Time (s)"
                    # if max(x_data) > 3600:
                    #     x_data = x_data / 3600
                    #     x_label = "Time (hr)"
                    # elif max(x_data) > 120:
                    #     x_data = x_data / 60
                    #     x_label = "Time (min)"

                    if not plt.fignum_exists(self.fig.number):
                        self.fig.show()

                    y_lim = self.ax.get_ylim()
                    is_auto = self.ax.get_autoscale_on()

                    self.ax.clear()
                    self.ax.set_prop_cycle(cc)
                    for name in self.df.columns[2:]:
                        self.ax.plot(
                            x_data, self.df[name], label=f"{name}"
                        )
                    if not is_auto:
                        self.ax.set_ylim(*y_lim)

                    self.ax.set_xlabel(x_label)
                    self.ax.set_ylabel("Temperature (°C)")
                    self.ax.grid(self._plot_kwargs.get("grid", True))
                    self.ax.legend()
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
            time.sleep(self.interval)

    def start_logging(self):
        """
        Start logging data.
        """
        thread_name = "logging: " + ", ".join(self.df.columns[2:])
        if not self._running:
            self.start_time = datetime.now()
            self._running = True
            self._paused = False
            self._thread = SThread(target=self._log_data, name=thread_name)
            self._thread.start()

    def stop_logging(self):
        """
        Stop logging data.
        """
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()

    def pause_logging(self):
        """
        Pause logging data.
        """
        self._paused = True

    def resume_logging(self):
        """
        Resume logging data.
        """
        self._paused = False

    def plot_data(self, **kwargs):
        """
        Control whether to plot the logged data.
        """
        thread_name = "plotting: " + ", ".join(self.df.columns[2:])
        self._plot_kwargs = kwargs
        if not self._plotting:
            self._plotting = True
            self.fig, self.ax = plt.subplots()
            self._plot_thread = SThread(
                target=self._plot_data, name=thread_name
            )
            self._plot_thread.start()
        else:
            self._plotting = False
            if self._plot_thread.is_alive():
                self._plot_thread.join()
            self.plot_data()

    def save_data(self):
        """
        Save the logged data to a CSV file.
        """
        f_path = (
            self.save_path
            / (str(self.start_time.strftime("%Y%m%d %H%M%S")) + ".csv")
            if self.save_path.is_dir()
            else self.save_path
        )
        with self._lock:
            self.df.to_csv(f_path, index=False)

class Incrementor:
    """
    Incrementor class to set a temperature controller to requested values with a delay.

    Parameters
    ----------
    signal_manager : SignalManager
        SignalManager instance to manage signals.
    temp_controller : str
        Name of the settable signal in the SignalManager.
    values : list
        List of values to set the temperature controller to.
    delay : float
        Delay time between setting values.
    delay_unit : str, optional
        Unit of delay time ('hours', 'minutes', 'seconds'). Default is 'seconds'.
    loop : bool, optional
        Whether to loop the values continuously. Default is False.
    n_loops : int, optional
        Number of loops to run if loop is True. Default is 1.
    stability_check : bool, optional
        Whether to check for stability before proceeding. Default is False.
    stability_signal : str, optional
        Name of the signal in the SignalManager to check for stability. Default is None.
    stability_tol : float, optional
        Tolerance within which the value is considered stable. Default is 0.5.
    stability_time : float, optional
        Time to wait for stability check. Default is 120.
    """

    def __init__(
        self,
        signal_manager: SignalManager,
        temp_controller: str,
        steps: List[float],
        delay: float,
        delay_unit: str = "seconds",
        loop: bool = False,
        n_loops: int = 1,
        stability_check: bool = False,
        stability_signal: Optional[str] = None,
        stability_tol: float = 0.5,
        stability_time: float = 120,
    ):
        self.signal_manager = signal_manager
        self.temp_controller = temp_controller
        self.steps = steps
        self.delay = delay
        self.delay_unit = delay_unit
        self.loop = loop
        self.n_loops = n_loops
        self.stability_check = stability_check
        self.stability_signal = stability_signal
        self.stability_tol = stability_tol
        self.stability_time = stability_time
        self._step = 0
        self._running = False
        self._paused = False
        self._thread = None
        self._start_time = datetime.now()
        self._step_start = datetime.now()
        self._total_elapsed = 0
        self._elapsed_time = 0

        if not isinstance(stability_signal, str) or stability_signal not in self.signal_manager.signals:
            self.stability_check = False


    def to_seconds(self, val: float = 0, unit: str = "") -> float:
        """
        Convert the delay time to seconds based on the delay unit.

        Parameters
        ----------
        val : float, optional
            Delay value. Default is 0.
        unit : str, optional
            Delay unit. Default is ''.

        Returns
        -------
        float
            Delay time in seconds.
        """
        val = val or self.delay
        unit = unit or self.delay_unit
        if not isinstance(val, (int, float)) or not isinstance(unit, str):
            raise ValueError("Invalid delay value or unit.")
        if unit.startswith("h"):
            res = val * 3600
        elif unit.startswith("min"):
            res = val * 60
        elif unit.startswith("sec"):
            res = val
        else:
            raise ValueError("Invalid delay unit. Choose from 'hours', 'minutes', or 'seconds'.")

        return res

    @property
    def total_runtime(self):
        time = (datetime.now() - self._start_time).total_seconds()
        if self.delay_unit.startswith("min"):
            return time / 60
        if self.delay_unit.startswith("h"):
            return time / 3600
        return self._total_elapsed

    @property
    def step_runtime(self):
        if not self._step_start:
            return
        time = (datetime.now() - self._step_start).total_seconds()
        if self.delay_unit.startswith("min"):
            return time / 60
        if self.delay_unit.startswith("h"):
            return time / 3600
        return self._total_elapsed

    @property
    def time_left(self):
        time = self.to_seconds() - self._elapsed_time

        if self.delay_unit.startswith("min"):
            return time / 60
        if self.delay_unit.startswith("h"):
            return time / 3600
        return self._total_elapsed
    
    @property
    def time_stable(self):
        time = self._elapsed_time

        if self.delay_unit.startswith("min"):
            return time / 60
        if self.delay_unit.startswith("h"):
            return time / 3600
        return self._total_elapsed

    @property
    def step(self) -> float:
        """Return the current step value."""
        return self.steps[self._step]

    @step.setter
    def step(self, value: Union[float, int]):
        if value in self.steps:
            self._step = self.steps.index(value)
        elif isinstance(value, int) and 0 <= value < len(self.steps):
            self._step = value
        else:
            return
        if self._running:
            self._running = False
            self._start()

    def run(self):
        """
        Runs the incrementor to set the temperature controller to the requested values.
        """
        delay_seconds = self.to_seconds()
        loop_count = 0
        self._running = True

        while self._running:
            if self._thread.thread_killed():
                break
            while self._step < len(self.steps):
                if self._thread.thread_killed() or not self._running:
                    break
                while self._paused and not self._thread.thread_killed():
                    time.sleep(0.1)

                self._step_start = datetime.now()
                self._elapsed_time = 0

                value = self.steps[self._step]
                print(f"Setting temperature to: {value}°C")
                self.signal_manager.set_to(self.temp_controller, value)

                sleep_interval = min(1, delay_seconds)
                stability_interval = min(sleep_interval * 60, max(1, delay_seconds / 5))
                time_since_check = 0
                if self.stability_check:
                    temps = np.array([self.signal_manager.get(self.stability_signal)], dtype=float)
                while self._elapsed_time < delay_seconds:
                    if not self._running or self._thread.thread_killed():
                        break
                    if self.stability_check and time_since_check > stability_interval:
                        temp = self.signal_manager.get(self.stability_signal)
                        if temp > 0:
                            time_since_check = 0
                            temps = np.append(temps, temp)
                        temp_std = np.ptp(temps)
                        if temp_std > self.stability_tol:
                            temps = np.array([temps[-1]], dtype=float)
                            self._elapsed_time = 0

                    time.sleep(sleep_interval)
                    self._elapsed_time += sleep_interval
                    time_since_check += sleep_interval

                self._step += 1

            if not self.loop:
                break

            loop_count += 1
            if self.n_loops > 0 and loop_count >= self.n_loops:
                break

            self._step = 0

    def start(self, step=None):
        """
        Starts the incrementor in a separate thread.
        """
        if self._running:
            self.stop()
        if step is not None:
            self.step = step
        self._start()

    def _start(self):
        """
        Starts the incrementor in a separate thread.
        """
        thread_name = "Increment: " + self.temp_controller
        if self._running:
            self.stop()
        self._start_time = datetime.now()
        self._thread = SThread(target=self.run, name=thread_name)
        self._thread.start()

    def stop(self):
        """
        Stops the incrementor.
        """
        self._running = False
        if self._thread is not None:
            self._thread.join()
            self._thread = None
            self._step = 0

    def pause(self):
        """
        Pauses the incrementor.
        """
        self._paused = True

    def resume(self):
        """
        Resumes the incrementor.
        """
        self._paused = False

    def next_step(self):
        """
        Skips to the next step.
        """
        step = self._step + 1
        if step >= len(self.steps):
            step = 0
        self.start(step)

    def previous(self):
        """
        Goes back to the previous step.
        """
        step = self._step - 1
        if step < 0:
            step = len(self.steps) - 1
        self.start(step)
 
#%%
if __name__ == "__main__":
    # import numpy as np
    from pathlib import Path
    import sys
    # try:
    #     from .utilities.thread_tools import DeviceTrigger
    #     from .mfia_base import MFIABase
    # except ImportError:
    #     from equipment.utilities.thread_tools import DeviceTrigger
    #     from equipment.mfia_base import MFIABase

    # Set the threading excepthook to print exceptions
    def thread_excepthook(args):
        print(f"Exception in thread {args.thread.name}: {args.exc_type.__name__}: {args.exc_value}", file=sys.stderr)

    threading.excepthook = thread_excepthook

    folder_path = Path(r"C:\Users\DEfECT\Documents\Omega")
    #%%

    # ez_zone_logger = WatlowLogger(port="com4", timeout=1, interval=60)
    # ez_zone_logger.start_logging()
    # ez_zone_logger = WatlowLogger(port="com4", interval=2)
    # ez_zone_logger.start_logging()
    # ez_zone_logger.plot_data()

    watlow_device = Watlow(
        port="sim COM4", #COM4
        max_setpoint=90,
        mode="normal",
        duration="auto",
        overshoot="auto"
        # duration=120,
        # overshoot=2.5
    )
    uwtc_device = UWTC(port="sim COM3") #COM3
    # uwtc_device._lock = threading.Lock()
    # mfia_device = MFIABase("localhost")

    device_manager = SignalManager()
    device_manager.add_device(
        uwtc_device,
        ("temp", "Top Temp"),
        # (lambda: uwtc_device.previous_reply.ambient, "ambient"),
        (lambda: uwtc_device.temp('ambient'), "ambient"),
    )
    device_manager.add_device(watlow_device, ("temp", "Bottom Temp"), ("setpoint", "setpoint", True))
    device_manager.define_signal("Ave Temp", ["Top Temp", "Bottom Temp"])
    device_manager.define_signal(watlow_device.__name__, ["setpoint"])
    
    device_manager.get()


    #%%
    temp_logger = TempLogger(
        signal_manager=device_manager,
        signal_names=["Top Temp", "Ave Temp", "Bottom Temp", "setpoint"],
        interval=15,
        ignore_below=0,
        ignore_above=200,
        auto_save=False,
        save_path=folder_path,
    )


    temp_logger.start_logging()
    temp_logger.plot_data()


    values = list(np.arange(60, 90, 5))
    # delay = 8  # 57.2  # 1 hour
    incrementor = Incrementor(
        signal_manager=device_manager,
        temp_controller=watlow_device.__name__,
        steps=values,
        delay=2,
        delay_unit="min",
        loop=True,
        n_loops=0,
        stability_check=False,
        stability_signal="Ave Temp",
        stability_tol=0.6,
        stability_time=150,
    )
    # incrementor.start()
    
    

    # def check_condition():
    #     # value = mfia_device.device["demods/0/sample"]()["frequency"].item()
    #     return mfia_device.frequency >= 4.1e6  # Example condition

    # def trigger_action():
    #     t_stable = incrementor.to_seconds(incrementor.time_stable)
    #     if mfia_device.impedance_run_status == 1:
    #         mfia_device.toggle_imps_module()
    #         # if stable for more than 5 min then this isnt a case of bad stepping
    #         if t_stable > 5*60:
    #             incrementor.next_step()
    #     # if a bad step, while loop would just move on, otherwise will continue after stable for 5 min
    #     n=0
    #     while incrementor.to_seconds(incrementor.time_stable) < 5*60 and n < 30:
    #         n+=1
    #         time.sleep(1)
    #     mfia_device.toggle_imps_module()
    #     print(f"Condition met @ {datetime.now().strftime('%y%m%d %H%M%S')}")
    
    # # poller = DevicePoller(mfia_base.device, "demods/0/sample", "frequency", check_condition, trigger_action, true_count=1, wait_time=2.0)
    # poller = DeviceTrigger(check_condition, trigger_action, true_count=1, wait_time=2.0)
    # poller.start()