import pandas as pd
import matplotlib.pyplot as plt
import threading
import time
from datetime import datetime

from .watlow import Watlow
from .uwtc import UWTC

# class WatlowLogger(Watlow):
#     '''
#     Subclass of Watlow that logs temperature and setpoint readings every n seconds.
#     '''
#     def __init__(self, port=None, baudrate=38400, timeout=0.5, address=1, **kwargs):
#         super().__init__(port, baudrate, timeout, address, **kwargs)
#         self.interval = kwargs.get('interval', 30)
#         self.integration_percent = kwargs.get('integration_percent', 0.2)
#         self.auto_save = kwargs.get('auto_save', False)
#         self.save_path = kwargs.get('save_path', 'data_log.csv')
#         self.df = pd.DataFrame(columns=['timestamp', 'time', 'temp', 'setpoint'])
#         self._running = False
#         self._paused = False
#         self._plotting = False
#         self._lock = threading.Lock()


#     def _log_data(self):
#         start_time = time.time()
#         integration_time = self.interval * self.integration_percent
#         temp_values = []
#         setpoint_values = []
        
#         while self._running:
#             if not self._paused:
#                 timestamp = datetime.now()
#                 elapsed_time = time.time() - start_time
#                 try:
#                     temp = self.temp()
#                     setpoint = self.setpoint()
#                 except Exception as e:
#                     print(f"Error reading temperature or setpoint: {e}")
#                     continue
#                 else:
#                     temp_values.append(temp)
#                     setpoint_values.append(setpoint)
                    
#                     if elapsed_time % self.interval < integration_time:
#                         avg_temp = sum(temp_values) / len(temp_values)
#                         avg_setpoint = sum(setpoint_values) / len(setpoint_values)
#                         new_data = pd.DataFrame([{
#                             'timestamp': timestamp,
#                             'time': elapsed_time,
#                             'temp': avg_temp,
#                             'setpoint': avg_setpoint
#                         }])
#                         temp_values.clear()
#                         setpoint_values.clear()
                        
#                         if self.df.empty:
#                             self.df = new_data
#                         elif not new_data.empty:
#                             with self._lock:
#                                 self.df = pd.concat([self.df, new_data], ignore_index=True)
#                         if self.auto_save:
#                             self.save_data()
#             # Calculate the time taken for the computation
#             computation_time = time.time() - elapsed_time - start_time
#             # Adjust the sleep time to account for computation time
#             adjusted_sleep_time = integration_time - computation_time
#             if adjusted_sleep_time > 0:
#                 time.sleep(adjusted_sleep_time)
    
#     def _plot_data(self):
        
#         while self._plotting:
#             with self._lock:
#                 x_data = self.df['time'].to_numpy()
#                 x_label = 'Time (s)'
#                 if max(x_data) > 3600:
#                     x_data = x_data / 3600 # Convert time to minutes
#                     x_label = 'Time (hr)'
#                 elif max(x_data) > 120:
#                     x_data = x_data / 60
#                     x_label = 'Time (min)'
                
#                 self.ax.clear()
#                 self.ax.plot(x_data, self.df['temp'], label='Temperature (C)')
#                 self.ax.plot(x_data, self.df['setpoint'], label='Setpoint (C)')
#                 self.ax.set_xlabel(x_label)
#                 self.ax.set_ylabel('Temperature (C)')
#                 self.ax.legend()
#                 self.fig.canvas.draw()
#                 self.fig.canvas.flush_events()
#             time.sleep(self.interval)
#                 # plt.pause(self.interval)
        

#     def start_logging(self):
#         '''
#         Start logging temperature and setpoint readings.
#         '''
#         if not self._running:
#             self._running = True
#             self._paused = False
#             self._thread = threading.Thread(target=self._log_data)
#             self._thread.start()

#     def stop_logging(self):
#         '''
#         Stop logging temperature and setpoint readings.
#         '''
#         self._running = False
#         if self._thread.is_alive():
#             self._thread.join()

#     def pause_logging(self):
#         '''
#         Pause logging temperature and setpoint readings.
#         '''
#         self._paused = True

#     def resume_logging(self):
#         '''
#         Resume logging temperature and setpoint readings.
#         '''
#         self._paused = False

#     def plot_data(self):
#         '''
#         Control whether to plot the logged data.
#         '''
#         if not self._plotting:
#             self._plotting = True
#             # plt.ion()
#             self.fig, self.ax = plt.subplots()
#             self._plot_thread = threading.Thread(target=self._plot_data)
#             self._plot_thread.start()
#             # plt.ioff()
#             plt.show()
#         else:
#             self._plotting = False
#             if self._plot_thread.is_alive():
#                 self._plot_thread.join()

#     def save_data(self):
#         '''
#         Save the logged data to a CSV file.
#         '''
#         with self._lock:
#             self.df.to_csv(self.save_path, index=False)

class TempLogger:
    """
    Generic temperature logger that logs readings from multiple objects every n seconds.
    """

    def __init__(self, interval=30, auto_save=False, save_path='data_log.csv'):
        self.objects = []
        self.functions = []
        self.names = []
        self.interval = interval
        self.auto_save = auto_save
        self.save_path = save_path
        self.df = pd.DataFrame(columns=['timestamp', 'time'] + self.names)
        self._running = False
        self._paused = False
        self._plotting = False
        self._lock = threading.Lock()

    def add_device(self, obj, *args):
        """
        Add a device to the logger.

        Parameters
        ----------
        obj : object
            The object to log data from.
        *args : tuple
            Each argument can be a string, a callable pair, or just a string.
        """
        if len(args) == 2 and any(callable(arg) for arg in args):
            self.add_device(obj, args)
        else:
            for arg in args:
                if isinstance(arg, str):
                    func = eval(f'obj.{arg}', {}, {'obj': obj})
                    name = arg
                elif isinstance(arg, tuple) and len(arg) == 2:
                    if callable(arg[0]):
                        func, name = arg
                    elif callable(arg[1]):
                        name, func = arg
                    else:
                        try:
                            func = eval(f'obj.{arg[0]}', {}, {'obj': obj})
                            name = arg[1]
                        except (NameError, AttributeError):
                            try:
                                func = eval(f'obj.{arg[1]}', {}, {'obj': obj})
                                name = arg[0]
                            except (NameError, AttributeError) as exc:
                                raise ValueError("Each argument must be a string or a (callable, name) pair.") from exc
                else:
                    raise ValueError("Each argument must be a string or a (callable, name) pair.")
                self.objects.append(obj)
                self.functions.append(func)
                self.names.append(name)
                self.df[name] = None

    def _log_data(self):
        start_time = time.time()

        while self._running:
            if not self._paused:
                timestamp = datetime.now()
                elapsed_time = time.time() - start_time
                data = []
                for func in self.functions:
                    try:
                        data.append(func())
                    except Exception:
                        data.append(0)
                new_data = pd.DataFrame([{
                    'timestamp': timestamp,
                    'time': elapsed_time,
                    **dict(zip(self.names, data))
                }])

                if self.df.empty:
                    self.df = new_data
                elif not new_data.empty:
                    with self._lock:
                        self.df = pd.concat([self.df, new_data], ignore_index=True)
                if self.auto_save:
                    self.save_data()
            # time.sleep(self.interval)
            # Calculate the time taken for the computation
            computation_time = time.time() - elapsed_time - start_time
            # Adjust the sleep time to account for computation time
            adjusted_sleep_time = self.interval - computation_time
            if adjusted_sleep_time > 0:
                time.sleep(adjusted_sleep_time)
            

    def _plot_data(self):
        while self._plotting:
            with self._lock:
                if self.df.empty:
                    continue
                x_data = self.df['time'].to_numpy()
                x_label = 'Time (s)'
                if max(x_data) > 3600:
                    x_data = x_data / 3600
                    x_label = 'Time (hr)'
                elif max(x_data) > 120:
                    x_data = x_data / 60
                    x_label = 'Time (min)'

                self.ax.clear()
                for name in self.names:
                    self.ax.plot(x_data, self.df[name], label=f'{name}')
                self.ax.set_xlabel(x_label)
                self.ax.set_ylabel('Value')
                self.ax.grid(True)
                self.ax.legend()
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            time.sleep(self.interval)

    def start_logging(self):
        """
        Start logging data.
        """
        if not self._running:
            self._running = True
            self._paused = False
            self._thread = threading.Thread(target=self._log_data)
            self._thread.start()

    def stop_logging(self):
        """
        Stop logging data.
        """
        self._running = False
        if self._thread.is_alive():
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

    def plot_data(self):
        """
        Control whether to plot the logged data.
        """
        if not self._plotting:
            self._plotting = True
            self.fig, self.ax = plt.subplots()
            self._plot_thread = threading.Thread(target=self._plot_data)
            self._plot_thread.start()
            plt.show()
        else:
            self._plotting = False
            if self._plot_thread.is_alive():
                self._plot_thread.join()

    def save_data(self):
        """
        Save the logged data to a CSV file.
        """
        with self._lock:
            self.df.to_csv(self.save_path, index=False)

class SignalStability:
    """
    SignalStability class to check the stability of a signal.

    Attributes
    ----------
    init_signal_func : callable
        Optional callable which represents an additional signal to evaluate with a faster response time than the primary signal.
    signal_func : callable
        Primary signal callable to check for stability.
    stability_tol : float
        Tolerance within which the value is considered stable.
    stability_time : float
        Time to wait for stability check.
    stability_n : int
        Number of consecutive measurements required for stability.

    Methods
    -------
    check_stability()
        Checks if the value from the callables is stable.
    """

    def __init__(self, signal_func=None, check=True, tol=0.5, time=120, count=5, init_signal_func=None):
        self._init_signal_func = None
        self._signal_func = None
        
        if isinstance(signal_func, (list, tuple)) and len(signal_func) == 2:
            self.init_signal_func, self.signal_func = signal_func
        elif signal_func is None and init_signal_func is not None:
            self.init_signal_func = None
            self.signal_func = init_signal_func
        else:
            self.init_signal_func = init_signal_func
            self.signal_func = signal_func
        
        if self.signal_func is None and self.init_signal_func is None:
            raise ValueError("signal_func must be provided")
        
        self.check = check
        self.tol = tol
        self.time = time if isinstance(time, (int, float)) else self._to_seconds(*time)
        self.count = count

    @property
    def init_signal_func(self):
        return self._init_signal_func or self._signal_func

    @init_signal_func.setter
    def init_signal_func(self, func):
        if func is None:
            self._init_signal_func = None
        elif callable(func):
            func_res = func()
            if not isinstance(func_res, (int,float)):
                raise ValueError("init_signal_func must return an int or float.")
            self._init_signal_func = func
        else:
            raise ValueError("init_signal_func must be a callable.")
        

    @property
    def signal_func(self):
        return self._signal_func

    @signal_func.setter
    def signal_func(self, func):
        if callable(func):
            func_res = func()
            if not isinstance(func_res, (int,float)):
                raise ValueError("init_signal_func must return an int or float.")
            self._signal_func = func
        else:
            raise ValueError("init_signal_func must be a callable.")

    def _to_seconds(self, val=0, unit='s'):
        """
        Convert the delay time to seconds based on the delay unit.
        """
        if not isinstance(val, (int, float)) or not isinstance(unit, str):
            raise ValueError("Invalid delay value or unit.")
        if unit.startswith('h'):
            res = val * 3600
        elif unit.startswith('min'):
            res = val * 60
        elif unit.startswith('sec'):
            res = val
        else:
            raise ValueError("Invalid delay unit. Choose from 'hours', 'minutes', or 'seconds'.")
        return res

    def check_stability(self):
        """
        Check if the value from the callables is stable.
        """
        if self.check:
            func = self.init_signal_func
            stable_count = 0
            previous_value = 0
            while stable_count < self.count:
                current_value = func()
                if not isinstance(current_value, float):
                    raise ValueError("init_signal_func must return a float.")
                if abs(current_value - previous_value) <= self.tol:
                    stable_count += 1
                else:
                    stable_count = 0
                previous_value = current_value
                time.sleep(self.time / self.count)
            
            func = self.signal_func
            stable_count = 0
            previous_value = 0
            while stable_count < self.count:
                current_value = func()
                if not isinstance(current_value, float):
                    raise ValueError("signal_func must return a float.")
                if abs(current_value - previous_value) <= self.tol:
                    stable_count += 1
                else:
                    stable_count = 0
                previous_value = current_value
                time.sleep(self.time * (self.count - 1) / self.count)


class Incrementor:
    """
    Incrementor class to set a temperature controller to requested values with a delay.

    Attributes
    ----------
    values : list
        List of values to set the temperature controller to.
    delay : float
        Delay time between setting values.
    delay_unit : str
        Unit of delay time ('hours', 'minutes', 'seconds').
    loop : bool
        Whether to loop the values continuously.
    n_loops : int
        Number of loops to run if loop is True.
    stability_check : bool
        Whether to check for stability before proceeding.
    stability : SignalStability
        SignalStability instance to check for stability.

    Methods
    -------
    run()
        Runs the incrementor to set the temperature controller to the requested values.
    start()
        Starts the incrementor in a separate thread.
    stop()
        Stops the incrementor.
    pause()
        Pauses the incrementor.
    resume()
        Resumes the incrementor.
    next()
        Skips to the next step.
    previous()
        Goes back to the previous step.
    """

    def __init__(self, temp_controller, values, delay, delay_unit='seconds', loop=False, n_loops=1, stability_check=False, stability_callable=None, stability_tol=0.5, stability_time=120):
        self.temp_controller = temp_controller
        self.values = values
        self.delay = delay
        self.delay_unit = delay_unit
        self.loop = loop
        self.n_loops = n_loops
        # self.stability_check = stability_check
        self.stability = SignalStability([temp_controller.temp, stability_callable], stability_check, stability_tol, stability_time, 5)
        self._step = 0
        self._running = False
        self._paused = False
        self._thread = None

    def _to_seconds(self, val=0, unit=''):
        """
        Convert the delay time to seconds based on the delay unit.
        """
        val = val or self.delay
        unit = unit or self.delay_unit
        if not isinstance(val, (int, float)) or not isinstance(unit, str):
            raise ValueError("Invalid delay value or unit.")
        if unit.startswith('h'):
            res = val * 3600
        elif unit.startswith('min'):
            res = val * 60
        elif unit.startswith('sec'):
            res = val
        else:
            raise ValueError("Invalid delay unit. Choose from 'hours', 'minutes', or 'seconds'.")
        if self.stability.check:
            res -= self.stability.time
        return res

    @property
    def step(self):
        return self.values[self._step]
    
    @step.setter
    def step(self, value):
        if value in self.values:
            self._step = self.values.index(value)
        elif isinstance(value, int) and 0 <= value < len(self.values):
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
        delay_seconds = self._to_seconds()
        loop_count = 0
        self._running = True

        while self._running:
            while self._step < len(self.values):
                if not self._running:
                    break
                while self._paused:
                    time.sleep(0.1)
                value = self.values[self._step]
                print(f"Setting temperature to: {value}°C")
                self.temp_controller.setpoint(value)
                
                # if self.stability.check:
                self.stability.check_stability()
                
                # Break down the sleep time into smaller intervals
                sleep_interval = min(1, delay_seconds)
                elapsed_time = 0
                while elapsed_time < delay_seconds:
                    if not self._running:
                        break
                    time.sleep(sleep_interval)
                    elapsed_time += sleep_interval

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
        if self._running:
            self.stop()
        self._thread = threading.Thread(target=self.run)
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

    def next(self):
        """
        Skips to the next step.
        """
        step = self._step + 1
        if step >= len(self.values):
            step = 0
        self.start(step)

    def previous(self):
        """
        Goes back to the previous step.
        """
        step = self._step - 1
        if step < 0:
            step = len(self.values) - 1
        self.start(step)


if __name__ == "__main__":
    import numpy as np
    # ez_zone_logger = WatlowLogger(port="com4", interval=2)
    # ez_zone_logger.start_logging()
    # time.sleep(10)  # Log data for 10 seconds
    # ez_zone_logger.pause_logging()
    # time.sleep(5)  # Pause logging for 5 seconds
    # ez_zone_logger.resume_logging()
    # time.sleep(10)  # Resume logging for another 10 seconds
    # ez_zone_logger.stop_logging()
    # ez_zone_logger.plot_data()

    watlow_device = Watlow(port="COM4")
    uwtc_device = UWTC(port="COM.")

    temp_logger = TempLogger(interval=2)
    temp_logger.add_device(watlow_device, 'temp', 'setpoint')
    temp_logger.add_device(uwtc_device, ('temp', 'uwtc_measured'), (lambda: uwtc_device.previous_reply["ambient"], 'uwtc_ambient'))

    temp_logger.start_logging()
    temp_logger.plot_data()

    values = list(np.arange(85,57.5,-2.5))
    delay = 40 #57.2  # 1 hour
    incrementor = Incrementor(temp_controller=watlow_device, values=values, delay=delay, delay_unit='min', loop=True, n_loops=0, stability_check=True, stability_callable=uwtc_device.temp)
    incrementor.start()