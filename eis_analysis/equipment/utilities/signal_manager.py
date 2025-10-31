import time
import threading
from typing import Any, Literal, NamedTuple, overload
from collections.abc import Callable

import numpy as np


def is_number(x):
    return isinstance(x, (int, float, np.integer, np.floating))


class Signal(NamedTuple):
    obj: object
    function: Callable
    name: str
    settable: bool = False
    conditional: Callable = lambda x: True
    notes: str = ""


class SignalManager:
    """
    SignalManager class to manage devices and their signals.

    Attributes
    ----------
    devices : list
        list of devices added to the manager.
    signals : dict
        Dictionary of signals with their associated callables and settable flags.
    signal_groups : dict
        Dictionary of signal groups with their associated signal names.

    Methods
    -------
    add_device(obj, *args)
        Adds a device and its signals to the manager.
    define_signal(name, signals)
        Defines a signal group by associating a name with a list of signals.
    get(name)
        Retrieves the value(s) of the specified signal or signal group.
    set(name, value)
        Sets the value of the specified signal if it is settable.
    """

    def __init__(self):
        self.devices = []
        self.signals = {}
        # self.signal_groups = {}
        self._lock = threading.Lock()

    def add_device(self, obj: object, *args: Any):
        """
        Add a device and its signals to the manager.

        Parameters
        ----------
        obj : object
            The device object to add.
        *args : tuple
            Each argument can be a string or a tuple (callable, name, settable).
            [str | tuple[Callable, str, bool, Callable]]
        """
        self.devices.append(obj)
        for arg in args:
            if isinstance(arg, str):
                arg = (arg, arg)
            elif isinstance(arg, Signal):
                # self._validate_signal_function(arg.function)
                self.signals[arg.name] = arg
                continue
            elif not isinstance(arg, (tuple, list)) or len(arg) < 2:
                raise ValueError(
                    "Each argument must be a tuple (callable or str, name, settable, conditional)."
                )

            func, name = arg[0], arg[1]
            settable = arg[2] if len(arg) > 2 else False
            conditional = arg[3] if len(arg) > 3 else is_number

            if isinstance(func, str):
                if hasattr(obj, func):
                    func = eval(f"obj.{func}", {}, {"obj": obj})
                else:
                    raise AttributeError(f"Attribute '{func}' not found in the object.")

            # self._validate_signal_function(func)

            self.signals[name] = Signal(obj, func, name, settable, conditional)

    def define_signal(
        self,
        name: str,
        signals: list[str],
        combining_function: Callable | None = None,
        **kwargs,
    ):
        """
        Define a new signal by associating a name with a list of signals.

        Parameters
        ----------
        name : str
            The name of the new signal.
        signals : list
            list of signal names to include in the new signal.
        """
        if name in self.signals:
            raise ValueError(f"Signal '{name}' already exists.")

        def make_comb_getter(source, comb_func: Callable | None = None):
            comb_func = comb_func if callable(comb_func) else np.mean

            def get_combined_signal(values=None):
                try:
                    if values is not None:
                        return float(comb_func(values))
                    return float(comb_func(source.get(output_format="list")))
                except (TypeError, ValueError):
                    return values if values is not None else source.get()

            return get_combined_signal

        if isinstance(signals, str):
            signals = [signals]

        if len(signals) == 1:
            self.signals[name] = self.signals[signals[0]]
        else:
            sub_signal_manager = SignalManager()
            for signal in signals:
                sub_signal_manager.signals[signal] = self.signals[signal]
            func = make_comb_getter(sub_signal_manager, combining_function)
            self.signals[name] = Signal(
                sub_signal_manager, func, name, False, is_number, ", ".join(signals)
            )

    # def _validate_signal_function(self, func: Callable):
    #     """
    #     Validate that the signal function returns an int, float, or numpy equivalent.

    #     Parameters
    #     ----------
    #     func : Callable
    #         The signal function to validate.

    #     Raises
    #     ------
    #     ValueError
    #         If the function does not return an int, float, or numpy equivalent.
    #     """
    #     result = None
    #     if callable(func):
    #         result = func()
    #     else:
    #         raise ValueError("Signal function must be callable.")
    #     if not isinstance(result, (int, float, np.integer, np.floating)):
    #         raise ValueError("Signal function must return an int, float, or numpy equivalent.")
    # @overload
    # def get(self, name: str | list[str], output_format: str = "list") -> list: ...
    # @overload
    # def get(self, name: str | list[str], output_format: str = "tuple") -> list: ...
    @overload
    def get(self, name: str | list[str], output_format: Literal["list", "tuple"]) -> list: ...
    @overload
    def get(self, name: str | list[str], output_format: Literal["mean", "float"]) -> float: ...
    @overload
    def get(self, name: str | list[str], output_format: None = ...) -> dict: ...
    @overload
    def get(self, name: str | list[str]) -> dict: ...

    def get(
        self, name: str | list[str] = "all", output_format: str | None = None
    ) -> dict | list | float:
        """
        Retrieve the value(s) of the specified signal or signal group.

        Parameters
        ----------
        name : str | list[str]
            The name of the signal or signal group, or a list of signal names.
        output_format : str, optional
            The format to return the value(s) in. Default is None which returns single
            values as floats and multiple values as lists. Other options are
            'list', 'dict', 'mean', or 'float'.

        Returns
        -------
        dict or list
            The value(s) of the specified signal or signal group.
        """

        def get_signal_value(signal: Signal):
            """
            Get the value of a signal.

            Parameters
            ----------
            signal : Signal
                The signal to get the value of.

            Returns
            -------
            float
                The value of the signal.
            """
            val = 0
            try:
                if hasattr(signal.obj, "lock"):
                    with signal.obj.lock:  # type: ignore
                        val = signal.function() if callable(signal.function) else signal.function
                else:
                    for _ in range(5):
                        val = signal.function() if callable(signal.function) else signal.function
                        if signal.conditional(val):
                            break
                        time.sleep(0.5)
            except (ValueError, PermissionError, FileNotFoundError, IOError):
                val = 0
            return val

        with self._lock:
            if isinstance(name, str):
                names = list(self.signals.keys()) if name.lower() == "all" else [name]
            else:
                names = name

            res = {}
            combined_signals = {}
            for n in names:
                if n in self.signals:
                    signal = self.signals[n]
                    if signal.notes:
                        # notes typically contain the list of source signals
                        combined_signals[n] = signal
                    else:
                        res[n] = get_signal_value(signal)
                else:
                    raise ValueError(f"Signal '{n}' not found.")

            for n, signal in combined_signals.items():
                source_signals = signal.notes.split(", ")
                source_vals = []
                for src in source_signals:
                    if src in res:
                        source_vals.append(res[src])
                    elif src in self.signals:
                        source_vals.append(get_signal_value(self.signals[src]))
                    else:
                        raise ValueError(f"Source signal '{src}' not found.")
                res[n] = signal.function(source_vals)
                # if all(src in res for src in source_signals):
                #     values = [res[src] for src in source_signals]
                #     res[n] = signal.function(values)
                # else:
                #     res[n] = get_signal_value(signal)

        if str(output_format).lower() in ["list", "tuple"]:
            return list(res.values())
        if str(output_format).lower() in ["mean", "float"]:
            return float(np.mean(list(res.values())))
        return res

    def set_to(self, name: str, value: float):
        """
        Set the value of the specified signal if it is settable.

        Parameters
        ----------
        name : str
            The name of the signal.
        value : float
            The value to set.

        Raises
        ------
        ValueError
            If the signal is not settable or not found.
        """
        # with self._lock:
        signal = self.signals.get(name)
        if isinstance(signal, Signal) and signal.settable:
            signal.function(value)
        else:
            raise ValueError(f"Signal '{name}' is not settable or not found.")
