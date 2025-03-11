# -*- coding: utf-8 -*-
"""
A module to manage serial port connections using the Singleton pattern.

This module provides a class to manage serial port connections using the Singleton pattern.
The class ensures that only one instance of a serial port connection is created for each unique port.

Created with the help of the following resources:
- https://stackoverflow.com/a/6798042
- GitHub Copilot
"""


import threading
import warnings

import serial as ser


class SerialManager:
    """
    A class to manage serial port connections using the Singleton pattern.

    This class ensures that only one instance of a serial port connection is created
    for each unique port. If an attempt is made to create a new instance with a different
    baudrate for an already existing port, a ValueError is raised. If the timeout is different,
    a warning is issued, and the old timeout is maintained.

    Attributes
    ----------
    port : str
        The serial port to connect to.
    baudrate : int
        The baudrate for the serial connection.
    timeout : float
        The timeout for the serial connection.
    serial : serial.Serial
        The serial connection object.

    Methods
    -------
    open()
        Opens the serial connection.
    close()
        Closes the serial connection.
    write(data)
        Writes data to the serial connection.
    read(size)
        Reads data from the serial connection.
    __enter__()
        Enters the runtime context related to this object.
    __exit__(exc_type, exc_val, exc_tb)
        Exits the runtime context related to this object.
    """

    _instances = {}
    _lock = threading.Lock()

    def __new__(cls, port, baudrate=38400, timeout=0.5):
        """
        Creates a new instance of SerialManager or returns the existing instance for the given port.

        Parameters
        ----------
        port : str
            The serial port to connect to.
        baudrate : int, optional
            The baudrate for the serial connection (default is 38400).
        timeout : float, optional
            The timeout for the serial connection (default is 0.5).

        Returns
        -------
        SerialManager
            The instance of SerialManager for the given port.

        Raises
        ------
        ValueError
            If the port is already in use with a different baudrate.
        """
        key = port
        with cls._lock:
            if key in cls._instances:
                instance = cls._instances[key]
                if instance.baudrate != baudrate:
                    raise ValueError(f"Port {port} is already in use with a different baudrate.")
                if instance.timeout != timeout:
                    warnings.warn(
                        f"Port {port} is already in use with a different timeout. "
                        f"Using the existing timeout of {instance.timeout} seconds."
                    )
            else:
                instance = super().__new__(cls)
                instance._initialize(port, baudrate, timeout)
                cls._instances[key] = instance
            return instance

    def _initialize(self, port, baudrate, timeout):
        """
        Initializes the SerialManager instance.

        Parameters
        ----------
        port : str
            The serial port to connect to.
        baudrate : int
            The baudrate for the serial connection.
        timeout : float
            The timeout for the serial connection.
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._lock = threading.Lock()
        self.serial = None

    def open(self):
        """
        Opens the serial connection.
        """
        with self._lock:
            if self.serial is None or not self.serial.is_open:
                self.serial = ser.Serial(self.port, self.baudrate, timeout=self.timeout)

    def close(self):
        """
        Closes the serial connection.
        """
        with self._lock:
            if self.serial and self.serial.is_open:
                self.serial.close()

    def write(self, data):
        """
        Writes data to the serial connection.

        Parameters
        ----------
        data : bytes
            The data to write to the serial connection.
        """
        with self._lock:
            self.serial.write(data)

    def read(self, size):
        """
        Reads data from the serial connection.

        Parameters
        ----------
        size : int
            The number of bytes to read from the serial connection.

        Returns
        -------
        bytes
            The data read from the serial connection.
        """
        with self._lock:
            return self.serial.read(size)

    def __enter__(self):
        """
        Enters the runtime context related to this object.

        Returns
        -------
        SerialManager
            The instance of SerialManager.
        """
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exits the runtime context related to this object.

        Parameters
        ----------
        exc_type : type
            The exception type.
        exc_val : Exception
            The exception value.
        exc_tb : traceback
            The traceback object.
        """
        self.close()


# Usage example
if __name__ == "__main__":
    try:
        with SerialManager(port="COM4", baudrate=38400, timeout=0.5) as serial_manager:
            serial_manager.write(b"some data")
            response = serial_manager.read(20)
            print(response)
    except ValueError as e:
        print(e)
