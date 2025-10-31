# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:14:23 2025

This module provides classes for interfacing with Watlow PID temperature controllers
via BACnet TP/MS messages. The code is adapted from the pywatlow project available at
https://github.com/BrendanSweeny/pywatlow.git, which is licensed under the GNU LGPL v3.

Classes:
    WatlowSerial: Base class for generating and parsing BACnet TP/MS messages.
    Watlow: Extended class with additional functionality for temperature control.


@author: DEfECT
"""

__all__ = ["Watlow"]

import time
import struct
from typing import Any, NamedTuple
from binascii import hexlify, unhexlify

import numpy as np
import crcmod

try:
    from ..utilities.thread_tools import SThread
    from ..utilities.watlow_tables import crc_8_table, watlow_params
    from ..utilities.serial_manager import SerialManager
except ImportError:
    from eis_analysis.equipment.utilities.thread_tools import SThread
    from eis_analysis.equipment.utilities.watlow_tables import crc_8_table, watlow_params
    from eis_analysis.equipment.utilities.serial_manager import SerialManager


BASIC_CONVERSIONS = {
    "f_to_c": lambda f: (f - 32.0) / (9.0 / 5.0),
    "k_to_c": lambda k: k - 273.15,
    "c_to_f": lambda c: c * (9.0 / 5.0) + 32.0,
    "c_to_k": lambda c: c + 273.15,
}


class WatlowOutput(NamedTuple):
    """
    A named tuple representing the output from a Watlow temperature controller.

    Attributes
    ----------
    address : int
        The address of the Watlow controller.
    param : int
        The parameter ID.
    data : int | float
        The data value.
    error : Exception  | None
        The error message if an error occurred.
    """

    address: int | None = None
    param: int | None = None
    data: int | float | None = None
    error: Exception | str | None = None


class WatlowSerial:
    """
    Object representing a Watlow PID temperature controller. This class
    facilitates the generation and parsing of BACnet TP/MS messages to and from
    Watlow temperature controllers.

    Parameters
    ----------
    port : str, optional
        String representing the serial port or `None`.
    baudrate : int, optional
        Baudrate for the serial communication. Default is 38400.
    timeout : float, optional
        Read timeout value in seconds. Default is 0.5.
    address : int, optional
        Watlow controller address (found in the setup menu). Acceptable values are 1 through 16. Default is 1.
    retries : int, optional
        Number of retries for communication. Default is 3.

    Attributes
    ----------
    timeout : float
        Read timeout value in seconds.
    baudrate : int
        Baudrate for the serial communication.
    address : int
        Watlow controller address.
    port : str
        String representing the serial port.
    retries : int
        Number of retries for communication.
    serial_manager : SerialManager
        Instance of SerialManager for managing the serial connection.
    """

    KNOWN_PARAMS = watlow_params

    def __init__(
        self,
        port: str = "",
        baudrate: int = 38400,
        timeout: float = 0.5,
        address: int = 1,
        retries: int = 3,
    ):
        self.timeout = timeout
        self.baudrate = baudrate
        self.address = address
        self.port = port
        self.retries = retries
        self.__name__ = f"watlow_{address} (port: {port})"
        self.serial_manager = SerialManager(port, baudrate, timeout)
        self._simulate = False
        self._sim_value = 124.0
        self._print_sim_events = True

        if "sim" in port.lower():
            self._simulation_mode()

    def _simulation_mode(self, print_events=True) -> None:
        """
        Sets the simulation mode for the Watlow controller.
        """
        self._simulate = not self._simulate
        self._sim_value = 124.0
        self._print_sim_events = print_events

    def _header_check_byte(self, header_bytes: bytes | bytearray) -> bytes:
        """
        Takes the full header byte array bytes[0] through bytes[6] of the full
        command and returns a check byte (bytearray of length one) using Watlow's
        algorithm.

        Implementation relies on this post:
        https://reverseengineering.stackexchange.com/questions/8303/rs-485-checksum-reverse-engineering-watlow-ez-zone-pm

        Parameters
        ----------
        header_bytes : bytearray
            The header byte array.

        Returns
        -------
        bytes
            The check byte.
        """
        intCheck = ~crc_8_table[
            header_bytes[6]
            ^ crc_8_table[
                header_bytes[5]
                ^ crc_8_table[
                    header_bytes[4] ^ crc_8_table[header_bytes[3] ^ crc_8_table[~header_bytes[2]]]
                ]
            ]
        ] & (2**8 - 1)
        return bytes([intCheck])

    def _data_check_byte(self, data_bytes: bytes | bytearray) -> bytes:
        """
        Takes the full data byte array, bytes[8] through bytes[13] of the full
        command and calculates the data check byte using BacNET CRC-16.

        Parameters
        ----------
        data_bytes : bytearray
            The data byte array.

        Returns
        -------
        bytes
            The data check byte.
        """
        # CRC-16 with 0xFFFF as initial value, 0x1021 as polynomial, bit reversed
        crc_fun = crcmod.mkCrcFun(poly=0x11021, initCrc=0, rev=True, xorOut=0xFFFF)
        # bytes object packed using C-type unsigned short, little-endian:
        byte_str = struct.pack("<H", crc_fun(data_bytes))
        return byte_str

    def _build_request(
        self,
        data_param: int,
        request_type: str,
        value: float | None = None,
        instance: str = "01",
    ) -> bytearray:
        """
        Helper function to build a request byte array for read, write_float, and write_int.

        Parameters
        ----------
        data_param : int
            The data parameter ID.
        request_type : str
            The type of request ('read', 'write_float', 'write_int').
        value : int or float, optional
            The value to write for write requests. Default is None.
        instance : str, optional
            The instance to read/write. Default is "01".

        Returns
        -------
        bytearray
            The request byte array.
        """
        # Reformats data param from notation in the manual to hex string
        s_data_param = format(int(data_param), "05d")
        s_data_param = hexlify(
            int(s_data_param[:2]).to_bytes(1, "big") + int(s_data_param[2:]).to_bytes(1, "big")
        ).decode("utf-8")

        b_value = None
        if request_type == "read":
            header_suffix = "000006"
            data_prefix = "010301"
            data_suffix = ""
        elif request_type == "write_float":
            if value is None:
                raise ValueError("Value must be provided for write_float request")
            header_suffix = "00000a"
            data_prefix = "0104"
            data_suffix = "08"
            b_value = struct.pack(">f", float(value))
        elif request_type == "write_int":
            if value is None:
                raise ValueError("Value must be provided for write_int request")
            header_suffix = "030009"
            data_prefix = "0104"
            data_suffix = "0f01"
            b_value = int(value).to_bytes(2, "big")
        else:
            raise ValueError("Invalid request type")

        # Request Header: BACnet preamble, request type, zone, additional header
        hex_header = "55ff" + "05" + format(self.address + 15, "x") + header_suffix
        hex_header = unhexlify(hex_header)
        header_chk = self._header_check_byte(hex_header)

        # Request Data: data prefix, data param in hex, instance, data suffix
        hex_data = data_prefix + s_data_param + instance + data_suffix
        hex_data = unhexlify(hex_data)
        if b_value is not None:
            hex_data += b_value
        data_chk = self._data_check_byte(hex_data)

        # Assemble request byte array:
        request = bytearray(hex_header)
        request += bytearray(header_chk)
        request += bytearray(hex_data)
        request += data_chk

        return request

    def _validate_response(self, bytes_response: bytearray) -> bool:
        """
        Compares check bytes received in response to those calculated.

        Evaluate header_chk as bytearray instead of as an int (which is how
        python will interpret a single hex character)

        Parameters
        ----------
        bytes_response : bytearray
            The response byte array.

        Returns
        -------
        bool
            True if the response is valid, False otherwise.
        """
        is_valid = False

        header_chk_received = bytearray([bytes_response[7]])
        data_check_received = bytes_response[-2:]
        address_received = bytes_response[4] - 15
        if (
            header_chk_received == self._header_check_byte(bytes_response[0:7])
            and data_check_received == self._data_check_byte(bytes_response[8:-2])
            and address_received == self.address
        ):
            is_valid = True
        return is_valid

    def _parse_response(self, bytes_response: bytearray) -> WatlowOutput:
        """
        Takes the full response byte array and extracts the relevant data (e.g.
        current temperature), constructs a WatlowOutput named tuple, and returns it.

        Parameters
        ----------
        bytes_response : bytearray
            The response byte array.

        Returns
        -------
        WatlowOutput
            The response data.
        """
        # output = WatlowOutput(address=self.address)

        def byte_to_int(hex_param: bytes) -> int:
            r"""
            Reformats data parameter from bytes string to integer (e.g. b'\x1a\x1d' to 26029).
            """
            return int(str(hex_param[0]) + format(hex_param[1], "03d"))

        try:
            if bytes_response == b"" or bytes_response == bytearray(len(bytes_response)):
                raise ValueError(f"No response from address {self.address}")
            if not self._validate_response(bytes_response):
                print(
                    f"Invalid Response at address {self.address}: ",
                    hexlify(bytes_response),
                )
                raise ValueError(f"Invalid response received from address {self.address}")
        except (ValueError, PermissionError, FileNotFoundError, IOError) as e:
            return WatlowOutput(address=self.address, error=e)
        else:
            # Case where response data value is an int used to represent a state defined
            # in the manual (e.g. param 8003, heat algorithm, where 62 means 'PID')
            # from a read request
            # Hex byte 7: '0a', Hex bytes 15, 16: 0F, 01
            if bytes_response[6] == 10 and bytes_response[-6] == 15 and bytes_response[-5] == 1:
                data = bytes_response[-4:-2]
                output = WatlowOutput(
                    address=self.address,
                    param=byte_to_int(bytes_response[11:13]),
                    data=int.from_bytes(data, byteorder="big"),
                )
            # Case where response data value is a float from a set param request
            # (e.g. 7001, process value setpoint)
            # Hex byte 7: '0a', Hex byte 14: '08'
            elif bytes_response[6] == 10 and bytes_response[-7] == 8:
                data = hexlify(bytes_response[-6:-2])
                output = WatlowOutput(
                    address=self.address,
                    param=byte_to_int(bytes_response[10:12]),
                    data=struct.unpack(">f", unhexlify(data))[0],
                )
            # Case where response data value is an integer from a set param
            # request (e.g. param 8003, heat algorithm, where 62 means 'PID')
            # Hex byte 7: '09'
            elif bytes_response[6] == 9:
                data = bytes_response[-4:-2]
                output = WatlowOutput(
                    address=self.address,
                    param=byte_to_int(bytes_response[10:12]),
                    data=int.from_bytes(data, byteorder="big"),
                )
            # Case where data value is a float representing a process value
            # (e.g. 4001, where current temp of 50.0 is returned)
            # Hex byte 7: '0b'
            elif bytes_response[6] == 11:
                data = bytes_response[-6:-2]
                output = WatlowOutput(
                    address=self.address,
                    param=byte_to_int(bytes_response[11:13]),
                    data=struct.unpack(">f", data)[0],
                )
            # Other cases, such as response from trying to write a read-only parameter:
            else:
                output = WatlowOutput(
                    address=self.address,
                    error=ValueError(
                        f"Received a message that could not be parsed from address {self.address}"
                    ),
                )

            return output

    def read_param(
        self,
        param: int,
        data_type: type,
        *,
        instance: str = "01",
        retries: int | None = None,
        **_,
    ) -> Any:
        """
        Takes a parameter and writes data to the watlow controller at
        object's internal address. Using this function requires knowing the data
        type for the parameter (int or float). See the Watlow
        `user manual <https://www.watlow.com/-/media/documents/user-manuals/pm-pid-1.ashx>`_
        for individual parameters and the Usage section of these docs.

        Parameters
        ----------
        param : int
            A four digit integer corresponding to a Watlow parameter (e.g. 4001, 7001).
        data_type : type
            The Python type representing the data value type (i.e. `int` or `float`).
        instance : str, optional
            A two digit string corresponding to the channel to read (e.g. '01', '05'). Default is "01".
        retries : int, optional
            Number of retries for communication. Default is None.

        Returns
        -------
        WatlowOutput
            A named tuple containing the response data, parameter ID, and address.
        """
        if self._simulate:
            val = self._sim_value if data_type == float else int(self._sim_value)
            val = val + np.random.normal(0, 0.1)
            if self._print_sim_events:
                print(f"Watlow simulated read for {param} at: {val}")
            return WatlowOutput(self.address, param, val, "Simulation mode")
        request = self._build_request(param, "read", instance=instance)
        attempt = 0
        retries = retries if isinstance(retries, int) else self.retries
        while attempt < retries:
            try:
                with self.serial_manager as serial:
                    serial.write(request)
                    response = serial.read(21) if data_type == float else serial.read(20)
                    output = self._parse_response(response)
                    return output
            except (ValueError, PermissionError, FileNotFoundError, IOError) as e:
                attempt += 1
                if attempt >= retries:
                    return WatlowOutput(address=self.address, error=e)
                time.sleep(self.timeout)

    def write_param(
        self,
        param: int,
        value: int | float,
        data_type: type,
        *,
        instance: str = "01",
        retries: int | None = None,
        **_,
    ) -> Any:
        """
        Changes the value of the passed watlow parameter ID. Using this function
        requires knowing the data type for the parameter (int or float).
        See the Watlow
        `user manual <https://www.watlow.com/-/media/documents/user-manuals/pm-pid-1.ashx>`_
        for individual parameters and the Usage section of these docs.

        Parameters
        ----------
        param : int
            A four digit integer corresponding to a Watlow parameter (e.g. 4001, 7001).
        value : int or float
            An int or float representing the new target setpoint in degrees F by default.
        data_type : type
            The Python type representing the data value type (i.e. `int` or `float`).
        instance : str, optional
            A two digit string corresponding to the channel to read (e.g. '01', '05'). Default is "01".
        retries : int, optional
            Number of retries for communication. Default is None.

        Returns
        -------
        WatlowOutput
            A named tuple containing the response data, parameter ID, and address.
        """
        if self._simulate:
            self._sim_value = float(value)
            if self._print_sim_events:
                print(f"Watlow simulated write for {param} at: {self._sim_value}")
            return WatlowOutput(self.address, param, value, "Simulation mode")
        request_type = "write_float" if data_type == float else "write_int"
        request = self._build_request(param, request_type, value, instance)
        attempt = 0
        retries = retries if isinstance(retries, int) else self.retries
        while attempt < retries:
            try:
                with self.serial_manager as serial:
                    serial.write(request)
                    response = serial.read(20) if data_type == float else serial.read(19)
                    output = self._parse_response(response)
                    return output
            except (ValueError, PermissionError, FileNotFoundError, IOError) as e:
                attempt += 1
                if attempt >= retries:
                    return WatlowOutput(address=self.address, error=e)
                time.sleep(self.timeout)


class Watlow(WatlowSerial):
    """
    Object representing a Watlow PID temperature controller. This class
    facilitates the generation and parsing of BACnet TP/MS messages to and from
    Watlow temperature controllers.

    Parameters
    ----------
    port : str, optional
        String representing the serial port or `None`.
    baudrate : int, optional
        Baudrate for the serial communication. Default is 38400.
    timeout : float, optional
        Read timeout value in seconds. Default is 0.5.
    address : int, optional
        Watlow controller address (found in the setup menu). Acceptable values are 1 through 16. Default is 1.
    **kwargs : dict, optional
        Additional keyword arguments:
        - full_output : bool, optional
            Whether to return full output including metadata. Default is False.
        - temp_unit : str, optional
            Temperature unit ('C' for Celsius, 'F' for Fahrenheit). Default is 'C'.
        - mode : str, optional
            Mode for changing setpoint ('rapid', 'gradual', 'normal'). Default is 'normal'.
        - mode_duration : int, optional
            Duration in seconds for setpoint change. Default is 300.
        - mode_offset : float, optional
            Offset for overshoot/undershoot in setpoint change. Default is 5.
        - max_setpoint : float, optional
            Maximum allowable setpoint. Default is None.
        - min_setpoint : float, optional
            Minimum allowable setpoint. Default is None.

    Attributes
    ----------
    full_output : bool
        Whether to return full output including metadata.
    temp_unit : str
        Temperature unit ('C' for Celsius, 'F' for Fahrenheit).
    mode : str
        Mode for changing setpoint ('rapid', 'gradual', 'normal').
    mode_duration : int
        Duration in seconds for setpoint change.
    mode_offset : float
        Offset for overshoot/undershoot in setpoint change.
    max_setpoint : float
        Maximum allowable setpoint.
    min_setpoint : float
        Minimum allowable setpoint.

    This class extends WatlowSerial with additional functionality for temperature control.

    This class is adapted from the pywatlow project available at
    https://github.com/BrendanSweeny/pywatlow.git, which is licensed under the GNU LGPL v3.
    """

    def __init__(self, port="", baudrate=38400, timeout=0.5, address=1, **kwargs):
        super().__init__(port, baudrate, timeout, address)
        self.full_output = kwargs.get("full_output", False)
        self.temp_unit = kwargs.get("temp_unit", "C")
        self.mode = kwargs.get("mode", "normal")
        self.mode_duration = kwargs.get("mode_duration", "auto")
        self.mode_offset = kwargs.get("mode_offset", "auto")
        self.max_setpoint = kwargs.get("max_setpoint", None)
        self.min_setpoint = kwargs.get("min_setpoint", None)

        if self.min_setpoint is None:
            self.min_setpoint = self.read_param(
                7003, float, convert=self.temp_unit, full_output=False
            )
            if not isinstance(self.min_setpoint, (float, int)):
                self.min_setpoint = 0.0
        if self.max_setpoint is None:
            self.max_setpoint = self.read_param(
                7004, float, convert=self.temp_unit, full_output=False
            )
            if not isinstance(self.max_setpoint, (float, int)):
                self.max_setpoint = 400.0

    def read_param(self, param, data_type, convert=None, **kwargs):
        """
        Reads parameter from the Watlow device.

        Parameters
        ----------
        param : int
            The parameter ID to read.
        data_type : type
            The Python type representing the data value type (i.e. `int` or `float`).
        convert : str, optional
            The unit to convert the data to ('C' for Celsius, 'F' for Fahrenheit). Default is None.
        **kwargs : dict, optional
            Additional keyword arguments:
            - full_output : bool, optional
                Whether to return full output including metadata. Default is False.

        Returns
        -------
        dict or float
            The resulting values or the full output including metadata.
        """
        result = super().read_param(param, data_type, **kwargs)
        if result.error and not self._simulate:
            if kwargs.get("full_output", self.full_output):
                return result
            return result.error
        if str(convert).lower() == "c":
            result = result._replace(data=BASIC_CONVERSIONS["f_to_c"](result.data))
            # result.data = BASIC_CONVERSIONS["f_to_c"](result.data)
        if kwargs.get("full_output", self.full_output):
            return result
        return result.data

    def write_param(self, param, value, data_type, convert=None, **kwargs):
        """
        Writes parameter to the Watlow device.

        Parameters
        ----------
        param : int
            The parameter ID to write.
        value : int or float
            The value to write to the parameter.
        data_type : type
            The Python type representing the data value type (i.e. `int` or `float`).
        convert : str, optional
            The unit to convert the data to ('C' for Celsius, 'F' for Fahrenheit). Default is None.
        **kwargs : dict, optional
            Additional keyword arguments:
            - full_output : bool, optional
                Whether to return full output including metadata. Default is False.

        Returns
        -------
        dict or float
            The resulting values or the full output including metadata.
        """
        if str(convert).lower() == "c":
            value = BASIC_CONVERSIONS["c_to_f"](value)
        result = super().write_param(param, value, data_type, **kwargs)
        if result.error and not self._simulate:
            if kwargs.get("full_output", self.full_output):
                return result
            return result.error
        if str(convert).lower() == "c":
            result = result._replace(data=BASIC_CONVERSIONS["f_to_c"](result.data))
            # result.data = BASIC_CONVERSIONS["f_to_c"](result.data)
        if kwargs.get("full_output", self.full_output):
            return result
        return result.data

    def temp(self, unit=None, full_output=False):
        """
        Reads the current temperature.

        This is a wrapper around `read_param()` and is equivalent to `read_param(4001, float, instance)`.

        Parameters
        ----------
        unit : str, optional
            The temperature unit ('C' for Celsius, 'F' for Fahrenheit). Default is None.
        full_output : bool, optional
            Whether to return full output including metadata. Default is False.

        Returns
        -------
        dict or float
            The resulting values or the full output including metadata.
        """
        if unit is None:
            unit = self.temp_unit
        return self.read_param(4001, float, convert=unit, full_output=full_output)

    def setpoint(self, value=None, unit=None, **kwargs):
        """
        Changes the Watlow temperature setpoint.

        Takes a value (in degrees F by default), builds request, writes to Watlow, receives and returns response object.

        Parameters
        ----------
        value : int or float, optional
            The new target setpoint in degrees F by default.
        unit : str, optional
            The temperature unit ('C' for Celsius, 'F' for Fahrenheit). Default is None.
        **kwargs : dict, optional
            Additional keyword arguments:
            - mode : str, optional
                Mode for changing setpoint ('rapid', 'gradual', 'normal'). Default is 'normal'.
            - duration : int, optional
                Duration in seconds for setpoint change. Default is 300.
            - offset : float, optional
                Offset for overshoot/undershoot in setpoint change. Default is 5.

        Returns
        -------
        dict
            A dict containing the response data, parameter ID, and address.
        """
        if unit is None:
            unit = self.temp_unit
        if value is None:
            return self.read_param(7001, float, convert=unit)
        elif value == self.read_param(7001, float, convert=unit):
            return
        elif value > self.max_setpoint:
            value = self.max_setpoint
        elif value < self.min_setpoint:
            value = self.min_setpoint

        mode = kwargs.get("mode", self.mode).lower()
        duration = kwargs.get("duration", self.mode_duration)
        offset = kwargs.get("offset", self.mode_offset)

        if mode == "rapid" or mode == "gradual":
            thread = SThread(
                target=self._setpoint_mode_control,
                name=self.__name__ + "mode_control_run",
                args=(value, duration, offset, unit, mode),
            )
            thread.start()
            return
        return self.write_param(7001, value, float, convert=unit)

    def _setpoint_mode_control(
        self,
        target_value,
        duration: str | int | float = "auto",
        offset: str | float = "auto",
        unit=None,
        mode="rapid",
    ):
        """
        Change the setpoint either aggressively (with an overshoot) or gradually (with an undershoot).

        Parameters
        ----------
        target_value : float
            The target setpoint in degrees F by default.
        offset : float, optional
            The number of degrees to overshoot/undershoot the setpoint. Default is 5.
        duration : int, optional
            The time span in seconds over which to dial back to the setpoint. Default is 300.
        unit : str, optional
            The temperature unit ('C' for Celsius, 'F' for Fahrenheit). Default is None.
        mode : str, optional
            The mode of setpoint change, either "rapid" for aggressive change or "gradual" for slow change. Default is "rapid".

        Returns
        -------
        dict
            A dict containing the response data, parameter ID, and address.
        """
        if unit is None:
            unit = self.temp_unit

        current_value = self.setpoint(unit=unit, mode="normal")
        if str(offset).lower() == "auto":
            offset = abs(target_value - current_value) / 2
        offset = float(offset)
        if abs(target_value - current_value) < offset or offset < 0.5:
            return self.setpoint(target_value, unit=unit, mode="normal")

        if mode == "rapid":
            initial_value = (
                target_value - offset if target_value < current_value else target_value + offset
            )
        elif mode == "gradual":  # gradual mode
            initial_value = (
                target_value + offset if target_value < current_value else target_value - offset
            )
        else:
            return self.setpoint(target_value, unit=unit, mode="normal")

        # Set the initial setpoint
        self.setpoint(initial_value, unit=unit, mode="normal")

        if str(duration).lower() == "auto":
            while not target_value - offset < self.temp() < target_value + offset:
                time.sleep(1)
            step_size = max(0.1, offset / 1.25**10)
            adjustment_point = (target_value - initial_value) / 1.25
            self.setpoint(target_value - adjustment_point, unit=unit, mode="normal")
            n = 0

            while abs(adjustment_point) > step_size and n < 15 * 60:
                deviation = abs(self.temp() - (target_value - adjustment_point))
                if deviation < step_size:
                    adjustment_point /= 1.25
                    self.setpoint(target_value - adjustment_point, unit=unit, mode="normal")
                n += 10
                time.sleep(10)
        else:
            duration = float(duration)
            time.sleep(duration / 4)

            # Calculate the number of steps and the step size
            steps = 10
            step_size = (initial_value - target_value) / steps
            step_duration = (duration * 3 / 4) / steps

            # Incrementally dial back to the target setpoint
            for i in range(steps):
                current_value = initial_value - (i + 1) * step_size
                self.setpoint(current_value, unit=unit, mode="normal")
                time.sleep(step_duration)

        # Ensure the final setpoint is set
        self.setpoint(target_value, unit=unit, mode="normal")

        return self.setpoint(unit=unit, mode="normal")


if __name__ == "__main__":
    ez_zone = Watlow(port="sim com4")
    print(ez_zone.temp())
