# -*- coding: utf-8 -*-
"""
Module for interfacing with the OMEGA UWTC thermocouple device.

Decoding information can be found in on their website:
TODO: Add link to the manual

"""
__all__ = ["UWTC"]

# import time
# from serial import Serial

from typing import  Optional, Union, NamedTuple

import numpy as np

try:
    from ..utilities.serial_manager import SerialManager
except ImportError:
    from equipment.utilities.serial_manager import SerialManager

BASIC_CONVERSIONS = {
    "f_to_c": lambda f: (f - 32.0) / (9.0 / 5.0),
    "k_to_c": lambda k: k - 273.15,
    "c_to_f": lambda c: c * (9.0 / 5.0) + 32.0,
    "c_to_k": lambda c: c + 273.15,
}


class UWTCOutput(NamedTuple):
    """
    Named tuple for the output data from the UWTC device.
    """
    address: Optional[int] = None
    signal_strength: int = 0
    sensor_type: str = "None"
    measured: float = 0
    ambient: float = 0
    battery_voltage: float = 0
    error: str = ""


class UWTC:
    """
    Class representing a UWTC thermocouple device.

    Attributes
    ----------
    port : str
        The serial port to which the thermocouple device is connected.
    baudrate : int
        The baud rate for the serial communication.
    temp_unit : str
        The temperature unit ('C' for Celsius, 'F' for Fahrenheit).
    full_output : bool
        Whether to return full output or just the temperature.
    retries : int
        Number of retries for communication.
    timeout : int
        Read timeout value in seconds.

    Methods
    -------
    read_temperature()
        Reads the temperature from the thermocouple device.
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 9600,
        temp_unit: str = "C",
        full_output: bool = False,
        retries: int = 3,
        timeout: int = 5,
    ):
        self.port = port
        self.baudrate = baudrate
        self.retries = retries
        self.timeout = timeout
        self.temp_unit = temp_unit
        self.full_output = full_output
        self.serial_manager = SerialManager(port, baudrate, timeout)
        self.previous_reply = UWTCOutput()
        self._simulate = False
        self._sim_value = 124.0
        self._sim_ambient = 75.5
        self._print_sim_events = True

        if "sim" in port.lower():
            self._simulation_mode()

    def _simulation_mode(self, print_events=True) -> None:
        """
        Sets the simulation mode for the Watlow controller.
        """
        self._simulate = not self._simulate
        self._sim_value = 124.0
        self._sim_ambient = 75.5
        self._print_sim_events = print_events

    def _parse_response(self, response: bytes) -> Optional[UWTCOutput]:
        """
        Parse the response from the thermocouple device.

        Parameters
        ----------
        response : bytes
            The response from the device.

        Returns
        -------
        Optional[UWTCOutput]
            A named tuple containing the parsed data.
        """
        if len(response) != 16:
            return None

        address = response[4] * 256 + response[5]
        signal_strength = response[6]
        sensor_type = response[8:9].decode()
        measured = response[9] * 256 + response[10]
        ambient = (response[11] * 256 + response[12]) / 10.0
        battery_voltage = response[13] * 256 + response[14]

        return UWTCOutput(
            address=address,
            signal_strength=signal_strength,
            sensor_type=sensor_type,
            measured=measured,
            ambient=ambient,
            battery_voltage=battery_voltage,
        )

    def _build_request(self, param: Optional[str], value: Optional[str] = None) -> bytes:
        """
        Build the read request for the UWTC device.

        Parameters
        ----------
        param : Optional[str]
            The parameter to read.
        value : Optional[str]
            The value to write.

        Returns
        -------
        bytes
            The encoded read request.
        """
        if param is None:
            return b"\r"
        if value is not None:
            param = f"{param} {value}"
        if not param.endswith("\r"):
            param += "\r"
        return param.encode()

    def write_param(
        self,
        param: Optional[str] = None,
        value: Optional[str] = None,
        retries: Optional[int] = None,
    ) -> UWTCOutput:
        """
        Writes a parameter to the UWTC device.

        Parameters
        ----------
        param : Optional[str]
            The parameter to write.
        value : Optional[str]
            The value to write.
        retries : Optional[int]
            Number of retries for communication.

        Returns
        -------
        UWTCOutput
            A named tuple containing the response data.
        """
        if self._simulate:
            read_write = "write"
            if value is None:
                value = self._sim_value + np.random.normal(0, 0.1)
                read_write = "read"
            if self._print_sim_events:
                print(f"UWTC simulated {read_write}: {value}")
            return UWTCOutput(
                    address=201,
                    signal_strength=25,
                    sensor_type="k",
                    measured=value,
                    ambient=self._sim_ambient,
                    battery_voltage=1300.0,
                    error="Simulation mode",
                )       
            
        request = self._build_request(param, value)
        attempt = 0
        retries = retries if isinstance(retries, int) else self.retries
        while attempt < retries:
            try:
                with self.serial_manager as serial:
                    if request:
                        serial.write(request)
                    response = serial.read(16)

                    output = self._parse_response(response)
                    if not output:
                        raise ValueError("No response received from the device.")
                    self.previous_reply = output
                    return output
            except (ValueError, PermissionError, FileNotFoundError) as e:
                attempt += 1
                if attempt >= retries:
                    output = UWTCOutput(error=str(e))
                    self.previous_reply = output
                    return output
                # time.sleep(self.timeout)

    def read_param(self, param: Optional[str] = None, retries: Optional[int] = None) -> UWTCOutput:
        """
        Reads a parameter from the UWTC device.

        Parameters
        ----------
        param : Optional[str]
            The parameter to read.
        retries : Optional[int]
            Number of retries for communication.

        Returns
        -------
        UWTCOutput
            A named tuple containing the response data.
        """
        return self.write_param(param, retries=retries)

    def temp(
        self, param: Optional[str] = None, temp_unit: Optional[str] = None
    ) -> Union[float, UWTCOutput]:
        """
        Reads the temperature from the thermocouple device.

        Parameters
        ----------
        param : Optional[str]
            The parameter to read.
        temp_unit : Optional[str]
            The temperature unit ('C' for Celsius, 'F' for Fahrenheit).

        Returns
        -------
        Union[float, UWTCOutput]
            The temperature in the specified unit or the full output if requested.
        """
        parsed_data = self.read_param()

        temp_unit = temp_unit or self.temp_unit
        # Convert temperature if needed
        if temp_unit == "C":
            parsed_data = parsed_data._replace(
                ambient=BASIC_CONVERSIONS["f_to_c"](parsed_data.ambient),
                measured=BASIC_CONVERSIONS["f_to_c"](parsed_data.measured),
            )

        if self.full_output:
            return parsed_data
        if param is not None and param in parsed_data._fields:
            return getattr(parsed_data, param)
        return parsed_data.measured


# Example usage
if __name__ == "__main__":
    uwtc_device = UWTC(port="sim COM3", temp_unit="C", full_output=True)
    temperature_data = uwtc_device.temp()
    if temperature_data is not None:
        print(f"Temperature Data: {temperature_data}")
