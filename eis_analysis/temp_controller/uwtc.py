from serial import Serial

def f_to_c(f):
    """Convert Fahrenheit to Celsius."""
    return (f - 32.0) / 1.8

def c_to_f(c):
    """Convert Celsius to Fahrenheit."""
    return c * 1.8 + 32.0


class UWTC:
    """
    Class representing a UWTC thermocouple device.

    Attributes
    ----------
    port : str
        The serial port to which the thermocouple device is connected.
    baudrate : int
        The baud rate for the serial communication.
    databits : int
        The number of data bits.
    parity : str
        The parity setting ('N' for None, 'E' for Even, 'O' for Odd).
    stopbits : int
        The number of stop bits.
    flowcontrol : str
        The flow control setting ('None', 'RTS/CTS', 'XON/XOFF').
    temp_unit : str
        The temperature unit ('C' for Celsius, 'F' for Fahrenheit).
    full_output : bool
        Whether to return full output or just the temperature.

    Methods
    -------
    read_temperature()
        Reads the temperature from the thermocouple device.
    """

    def __init__(self, port, baudrate=9600, temp_unit='C', full_output=False, retries=3, timeout=5):
        self.port = port
        self.baudrate = baudrate
        self.retries = retries
        self.timeout = timeout
        self.temp_unit = temp_unit
        self.full_output = full_output
        self.previous_reply = {
                        'address': None,
                        'signal_strength': 0,
                        'sensor_type': "None",
                        'measured': 0,
                        'ambient': 0,
                        'battery_voltage': 0,
                        'error': "",
                    }

    def f_to_c(self, temp_f):
        """Convert Fahrenheit to Celsius."""
        return (temp_f - 32.0) / 1.8

    def _parseResponse(self, response):
        """
        Parse the response from the thermocouple device.

        Parameters
        ----------
        response : bytes
            The response from the device.

        Returns
        -------
        dict
            A dictionary containing the parsed data.
        """
        if len(response) != 16:
            return None
        
        address = response[4] * 256 + response[5]
        signal_strength = response[6]
        sensor_type = response[8:9].decode()
        measured = response[9] * 256 + response[10]
        ambient = (response[11] * 256 + response[12]) / 10.0
        battery_voltage = response[13] * 256 + response[14]

        return {
            'address': address,
            'signal_strength': signal_strength,
            'sensor_type': sensor_type,
            'measured': measured,
            'ambient': ambient,
            'battery_voltage': battery_voltage
        }
    def _buildRequest(self, param, value=None):
        """
        Build the read request for the UWTC device.

        Parameters
        ----------
        param : str
            The parameter to read.
        Returns
        -------
        bytes
            The encoded read request.
        """
        if param is None:
            return b'\r'
        if value is not None:
            param = f"{param} {value}"
        if not param.endswith('\r'):
            param += '\r'
        return param.encode()
    
    def writeParam(self, param=None, value=None, retries=None):
        request = self._buildRequest(param, value)
        attempt = 0
        retries = retries if isinstance(retries, int) else self.retries
        while attempt < retries:
            try:
                with Serial(self.port, self.baudrate, timeout=self.timeout) as serial:
                    if request:
                        serial.write(request)
                    response = serial.read(16)
                    
                    output = self._parseResponse(response)
                    if not output:
                        raise Exception("No response received from the device.")
                    self.previous_reply = output
                    return output
            except Exception as e:
                attempt += 1
                if attempt >= retries:
                    output = {
                        'address': None,
                        'signal_strength': 0,
                        'sensor_type': "None",
                        'measured': 0,
                        'ambient': 0,
                        'battery_voltage': 0,
                        'error': str(e),
                    }
                    self.previous_reply = output
                    return output
                # time.sleep(self.timeout)
    
    def readParam(self, param=None, retries=None):
        return self.writeParam(param, retries=retries)

    
    def temp(self, param=None, temp_unit=None):
        """
        Reads the temperature from the thermocouple device.

        Parameters
        ----------
        timeout : int
            The maximum time in seconds to wait for a response.

        Returns
        -------
        float or dict
            The temperature in the specified unit or the full output if requested.
        """

        parsed_data = self.readParam()

        temp_unit = temp_unit or self.temp_unit
        # Convert temperature if needed
        if temp_unit == 'C':
            parsed_data['ambient'] = self.f_to_c(parsed_data['ambient'])
            parsed_data['measured'] = self.f_to_c(parsed_data['measured'])
        
        if self.full_output:
            return parsed_data
        if param is not None and param in parsed_data:
            return parsed_data[param]
        return parsed_data['measured']



# Example usage
if __name__ == "__main__":
    uwtc_device = UWTC(port="COM3", temp_unit='C', full_output=True)
    temperature_data = uwtc_device.temp()
    if temperature_data is not None:
        print(f"Temperature Data: {temperature_data}")


