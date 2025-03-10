# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:14:23 2025

@author: DEfECT
"""

import struct
import time
from binascii import hexlify
from binascii import unhexlify

import crcmod
import serial as ser


def f_to_c(f):
    """Convert Fahrenheit to Celsius."""
    return (f - 32.0) / 1.8


def c_to_f(c):
    """Convert Celsius to Fahrenheit."""
    return c * 1.8 + 32.0

class WatlowSerial():
    '''
    Object representing a Watlow PID temperature controller. This class
    facilitates the generation and parsing of BACnet TP/MS messages to and from
    Watlow temperature controllers.

    * **serial**: serial object (see pySerial's serial.Serial class) or `None`
    * **port** (str): string representing the serial port or `None`
    * **timeout** (float): Read timeout value in seconds
    * **address** (int): Watlow controller address (found in the setup menu). Acceptable values are 1 through 16.

    `timeout` and `port` are not necessary if a serial object was already passed
    with those arguments. The baudrate for Watlow temperature controllers is 38400
    and hardcoded.
    '''
    def __init__(self, port=None, baudrate=38400, timeout=0.5, address=1, retries=3):
        self.timeout = timeout
        self.baudrate = baudrate
        self.address = address
        self.port = port
        self.retries = retries


    def _headerCheckByte(self, headerBytes):
        '''
        Takes the full header byte array bytes[0] through bytes[6] of the full
        command and returns a check byte (bytearray of length one) using Watlow's
        algorithm.

        Implementation relies on this post:
        https://reverseengineering.stackexchange.com/questions/8303/rs-485-checksum-reverse-engineering-watlow-ez-zone-pm
        '''
        crc_8_table = [
            0x00, 0xfe, 0xff, 0x01, 0xfd, 0x03, 0x02, 0xfc,
            0xf9, 0x07, 0x06, 0xf8, 0x04, 0xfa, 0xfb, 0x05,
            0xf1, 0x0f, 0x0e, 0xf0, 0x0c, 0xf2, 0xf3, 0x0d,
            0x08, 0xf6, 0xf7, 0x09, 0xf5, 0x0b, 0x0a, 0xf4,
            0xe1, 0x1f, 0x1e, 0xe0, 0x1c, 0xe2, 0xe3, 0x1d,
            0x18, 0xe6, 0xe7, 0x19, 0xe5, 0x1b, 0x1a, 0xe4,
            0x10, 0xee, 0xef, 0x11, 0xed, 0x13, 0x12, 0xec,
            0xe9, 0x17, 0x16, 0xe8, 0x14, 0xea, 0xeb, 0x15,
            0xc1, 0x3f, 0x3e, 0xc0, 0x3c, 0xc2, 0xc3, 0x3d,
            0x38, 0xc6, 0xc7, 0x39, 0xc5, 0x3b, 0x3a, 0xc4,
            0x30, 0xce, 0xcf, 0x31, 0xcd, 0x33, 0x32, 0xcc,
            0xc9, 0x37, 0x36, 0xc8, 0x34, 0xca, 0xcb, 0x35,
            0x20, 0xde, 0xdf, 0x21, 0xdd, 0x23, 0x22, 0xdc,
            0xd9, 0x27, 0x26, 0xd8, 0x24, 0xda, 0xdb, 0x25,
            0xd1, 0x2f, 0x2e, 0xd0, 0x2c, 0xd2, 0xd3, 0x2d,
            0x28, 0xd6, 0xd7, 0x29, 0xd5, 0x2b, 0x2a, 0xd4,
            0x81, 0x7f, 0x7e, 0x80, 0x7c, 0x82, 0x83, 0x7d,
            0x78, 0x86, 0x87, 0x79, 0x85, 0x7b, 0x7a, 0x84,
            0x70, 0x8e, 0x8f, 0x71, 0x8d, 0x73, 0x72, 0x8c,
            0x89, 0x77, 0x76, 0x88, 0x74, 0x8a, 0x8b, 0x75,
            0x60, 0x9e, 0x9f, 0x61, 0x9d, 0x63, 0x62, 0x9c,
            0x99, 0x67, 0x66, 0x98, 0x64, 0x9a, 0x9b, 0x65,
            0x91, 0x6f, 0x6e, 0x90, 0x6c, 0x92, 0x93, 0x6d,
            0x68, 0x96, 0x97, 0x69, 0x95, 0x6b, 0x6a, 0x94,
            0x40, 0xbe, 0xbf, 0x41, 0xbd, 0x43, 0x42, 0xbc,
            0xb9, 0x47, 0x46, 0xb8, 0x44, 0xba, 0xbb, 0x45,
            0xb1, 0x4f, 0x4e, 0xb0, 0x4c, 0xb2, 0xb3, 0x4d,
            0x48, 0xb6, 0xb7, 0x49, 0xb5, 0x4b, 0x4a, 0xb4,
            0xa1, 0x5f, 0x5e, 0xa0, 0x5c, 0xa2, 0xa3, 0x5d,
            0x58, 0xa6, 0xa7, 0x59, 0xa5, 0x5b, 0x5a, 0xa4,
            0x50, 0xae, 0xaf, 0x51, 0xad, 0x53, 0x52, 0xac,
            0xa9, 0x57, 0x56, 0xa8, 0x54, 0xaa, 0xab, 0x55
        ]

        # Watlow's header check byte algorithm:
        intCheck = ~crc_8_table[headerBytes[6] ^ crc_8_table[headerBytes[5] ^
                                crc_8_table[headerBytes[4] ^ crc_8_table[headerBytes[3] ^
                                            crc_8_table[~headerBytes[2]]]]]] & (2**8-1)
        return bytes([intCheck])

    def _dataCheckByte(self, dataBytes):
        '''
        Takes the full data byte array, bytes[8] through bytes[13] of the full
        command and calculates the data check byte using BacNET CRC-16.
        '''
        # CRC-16 with 0xFFFF as initial value, 0x1021 as polynomial, bit reversed
        crc_fun = crcmod.mkCrcFun(poly=0x11021, initCrc=0, rev=True, xorOut=0xFFFF)
        # bytes object packed using C-type unsigned short, little-endian:
        byte_str = struct.pack('<H', crc_fun(dataBytes))
        return byte_str

    def _intDataParamToHex(self, dataParam):
        # Reformats data param from notation in the manual to hex string
        # (e.g. '4001' becomes '04' and '001', returned as '0401')
        dataParam = format(int(dataParam), '05d')
        dataParam = hexlify(int(dataParam[:2]).to_bytes(1, 'big') + int(dataParam[2:]).to_bytes(1, 'big')).decode('utf-8')
        return dataParam

    def _byteDataParamToInt(self, hexParam):
        # Reformats data parameter from bytes string to integer
        # (e.g. b'\x1a\x1d' to 26029)
        return int(str(hexParam[0]) + format(hexParam[1], '03d'))

    def _buildZone(self, address):
        '''
        Generates the zone portion of a message from the watlow address.
        Returns the hexidecimal form as a string (without "0x")
        '''
        return format(address + 15, 'x')

    def _buildReadRequest(self, dataParam, instance='01'):
        '''
        Takes the watlow parameter ID, converts to bytes objects, calls
        internal functions to calc check bytes, and assembles/returns the request
        byte array.
        '''
        # Request Header:
        BACnetPreamble = '55ff'
        requestParam = '05'
        # Zone corresponds to the address parameter in setup (e.g. '10' = 1, '11' = 2, etc.)
        zone = self._buildZone(self.address)
        additionalHeader = '000006'
        hexHeader = BACnetPreamble + requestParam + zone + additionalHeader

        # Request Data Parameters
        additionalData = '010301'
        dataParam = self._intDataParamToHex(dataParam)
        hexData = additionalData + dataParam + instance

        # Convert input strings to bytes:
        hexHeader = unhexlify(hexHeader)
        hexData = unhexlify(hexData)

        # Calculate check bytes:
        headerChk = self._headerCheckByte(hexHeader)
        dataChk = self._dataCheckByte(hexData)

        # Assemble request byte array:
        request = bytearray(hexHeader)
        request += bytearray(headerChk)
        request += bytearray(hexData)
        request += dataChk

        return request

    def _buildWriteRequest(self, dataParam, value, data_type, instance='01'):
        '''
        Takes the set point temperature value, converts to bytes objects, calls
        internal functions to calc check bytes, and assembles/returns the request
        byte array.

        Much of this function is hard coded until I figure out how each
        part of the hex command is assembled. It is different than a normal read
        command.
        '''
        BACnetPreamble = '55ff'
        requestParam = '05'
        zone = self._buildZone(self.address)
        dataParam = self._intDataParamToHex(dataParam)
        if data_type == float:
            additionalHeader = '00000a'
            hexData = '0104' + dataParam + instance + '08'
            value = struct.pack('>f', float(value))
        else:
            additionalHeader = '030009'
            hexData = '0104' + dataParam + instance + '0f01'
            value = value.to_bytes(2, 'big')

        # Request Header String:
        hexHeader = BACnetPreamble + requestParam + zone + additionalHeader
        # Convert input strings to bytes:
        hexHeader = unhexlify(hexHeader)

        # Data portion of request (here the set point value is appended)
        hexData = unhexlify(hexData) + value

        # Calculate check bytes:
        headerChk = self._headerCheckByte(hexHeader)
        dataChk = self._dataCheckByte(hexData)

        # Assemble request byte array:
        request = bytearray(hexHeader)
        request += bytearray(headerChk)
        request += bytearray(hexData)
        request += dataChk

        return request

    def _validateResponse(self, bytesResponse):
        '''
        Compares check bytes received in response to those calculated.
        '''
        isValid = False
        # Evaluate headerChk as bytearray instead of as an int (which is how
        # python will interpret a single hex character)
        headerChkReceived = bytearray([bytesResponse[7]])
        dataCheckRecieved = bytesResponse[-2:]
        addressReceived = bytesResponse[4] - 15
        if (headerChkReceived == self._headerCheckByte(bytesResponse[0:7]) and
                dataCheckRecieved == self._dataCheckByte(bytesResponse[8:-2]) and
                addressReceived == self.address):
            isValid = True
        return isValid

    def _parseResponse(self, bytesResponse):
        '''
        Takes the full response byte array and extracts the relevant data (e.g.
        current temperature), constructs response dict, and returns it.
        '''
        output = {
                    'address': self.address,
                    'param': None,
                    'data': None,
                    'error': None
                 }
        try:
            if bytesResponse == b'' or bytesResponse == bytearray(len(bytesResponse)):
                raise Exception('Exception: No response from address {0}'.format(self.address))
            if not self._validateResponse(bytesResponse):
                print('Invalid Response at address {0}: '.format(self.address), hexlify(bytesResponse))
                raise Exception('Exception: Invalid response received from address {0}'.format(self.address))
        except Exception as e:
            output['error'] = e
            return output
        else:
            # Case where response data value is an int used to represent a state defined
            # in the manual (e.g. param 8003, heat algorithm, where 62 means 'PID')
            # from a read request
            # Hex byte 7: '0a', Hex bytes 15, 16: 0F, 01
            if bytesResponse[6] == 10 and bytesResponse[-6] == 15 and bytesResponse[-5] == 1:
                data = bytesResponse[-4:-2]
                output['param'] = self._byteDataParamToInt(bytesResponse[11:13])
                output['data'] = int.from_bytes(data, byteorder='big')
            # Case where response data value is a float from a set param request
            # (e.g. 7001, process value setpoint)
            # Hex byte 7: '0a', Hex byte 14: '08'
            elif bytesResponse[6] == 10 and bytesResponse[-7] == 8:
                ieee_754 = hexlify(bytesResponse[-6:-2])
                output['data'] = struct.unpack('>f', unhexlify(ieee_754))[0]
                output['param'] = self._byteDataParamToInt(bytesResponse[10:12])
            # Case where response data value is an integer from a set param
            # request (e.g. param 8003, heat algorithm, where 62 means 'PID')
            # Hex byte 7: '09'
            elif bytesResponse[6] == 9:
                data = bytesResponse[-4:-2]
                output['param'] = self._byteDataParamToInt(bytesResponse[10:12])
                output['data'] = int.from_bytes(data, byteorder='big')
            # Case where data value is a float representing a process value
            # (e.g. 4001, where current temp of 50.0 is returned)
            # Hex byte 7: '0b'
            elif bytesResponse[6] == 11:
                ieee_754 = bytesResponse[-6:-2]
                output['param'] = self._byteDataParamToInt(bytesResponse[11:13])
                output['data'] = struct.unpack('>f', ieee_754)[0]
            # Other cases, such as response from trying to write a read-only parameter:
            else:
                output['error'] = Exception('Received a message that could not be parsed from address {0}'.format(self.address))

            return output

    def readParam(self, param, data_type, instance='01', retries=None):
        '''
        Takes a parameter and writes data to the watlow controller at
        object's internal address. Using this function requires knowing the data
        type for the parameter (int or float). See the Watlow
        `user manual <https://www.watlow.com/-/media/documents/user-manuals/pm-pid-1.ashx>`_
        for individual parameters and the Usage section of these docs.

        * **param**: a four digit integer corresponding to a Watlow parameter (e.g. 4001, 7001)
        * **data_type**: the Python type representing the data value type (i.e. `int` or `float`)
        * **instance**: a two digit string corresponding to the channel to read (e.g. '01', '05')

        `data_type` is used to determine how many characters to read
        following the controller's response. If `int` is passed when the data type
        should be `float`, it will not read the entire message and will throw an
        error. If `float` is passed when it should be `int`, it will timeout,
        possibly reading correctly. If multiple instances of `Watlow()` are using
        the same serial port for different controllers it will read too many
        characters. It is best to be completely sure which data type is being used
        by each parameter (`int` or `float`).

        Returns a dict containing the response data, parameter ID, and address.
        '''
        request = self._buildReadRequest(param, instance)
        attempt = 0
        retries = retries if isinstance(retries, int) else self.retries
        while attempt < retries:
            try:
                with ser.Serial(self.port, self.baudrate, timeout=self.timeout) as serial:
                    serial.write(request)
                    if data_type == float:
                        response = serial.read(21)
                    else:
                        response = serial.read(20)
                    output = self._parseResponse(response)
                    return output
            except Exception as e:
                attempt += 1
                if attempt >= retries:
                    output = {
                        'address': self.address,
                        'param': None,
                        'data': None,
                        'error': str(e)
                    }
                    return output
                time.sleep(self.timeout)

    def writeParam(self, param, value, data_type, instance='01', retries=None):
        '''
        Changes the value of the passed watlow parameter ID. Using this function
        requires knowing the data type for the parameter (int or float).
        See the Watlow
        `user manual <https://www.watlow.com/-/media/documents/user-manuals/pm-pid-1.ashx>`_
        for individual parameters and the Usage section of these docs.

        * **value**: an int or float representing the new target setpoint in degrees F by default
        * **data_type**: the Python type representing the data value type (i.e. `int` or `float`)
        * **instance**: a two digit string corresponding to the channel to read (e.g. '01', '05')

        `data_type` is used to determine how the BACnet TP/MS message will be constructed
        and how many serial characters to read following the controller's response.

        Returns a dict containing the response data, parameter ID, and address.
        '''
        request = self._buildWriteRequest(param, value, data_type, instance)
        attempt = 0
        retries = retries if isinstance(retries, int) else self.retries
        while attempt < retries:
            try:
                with ser.Serial(self.port, self.baudrate, timeout=self.timeout) as serial:
                    serial.write(request)
                    if data_type == float:
                        bytesResponse = serial.read(20)
                    else:
                        bytesResponse = serial.read(19)
                    output = self._parseResponse(bytesResponse)
                    return output
            except Exception as e:
                attempt += 1
                if attempt >= retries:
                    output = {
                        'address': self.address,
                        'param': None,
                        'data': None,
                        'error': str(e)
                    }
                    return output
                
                time.sleep(self.timeout)
        

class Watlow(WatlowSerial):
    '''
    Object representing a Watlow PID temperature controller. This class
    facilitates the generation and parsing of BACnet TP/MS messages to and from
    Watlow temperature controllers.

    * **serial**: serial object (see pySerial's serial.Serial class) or `None`
    * **port** (str): string representing the serial port or `None`
    * **timeout** (float): Read timeout value in seconds
    * **address** (int): Watlow controller address (found in the setup menu). Acceptable values are 1 through 16.

    `timeout` and `port` are not necessary if a serial object was already passed
    with those arguments. The baudrate for Watlow temperature controllers is 38400
    and hardcoded.
    '''
    def __init__(self, port=None, baudrate=38400, timeout=0.5, address=1, **kwargs):
        super().__init__(port, baudrate, timeout, address)
        self.full_output = kwargs.get('full_output', False)
        self.temp_unit = kwargs.get('temp_unit', 'C')

    def readParam(self, param, data_type, instance='01', convert=None, **kwargs):
        result = super().readParam(param, data_type, instance)
        if result['error']:
            if kwargs.get("full_output", self.full_output):
                return result
            return result['error']
        if str(convert).lower() == 'c':
            result['data'] = f_to_c(result['data'])
        if kwargs.get("full_output", self.full_output):
            return result
        return result['data']
        

    def writeParam(self, param, value, data_type, instance='01', convert=None, **kwargs):
        if str(convert).lower() == 'c':
            value = c_to_f(value)
        result = super().writeParam(param, value, data_type, instance)
        if result['error']:
            if kwargs.get("full_output", self.full_output):
                return result
            return result['error']
        if str(convert).lower() == 'c':
            result['data'] = f_to_c(result['data'])
        if kwargs.get("full_output", self.full_output):
            return result
        return result['data']

    def temp(self, unit=None):
        '''
        Reads the current temperature. This is a wrapper around `readParam()`
        and is equivalent to `readParam(4001, float, instance)`.

        Returns a dict containing the response data, parameter ID, and address.
        * **instance**: a two digit string corresponding to the channel to read (e.g. '01', '05')
        '''
        if unit is None:
            unit = self.temp_unit
        return self.readParam(4001, float, instance="01", convert=unit)

    def setpoint(self, value=None, unit=None, **kwargs):
        '''
        Changes the watlow temperature setpoint. Takes a value (in degrees F by
        default), builds request, writes to watlow, receives and returns response
        object.

        * **value**: an int or float representing the new target setpoint in degrees F by default
        * **instance**: a two digit string corresponding to the channel to set (e.g. '01', '05')

        This is a wrapper around `writeParam()` and is equivalent to
        `writeParam(7001, value, float, instance)`.

        Returns a dict containing the response data, parameter ID, and address.
        '''
        if unit is None:
            unit = self.temp_unit
        if value is None:
            return self.readParam(7001, float, instance="01", convert=unit)
        if kwargs.get("fast", False):
            return self.fast_setpoint(value, kwargs.get("duration", 300), kwargs.get("overshoot", 5), unit=unit)
        return self.writeParam(7001, value, float, instance="01", convert=unit)
    
    def fast_setpoint(self, setpoint, duration=300, overshoot=5, unit=None):
        '''
        Aggressively sets the setpoint by setting an overshoot value, then incrementally dialing back to the setpoint over a defined time span.

        * **setpoint**: an int or float representing the target setpoint in degrees F by default
        * **overshoot**: an int or float representing the number of degrees to overshoot the setpoint
        * **duration**: the time span in seconds over which to dial back to the setpoint
        * **unit**: the temperature unit ('C' for Celsius, 'F' for Fahrenheit)

        Returns a dict containing the response data, parameter ID, and address.
        '''
        if unit is None:
            unit = self.temp_unit
        if setpoint < self.setpoint(unit=unit):
            alt_setpoint =  setpoint - overshoot
        else:
            alt_setpoint = setpoint + overshoot
        # Set the overshoot value
        self.setpoint(alt_setpoint, unit=unit)
        time.sleep(duration / 4)

        # Calculate the number of steps and the step size
        steps = 10
        step_size = (alt_setpoint - setpoint) / steps
        step_duration = (duration * 3 / 4) / steps

        # Incrementally dial back to the target setpoint
        for i in range(steps):
            current_value = alt_setpoint - (i + 1) * step_size
            self.setpoint(current_value, unit=unit)
            time.sleep(step_duration)

        # Ensure the final setpoint is set
        self.setpoint(setpoint, unit=unit)

        return self.setpoint(unit=unit)

if __name__ == "__main__":
    ez_zone = Watlow(port="com4")
    print(ez_zone.temp())


# """Drivers for Watlow EZ-Zone temperature controllers."""
# from __future__ import annotations

# import logging
# import re
# import struct
# from binascii import unhexlify
# from typing import ClassVar

# import crcmod  # type: ignore
# import serial
# # from pymodbus.constants import Endian
# # from pymodbus.payload import BinaryPayloadBuilder, BinaryPayloadDecoder

# # from .util import AsyncioModbusClient

# # BACnet CRC: https://sourceforge.net/p/bacnet/mailman/message/1259086/
# # CRC8 polynominal: X^8 + X^7 + 1 (110000001)
# crc8 = crcmod.mkCrcFun(0b110000001)
# # CRC16 polynominal: X^16 + X^12 + X^5 + 1 (10001000000100001)
# crc16 = crcmod.mkCrcFun(0b10001000000100001)


# def f_to_c(f):
#     """Convert Fahrenheit to Celsius."""
#     return (f - 32.0) / 1.8


# def c_to_f(c):
#     """Convert Celsius to Fahrenheit."""
#     return c * 1.8 + 32.0


# class TemperatureController:
#     """Driver for the Watlow EZ-ZONE temperature controller.

#     This driver borrows heavily from this StackOverflow post:
#         https://reverseengineering.stackexchange.com/questions/8303/
#         rs-485-checksum-reverse-engineering-watlow-ez-zone-pm

#     The EZ-Zone communication protocol is Bacnet MS/TP over a serial line.
#     There are libraries for this protocol, namely bacpypes, but they don't
#     support serial devices. As we only need three commands, we're going to
#     manually build the appropriate request strings.

#     The request breakdown is:

#     Preamble Req/Res Zone ???    Check ???    Register Instance Value    Check
#     55ff     05      10   000006 e8    010301 0401     01       00000000 e399

#      * Preamble is always 55ff for BACNET MS/TP.
#      * Req/Res is a guess. It looks like all requests are 05 and responses are 06.
#      * Zone, only 10 works. Maybe other zones are for splitting RS-485 out into
#        a many-to-one configuration.
#      * Nothings don't seem to change between valid requests.
#      * First checksum is a custom protocol.
#      * Only known registers are 0401 for PV and 0701 for SP. Other registers
#        return data, so we could hunt around for PID params if needed.
#      * Instance, only 01 works. Current understanding is similar to zone.
#      * Second checksum is a custom CRC-16 following Bacnet spec.
#     """

#     commands: ClassVar[dict] = {
#         'actual':
#             {'header': unhexlify('0510000006'),
#              'body':   unhexlify('010301040101')},
#         'setpoint':
#             {'header': unhexlify('0510000006'),
#              'body':   unhexlify('010301070101')},
#         'set':
#             {'header': unhexlify('051000000a'),
#              'body':   unhexlify('010407010108')},
#     }
#     responses: ClassVar[dict] = {
#         'actual': re.compile('^55ff060010000b8802030104010108'
#                              '([0-9a-f]{8})([0-9a-f]{4})$'),
#         'setpoint': re.compile('^55ff060010000b8802030107010108'
#                                '([0-9a-f]{8})([0-9a-f]{4})$'),
#         'set': re.compile('^55ff060010000a76020407010108'
#                           '([0-9a-f]{8})([0-9a-f]{4})$')
#     }

#     def __init__(self, port, timeout=0.5):
#         """Open up a serial connection to the controller.

#         This device uses RS-422 instead of RS-232. You will likely need a
#         custom converter.
#         """
#         self.port = port
#         self.baudrate = 38400
#         self.timeout = timeout
#         self.connection = self.open()

#     def open(self):
#         """Open up a serial connection to the oven."""
#         return serial.Serial(
#             self.port,
#             self.baudrate,
#             timeout=self.timeout
#         )

#     def close(self):
#         """Close the serial connection. Use on cleanup."""
#         self.connection.flush()
#         self.connection.close()

#     def get(self):
#         """Get the current temperature and setpoint, in C."""
#         preamble = unhexlify('55ff')
#         output = {'actual': None, 'setpoint': None}
#         for key in output:
#             header = self.commands[key]['header']
#             body = self.commands[key]['body']
#             # Calculate header and data checksums based on BACnet CRC
#             header_checksum = struct.pack('<H', ~crc8(self.commands[key]['header']) & 0xff)
#             data_checksum = struct.pack('<H', ~crc16(self.commands[key]['body']) & 0xffff)

#             # send command to controller, formatting preamble, header, crc8, body and crc16
#             output[key] = self._write_and_read(
#                 request=preamble + header + header_checksum[:1] + body + data_checksum,
#                 length=21,
#                 check=self.responses[key]
#             )
#         return output

#     def set(self, setpoint):
#         """Set the setpoint temperature, in C."""
#         preamble = unhexlify('55ff')
#         header = self.commands['set']['header']
#         body = self.commands['set']['body'] + struct.pack('>f', c_to_f(setpoint))
#         # Calculate header and data checksums based on BACnet CRC
#         header_checksum = struct.pack('<H', ~crc8(self.commands['set']['header']) & 0xff)
#         data_checksum = struct.pack('<H', ~crc16(body) & 0xffff)

#         response = self._write_and_read(
#             request=preamble + header + header_checksum[:1] + body + data_checksum,
#             length=20,
#             check=self.responses['set']
#         )

#         # check setpoint versus response, if not the same raise an error
#         if round(setpoint, 2) != round(response, 2):
#             raise OSError(f"Could not change setpoint from "
#                           f"{response:.2f}°C to {setpoint:.2f}°C.")

#     def _write_and_read(self, request, length, check, retries=3):
#         """Write to and read from the device.

#         This function abstracts a whole lot of validation checks and error
#         handling. The goal is for this driver to be stable to both incomplete
#         messages and temporary disconnects.

#         The regex parses out the response checksum but does not use it. A
#         description of how to calculate it is in the following manual:
#             http://www.bacnet.org/Addenda/Add-135-2010an-APR1-1_chair-approved.pdf
#         However, my attempts at reproducing did not go well.
#         """
#         if not self.connection.is_open:
#             self.open()
#         if retries <= 0:
#             self.close()
#             raise OSError("Could not communicate with Watlow.")
#         self.connection.flush()
#         try:
#             logging.debug('Formatted Request: ' + str(bytes.hex(request)))
#             self.connection.write(request)
#             response = self.connection.read(length)
#         except serial.serialutil.SerialException:
#             return self._write_and_read(request, length, check, retries - 1)
#         match = check.match(bytes.hex(response))
#         logging.debug('Formatted Response: ' + str(bytes.hex(response)))
#         if not match:
#             return self._write_and_read(request, length, check, retries - 1)
#         value = match.group(1)
#         # From docstring, `checksum = match.group(2)` could be added and checked.
#         temperature = f_to_c(struct.unpack('>f', unhexlify(value))[0])
#         if temperature < 0 or temperature > 250:
#             return self._write_and_read(request, length, check, retries - 1)
#         return temperature


# class Gateway(AsyncioModbusClient):
#     """Watlow communication using their EZ-Zone Modbus Gateway."""

#     def __init__(self, address, timeout=1, modbus_offset=5000, max_temp=220):
#         """Open connection to gateway."""
#         super().__init__(address, timeout)
#         self.modbus_offset = modbus_offset
#         self.actual_temp_address = 360
#         self.setpoint_address = 2160
#         self.output_address = 1904
#         self.setpoint_range = (10, max_temp)

#     async def get(self, zone: int):
#         """Get oven data for a zone.

#         For more information on a 'Zone', refer to Watlow manuals.
#         """
#         output: dict[str, int | None] = {
#             'actual': self.actual_temp_address,
#             'setpoint': self.setpoint_address,
#             'output': self.output_address,
#         }
#         endian = Endian.BIG if self.pymodbus35plus else Endian.big  # type: ignore[attr-defined]
#         for k, v in output.items():
#             address = (zone - 1) * self.modbus_offset + v
#             try:
#                 result = await self.read_registers(address, 2)
#                 output[k] = BinaryPayloadDecoder.fromRegisters(
#                     result,
#                     byteorder=endian
#                 ).decode_32bit_float()
#             except AttributeError:
#                 output[k] = None
#         return output

#     async def set_setpoint(self, zone: int, setpoint: float):
#         """Set the temperature setpoint for a zone.

#         For more information on a 'Zone', refer to Watlow manuals.
#         """
#         if not self.setpoint_range[0] <= setpoint <= self.setpoint_range[1]:
#             raise ValueError(f"Setpoint ({setpoint}) is not in the valid range from"
#                              f" {self.setpoint_range[0]} to {self.setpoint_range[1]}")
#         address = (zone - 1) * self.modbus_offset + self.setpoint_address
#         endian = Endian.BIG if self.pymodbus35plus else Endian.big  # type: ignore[attr-defined]
#         builder = BinaryPayloadBuilder(byteorder=endian)
#         builder.add_32bit_float(setpoint)
#         await self.write_registers(address, builder.build(),
#                                    skip_encode=True)