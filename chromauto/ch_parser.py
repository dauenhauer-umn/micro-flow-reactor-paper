"""
NODOC:
    File parser for Chemstation files
    Copyright (C) 2015-2018 CINF team on GitHub: https://github.com/CINF
    The General Stepped Program Runner is free software: you can
    redistribute it and/or modify it under the terms of the GNU
    General Public License as published by the Free Software
    Foundation, either version 3 of the License, or
    (at your option) any later version.
    The General Stepped Program Runner is distributed in the hope
    that it will be useful, but WITHOUT ANY WARRANTY; without even
    the implied warranty of MERCHANTABILITY or FITNESS FOR A
    PARTICULAR PURPOSE.  See the GNU General Public License for more
    details.
    You should have received a copy of the GNU General Public License
    along with The CINF Data Presentation Website.  If not, see
    <http://www.gnu.org/licenses/>.
    .. note:: This file parser went through a large re-write on ??? which
       changed the data structures of the resulting objects. This means
       that upon upgrading it *will* be necessary to update code. The
       re-write was done to fix some serious errors from the first
       version, like relying on the Report.TXT file for injections
       summaries. These are now fetched from the more ordered CSV files.
NODOC
"""

import struct
from struct import unpack
import numpy
from dateutil.parser import parse as date_parser

ENDIAN = '>'
STRING = ENDIAN + '{}s'
UINT8 = ENDIAN + 'B'
UINT16 = ENDIAN + 'H'
INT16 = ENDIAN + 'h'
INT32 = ENDIAN + 'i'
UINT32 = ENDIAN + 'I'


def parse_utf16_string(file_, encoding='UTF16'):
    """Parse a pascal type UTF16 encoded string from a binary file object"""
    string_length = unpack(UINT8, file_.read(1))[0]
    parsed = unpack(STRING.format(2 * string_length), file_.read(2 * string_length))
    return parsed[0].decode(encoding)


class CHFile(object):
    """Class that implements the Agilent .ch file format version 179 or 181

    .. warning:: Not all aspects of the file header is understood, so there may and probably
       is information that is not parsed. See the method :meth:`._parse_header_status` for
       an overview of which parts of the header is understood.

    .. note:: Although the fundamental storage of the actual data has change, lots of
       inspiration for the parsing of the header has been drawn from the parser in the
       `ImportAgilent.m file <https://github.com/chemplexity/chromatography/blob/dev/
       Methods/Import/ImportAgilent.m>`_ in the `chemplexity/chromatography project
       <https://github.com/chemplexity/chromatography>`_ project. All credit for the parts
       of the header parsing that could be reused goes to the author of that project.

    Attributes:
        values (numpy.array): The intensity values (y-value) or the spectrum. The unit
            for the values is given in `metadata['units']`
        metadata (dict): The extracted metadata
        filepath (str): The filepath this object was loaded from

    """
    fields = (
        ('sequence_line_or_injection', 252, UINT16),
        ('injection_or_sequence_line', 256, UINT16),
        ('start_time', 282, 'x-time'),
        ('end_time', 286, 'x-time'),
        ('version_string', 326, 'utf16'),
        ('description', 347, 'utf16'),
        ('sample', 858, 'utf16'),
        ('operator', 1880, 'utf16'),
        ('date', 2391, 'utf16'),
        ('inlet', 2492, 'utf16'),
        ('instrument', 2533, 'utf16'),
        ('method', 2574, 'utf16'),
        ('software version', 3601, 'utf16'),
        ('software name', 3089, 'utf16'),
        ('software revision', 3802, 'utf16'),
        ('units', 4172, 'utf16'),
        ('detector', 4213, 'utf16'),
        ('yscaling', 4732, ENDIAN + 'd'))
    fields_revtwo = tuple(filter(lambda x: x[1] not in (3601, 3089, 3802), fields))
    data_start = 6144
    supported_versions = {
        179, 181}

    def __init__(self, filepath):
        """Instantiate object

        Args:
            filepath (str): The path of the data file
        """
        self.filepath = filepath
        self.metadata = {}
        with open(self.filepath, 'rb') as (file_):
            self._parse_header(file_)
            if self.metadata['magic_number_version'] == 179:
                self.values = self._parse_data(file_)
            else:
                if self.metadata['magic_number_version'] == 181:
                    self.values = self._parse_data_decompress(file_)

    def _parse_header(self, file_):
        """Parse the header"""
        length = unpack(UINT8, file_.read(1))[0]
        parsed = unpack(STRING.format(length), file_.read(length))
        version = int(parsed[0])
        if version not in self.supported_versions:
            raise ValueError(('Unsupported file version {}').format(version))
        self.metadata['magic_number_version'] = version
        if version == 181:
            self.fields = self.fields_revtwo
        for name, offset, type_ in self.fields:
            file_.seek(offset)
            if type_ == 'utf16':
                self.metadata[name] = parse_utf16_string(file_)
            elif type_ == 'x-time':
                self.metadata[name] = unpack(ENDIAN + 'f', file_.read(4))[0] / 60000
            else:
                self.metadata[name] = unpack(type_, file_.read(struct.calcsize(type_)))[0]


        self.metadata['datetime'] = date_parser(self.metadata['date']) if self.metadata['date'] else None

    def _parse_data(self, file_):
        """Parse the data"""
        file_.seek(0, 2)
        n_points = (file_.tell() - self.data_start) // 8
        file_.seek(self.data_start)
        return numpy.fromfile(file_, dtype=b'<d', count=n_points) * self.metadata['yscaling']

    def _parse_data_decompress(self, file_):
        """Parse the data"""
        file_.seek(0, 2)
        n_points = file_.tell() - self.data_start
        file_length = file_.tell()
        buf = [
            0, 0, 0]
        signal = [0] * int(n_points / 2)
        file_.seek(self.data_start)
        count = 0
        while file_.tell() < file_length:
            buf[2] = unpack(INT16, file_.read(2))[0]
            if buf[2] != 32767:
                buf[1] = buf[1] + buf[2]
                buf[0] = buf[0] + buf[1]
            else:
                buf[0] = unpack(INT16, file_.read(2))[0] * 4294967296
                buf[0] = unpack(UINT32, file_.read(4))[0] + buf[0]
                buf[1] = 0
            signal[count] = buf[0]
            count += 1

        signal[count:] = []
        return numpy.array(signal) * self.metadata['yscaling']

    def times(self):
        """The time values (x-value) for the data set in minutes"""
        return numpy.linspace(self.metadata['start_time'], self.metadata['end_time'], len(self.values))


def parse_utf_string(file_, encoding='UTF8'):
    """Parse a pascal type UTF16 encoded string from a binary file object"""
    string_length = unpack(UINT8, file_.read(1))[0]
    parsed = unpack(STRING.format(string_length), file_.read(string_length))
    return parsed[0].decode(encoding)


class MSFile(object):
    """Class that implements the Agilent .ms file format (mass spectrometry instrument)

    Attributes:
        xic_values (numpy.array): The intensity values of the mass spec
        tic_values (list): The intensity values (y-value)
        times (list): Time for the x-values
        metadata (dict): The extracted metadata
        filepath (str): The filepath this object was loaded from

    """
    fields = (
        ('sample', 24, 'utf8'),
        ('description', 86, 'utf8'),
        ('sequence', 252, INT16),
        ('vial', 253, INT16),
        ('replicate', 254, INT16),
        ('method', 228, 'utf8'),
        ('operator', 148, 'utf8'),
        ('date', 178, 'utf8'),
        ('instrument', 208, 'utf8'),
        ('scans', 278, UINT32))
    data_start = 5771
    supported_versions = [2]

    def __init__(self, filepath):
        """Instantiate object

        Args:
            filepath (str): The path of the data file
        """
        self.filepath = filepath
        self.metadata = {}
        self.times = []
        self.tic_values = []
        self.xic_values = []
        self.intensity = []
        self.mz = []
        self.precision = 3
        with open(self.filepath, 'rb') as (file_):
            self._parse_header(file_)
            if self.metadata['magic_number_version'] == 2:
                self._parse_tic_data(file_)
                self._parse_xic_data(file_)

    def _parse_header(self, file_):
        """Parse the header"""
        length = unpack(UINT8, file_.read(1))[0]
        parsed = unpack(STRING.format(length), file_.read(length))
        version = int(parsed[0])
        if version not in self.supported_versions:
            raise ValueError(('Unsupported file version {}').format(version))
        self.metadata['magic_number_version'] = version
        for name, offset, type_ in self.fields:
            file_.seek(offset)
            if type_ == 'utf8':
                self.metadata[name] = parse_utf_string(file_)
            else:
                if type_ == 'x-time':
                    self.metadata[name] = unpack(ENDIAN + 'f', file_.read(4))[0] / 60000
                else:
                    self.metadata[name] = unpack(type_, file_.read(struct.calcsize(type_)))[0]

        if self.metadata['scans']:
            file_.seek(260)
            self.metadata['tic'] = unpack(INT32, file_.read(struct.calcsize(INT32)))[0] * 2 - 2

            file_.seek(self.metadata['tic'])
            self.metadata['xic'] = []
            xic_buffer = file_.read(self.metadata['scans'] * 12)
            for i in range(0, len(xic_buffer), 12):
                self.metadata['xic'].append(unpack(INT32, xic_buffer[i:i + struct.calcsize(INT32)])[0] * 2 - 2)

            xic_buffer = None
            file_.seek(272)
            self.metadata['normalization'] = unpack(INT32, file_.read(struct.calcsize(INT32)))[0] * 2 - 2

    def _parse_xic_data(self, file_):
        """Parse the data"""
        self.mz = set()

        self.xic_values = [0] * self.metadata['scans']
        for i in range(0, self.metadata['scans']):
            file_.seek(self.metadata['xic'][i])
            xic_offset = int((unpack(INT16, file_.read(struct.calcsize(INT16)))[0] - 18) / 2 + 2)
            file_.seek(self.metadata['xic'][i] + 18)
            mz_buffer = file_.read(xic_offset * 4)

            self.xic_values[i] = []
            for idx, m in enumerate(range(0, len(mz_buffer), 4)):
                self.xic_values[i].append(unpack(UINT16, mz_buffer[m + 2:m + 4])[0])
                self.mz.add(unpack(UINT16, mz_buffer[m:m + 2])[0])

            self.xic_values[i] = numpy.array(self.xic_values[i])
            mz_buffer = None

        self.xic_values = numpy.array(self.xic_values)
        self.xic_values = numpy.bitwise_and(self.xic_values, 16383) * numpy.power(8, numpy.abs(
            numpy.right_shift(self.xic_values, 14)))
        self.mz = numpy.array(list(self.mz))
        self.mz = self.mz / 20.0
        self.mz = numpy.round(self.mz, self.precision)


    def _parse_tic_data(self, file_):
        """The time values (x-value) for the data set in minutes"""
        file_.seek(self.metadata['tic'] + 4)
        xic_buffer = file_.read(self.metadata['scans'] * 12)
        self.times = []
        self.tic_values = []
        for i in range(0, len(xic_buffer), 12):
            self.times.append(unpack(INT32, xic_buffer[i:i + 4])[0] / 60000)
            self.tic_values.append(unpack(INT32, xic_buffer[i + 4:i + 8])[0])

        xic_buffer = None
