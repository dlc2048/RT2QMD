# -*- coding: utf-8 -*-
from __future__ import annotations

#
# Copyright (C) 2025 CM Lee, SJ Ye, Seoul Sational University
#
# Licensed to the Apache Software Foundation(ASF) under one
# or more contributor license agreements.See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.You may obtain a copy of the License at
# 
# http ://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.See the License for the
# specific language governing permissionsand limitations
# under the License.

"""
RT2 tally I/O and interface
"""

__author__    = "Chang-Min Lee"
__copyright__ = "Copyright 2025, Seoul National University"
__credits__   = ["Chang-Min Lee", "Sung-Joon Ye"]
__license__   = "Apache-2.0"
__email__     = "dlc2048@snu.ac.kr"
__status__    = "Production"

from enum import Enum
from copy import deepcopy
from typing import Union

import numpy as np

from rt2qmd.fortran import Fortran, UnformattedFortran
from rt2qmd.particle import PID_TO_PNAME, DTYPE_PHASE_SPACE
from rt2qmd.print import fieldFormat, nameFormat


class ENERGY_TYPE(Enum):
    LINEAR  = 0
    LOG     = 1


class DENSITY_TYPE(Enum):
    DEPO = 0
    DOSE = 1


class _TallyContext:
    def __init__(self):
        self.name = ""
        self.unit = ""
        self.data = None
        self.unc  = None

    def _readHeader(self, stream: Fortran):
        name_byte = stream.read(np.byte).tostring()
        self.name = name_byte.decode('utf-8')
        part_byte = stream.read(np.byte).tostring()
        self.unit = part_byte.decode('utf-8')

    def _writeHeader(self, stream: Fortran):
        stream.write(self.name)
        stream.write(self.unit)

    @staticmethod
    def _readData(stream: Fortran):
        data_1d = stream.read(np.float32)
        unc_1d  = stream.read(np.float32)
        return data_1d, unc_1d

    def _writeData(self, stream: Fortran):
        stream.write(self.data.flatten())
        stream.write(self.unc.flatten())

    def _add(self, other: Union[_TallyContext, float, int, np.ndarray]):
        if isinstance(other, _TallyContext):
            if self.unit != other.unit:
                raise TypeError()
            # Error propagation
            var = (self.unc * self.data) ** 2 + (other.unc * other.data) ** 2
            self.data += other.data
        elif isinstance(other, (float, int, np.ndarray)):
            var = (self.unc * self.data) ** 2
            self.data += other
        else:
            raise TypeError()
        self.unc = np.divide(np.sqrt(var), self.data, out=np.zeros_like(self.data), where=self.data != 0)

    def _sub(self, other: Union[_TallyContext, float, int, np.ndarray]):
        if isinstance(other, _TallyContext):
            if self.unit != other.unit:
                raise TypeError()
            # Error propagation
            var = (self.unc * self.data) ** 2 + (other.unc * other.data) ** 2
            self.data -= other.data  # Boundary check will be processed in child
        elif isinstance(other, (float, int, np.ndarray)):
            var = (self.unc * self.data) ** 2
            self.data -= other
            # Same unc
        else:
            raise TypeError()
        self.unc = np.divide(np.sqrt(var), self.data, out=np.zeros_like(self.data), where=self.data != 0)

    def _mul(self, other: Union[float, int, np.ndarray]):
        if isinstance(other, (float, int, np.ndarray)):
            self.data *= other
            # Same unc
        else:
            raise TypeError()

    def _truediv(self, other: Union[float, int, np.ndarray]):
        if isinstance(other, (float, int, np.ndarray)):
            self.data /= other
            # Same unc
        else:
            raise TypeError()

    def _summary(self):
        message = ""
        message += fieldFormat("Name", self.name)
        message += fieldFormat("Unit", self.unit)
        return message


class _FilterContext:
    def __init__(self):
        self.part = ""

    def _readFilter(self, stream: Fortran):
        name_byte = stream.read(np.byte).tostring()
        self.part = name_byte.decode('utf-8')

    def _writeFilter(self, stream: Fortran):
        stream.write(self.part)

    def _combine(self, other: Union[_FilterContext, float, int, np.ndarray]):
        if isinstance(other, _FilterContext):
            if self.part == other.part:
                pass
            else:
                self.part = "mixed"
        else:
            pass

    def _summary(self):
        message = ""
        message += fieldFormat("Part", self.part)
        return message


class _FluenceContext:
    def __init__(self):
        self._etype  = ENERGY_TYPE(0)
        self._erange = np.empty(2)
        self._nbin   = 0

    def _readEnergyStructure(self, stream: Fortran):
        etype = stream.read(np.int32)[0]
        self._etype  = ENERGY_TYPE(etype)
        self._erange = stream.read(float)
        self._nbin   = stream.read(np.int32)[0]

    def _writeEnergyStructure(self, stream: Fortran):
        etype = np.array([self._etype.value], dtype=np.int32)
        stream.write(etype)
        stream.write(self._erange)
        nbin = np.array([self._nbin], dtype=np.int32)
        stream.write(nbin)

    def _operatorCheck(self, other: Union[_FluenceContext, float, int]):
        if isinstance(other, _FluenceContext):
            if self._etype != other._etype:
                raise TypeError("Energy bin type must be same")
            if not (self._erange == other._erange).all():
                raise ValueError("Energy boundary must be same")
            if self._nbin != other._nbin:
                raise ValueError("Number of energy bin must be same")

    def _setEnergyBoundary(self, index: slice):
        start = index.start
        if not start:  # None
            start = 0
        elif start < 0:
            start += self._nbin
        stop = index.stop
        if not stop:  # None
            stop = self._nbin
        elif stop < 0:
            stop += self._nbin

        if index.step and index.step > 1:
            raise IndexError
        if not 0 <= start < stop <= self._nbin:
            raise IndexError("Index out of range")

        efrom = self._erange[0]
        eto   = self._erange[1]
        if self._etype == ENERGY_TYPE.LOG:
            efrom = np.log10(efrom)
            eto = np.log10(eto)
        estep = (eto - efrom) / self._nbin
        eto   = efrom + estep * stop
        efrom = efrom + estep * start
        self._nbin = stop - start
        if self._etype == ENERGY_TYPE.LOG:
            efrom = 10 ** efrom
            eto   = 10 ** eto
        self._erange[0] = efrom
        self._erange[1] = eto

    def _summary(self):
        message = ""
        message += fieldFormat("Ebin type", self.etype())
        message += fieldFormat("Ebin range", tuple(self._erange), "MeV")
        message += fieldFormat("# of ebin", self._nbin)
        return message

    def etype(self):
        return "Linear" if self._etype == ENERGY_TYPE.LINEAR else "Log"

    def ebin(self):
        if self._etype == ENERGY_TYPE.LINEAR:
            return np.linspace(self._erange[0], self._erange[1], self._nbin + 1)
        else:
            return np.logspace(np.log10(self._erange[0]), np.log10(self._erange[1]), self._nbin + 1)

    def eindex(self, energy: float):
        """
        Get the energy bin index from energy

        :param energy: Point energy (MeV)
        :return: Energy bin index
        """
        if not self._erange[0] <= energy < self._erange[1]:
            raise IndexError("Point out of range")
        return np.argmax(energy < self.ebin()) - 1


class Yield(_TallyContext, _FilterContext, _FluenceContext):
    def __init__(self, file_name: str, mode="r"):
        _TallyContext.__init__(self)
        _FilterContext.__init__(self)
        _FluenceContext.__init__(self)

        if mode == "r":
            stream = Fortran(file_name)
            _TallyContext._readHeader(self, stream)
            _FilterContext._readFilter(self, stream)
            _FluenceContext._readEnergyStructure(self, stream)

            data_1d, err_1d = _TallyContext._readData(stream)
            # Get dimension info
            shape = self._nbin
            self.data = data_1d.reshape(shape)
            self.unc  = err_1d.reshape(shape)

            stream.close()

    def __add__(self, other: Union[Cross, float, int]):
        if isinstance(other, _TallyContext) and not isinstance(other, Cross):
            raise TypeError
        _FluenceContext._operatorCheck(self, other)
        new = deepcopy(self)
        new._add(other)
        new._combine(other)
        return new

    def __sub__(self, other: Union[Cross, float, int]):
        if isinstance(other, _TallyContext) and not isinstance(other, Cross):
            raise TypeError
        _FluenceContext._operatorCheck(self, other)
        new = deepcopy(self)
        new._sub(other)
        return new

    def __mul__(self, other: Union[Cross, float, int]):
        _FluenceContext._operatorCheck(self, other)
        new = deepcopy(self)
        new._mul(other)
        return new

    def __truediv__(self, other: Union[Cross, float, int]):
        _FluenceContext._operatorCheck(self, other)
        new = deepcopy(self)
        new._truediv(other)
        return new

    def __getitem__(self, index):
        index = Slicer(1)[index]
        new = deepcopy(self)
        new._setEnergyBoundary(index[-1])
        new.data = self.data[index]
        new.unc  = self.unc[index]
        return new

    def summary(self):
        message = ""
        message += _TallyContext._summary(self)
        message += _FilterContext._summary(self)
        message += _FluenceContext._summary(self)
        return message

    def write(self, file_name: str):
        stream = Fortran(file_name, mode='w')
        self._writeHeader(stream)
        self._writeFilter(stream)
        self._writeEnergyStructure(stream)
        self._writeData(stream)
        stream.close()

    def __repr__(self):
        return self.summary()


class PhaseSpace:
    def __init__(self, file_name: str = '', max_counts=-1):
        self._capacity = 100
        self._size     = 0
        self._ps       = np.empty(self._capacity, dtype=DTYPE_PHASE_SPACE)
        if file_name == '':
            pass
        else:
            stream = UnformattedFortran(file_name, recl=36)
            bytes_array = stream.read(0, max_counts)
            stream.close()
            ps_temp = np.frombuffer(bytes_array, dtype=DTYPE_PHASE_SPACE)
            self.append(ps_temp)

    def reserve(self, size: int):
        if size > self._capacity:
            ps_temp = self._ps
            self._ps = np.empty(size, dtype=DTYPE_PHASE_SPACE)
            self._ps[:self._capacity] = ps_temp[:self._capacity]
            self._capacity = size

    def resize(self, size: int):
        if size > self._capacity:
            self.reserve(size)
        if size > self._size:
            self._size = size

    def data(self):
        return self._ps[:self._size]

    def append(self, arr: np.ndarray):
        if arr.dtype != DTYPE_PHASE_SPACE:
            raise ValueError("'arr' dtype must be 'DTYPE_PHASE_SPACE'")
        arr_1d = arr.flatten()

        while self._capacity < self._size + len(arr_1d):
            self.reserve(self._capacity * 2)

        len_origin = self._size
        self.resize(len_origin + len(arr_1d))
        self._ps[len_origin:self._size] = arr_1d
        return

    def write(self, file_name: str):
        stream = UnformattedFortran(file_name, mode="w", recl=36)
        stream.write(self.data())
        stream.close()

    def summary(self):
        ps = self.data()
        total_weight = np.sum(ps['wee'])
        total_count = len(ps)
        message = ""
        message += fieldFormat("Total counts", total_count)
        message += fieldFormat("Total weights", total_weight)

        pid_list = np.unique(ps['pid'])
        for pid in pid_list:
            part = PID_TO_PNAME[pid]
            weight = np.sum(ps[ps['pid'] == pid]['wee'])
            count = len(ps[ps['pid'] == pid])
            message += nameFormat(part)
            message += fieldFormat("counts", count)
            message += fieldFormat("weights", weight)

        return message

    def __repr__(self):
        return self.summary()
