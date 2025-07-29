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
RT2 hadron data
"""

__author__    = "Chang-Min Lee"
__copyright__ = "Copyright 2025, Seoul National University"
__credits__   = ["Chang-Min Lee", "Sung-Joon Ye"]
__license__   = "Apache-2.0"
__email__     = "dlc2048@snu.ac.kr"
__status__    = "Production"

import os
import numpy as np

import rt2qmd.constant as const
from rt2qmd.fortran import Fortran
from rt2qmd.singleton import _SingletonTemplate


class NuclearMassData(_SingletonTemplate):
    __file = "resource\\hadron\\walletlifetime.dat"

    def __init__(self):
        self._mass_table = {}

        if 'MCRT2_HOME' not in os.environ:
            assert 'Environment variable "MCRT2_HOME" is missing'
        home  = os.environ['MCRT2_HOME']
        fname = os.path.join(home, NuclearMassData.__file)

        with open(fname) as file:
            for line in file:
                items  = line.split()
                a      = int(items[0])
                z      = int(items[1])
                excess = float(items[2])
                mass   = a * const.ATOMIC_MASS_UNIT + excess - self._electronMass(z)
                self._mass_table[z * 1000 + a] = mass

    @staticmethod
    def _electronMass(z: int) -> float:
        emass   = z * const.MASS_ELECTRON
        binding = 14.4381 * z**2.39 + 1.55468e-6 * z**5.35
        return emass - binding * 1e-6

    def getMass(self, z: int, a: int) -> float:
        za = z * 1000 + a
        if za in self._mass_table.keys():
            return self._mass_table[za]
        else:
            return self._getWeizsaeckerMass(z, a)

    @staticmethod
    def _getWeizsaeckerMass(z: int, a: int) -> float:
        npair   = (a - z) % 2
        zpair   = z % 2
        binding = (
            - 15.67 * a
            + 17.23 * a**(2/3)
            + 93.15 * (a/2 - z)**2 / a
            + 0.6984523 * z**2 * a**(-1/3)
        )
        if npair == zpair:
            binding += (npair + zpair - 1) * 12 / a**0.5
        return z * const.MASS_PROTON + (a - z) * const.MASS_NEUTRON + binding


class NeutronGroup(_SingletonTemplate):
    __nfile = "resource\\neutron\\endf7_260_egn.bin"
    __gfile = "resource\\neutron\\endf7_260_egg.bin"

    def __init__(self):
        if 'MCRT2_HOME' not in os.environ:
            assert 'Environment variable "MCRT2_HOME" is missing'
        home = os.environ['MCRT2_HOME']

        nfname    = os.path.join(home, NeutronGroup.__nfile)
        nf        = Fortran(nfname)
        self._egn = nf.read(np.float32)
        nf.close()

        gfname    = os.path.join(home, NeutronGroup.__gfile)
        gf        = Fortran(gfname)
        self._egg = gf.read(np.float32)
        gf.close()

    def egn(self):
        return self._egn

    def egg(self):
        return self._egg
