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
RT2 particle dump
"""

__author__    = "Chang-Min Lee"
__copyright__ = "Copyright 2025, Seoul National University"
__credits__   = ["Chang-Min Lee", "Sung-Joon Ye"]
__license__   = "Apache-2.0"
__email__     = "dlc2048@snu.ac.kr"
__status__    = "Production"

import numpy as np
import enum

from rt2qmd.cstruct import getBitMaskAndOffset, FLAGS_DEFAULT, FLAGS_GENION, FLAGS_DEEX
from rt2qmd.hadron import NuclearMassData
from rt2qmd.fortran import Fortran
import rt2qmd.constant as const


PS_DTYPE = [
    ('type',  np.int32),
    ('flag',  np.uint32),
    ('hid',   np.int32),
    ('x',     np.float32),
    ('y',     np.float32),
    ('z',     np.float32),
    ('u',     np.float32),
    ('v',     np.float32),
    ('w',     np.float32),
    ('wee',   np.float32),
    ('e',     np.float32)
]


class BUFFER_TYPE(enum.Enum):
    SOURCE        = 0
    QMD           = 1
    ELECTRON      = 2
    PHOTON        = 3
    POSITRON      = 4
    NEUTRON       = 5
    GNEUTRON      = 6
    GENION        = 7
    RELAXATION    = 8
    RAYLEIGH      = 9
    PHOTO         = 10
    COMPTON       = 11
    PAIR          = 12
    EBREM         = 13
    EBREM_SP      = 14
    MOLLER        = 15
    PBREM         = 16
    PBREM_SP      = 17
    BHABHA        = 18
    ANNIHI        = 19
    NEU_SECONDARY = 20
    DELTA         = 21
    ION_NUCLEAR   = 22
    BME           = 23
    CN_FORMATION  = 24
    ABRASION      = 25
    NUC_SECONDARY = 26
    DEEXCITATION  = 27
    PHOTON_EVAP   = 28
    COMP_FISSION  = 29
    EOB           = 30


class ParticleDefinition:
    def __init__(self,
                 z          : int,
                 a          : int,
                 mass       : float
                 ):
        self._z          = z
        self._a          = a
        self._mass       = mass

    def z(self):
        return self._z

    def a(self):
        return self._a

    def mass(self):
        return self._mass


class convertingRuleDefault(ParticleDefinition):

    def __init__(self, target: int, z: int, a: int, mass: float):
        self._transform_target = target
        super().__init__(z, a, mass)

    def _inheritCommonMember(self, ps_formatted: np.ndarray, ps_unformatted: np.ndarray):
        # inherit
        # bit mask, size, and offset
        bmask, bshift = getBitMaskAndOffset(FLAGS_DEFAULT, 'region')

        ps_formatted['type']   = self._transform_target
        ps_formatted['target'] = 0
        ps_formatted['x']      = ps_unformatted['x']
        ps_formatted['y']      = ps_unformatted['y']
        ps_formatted['z']      = ps_unformatted['z']
        ps_formatted['wee']    = ps_unformatted['wee']
        ps_formatted['hid']    = ps_unformatted['hid']
        ps_formatted['region'] = (ps_unformatted['flag'] & bmask) >> bshift

    @staticmethod
    def _cvtDirectionToMomentum(ps_formatted: np.ndarray, ps_unformatted: np.ndarray, eke: np.ndarray):

        # set momentum
        dnorm = (ps_unformatted['u'] ** 2 +
                 ps_unformatted['v'] ** 2 +
                 ps_unformatted['w'] ** 2) ** 0.5
        m = ps_formatted['mass']
        etot = m + eke
        pnorm = etot ** 2 - m ** 2
        pnorm[pnorm < 0.0] = 0.0
        pnorm = pnorm ** 0.5
        ps_formatted['eke'] = eke
        ps_formatted['u'] = ps_unformatted['u'] * pnorm / dnorm
        ps_formatted['v'] = ps_unformatted['v'] * pnorm / dnorm
        ps_formatted['w'] = ps_unformatted['w'] * pnorm / dnorm

    def convert(self, ps_unformatted: np.ndarray):
        assert(np.all(ps_unformatted['type'] == self._transform_target))
        ps_formatted = np.empty(ps_unformatted.shape, dtype=PDEF_DTYPE)

        self._inheritCommonMember(ps_formatted, ps_unformatted)

        # using direction element, fixed mass
        ps_formatted['exc'] = 0.0

        # set mass & ZA
        z = self.z()
        a = self.a()
        ps_formatted['mass'] = self.mass()
        ps_formatted['za']   = z * 1000 + a

        # kinetic energy
        eke = ps_unformatted['e']

        self._cvtDirectionToMomentum(ps_formatted, ps_unformatted, eke)

        return ps_formatted


class convertingRuleInvalid(convertingRuleDefault):
    def __init__(self):
        super().__init__(-1, 0, 0, 0.0)
        pass

    def convert(self, ps_unformatted: np.ndarray):
        assert False


class convertingRuleGenion(convertingRuleDefault):
    __za_list = None

    def __init__(self, target: int):
        super().__init__(target, 0, 0, 0.0)

    @staticmethod
    def setZANumberList(za_list: np.ndarray):
        convertingRuleGenion.__za_list = za_list

    def convert(self, ps_unformatted: np.ndarray):
        assert (np.all(ps_unformatted['type'] == self._transform_target))
        ps_formatted = np.empty(ps_unformatted.shape, dtype=PDEF_DTYPE)

        self._inheritCommonMember(ps_formatted, ps_unformatted)

        # bit mask, size, and offset
        bmask, bshift = getBitMaskAndOffset(FLAGS_GENION, 'ion_idx')

        # ion mass table
        mass_table = NuclearMassData.instance()
        list_mass = []
        for za in convertingRuleGenion.__za_list:
            z = za // 1000
            a = za % 1000
            list_mass += [mass_table.getMass(z, a)]
        list_mass = np.array(list_mass)

        # set mass & ZA
        iid = (ps_unformatted['flag'] & bmask) >> bshift

        ps_formatted['exc']  = 0.0
        ps_formatted['mass'] = list_mass[iid]
        ps_formatted['za']   = convertingRuleGenion.__za_list[iid]

        a = ps_formatted['za'] % 1000

        # kinetic energy
        eke = ps_unformatted['e'] * a

        self._cvtDirectionToMomentum(ps_formatted, ps_unformatted, eke)

        return ps_formatted


class convertingRuleINC(convertingRuleDefault):
    def __init__(self, target: int):
        super().__init__(target, 0, 0, 0.0)

    def convert(self, ps_unformatted: np.ndarray):
        assert (np.all(ps_unformatted['type'] == self._transform_target))
        ps_formatted = np.empty(ps_unformatted.shape, dtype=PDEF_DTYPE)

        self._inheritCommonMember(ps_formatted, ps_unformatted)

        # mass table
        mass_table = NuclearMassData.instance()

        zapt = np.frombuffer(ps_unformatted['flag'].tobytes(), dtype=np.uint8)

        z_p = zapt[0::4]
        a_p = zapt[1::4]
        z_t = zapt[2::4]
        a_t = zapt[3::4]

        ps_formatted['za']     = z_p * 1000 + a_p
        ps_formatted['region'] = -1

        vfunc = np.vectorize(mass_table.getMass)
        ps_formatted['mass']   = vfunc(z_p, a_p)
        ps_formatted['exc']    = 0.0
        ps_formatted['target'] = z_t * 1000 + a_t

        eke = ps_unformatted['e'] * 1000 * a_p  # GeV -> MeV

        self._cvtDirectionToMomentum(ps_formatted, ps_unformatted, eke)

        return ps_formatted


class convertingRuleINCL(convertingRuleDefault):
    __za_list_target     = None
    __za_list_projectile = None
    
    def __init__(self, target: int):
        super().__init__(target, 0, 0, 0.0)

    @staticmethod
    def setZATable(za_list_target: np.ndarray, za_list_projectile: np.ndarray):
        convertingRuleINCL.__za_list_target     = za_list_target
        convertingRuleINCL.__za_list_projectile = za_list_projectile

    def convert(self, ps_unformatted: np.ndarray):
        assert (np.all(ps_unformatted['type'] == self._transform_target))
        ps_formatted = np.empty(ps_unformatted.shape, dtype=PDEF_DTYPE)

        self._inheritCommonMember(ps_formatted, ps_unformatted)

        # mass table
        mass_table = NuclearMassData.instance()

        tp_idx = np.frombuffer(ps_unformatted['flag'].tobytes(), dtype=np.uint8)
        t_idx  = tp_idx[0::4]
        p_idx  = tp_idx[1::4]

        za_t   = convertingRuleINCL.__za_list_target[t_idx]
        za_p   = convertingRuleINCL.__za_list_projectile[p_idx]

        ps_formatted['za']     = za_p['z'] * 1000 + za_p['a']
        ps_formatted['region'] = -1

        vfunc = np.vectorize(mass_table.getMass)
        ps_formatted['mass']   = vfunc(za_p['z'], za_p['a'])
        ps_formatted['exc']    = 0.0
        ps_formatted['target'] = za_t['z'] * 1000 + za_t['a']

        eke = ps_unformatted['e'] * za_p['a']

        self._cvtDirectionToMomentum(ps_formatted, ps_unformatted, eke)

        return ps_formatted


class convertingRuleDEEX(convertingRuleDefault):
    def __init__(self, target: int):
        super().__init__(target, 0, 0, 0.0)

    def convert(self, ps_unformatted: np.ndarray):
        assert (np.all(ps_unformatted['type'] == self._transform_target))
        ps_formatted = np.empty(ps_unformatted.shape, dtype=PDEF_DTYPE)

        self._inheritCommonMember(ps_formatted, ps_unformatted)

        # mass table
        mass_table = NuclearMassData.instance()

        bmask_z, bshift_z = getBitMaskAndOffset(FLAGS_DEEX, 'z')
        bmask_a, bshift_a = getBitMaskAndOffset(FLAGS_DEEX, 'a')

        # set mass & ZA
        z = (ps_unformatted['flag'] & bmask_z) >> bshift_z
        a = (ps_unformatted['flag'] & bmask_a) >> bshift_a
        ps_formatted['za']   = z.astype(int) * 1000 + a.astype(int)
        ps_formatted['exc']  = ps_unformatted['e']

        vfunc = np.vectorize(mass_table.getMass)
        ps_formatted['mass'] = vfunc(z, a)

        # get kinetic energy from momentum
        ps_formatted['u'] = ps_unformatted['u'] * 1e3  # GeV to MeV
        ps_formatted['v'] = ps_unformatted['v'] * 1e3
        ps_formatted['w'] = ps_unformatted['w'] * 1e3

        pnorm = (ps_formatted['u']**2 + ps_formatted['v']**2 + ps_formatted['w']**2)**0.5
        mtot  = ps_formatted['mass'] + ps_formatted['exc']
        eke   = (pnorm**2 + mtot**2)**0.5 - mtot
        eke[eke < 0.0] = 0.0

        ps_formatted['eke'] = eke

        return ps_formatted


class convertingRuleRelaxation(convertingRuleDefault):
    __za_list        = None
    __shell_offset   = None
    __binding_energy = None

    def __init__(self, target: int):
        super().__init__(target, 0, 0, 0.0)

    @staticmethod
    def setShellStructure(za_list: np.ndarray, shell_offset: np.ndarray, binding_energy: np.ndarray):
        convertingRuleRelaxation.__za_list        = za_list
        convertingRuleRelaxation.__shell_offset   = shell_offset
        convertingRuleRelaxation.__binding_energy = binding_energy

    def convert(self, ps_unformatted: np.ndarray):
        assert (np.all(ps_unformatted['type'] == self._transform_target))
        ps_formatted = np.empty(ps_unformatted.shape, dtype=PDEF_DTYPE)

        self._inheritCommonMember(ps_formatted, ps_unformatted)

        bmask, bshift = getBitMaskAndOffset(FLAGS_DEFAULT, 'fmask')

        # ZA
        shell_id  = (ps_unformatted['flag'] & bmask) >> bshift
        za_target = np.digitize(shell_id, convertingRuleRelaxation.__shell_offset) - 1
        z         = convertingRuleRelaxation.__za_list[za_target]
        za        = z * 1000 + shell_id - convertingRuleRelaxation.__shell_offset[za_target]
        ps_formatted['exc'] = convertingRuleRelaxation.__binding_energy[shell_id]

        ps_formatted['za']  = za

        return ps_formatted


FORMAT_RULE = {
    BUFFER_TYPE.SOURCE       : convertingRuleInvalid(),
    BUFFER_TYPE.QMD          : convertingRuleInvalid(),
    BUFFER_TYPE.ELECTRON     : convertingRuleDefault(BUFFER_TYPE.ELECTRON.value, -1, +0, const.MASS_ELECTRON),
    BUFFER_TYPE.PHOTON       : convertingRuleDefault(BUFFER_TYPE.PHOTON.value  , +0, +0, 0.0                ),
    BUFFER_TYPE.POSITRON     : convertingRuleDefault(BUFFER_TYPE.POSITRON.value, +1, +0, const.MASS_ELECTRON),
    BUFFER_TYPE.NEUTRON      : convertingRuleDefault(BUFFER_TYPE.NEUTRON.value , +0, +1, const.MASS_NEUTRON ),
    BUFFER_TYPE.GNEUTRON     : convertingRuleDefault(BUFFER_TYPE.GNEUTRON.value, +0, +1, const.MASS_NEUTRON ),
    BUFFER_TYPE.GENION       : convertingRuleGenion(BUFFER_TYPE.GENION.value),
    BUFFER_TYPE.RELAXATION   : convertingRuleRelaxation(BUFFER_TYPE.RELAXATION.value),
    BUFFER_TYPE.RAYLEIGH     : convertingRuleDefault(BUFFER_TYPE.RAYLEIGH.value, +0, +0, 0.0                ),
    BUFFER_TYPE.PHOTO        : convertingRuleDefault(BUFFER_TYPE.PHOTO.value   , +0, +0, 0.0                ),
    BUFFER_TYPE.COMPTON      : convertingRuleDefault(BUFFER_TYPE.COMPTON.value , +0, +0, 0.0                ),
    BUFFER_TYPE.PAIR         : convertingRuleDefault(BUFFER_TYPE.PAIR.value    , +0, +0, 0.0                ),
    BUFFER_TYPE.EBREM        : convertingRuleDefault(BUFFER_TYPE.EBREM.value   , -1, +0, const.MASS_ELECTRON),
    BUFFER_TYPE.EBREM_SP     : convertingRuleDefault(BUFFER_TYPE.EBREM_SP.value, -1, +0, const.MASS_ELECTRON),
    BUFFER_TYPE.MOLLER       : convertingRuleDefault(BUFFER_TYPE.MOLLER.value  , -1, +0, const.MASS_ELECTRON),
    BUFFER_TYPE.PBREM        : convertingRuleDefault(BUFFER_TYPE.PBREM.value   , +1, +0, const.MASS_ELECTRON),
    BUFFER_TYPE.PBREM_SP     : convertingRuleDefault(BUFFER_TYPE.PBREM_SP.value, +1, +0, const.MASS_ELECTRON),
    BUFFER_TYPE.BHABHA       : convertingRuleDefault(BUFFER_TYPE.BHABHA.value  , +1, +0, const.MASS_ELECTRON),
    BUFFER_TYPE.ANNIHI       : convertingRuleDefault(BUFFER_TYPE.ANNIHI.value  , +1, +0, const.MASS_ELECTRON),
    BUFFER_TYPE.DELTA        : convertingRuleGenion(BUFFER_TYPE.DELTA.value),
    BUFFER_TYPE.ION_NUCLEAR  : convertingRuleGenion(BUFFER_TYPE.ION_NUCLEAR.value),
    BUFFER_TYPE.BME          : convertingRuleINC(BUFFER_TYPE.BME.value),
    BUFFER_TYPE.CN_FORMATION : convertingRuleINCL(BUFFER_TYPE.CN_FORMATION.value),
    BUFFER_TYPE.ABRASION     : convertingRuleINC(BUFFER_TYPE.ABRASION.value),
    BUFFER_TYPE.NUC_SECONDARY: convertingRuleDEEX(BUFFER_TYPE.NUC_SECONDARY.value),
    BUFFER_TYPE.DEEXCITATION : convertingRuleDEEX(BUFFER_TYPE.DEEXCITATION.value),
    BUFFER_TYPE.PHOTON_EVAP  : convertingRuleDEEX(BUFFER_TYPE.PHOTON_EVAP.value),
    BUFFER_TYPE.COMP_FISSION : convertingRuleDEEX(BUFFER_TYPE.COMP_FISSION.value),
    BUFFER_TYPE.EOB          : convertingRuleInvalid()
}


PDEF_DTYPE = [
    ('type'  , np.int32),
    ('za'    , np.int32),
    ('target', np.int32),
    ('region', np.int32),
    ('hid'   , np.int32),
    ('wee'   , np.float64),
    ('x'     , np.float64),
    ('y'     , np.float64),
    ('z'     , np.float64),
    ('u'     , np.float64),
    ('v'     , np.float64),
    ('w'     , np.float64),
    ('mass'  , np.float64),
    ('eke'   , np.float64),
    ('exc'   , np.float64)
]

INCL_ZA_DTYPE = [
    ('z', np.uint16),
    ('a', np.uint16)
]


class Dump:
    def __init__(self, file_name: str, from_npy: bool = False):
        self._ps = []

        if from_npy:
            ps_zip = np.load(file_name)

            for f in ps_zip.files:
                self._ps += [ps_zip[f]]

        else:
            fort = Fortran(file_name)
            proj_za_list = fort.read(np.int32)

            # Generic ion
            convertingRuleGenion.setZANumberList(proj_za_list)

            # Relaxation
            list_atomic_number  = fort.read(np.int32)
            list_shell_offset   = fort.read(np.int32)
            list_n_electron     = fort.read(np.float64)
            list_binding_energy = fort.read(np.float64)

            # INCL
            use_incl = fort.read(np.int32)[0]
            if use_incl:
                list_za_target     = fort.read(INCL_ZA_DTYPE)
                list_za_projectile = fort.read(INCL_ZA_DTYPE)
                convertingRuleINCL.setZATable(list_za_target, list_za_projectile)

            convertingRuleRelaxation.setShellStructure(
                list_atomic_number,
                list_shell_offset,
                list_binding_energy
            )

            n = fort.read(np.int32)[0]
            for i in range(n):
                ps_seg = self._cvtFormattedPhaseSpace(fort.read(PS_DTYPE))
                self._ps += [ps_seg]

    @staticmethod
    def _cvtFormattedPhaseSpace(ps_unformatted: np.ndarray) -> np.ndarray:
        ps_formatted = np.empty(0, dtype=PDEF_DTYPE)

        for rule_key in FORMAT_RULE:
            mask = ps_unformatted['type'] == rule_key.value

            if not np.any(mask):
                continue

            ps_formatted = np.append(ps_formatted, FORMAT_RULE[rule_key].convert(ps_unformatted[mask]))

        return ps_formatted

    def save(self, file_name: str) -> None:
        np.savez(file_name, *self._ps)

    def __getitem__(self, item):
        return self._ps[item]
