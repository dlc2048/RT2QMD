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
Bit field structure of RT2 C++ interface
"""

__author__    = "Chang-Min Lee"
__copyright__ = "Copyright 2025, Seoul National University"
__credits__   = ["Chang-Min Lee", "Sung-Joon Ye"]
__license__   = "Apache-2.0"
__email__     = "dlc2048@snu.ac.kr"
__status__    = "Production"

import ctypes
import numpy as np


class FLAGS_DEFAULT(ctypes.Structure):
    _fields_ = [
        ("region", ctypes.c_uint16, 16),
        ("fmask" , ctypes.c_uint16, 16)
    ]


class FLAGS_NEUTRON_P(ctypes.Structure):
    _fields_ = [
        ("region", ctypes.c_uint16, 16),
        ("group" , ctypes.c_short,  12),
        ("fmask" , ctypes.c_uint16, 4 )
    ]


class FLAGS_NEUTRON_S(ctypes.Structure):
    _fields_ = [
        ("iso_idx", ctypes.c_uint16, 16),
        ("rea_idx", ctypes.c_uint8,  8 ),
        ("sec_pos", ctypes.c_uint8,  8 )
    ]


class FLAGS_GENION(ctypes.Structure):
    _fields_ = [
        ("region" , ctypes.c_uint16, 16),
        ("fmask"  , ctypes.c_uint8,  8 ),
        ("ion_idx", ctypes.c_uint8,  8 )
    ]


class FLAGS_DEEX(ctypes.Structure):
    _fields_ = [
        ("region", ctypes.c_uint16, 16),
        ("z"     , ctypes.c_uint8,  8 ),
        ("a"     , ctypes.c_uint8,  8 )
    ]


class FLAGS_INCL(ctypes.Structure):
    _fields_ = [
        ("region"    , ctypes.c_uint16, 16),
        ("target_idx", ctypes.c_uint8,  8 ),
        ("proj_idx"  , ctypes.c_uint8,  8 )
    ]


def getBitMaskAndOffset(flags_struct, name : str) -> tuple:
    offset = 0
    for fname, ftype, bit_size in flags_struct._fields_:
        bit_mask = ((1 << bit_size) - 1) << offset
        if fname == name:
            return bit_mask, offset
        offset += bit_size
    assert False
