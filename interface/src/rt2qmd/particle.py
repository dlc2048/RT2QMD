 -*- coding: utf-8 -*-
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
RT2 particle type and dump structure
"""

__author__    = "Chang-Min Lee"
__copyright__ = "Copyright 2025, Seoul National University"
__credits__   = ["Chang-Min Lee", "Sung-Joon Ye"]
__license__   = "Apache-2.0"
__email__     = "dlc2048@snu.ac.kr"
__status__    = "Production"

from enum import Enum


class PTYPE(Enum):
    ELECTRON = -1
    PHOTON   = 0
    POSITRON = 1
    NEUTRON  = 6
    FNEUTRON = 7
    PROTON   = 21
    HEAVYION = 26
    HEAVYNUC = 27


PID_TO_PNAME = {
    -99: 'Mixed',
    -11: 'Hounsfield',
    -1 : 'Electron',
    0  : 'Photon',
    1  : 'Positron',
    6  : 'Neutron',
    7  : 'Fast Neutron',
    21 : 'Proton',
    26 : 'Heavy Ion',
    27 : 'Untransportable'
}


DTYPE_PHASE_SPACE = [
    ('pid', 'int32'),
    ('x'  , 'float32'),
    ('y'  , 'float32'),
    ('z'  , 'float32'),
    ('u'  , 'float32'),
    ('v'  , 'float32'),
    ('w'  , 'float32'),
    ('wee', 'float32'),
    ('e'  , 'float32')
]

