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
RT2 physical quantities,
"""

__author__    = "Chang-Min Lee"
__copyright__ = "Copyright 2025, Seoul National University"
__credits__   = ["Chang-Min Lee", "Sung-Joon Ye"]
__license__   = "Apache-2.0"
__email__     = "dlc2048@snu.ac.kr"
__status__    = "Production"

ATOMIC_MASS_UNIT = 931.494028    # AMU [MeV]
MASS_PROTON      = 938.272013
MASS_NEUTRON     = 939.56536
MASS_ELECTRON    = 0.510998964   # mc^2 [MeV]
MASS_ELECTRON_D  = 1.021997928   # 2mc^2 [MeV]
MASS_ELECTRON_SQ = 0.2611199412  # sqrt(mc^2) [MeV^1/2]

HC_I      = 80.65506856998       # 1/hc [Armstrong MeV]
TWICE_HC2 = 0.000307444456       # [2 * (hc)^2]

# spin
MAXE_SPIN = 15
MAXE_SPI1 = 31
MAXQ_SPIN = 15
MAXU_SPIN = 31

# Rutherford scattering
MAXL_MS = 63
MAXQ_MS = 7
MAXU_MS = 31
