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

__author__    = "Chang-Min Lee"
__copyright__ = "Copyright 2025, Seoul National University"
__credits__   = ["Chang-Min Lee", "Sung-Joon Ye"]
__license__   = "Apache-2.0"
__email__     = "dlc2048@snu.ac.kr"
__status__    = "Production"

from collections.abc import Iterable


def fieldFormat(field: str, value: any, unit: str = ""):
    if isinstance(value, Iterable) and not isinstance(value, str):
        value = str(value.__repr__())
    if unit:
        return "{field: <14}: {value: >24} ({unit})\n".format(field=field, value=value, unit=unit)
    else:
        return "{field: <14}: {value: >24}\n".format(field=field, value=value)


def nameFormat(name: str, max_length: int = 14):
    return "  [ {name: >{max_length}} ]\n".format(name=name[:max_length], max_length=max_length)
