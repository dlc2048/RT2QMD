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
Fortran-like binary I/O
"""

__author__    = "Chang-Min Lee"
__copyright__ = "Copyright 2025, Seoul National University"
__credits__   = ["Chang-Min Lee", "Sung-Joon Ye"]
__license__   = "Apache-2.0"
__email__     = "dlc2048@snu.ac.kr"
__status__    = "Production"

import numpy as np


class Fortran:
    def __init__(self, file_name: str, mode="r"):
        """universal binary I/O"""
        self._file = open(file_name, mode=mode+"b")
        
    def close(self):
        self._file.close()

    def init(self):
        self._file.seek(0)

    def read(self, dtype):
        seg = self._file.read(4)
        if not seg:
            return seg  # EOF
        blen1 = np.frombuffer(seg, dtype=np.int32)[0]
        buffer = self._file.read(blen1)
        seg = self._file.read(4)
        blen2 = np.frombuffer(seg, dtype=np.int32)[0]

        if blen1 != blen2:
            raise ValueError
        if dtype == str:
            return buffer.decode()
        else:
            return np.frombuffer(buffer, dtype=dtype)
    
    def write(self, ndarray: np.ndarray | str):
        """write ndarray binary segment

        ndarray: numpy array. dtype should be int32, float32 or float64
        """
        # write ndarray binary as below
        # check data type
        if type(ndarray) is str:  # string
            bs = ndarray.encode()
        else:  # ndarray
            bs = ndarray.flatten().tobytes()

        length = len(bs)
        blen = np.array([length], dtype=np.int32).tobytes()
        self._file.write(blen)
        self._file.write(bs)
        self._file.write(blen)


class UnformattedFortran:
    def __init__(self, file_name: str, mode="r", recl: int = 1):
        """universal binary I/O"""
        self._file = open(file_name, mode=mode+"b")
        self._recl = recl

    def close(self):
        self._file.close()

    def init(self):
        self._file.seek(0)

    def read(self, rec: int, size: int = -1):
        self._file.seek(self._recl * rec)
        return self._file.read(size)

    def write(self, ndarray: np.ndarray | str):
        # write ndarray binary as below
        # check data type
        if type(ndarray) is str:  # string
            bs = ndarray.encode()
        else:  # ndarray
            bs = ndarray.flatten().tobytes()
        self._file.write(bs)
