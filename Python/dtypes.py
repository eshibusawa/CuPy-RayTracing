# This file is part of CuPy-RayTracing.
# Copyright (c) 2025, Eijiro Shibusawa <phd_kimberlite@yahoo.co.jp>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np

type_and_index = np.dtype({
    'names': ['type', 'index'],
    'formats': [np.int64, np.int64],
    'offsets': [0, 8]
})

sphere = np.dtype({
    'names': ['center', 'radius', 'material_ti'],
    'formats': [(np.float32, 3), np.float32, type_and_index],
    'offsets': [0, 12, 16]
})

lambertian = np.dtype({
    'names': ['albedo'],
    'formats': [(np.float32, 3)],
    'offsets': [0]
})

metal = np.dtype({
    'names': ['albedo', 'fuzz'],
    'formats': [(np.float32, 3), np.float32],
    'offsets': [0, 12]
})

dielectric = np.dtype({
    'names': ['refraction_index'],
    'formats': [np.float32],
    'offsets': [0]
})

hittable_list = np.dtype({
    'names': ['hittable_ti', 'object_count'],
    'formats': [np.uint64, np.int32],
    'offsets': [0, 8]
})

world = np.dtype({
    'names': ['pointers'],
    'formats': [(np.uint64, 6)],
    'offsets': [0]
})
