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

from typing import Tuple

import numpy as np
import cupy as cp

import dtypes
from util_cuda import upload_constant

class world():
    def __init__(self):
        self.spheres = []
        self.lambertians = []
        self.metals = []
        self.dielectrics = []
        self.spheres_gpu = None
        self.lambertians_gpu = None
        self.metals_gpu = None
        self.dielectrics_gpu = None

    def add_lambertian(self, albedo: Tuple[float, float, float]) -> Tuple[int, int]:
        material_type = 0
        material_index = len(self.lambertians)
        self.lambertians.append((albedo,))
        return material_type, material_index

    def add_metal(self, albedo: Tuple[float, float, float], fuzz: float) -> Tuple[int, int]:
        material_type = 1
        material_index = len(self.metals)
        self.metals.append((albedo, fuzz))
        return material_type, material_index

    def add_dielectrics(self, refraction_index: float) -> Tuple[int, int]:
        material_type = 2
        material_index = len(self.dielectrics)
        self.dielectrics.append(refraction_index)
        return material_type, material_index

    def add_sphere(self, center: Tuple[float, float, float], radius: float, type_and_index: Tuple[int, int]) -> None:
        sph = center, radius, type_and_index
        self.spheres.append(sph)

    def create(self) -> None:
        self.spheres = []
        self.lambertians = []
        self.metals = []
        self.dielectrics = []

        material_ti = self.add_lambertian((0.5, 0.5, 0.5))
        self.add_sphere((0, -1000, 0), 1000, material_ti)

        material_ti = self.add_dielectrics((1.5))
        self.add_sphere((0, 1, 0), 1, material_ti)

        material_ti = self.add_lambertian((0.4, 0.2, 0.1))
        self.add_sphere((-4, 1, 0), 1, material_ti)

        material_ti = self.add_metal((0.7, 0.6, 0.5), 0)
        self.add_sphere((4, 1, 0), 1, material_ti)

        rand_func = lambda sz: np.random.rand(sz).astype(np.float32)
        rand_min_max_func = lambda low, high, sz: np.random.uniform(low, high, sz).astype(np.float32)
        for a in range(-11, 11):
            for b in range(-11, 11):
                choose_mat = rand_func(1)
                center = np.array([a + 0.9 * rand_func(1)[0], 0.2, b + 0.9 * rand_func(1)[0]], dtype=np.float32)

                if np.linalg.norm((center - np.array([4, 0.2, 0], dtype=np.float32))) > 0.9:
                    if choose_mat < 0.8:
                        # diffuse
                        albedo = rand_func(3) * rand_func(3)
                        albedo = tuple(albedo.tolist())
                        material_ti = self.add_lambertian(albedo)
                    elif choose_mat < 0.95:
                        # metal
                        albedo = rand_min_max_func(0.5, 1.0, 3) * rand_min_max_func(0.5, 1.0, 3)
                        albedo = tuple(albedo.tolist())
                        fuzz = float(rand_min_max_func(0, 0.5, 1)[0])
                        material_ti = self.add_metal(albedo, fuzz)
                    else:
                        # glass
                        material_ti = self.add_dielectrics((1.5))

                    center = tuple(center.tolist())
                    self.add_sphere(center, 0.2, material_ti)

        self.hittable_ti = np.empty((len(self.spheres),), dtype=dtypes.type_and_index)
        self.hittable_ti['type'] = 0
        self.hittable_ti['index'] = np.arange(self.hittable_ti.shape[0])

        upload_dict = {}
        upload_dict['hittable_ti'] = (self.hittable_ti, dtypes.type_and_index)
        upload_dict['sphere'] = (self.spheres, dtypes.sphere)
        upload_dict['lambertian'] = (self.lambertians, dtypes.lambertian)
        upload_dict['metal'] = (self.metals, dtypes.metal)
        upload_dict['dielectric'] = (self.dielectrics, dtypes.dielectric)
        gpu_dict = {}
        for name, data in upload_dict.items():
            arr = np.array(data[0], dtype=data[1])
            arr_gpu = cp.frombuffer(arr.tobytes(), dtype=cp.byte)
            gpu_dict[name + '_gpu'] = arr_gpu
        self.hittable_ti_gpu = gpu_dict['hittable_ti_gpu']
        self.spheres_gpu = gpu_dict['sphere_gpu']
        self.lambertians_gpu = gpu_dict['lambertian_gpu']
        self.metals_gpu = gpu_dict['metal_gpu']
        self.dielectrics_gpu = gpu_dict['dielectric_gpu']

    def setup_module(self, module: cp.RawModule) -> None:
        hl = np.empty((1,), dtype=dtypes.hittable_list)
        hl['object_count'] = len(self.hittable_ti_gpu)
        hl['hittable_ti'] = self.hittable_ti_gpu.data.ptr

        self.hl_gpu = cp.frombuffer(hl.tobytes(), dtype=cp.byte)
        w = np.empty((1,), dtype=dtypes.world)
        w['pointers'] = (self.hl_gpu.data.ptr, self.hittable_ti_gpu.data.ptr, self.spheres_gpu.data.ptr, \
                         self.lambertians_gpu.data.ptr, self.metals_gpu.data.ptr, self.dielectrics_gpu.data.ptr)
        self.world_gpu = cp.frombuffer(w.tobytes(), dtype=cp.byte)
        upload_constant(module, np.array([self.world_gpu.data.ptr]), 'g_world', cp.uint64)
