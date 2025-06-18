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

import math
import numpy as np
import cupy as cp

from util_cuda import upload_constant
from world import world

class CameraSettings():
    def __init__(self) -> None:
        self.aspect_ratio = 1.0
        self.image_width = 100
        self.samples_per_pixel = 10

class Camera():
    def __init__(self, settings: CameraSettings, cuda_source: str) -> None:
        self.settings = settings
        image_height = int(self.settings.image_width / self.settings.aspect_ratio)
        self.settings.image_height = max(1, image_height)

        self.cuda_source = cuda_source

    def setup_module(self) -> None:
        image_width = self.settings.image_width
        image_height = self.settings.image_height

        cuda_source = self.cuda_source
        cuda_source = cuda_source.replace('RTOW_FLT_MAX', str(cp.finfo(cp.float32).max) + 'f')
        cuda_source = cuda_source.replace('RTOW_WIDTH', str(image_width))
        cuda_source = cuda_source.replace('RTOW_HEIGHT', str(image_height))

        module = cp.RawModule(code=cuda_source)
        module.compile()
        self.module = module

        focal_length = 1.0
        viewport_height = 2.0
        viewport_width = viewport_height * (float(image_width)/image_height)
        camera_center = np.array([0, 0, 0], dtype=np.float32)

        viewport_u = np.array([viewport_width, 0, 0], dtype=np.float32)
        viewport_v = np.array([0, -viewport_height, 0], dtype=np.float32)

        pixel_delta_u = viewport_u / image_width
        pixel_delta_v = viewport_v / image_height

        viewport_upper_left = camera_center - np.array([0, 0, focal_length], dtype=np.float32) - viewport_u/2 - viewport_v/2
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v)

        upload_constant(self.module, camera_center, 'g_cameraCenter')
        upload_constant(self.module, pixel_delta_u, 'g_pixelDeltaU')
        upload_constant(self.module, pixel_delta_v, 'g_pixelDeltaV')
        upload_constant(self.module, pixel00_loc, 'g_pixel00Loc')

    def render(self, world: world) -> None:
        image_width = self.settings.image_width
        image_height = self.settings.image_height

        img_gpu = cp.empty((image_height, image_width, 3), dtype=cp.float32)
        assert img_gpu.flags.c_contiguous

        gpu_func = self.module.get_function('render')
        sz_block = 32, 32
        sz_grid = math.ceil(image_width / sz_block[1]), math.ceil(image_height / sz_block[0])
        gpu_func(
        block=sz_block, grid=sz_grid,
        args=(img_gpu,
                world.world_ptr)
        )
        cp.cuda.runtime.deviceSynchronize()

        return img_gpu
