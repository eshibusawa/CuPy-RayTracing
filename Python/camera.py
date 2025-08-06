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
        self.aspect_ratio = 1.0 # Ratio of image width over height
        self.image_width = 100 # Rendered image width in pixel count
        self.samples_per_pixel = 10 # Count of random samples for each pixel
        self.max_depth = 10 # Maximum number of ray bounces into scene

        self.vfov = 90 # Vertical view angle (field of view) in degrees
        self.lookfrom = (0, 0, 0) # Point camera is looking from
        self.lookat = (0, 0, -1) # Point camera is looking at
        self.vup = (0, 1, 0) # Camera-relative "up" direction

        self.defocus_angle = 0 # Variation angle of rays through each pixel
        self.focus_dist = 10 # Distance from camera lookfrom point to plane of perfect focus

        self.sz_block = 16, 16

class Camera():
    def __init__(self, settings: CameraSettings, cuda_source: str) -> None:
        self.settings = settings
        image_height = int(self.settings.image_width / self.settings.aspect_ratio)
        self.settings.image_height = max(1, image_height)

        self.cuda_source = cuda_source

    def setup_module(self) -> None:
        image_width = self.settings.image_width
        image_height = self.settings.image_height
        samples_per_pixel = self.settings.samples_per_pixel
        max_depth = self.settings.max_depth
        defocus_angle = self.settings.defocus_angle

        cuda_source = self.cuda_source
        cuda_source = cuda_source.replace('RTOW_FLT_MAX', str(cp.finfo(cp.float32).max) + 'f')
        cuda_source = cuda_source.replace('RTOW_FLT_TINY', str(cp.finfo(cp.float32).tiny) + 'f')
        cuda_source = cuda_source.replace('RTOW_FLT_NEAR_ZERO', str(1e-8) + 'f')
        cuda_source = cuda_source.replace('RTOW_SAMPLES_PER_PIXEL', str(samples_per_pixel))
        cuda_source = cuda_source.replace('RTOW_PIXEL_SAMPLE_SCALE', str(1.0/samples_per_pixel) + 'f')
        cuda_source = cuda_source.replace('RTOW_MAX_DEPTH', str(max_depth))
        cuda_source = cuda_source.replace('RTOW_DEFOCUS_ANGLE', str(defocus_angle))
        cuda_source = cuda_source.replace('RTOW_WIDTH', str(image_width))
        cuda_source = cuda_source.replace('RTOW_HEIGHT', str(image_height))

        module = cp.RawModule(code=cuda_source)
        module.compile()
        self.module = module

        focus_dist = self.settings.focus_dist
        lookfrom = np.array(self.settings.lookfrom, dtype=np.float32)
        lookat = np.array(self.settings.lookat, dtype=np.float32)
        theta = np.deg2rad(self.settings.vfov)
        h = np.tan(theta/2)
        viewport_height = 2 * h * focus_dist
        viewport_width = viewport_height * (float(image_width)/image_height)
        camera_center = lookfrom

        w = lookfrom - lookat
        w /= np.linalg.norm(w)
        vup = np.array(self.settings.vup, dtype=np.float32)
        u = np.cross(vup, w)
        u /= np.linalg.norm(u)
        v = np.cross(w, u)

        viewport_u = viewport_width * u
        viewport_v = -viewport_height * v

        pixel_delta_u = viewport_u / image_width
        pixel_delta_v = viewport_v / image_height

        viewport_upper_left = camera_center - (focus_dist * w) - viewport_u/2 - viewport_v/2
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v)

        defocus_radius = focus_dist * np.tan(np.deg2rad(self.settings.defocus_angle / 2))
        defocus_disk_u = u * defocus_radius
        defocus_disk_v = v * defocus_radius

        upload_constant(self.module, camera_center, 'g_cameraCenter')
        upload_constant(self.module, pixel_delta_u, 'g_pixelDeltaU')
        upload_constant(self.module, pixel_delta_v, 'g_pixelDeltaV')
        upload_constant(self.module, pixel00_loc, 'g_pixel00Loc')
        upload_constant(self.module, defocus_disk_u, 'g_defocusDiskU')
        upload_constant(self.module, defocus_disk_v, 'g_defocusDiskV')

    def render(self, world: world) -> cp.ndarray:
        image_width = self.settings.image_width
        image_height = self.settings.image_height

        img_gpu = cp.empty((image_height, image_width, 3), dtype=cp.float32)
        assert img_gpu.flags.c_contiguous

        random_state = cp.random.randint(0, 2**64 - 1, (1,), dtype=cp.uint64)
        random_state = cp.uint64(random_state.item())

        gpu_func = self.module.get_function('render')
        sz_block = self.settings.sz_block
        sz_grid = math.ceil(image_width / sz_block[0]), math.ceil(image_height / sz_block[1])
        gpu_func(
        block=sz_block, grid=sz_grid,
        args=(img_gpu,
                world.world_ptr,
                random_state)
        )
        cp.cuda.runtime.deviceSynchronize()
        img_gpu = cp.sqrt(img_gpu)

        return img_gpu
