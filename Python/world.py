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

import cupy as cp

def check_pointer_size(module: cp.RawModule) -> None:
    sz = cp.empty((1), dtype=cp.int32)
    gpu_func = module.get_function('getPointerSize')
    sz_block = 1,
    sz_grid = 1,
    gpu_func(
        block=sz_block, grid=sz_grid,
        args=(sz)
    )
    cp.cuda.runtime.deviceSynchronize()
    sz = sz.get()
    sz = int(sz[0])

    assert sz == 8 # pointer size on GPU is assumed to 64bit

class world():
    def __init__(self, module: cp.RawModule) -> None:
        check_pointer_size(module)
        self.module = module
        self.count = 0
        self.is_world_created = False
    def __copy__(self):
        raise TypeError("copying of this object is not allowed")

    def __deepcopy__(self, memo):
        raise TypeError("deep copying of this object is not allowed")

    def create_world(self) -> None:
        if self.is_world_created:
            return

        self.max_count = 22 * 22 + 4
        spheres_ptr = cp.zeros((self.max_count,), dtype=cp.uint64)
        self.world_ptr = cp.zeros((1,), dtype=cp.uint64)
        count = cp.zeros((1,), dtype=cp.int32)

        random_state = cp.random.randint(0, 2**63 - 1, (3,), dtype=cp.uint64)

        sz_block = 1,
        sz_grid = 1,
        gpu_func = self.module.get_function('createSpheres')
        gpu_func(
            block=sz_block, grid=sz_grid,
            args=(spheres_ptr,
                  count,
                  cp.int32(self.max_count),
                  random_state)
        )
        cp.cuda.runtime.deviceSynchronize()
        self.count = int(count.get()[0])
        assert self.count <= self.max_count

        gpu_func = self.module.get_function('createWorld')
        gpu_func(
            block=sz_block, grid=sz_grid,
            args=(self.world_ptr,
                  spheres_ptr,
                  cp.int32(self.count))
        )
        cp.cuda.runtime.deviceSynchronize()
        self.is_world_created = True

    def destroy_world(self) -> None:
        if not self.is_world_created:
            return
        sz_block = 1,
        sz_grid = 1,
        gpu_func = self.module.get_function('destroyWorld')
        gpu_func(
            block=sz_block, grid=sz_grid,
            args=(self.world_ptr)
        )
        cp.cuda.runtime.deviceSynchronize()
        self.world_ptr[:] = 0
        self.is_world_created = False

    def __del__(self) -> None:
        if self.is_world_created:
            self.destroy_world()
