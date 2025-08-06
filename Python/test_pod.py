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

import os
import pytest
from typing import Dict, Any, Generator

import numpy as np
from numpy.testing import assert_allclose
import cupy as cp

from util_cuda import upload_constant
import dtypes

def create_test_kernel(cuda_source: str, type_name: str, placeholder) -> str:
    return cuda_source.replace(placeholder, type_name)

def recursive_assert(tut: np.dtypes.VoidDType, ref: np.ndarray, acc: np.ndarray, eps: float):
    for fk in tut.names:
        if len(ref[fk].dtype) == 0:
            assert_allclose(np.array(ref[fk]), np.array(acc[fk]), rtol=eps)
        else:
            recursive_assert(tut[fk], ref[fk], acc[fk], eps)

@pytest.fixture(scope='module')
def setup_module() -> Generator[Dict[str, Any], Any, None]:
    dn_base = os.path.dirname(os.path.dirname(__file__))
    dn = os.path.join(dn_base, 'kernels')
    fnl = []
    fnl.append(os.path.join(dn, 'vec3.cuh'))
    fnl.append(os.path.join(dn, 'util_rand.cuh'))
    fnl.append(os.path.join(dn, 'interval.cuh'))
    fnl.append(os.path.join(dn, 'ray.cuh'))
    fnl.append(os.path.join(dn, 'type_and_index.cuh'))
    fnl.append(os.path.join(dn, 'hit_record.cuh'))
    fnl.append(os.path.join(dn, 'material.cuh'))
    fnl.append(os.path.join(dn, 'sphere.cuh'))
    fnl.append(os.path.join(dn, 'world.cuh'))
    fnl.append(os.path.join(dn, 'dispatcher.cuh'))
    fnl.append(os.path.join(dn, 'hittable_list.cuh'))
    fnl.append(os.path.join(dn, 'render.cu'))
    cuda_source = None
    for fpfn in fnl:
        with open(fpfn, 'r') as f:
            if cuda_source is None:
                cuda_source = f.read()
            else:
                cuda_source += f.read()

    # DUMMY
    cuda_source = cuda_source.replace('RTOW_FLT_MAX', str(cp.finfo(cp.float32).max) + 'f')
    cuda_source = cuda_source.replace('RTOW_FLT_TINY', str(cp.finfo(cp.float32).tiny) + 'f')
    cuda_source = cuda_source.replace('RTOW_FLT_NEAR_ZERO', str(1e-8) + 'f')
    aspect_ratio = 16.0 / 9.0
    image_width = int(1200)
    samples_per_pixel = int(500)
    max_depth = int(50)
    defocus_angle = 0.6
    image_height = int(image_width / aspect_ratio)
    cuda_source = cuda_source.replace('RTOW_SAMPLES_PER_PIXEL', str(samples_per_pixel))
    cuda_source = cuda_source.replace('RTOW_PIXEL_SAMPLE_SCALE', str(1.0/samples_per_pixel) + 'f')
    cuda_source = cuda_source.replace('RTOW_MAX_DEPTH', str(max_depth))
    cuda_source = cuda_source.replace('RTOW_DEFOCUS_ANGLE', str(defocus_angle))
    cuda_source = cuda_source.replace('RTOW_WIDTH', str(image_width))
    cuda_source = cuda_source.replace('RTOW_HEIGHT', str(image_height))

    place_holder = 'RTOW_POD_TYPE_NAME'
    types_under_test = []
    types_under_test.append(('type_and_index', dtypes.type_and_index))
    types_under_test.append(('lambertian', dtypes.lambertian))
    types_under_test.append(('metal', dtypes.metal))
    types_under_test.append(('dielectric', dtypes.dielectric))
    types_under_test.append(('sphere', dtypes.sphere))
    types_under_test.append(('world', dtypes.world))
    types_under_test.append(('hittable_list', dtypes.hittable_list))

    fpfn = os.path.join(dn, 'test_pod.cu')
    with open(fpfn, 'r') as f:
        test_kernel_template = f.read()
    pod_copy_length = 100
    test_kernel_template = test_kernel_template.replace('RTOW_POD_COPY_LENGTH', str(pod_copy_length))

    test_kernels = []
    test_keys = []
    for tut in types_under_test:
        test_kernel = test_kernel_template.replace(place_holder, str(tut[0]))
        kernel_name_ptr_sz = 'getPointerSize_' + str(tut[0])
        assert kernel_name_ptr_sz in test_kernel
        kernel_name_sz = 'getPODSize_' + str(tut[0])
        assert kernel_name_sz in test_kernel
        kernel_name_copy = 'copyPOD_' + str(tut[0])
        assert kernel_name_copy in test_kernel
        kernel_name_copy_global = 'copyPODFromGlobal_' + str(tut[0])
        assert kernel_name_copy_global in test_kernel
        key_name_global = 'g_ptr_' + str(tut[0])
        assert key_name_global in test_kernel

        cuda_source += test_kernel
        test_kernels.append((kernel_name_ptr_sz, kernel_name_sz, kernel_name_copy, kernel_name_copy_global))
        test_keys.append(key_name_global)

    module = cp.RawModule(code=cuda_source, enable_cooperative_groups=True)
    module.compile()

    yield {
        'module':module,
        'types_under_test':types_under_test,
        'test_kernels':test_kernels,
        'test_keys':test_keys,
        'eps':1E-7,
        'pod_copy_length':pod_copy_length
    }

def test_pod_size(setup_module: Generator[Dict[str, Any], Any, None]) -> None:
    module = setup_module['module']
    types_under_test = setup_module['types_under_test']
    test_kernels = setup_module['test_kernels']

    sz_gpu = cp.zeros((1,), dtype=cp.int32)
    assert sz_gpu.flags.c_contiguous

    sz_block = 1,
    sz_grid = 1,
    for tut, tk in zip (types_under_test, test_kernels):
        gpu_func = module.get_function(tk[0])
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                sz_gpu,
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        sz = int(sz_gpu[0].get())
        assert sz == cp.dtype(cp.uint64).itemsize, 'expected POD {tut[0]} pointer size {cp.dtype(cp.uint64).itemsize}, but got {sz}'

        gpu_func = module.get_function(tk[1])
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                sz_gpu,
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        sz = int(sz_gpu[0].get())
        assert sz == tut[1].itemsize, 'expected POD {tut[0]} size {dtype_pod.itemsize}, but got {sz}'

def test_pod_copy(setup_module: Generator[Dict[str, Any], Any, None]) -> None:
    module = setup_module['module']
    types_under_test = setup_module['types_under_test']
    test_kernels = setup_module['test_kernels']
    eps = setup_module['eps']

    test_data = []
    test_data.append(np.array([((1234, 5678))], dtype=types_under_test[0][1]))
    test_data.append(np.array([((0.1, 0.2, 0.3),)], dtype=types_under_test[1][1]))
    test_data.append(np.array([((0.1, 0.2, 0.3), 0.4)], dtype=types_under_test[2][1]))
    test_data.append(np.array([((0.1))], dtype=types_under_test[3][1]))
    test_data.append(np.array([((1, 2, 3), 4, (5, 6))], dtype=types_under_test[4][1]))
    test_data.append(np.array([((1, 2, 3, 4, 5, 6),)], dtype=types_under_test[5][1]))
    test_data.append(np.array([((1, 2))], dtype=types_under_test[6][1]))

    sz_block = 1,
    sz_grid = 1,
    for tut, tk, td in zip (types_under_test, test_kernels, test_data):
        arr_in_gpu = cp.frombuffer(td.tobytes(), dtype=np.byte)
        assert arr_in_gpu.flags.c_contiguous
        arr_out_gpu = cp.empty_like(arr_in_gpu)
        assert arr_out_gpu.flags.c_contiguous

        gpu_func = module.get_function(tk[2])
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                arr_out_gpu,
                arr_in_gpu
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        arr_out = np.frombuffer(arr_out_gpu.get(), dtype=tut[1])
        recursive_assert(tut[1], td, arr_out, eps)

def test_pod_copy_from_global(setup_module: Generator[Dict[str, Any], Any, None]) -> None:
    module = setup_module['module']
    types_under_test = setup_module['types_under_test']
    test_kernels = setup_module['test_kernels']
    test_keys = setup_module['test_keys']
    eps = setup_module['eps']
    pod_copy_length = setup_module['pod_copy_length']

    sz_block = pod_copy_length,
    sz_grid = 1,
    gpu_ptr = cp.zeros((1,), dtype=cp.uint64)
    for tut, tk, key in zip (types_under_test, test_kernels, test_keys):
        random_bytes = np.random.bytes(tut[1].itemsize * pod_copy_length)
        td = np.frombuffer(random_bytes, dtype=tut[1])
        arr_in_gpu = cp.frombuffer(td.tobytes(), dtype=np.byte)
        assert arr_in_gpu.flags.c_contiguous
        arr_out_gpu = cp.zeros_like(arr_in_gpu)
        assert arr_out_gpu.flags.c_contiguous

        gpu_ptr[0] = arr_in_gpu.data.ptr
        upload_constant(module, gpu_ptr, key, gpu_ptr.dtype)
        gpu_func = module.get_function(tk[3])
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                arr_out_gpu,
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        arr_out = np.frombuffer(arr_out_gpu.get(), dtype=tut[1])
        recursive_assert(tut[1], td, arr_out, eps)
