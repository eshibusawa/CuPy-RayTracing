// This file is part of CuPy-RayTracing.
// Copyright (c) 2025, Eijiro Shibusawa <phd_kimberlite@yahoo.co.jp>
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

__constant__ RTOW_POD_TYPE_NAME *g_ptr_RTOW_POD_TYPE_NAME;

extern "C" __global__ void getPointerSize_RTOW_POD_TYPE_NAME(int *output)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index > 0)
  {
    return;
  }
  *output = sizeof(RTOW_POD_TYPE_NAME *);
}

extern "C" __global__ void getPODSize_RTOW_POD_TYPE_NAME(
        int *sz)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index > 0)
  {
    return;
  }
  *sz = sizeof(RTOW_POD_TYPE_NAME);
}

extern "C" __global__ void copyPOD_RTOW_POD_TYPE_NAME(
        RTOW_POD_TYPE_NAME *out,
        const RTOW_POD_TYPE_NAME *__restrict__ in)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index > 0)
  {
    return;
  }
  *out = *in;
}

extern "C" __global__ void copyPODFromGlobal_RTOW_POD_TYPE_NAME(
        RTOW_POD_TYPE_NAME *out)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index > RTOW_POD_COPY_LENGTH)
  {
    return;
  }
  out[index] = g_ptr_RTOW_POD_TYPE_NAME[index];
}
