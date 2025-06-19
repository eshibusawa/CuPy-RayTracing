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

#ifndef UTIL_RAND_CUH_
#define UTIL_RAND_CUH_

#include <curand_kernel.h>

__device__ float random_float(float min, float max, curandStateXORWOW_t &randomState)
{
  float u = curand_uniform(&randomState);
  return min + (max - min) * u;
}

__device__ vec3 random(curandStateXORWOW_t &randomState)
{
  return vec3(curand_uniform(&randomState), curand_uniform(&randomState), curand_uniform(&randomState));
}

__device__ vec3 random(float min, float max, curandStateXORWOW_t &randomState)
{
  return vec3(random_float(min, max, randomState), random_float(min, max, randomState), random_float(min, max, randomState));
}

__device__ vec3 random_unit_vector(curandStateXORWOW_t &randomState)
{
  while (true)
  {
    auto p = random(-1, 1, randomState);
    auto lensq = p.length_squared();
    if ((RTOW_FLT_TINY) < lensq && lensq <= 1)
      return p / sqrt(lensq);
  }
}

__device__ vec3 random_on_hemisphere(curandStateXORWOW_t &randomState, const vec3& normal)
{
  vec3 on_unit_sphere = random_unit_vector(randomState);
  if (dot(on_unit_sphere, normal) > 0.f)
    return on_unit_sphere;
  else
    return -on_unit_sphere;
}

#endif // UTIL_RAND_CUH_
