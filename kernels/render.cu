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

__constant__ vec3 g_cameraCenter;
__constant__ vec3 g_pixelDeltaU;
__constant__ vec3 g_pixelDeltaV;
__constant__ vec3 g_pixel00Loc;

using color = vec3;
using point3 = vec3;

__device__ float hit_sphere(const point3& center, double radius, const ray& r)
{
  vec3 oc = center - r.origin();
  auto a = r.direction().length_squared();
  auto h = dot(r.direction(), oc);
  auto c = oc.length_squared() - radius*radius;
  auto discriminant = h*h - a*c;
  if (discriminant < 0)
  {
    return -1.0f;
  }
  else
  {
    return (h - sqrtf(discriminant)) / a;
  }
}

__device__ vec3 ray_color(const ray& r)
{
  auto t = hit_sphere(point3(0, 0, -1), 0.5f, r);
  if (t > 0.0)
  {
    vec3 N = unit_vector(r.at(t) - vec3(0, 0, -1));
    return 0.5f * color(N.x()+1, N.y()+1, N.z()+1);
  }

  vec3 unit_direction = unit_vector(r.direction());
  float a = 0.5f * (unit_direction.y() + 1.0f);
  return (1.0f - a) * vec3(1.0f, 1.0f, 1.0f) + a *vec3(0.5f, 0.7f, 1.0f);
}

extern "C" __global__ void render(vec3 *output)
{
  const int indexX = threadIdx.x + blockIdx.x * blockDim.x;
  const int indexY = threadIdx.y + blockIdx.y * blockDim.y;
  if((indexX >= (RTOW_WIDTH)) || (indexY >= (RTOW_HEIGHT)))
  {
    return;
  }
  auto pixelCenter = g_pixel00Loc + (float(indexX) * g_pixelDeltaU) + (float(indexY) * g_pixelDeltaV);
  auto rayDirection = pixelCenter - g_cameraCenter;
  ray r(g_cameraCenter, rayDirection);

  const int index = indexY * (RTOW_WIDTH) + indexX;
  output[index] = ray_color(r);
}
