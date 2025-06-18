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

__device__ color ray_color(const ray& r, const hittable& world)
{
  hit_record rec;
  if (world.hit(r, interval(0, RTOW_FLT_MAX), rec))
  {
    return 0.5f * (rec.normal + color(1, 1, 1));
  }

  vec3 unit_direction = unit_vector(r.direction());
  float a = 0.5f * (unit_direction.y() + 1.0f);
  return (1.0f - a) * vec3(1.0f, 1.0f, 1.0f) + a *vec3(0.5f, 0.7f, 1.0f);
}

__device__ vec3 sample_square(unsigned long &randomState)
{
  // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
  return vec3(unifBetween(-.5f, .5f, randomState), unifBetween(-.5f, .5f, randomState), 0);
}

__device__ ray get_ray(int i, int j, unsigned long &randomState)
{
  // Construct a camera ray originating from the origin and directed at randomly sampled
  // point around the pixel location i, j.
  auto offset = sample_square(randomState);
  auto pixelSample = g_pixel00Loc + ((offset.x() + i) * g_pixelDeltaU) + ((offset.y() + j) * g_pixelDeltaV);
  auto rayDirection = pixelSample - g_cameraCenter;
  ray r(g_cameraCenter, rayDirection);

  return r;
}

extern "C" __global__ void render(vec3 *output, unsigned long *world_ptr, unsigned long int *randomState)
{
  const int indexX = threadIdx.x + blockIdx.x * blockDim.x;
  const int indexY = threadIdx.y + blockIdx.y * blockDim.y;
  if((indexX >= (RTOW_WIDTH)) || (indexY >= (RTOW_HEIGHT)))
  {
    return;
  }

  const int index = indexY * (RTOW_WIDTH) + indexX;
  hittable *world = reinterpret_cast<hittable *>(world_ptr[0]);
  unsigned long int lrs = randomState[index]; // local random state
  color pixel_color(0, 0, 0);
  #pragma unroll
  for (int sample = 0; sample < (RTOW_SAMPLES_PER_PIXEL); sample++)
  {
    ray r = get_ray(indexX, indexY, lrs);
    pixel_color += ray_color(r, *world);
  }
  pixel_color *= (RTOW_PIXEL_SAMPLE_SCALE);
  const interval intensity(0.f, 0.999f);
  output[index] = color(intensity.clamp(pixel_color.x()), intensity.clamp(pixel_color.y()), intensity.clamp(pixel_color.z()));
}
