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

#include <curand_kernel.h>

using color = vec3;
using point3 = vec3;

__constant__ vec3 g_cameraCenter;
__constant__ vec3 g_pixelDeltaU;
__constant__ vec3 g_pixelDeltaV;
__constant__ vec3 g_pixel00Loc;
__constant__ point3 g_defocusDiskU;
__constant__ point3 g_defocusDiskV;
__constant__ world *g_world;

__device__ bool hit(const type_and_index *p,  ray& r, interval ray_t, hit_record& rec)
{
  bool ret = false;
  const sphere *hittable_sphere = NULL;
  switch (p->type)
  {
  case 0:
    hittable_sphere = &(g_world->spheres[p->index]);
    ret = hit(hittable_sphere, r, ray_t, rec);
    break;

  default:
    ret = false;
    break;
  }

  return ret;
}

__device__ bool scatter(const type_and_index *p, const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandStateXORWOW_t &randomState)
{
  bool ret = false;
  const lambertian *material_lambertian = NULL;
  const metal *material_metal = NULL;
  const dielectric *material_dielectric = NULL;

  switch (p->type)
  {
  case 0:
    material_lambertian = &(g_world->lambertians[p->index]);
    ret = scatter(material_lambertian, r_in, rec, attenuation, scattered, randomState);
    break;

  case 1:
    material_metal = &(g_world->metals[p->index]);
    ret = scatter(material_metal, r_in, rec, attenuation, scattered, randomState);
    break;

  case 2:
    material_dielectric = &(g_world->dielectrics[p->index]);
    ret = scatter(material_dielectric, r_in, rec, attenuation, scattered, randomState);
    break;

  default:
    ret = false;
    break;
  }

  return ret;
}

__device__ color ray_color(const ray& r, const hittable_list& world, curandStateXORWOW_t &randomState)
{
  ray cur_ray = r;
  color cur_attenuation = color(1.f, 1.f, 1.f);

  for(int i = 0; i < (RTOW_MAX_DEPTH); i++)
  {
    hit_record rec;
    if (hit(&world, cur_ray, interval(0.001f, RTOW_FLT_MAX), rec))
    {
      ray scattered;
      color attenuation;
      if (scatter(&(rec.material_ti), cur_ray, rec, attenuation, scattered, randomState))
      {
        cur_attenuation = cur_attenuation * attenuation;
        cur_ray = scattered;
      }
      else
      {
        return vec3(0, 0, 0);
      }
    }
    else
    {
      vec3 unit_direction = unit_vector(cur_ray.direction());
      float a = 0.5f * (unit_direction.y() + 1.0f);
      color c = (1.0f - a) * vec3(1.0f, 1.0f, 1.0f) + a *vec3(0.5f, 0.7f, 1.0f);
      return cur_attenuation * c;
    }
  }

  return vec3(0, 0, 0);
}

__device__ vec3 sample_square(curandStateXORWOW_t &randomState)
{
  // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
  return vec3(curand_uniform(&randomState) - 0.5f, curand_uniform(&randomState) - 0.5f, 0);
}

__device__ point3 defocus_disk_sample(curandStateXORWOW_t &randomStat)
{
  // Returns a random point in the camera defocus disk.
  auto p = random_in_unit_disk(randomStat);
  return g_cameraCenter + (p.x() * g_defocusDiskU) + (p.y() * g_defocusDiskV);
}

__device__ ray get_ray(int i, int j, curandStateXORWOW_t &randomState)
{
  // Construct a camera ray originating from the origin and directed at randomly sampled
  // point around the pixel location i, j.
  auto offset = sample_square(randomState);
  auto pixelSample = g_pixel00Loc + ((offset.x() + i) * g_pixelDeltaU) + ((offset.y() + j) * g_pixelDeltaV);
  auto rayOrigin = ((RTOW_DEFOCUS_ANGLE) <= 0) ? g_cameraCenter : defocus_disk_sample(randomState);
  auto rayDirection = pixelSample - rayOrigin;
  ray r(rayOrigin, rayDirection);

  return r;
}

extern "C" __global__ void render(vec3 *output, unsigned long long randomState)
{
  const int indexX = threadIdx.x + blockIdx.x * blockDim.x;
  const int indexY = threadIdx.y + blockIdx.y * blockDim.y;
  if((indexX >= (RTOW_WIDTH)) || (indexY >= (RTOW_HEIGHT)))
  {
    return;
  }

  const int index = indexY * (RTOW_WIDTH) + indexX;
  curandStateXORWOW_t lrs;
  curand_init(randomState, index, 0, &lrs);
  color pixel_color(0, 0, 0);
  #pragma unroll
  for (int sample = 0; sample < (RTOW_SAMPLES_PER_PIXEL); sample++)
  {
    ray r = get_ray(indexX, indexY, lrs);
    pixel_color += ray_color(r, *(g_world->hittable_lists), lrs);
  }
  pixel_color *= (RTOW_PIXEL_SAMPLE_SCALE);
  const interval intensity(0.f, 0.999f);
  output[index] = color(intensity.clamp(pixel_color.x()), intensity.clamp(pixel_color.y()), intensity.clamp(pixel_color.z()));
}
