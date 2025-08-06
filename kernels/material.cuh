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

#ifndef MATERIAL_CUH_
#define MATERIAL_CUH_

using color = vec3;

#pragma pack(push, 4)
struct lambertian
{
  color albedo;
};

struct metal
{
  color albedo;
  float fuzz;
};

struct dielectric
{
  float refraction_index;
};
#pragma pack(pop)

struct hit_record;
using color = vec3;

__device__ inline vec3 reflect(const vec3& v, const vec3& n)
{
  return v - 2.0f * dot(v, n) * n;
}

__device__ inline vec3 refract(const vec3& uv, const vec3& n, float etai_over_etat)
{
  auto cos_theta = fminf(dot(-uv, n), 1.0f);
  vec3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
  vec3 r_out_parallel = -sqrtf(fabsf(1.0f - r_out_perp.length_squared())) * n;
  return r_out_perp + r_out_parallel;
}

__device__ bool scatter(const lambertian *p, const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandStateXORWOW_t &randomState)
{
  auto scatter_direction = rec.normal + random_unit_vector(randomState);
  if (scatter_direction.near_zero())
  {
      scatter_direction = rec.normal;
  }
  scattered = ray(rec.p, scatter_direction);
  attenuation = p->albedo;

  return true;
}

__device__ bool scatter(const metal *p, const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandStateXORWOW_t &randomState)
{
  vec3 reflected = reflect(r_in.direction(), rec.normal);
  reflected = unit_vector(reflected) + (p->fuzz * random_unit_vector(randomState));
  scattered = ray(rec.p, reflected);
  attenuation = p->albedo;

  return (dot(scattered.direction(), rec.normal) > 0);
}

__device__ float reflectance(const dielectric *p, float cosine, float refraction_index)
{
  // Use Schlick's approximation for reflectance.
  auto r0 = (1 - p->refraction_index) / (1 + p->refraction_index);
  r0 = r0 * r0;
  return r0 + (1 - r0) * powf((1 - cosine), 5);
}

__device__ bool scatter(const dielectric *p, const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandStateXORWOW_t &randomState)
{
  attenuation = color(1.0f, 1.0f, 1.0f);
  float ri = rec.front_face ? (1.0f/p->refraction_index) : p->refraction_index;

  vec3 unit_direction = unit_vector(r_in.direction());
  float cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0f);
  float sin_theta = sqrtf(1.0 - cos_theta*cos_theta);

  bool cannot_refract = ri * sin_theta > 1.0;
  vec3 direction;

  if (cannot_refract || reflectance(p, cos_theta, ri) > curand_uniform(&randomState))
    direction = reflect(unit_direction, rec.normal);
  else
    direction = refract(unit_direction, rec.normal, ri);

  scattered = ray(rec.p, direction);
  return true;
}

#endif // MATERIAL_CUH_
