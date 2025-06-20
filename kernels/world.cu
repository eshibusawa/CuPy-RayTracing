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

extern "C" __global__ void getPointerSize(int *output)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    *output = sizeof(hittable **);
  }
}

extern "C" __global__ void createSpheres(unsigned long *materials_ptr, unsigned long *spheres_ptr, int *count,
  int maxCount, unsigned long *randomState)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    int offset = 0;
    material *pm;
    hittable *ph;

    pm = new lambertian(color(0.5f, 0.5f, 0.5f));
    materials_ptr[offset] = reinterpret_cast<unsigned long>(pm);
    ph = new sphere(point3(0,-1000,0), 1000, pm);
    spheres_ptr[offset] = reinterpret_cast<unsigned long>(ph);
    offset++;

    pm = new dielectric(1.5f);
    materials_ptr[offset] = reinterpret_cast<unsigned long>(pm);
    ph = new sphere(point3(0, 1, 0), 1.0, pm);
    spheres_ptr[offset] = reinterpret_cast<unsigned long>(ph);
    offset++;

    pm = new lambertian(color(0.4f, 0.2f, 0.1f));
    materials_ptr[offset] = reinterpret_cast<unsigned long>(pm);
    ph = new sphere(point3(-4, 1, 0), 1.0, pm);
    spheres_ptr[offset] = reinterpret_cast<unsigned long>(ph);
    offset++;

    pm = new metal(color(0.7f, 0.6f, 0.5f), 0);
    materials_ptr[offset] = reinterpret_cast<unsigned long>(pm);
    ph = new sphere(point3(4, 1, 0), 1.0, pm);
    spheres_ptr[offset] = reinterpret_cast<unsigned long>(ph);
    offset++;

    curandStateXORWOW_t lrs;
    curand_init(randomState[0], randomState[1], randomState[2], &lrs);
    for (int a = -11; a < 11; a++)
    {
      for (int b = -11; b < 11; b++)
      {
        auto choose_mat = curand_uniform(&lrs);
        point3 center(a + 0.9*curand_uniform(&lrs), 0.2f, b + 0.9f*curand_uniform(&lrs));

        if ((center - point3(4, 0.2f, 0)).length() > 0.9f)
        {
          ph = nullptr;
          pm = nullptr;
          if (choose_mat < 0.8f)
          {
            // diffuse
            auto albedo = random(lrs) * random(lrs);
            pm = new lambertian(albedo);
          }
          else if (choose_mat < 0.95f)
          {
            // metal
            auto albedo = random(0.5f, 1.f, lrs);
            auto fuzz = random_float(0, 0.5f, lrs);
            pm = new metal(albedo, fuzz);
          }
          else
          {
            // glass
            pm = new dielectric(1.5f);
          }

          if (offset >= maxCount)
          {
            continue;
          }
          materials_ptr[offset] = reinterpret_cast<unsigned long>(pm);
          ph = new sphere(center, 0.2f, pm);
          spheres_ptr[offset] = reinterpret_cast<unsigned long>(ph);
          offset++;
        }
      }
    }
    *count = offset;
  }
}

extern "C" __global__ void createWorld(unsigned long *world_ptr, unsigned long *hittables_ptr, int count)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    hittable **hittables = new hittable* [count];
    for (int k = 0; k < count; k++)
    {
      hittables[k] = reinterpret_cast<hittable *>(hittables_ptr[k]);
    }

    world_ptr[0] = reinterpret_cast<unsigned long>(new hittable_list(hittables, count));
    delete [] hittables;
  }
}

extern "C" __global__ void destroyWorld(unsigned long *materials_ptr, unsigned long *hittables_ptr, unsigned long *world_ptr, int count)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    hittable *ph = nullptr;
    material *pm = nullptr;
    ph = reinterpret_cast<hittable *>(world_ptr[0]);
    delete ph;

    for (int k = 0; k < count; k++)
    {
      ph = reinterpret_cast<hittable *>(hittables_ptr[k]);
      delete ph;
      pm = reinterpret_cast<material *>(materials_ptr[k]);
      delete pm;
    }
  }
}
