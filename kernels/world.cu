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

extern "C" __global__ void createMatrials(unsigned long *materials_ptr)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    material *p;
    p = new lambertian(color(0.8f, 0.8f, 0));
    materials_ptr[0] = reinterpret_cast<unsigned long>(p);
    p = new lambertian(color(0.1f, 0.2f, 0.5f));
    materials_ptr[1] = reinterpret_cast<unsigned long>(p);
    p = new dielectric(1.5f);
    materials_ptr[2] = reinterpret_cast<unsigned long>(p);
    p = new dielectric(1.0f / 1.5f);
    materials_ptr[3] = reinterpret_cast<unsigned long>(p);
    p = new metal(color(0.8f, 0.6f, 0.2f), 1);
    materials_ptr[4] = reinterpret_cast<unsigned long>(p);
  }
}

extern "C" __global__ void createSpheres(unsigned long *spheres_ptr, unsigned long *materials_ptr)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    material *pm;
    hittable *ph;

    pm = reinterpret_cast<material *>(materials_ptr[0]);
    ph = new sphere(point3( 0, -100.5f, -1),    100,  pm);
    spheres_ptr[0] = reinterpret_cast<unsigned long>(ph);
    pm = reinterpret_cast<material *>(materials_ptr[1]);
    ph = new sphere(point3( 0, 0,       -1.2f), 0.5f, pm);
    spheres_ptr[1] = reinterpret_cast<unsigned long>(ph);
    pm = reinterpret_cast<material *>(materials_ptr[2]);
    ph = new sphere(point3(-1, 0,       -1),    0.5f, pm);
    spheres_ptr[2] = reinterpret_cast<unsigned long>(ph);
    pm = reinterpret_cast<material *>(materials_ptr[3]);
    ph = new sphere(point3(-1, 0,       -1),    0.4f, pm);
    spheres_ptr[3] = reinterpret_cast<unsigned long>(ph);
    pm = reinterpret_cast<material *>(materials_ptr[4]);
    ph = new sphere(point3( 1, 0,       -1),    0.5f, pm);
    spheres_ptr[4] = reinterpret_cast<unsigned long>(ph);
  }
}

extern "C" __global__ void createWorld(unsigned long *world_ptr, unsigned long *hittables_ptr)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    hittable *hittables[5];
    hittables[0] = reinterpret_cast<hittable *>(hittables_ptr[0]);
    hittables[1] = reinterpret_cast<hittable *>(hittables_ptr[1]);
    hittables[2] = reinterpret_cast<hittable *>(hittables_ptr[2]);
    hittables[3] = reinterpret_cast<hittable *>(hittables_ptr[3]);
    hittables[4] = reinterpret_cast<hittable *>(hittables_ptr[4]);

    world_ptr[0] = reinterpret_cast<unsigned long>(new hittable_list(hittables, 5));
  }
}

extern "C" __global__ void destroyWorld(unsigned long *materials_ptr, unsigned long *hittables_ptr, unsigned long *world_ptr)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    hittable *ph;
    ph = reinterpret_cast<hittable *>(world_ptr[0]);
    delete ph;

    ph = reinterpret_cast<hittable *>(hittables_ptr[0]);
    delete ph;
    ph = reinterpret_cast<hittable *>(hittables_ptr[1]);
    delete ph;
    ph = reinterpret_cast<hittable *>(hittables_ptr[2]);
    delete ph;
    ph = reinterpret_cast<hittable *>(hittables_ptr[3]);
    delete ph;
    ph = reinterpret_cast<hittable *>(hittables_ptr[4]);
    delete ph;

    material *pm = reinterpret_cast<material *>(materials_ptr[0]);
    delete pm;
    pm = reinterpret_cast<material *>(materials_ptr[1]);
    delete pm;
    pm = reinterpret_cast<material *>(materials_ptr[2]);
    delete pm;
    pm = reinterpret_cast<material *>(materials_ptr[3]);
    delete pm;
    pm = reinterpret_cast<material *>(materials_ptr[4]);
    delete pm;
  }
}
