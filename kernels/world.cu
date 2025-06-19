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

extern "C" __global__ void createWorld(unsigned long *objects_ptr, unsigned long *materials_ptr, unsigned long *world_ptr)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    material *materials[4];
    materials[0] = new lambertian(color(0.8f, 0.8f, 0));
    materials[1] = new lambertian(color(0.1f, 0.2f, 0.5f));
    materials[2] = new metal(color(0.8f, 0.8f, 0.8f), 0.3f);
    materials[3] = new metal(color(0.8f, 0.6f, 0.2f), 1);

    hittable *objects[4];
    objects[0] = new sphere(point3( 0, -100.5f, -1),    100,  materials[0]);
    objects[1] = new sphere(point3( 0, 0,       -1.2f), 0.5f, materials[1]);
    objects[2] = new sphere(point3(-1, 0,       -1),    0.5f, materials[2]);
    objects[3] = new sphere(point3( 1, 0,       -1),    0.5f, materials[3]);

    objects_ptr[0] = reinterpret_cast<unsigned long>(objects[0]);
    objects_ptr[1] = reinterpret_cast<unsigned long>(objects[1]);
    objects_ptr[2] = reinterpret_cast<unsigned long>(objects[2]);
    objects_ptr[3] = reinterpret_cast<unsigned long>(objects[3]);

    materials_ptr[0] = reinterpret_cast<unsigned long>(materials[0]);
    materials_ptr[1] = reinterpret_cast<unsigned long>(materials[1]);
    materials_ptr[2] = reinterpret_cast<unsigned long>(materials[2]);
    materials_ptr[3] = reinterpret_cast<unsigned long>(materials[3]);

    world_ptr[0] = reinterpret_cast<unsigned long>(new hittable_list(objects, 4));
  }
}

extern "C" __global__ void destroyWorld(unsigned long *objects_ptr, unsigned long *materials_ptr, unsigned long *world_ptr)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    hittable *objects = reinterpret_cast<hittable *>(objects_ptr[0]);
    delete objects;
    objects = reinterpret_cast<hittable *>(objects_ptr[1]);
    delete objects;
    objects = reinterpret_cast<hittable *>(objects_ptr[2]);
    delete objects;
    objects = reinterpret_cast<hittable *>(objects_ptr[3]);
    delete objects;
    objects_ptr[0] = 0;
    objects_ptr[1] = 0;
    objects_ptr[2] = 0;
    objects_ptr[3] = 0;

    material *materials = reinterpret_cast<material *>(materials_ptr[0]);
    delete materials;
    materials = reinterpret_cast<material *>(materials_ptr[1]);
    delete materials;
    materials = reinterpret_cast<material *>(materials_ptr[2]);
    delete materials;
    materials = reinterpret_cast<material *>(materials_ptr[3]);
    delete materials;
    materials_ptr[0] = 0;
    materials_ptr[1] = 0;
    materials_ptr[2] = 0;
    materials_ptr[3] = 0;

    hittable *world = reinterpret_cast<hittable *>(world_ptr[0]);
    delete world;
    world_ptr[0] = 0;
  }
}
