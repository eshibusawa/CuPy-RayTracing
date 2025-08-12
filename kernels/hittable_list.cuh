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

#ifndef HITTABLE_LIST_CUH_
#define HITTABLE_LIST_CUH_

#pragma pack(push, 4)
struct hittable_list
{
  type_and_index *hittable_ti;
  int object_count;
};
#pragma pack(pop)

__device__ bool hit(const hittable_list *p, ray& r, interval ray_t, hit_record& rec)
{
  hit_record temp_rec;
  bool hit_anything = false;
  auto closest_so_far = ray_t.max;

  for (int i = 0; i < p->object_count; i++)
  {
    if (hit(&(p->hittable_ti[i]), r, interval(ray_t.min, closest_so_far), temp_rec))
    {
      hit_anything = true;
      closest_so_far = temp_rec.t;
      rec = temp_rec;
    }
  }

  return hit_anything;
}

#endif // HITTABLE_LIST_CUH_
