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

#ifndef VEC3_CUH_
#define VEC3_CUH_

class vec3
{
public:
  __device__ vec3()
  {
  }
  __device__ vec3(float e0, float e1, float e2)
  {
    e[0] = e0;
    e[1] = e1;
    e[2] = e2;
  }
  __device__ inline float x() const
  {
    return e[0];
  }
  __device__ inline float y() const
  {
    return e[1];
  }
  __device__ inline float z() const
  {
    return e[2];
  }
  __device__ inline vec3 operator-() const
  {
    return vec3(-e[0], -e[1], -e[2]);
  }
  __device__ inline vec3& operator+=(const vec3& v)
  {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
  }
  __device__ inline vec3& operator*=(float t)
  {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
  }
  __device__ inline float length() const
  {
    return norm3df(e[0], e[1], e[2]);
  }
  __device__ inline float length_squared() const
  {
    return fmaf(e[2], e[2], fmaf(e[1], e[1], e[0]*e[0]));
  }
  float e[3];
};

__device__ inline vec3 operator+(const vec3& u, const vec3& v)
{
  return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__device__ inline vec3 operator-(const vec3& u, const vec3& v)
{
  return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__device__ inline vec3 operator*(float t, const vec3 &v)
{
  return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__device__ inline vec3 operator/(const vec3 &v, float t)
{
  return vec3(v.e[0]/t, v.e[1]/t, v.e[2]/t);
}

__device__ inline double dot(const vec3& u, const vec3& v)
{
  return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

__device__ inline vec3 unit_vector(const vec3 &v)
{
  return v / v.length();
}

#endif // VEC3_CUH_
