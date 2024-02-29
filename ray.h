#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>
#include <curand_kernel.h>
#include "vec3.h"


class ray
{
public:
    __device__ ray() {}
    __device__ ray(const vec3& a, const vec3& b) { A = a; B = b; tm = 0.0f; }
    __device__ ray(const vec3& a, const vec3& b, double time) { A = a; B = b; tm = time; }



    __device__ vec3 origin() const { return A; }
    __device__ vec3 direction() const { return B; }
    __device__ vec3 point_at_parameter(float t) const { return A + t * B; }
    __device__ double time() const { return tm; }


    vec3 A;
    vec3 B;
    double tm;
};

