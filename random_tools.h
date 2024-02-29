#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include "vec3.h"



#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__global__ inline void rand_init(curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__device__ inline vec3 random_in_unit_sphere(curandState* local_rand_state) {
    vec3 p;
    do {
        p = 2.0f * RANDVEC3 - vec3(1.0f, 1.0f, 1.0f);
    } while (p.squared_length() >= 1.0f);
    return p;
}

__device__ inline vec3 random_in_unit_disk(curandState* local_rand_state) {
    vec3 p;
    do {
        p = 2.0f * RANDVEC3 - vec3(1.0f, 1.0f, 0.0f);
    } while (p.squared_length() >= 1.0f);
    return p;
}

__device__ inline float random_float(curandState* local_rand_state) {
    return curand_uniform(local_rand_state);
}

__device__ inline float random_float(float min, float max, curandState* local_rand_state) {
    return min + max * random_float(local_rand_state);
}

