#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>
#include <curand_kernel.h>

#include "vec3.h"
#include "color.h"
#include "ray.h"
#include "camera.h"
#include "memery_allocate.h"



__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;

    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3* fb, int max_x, int max_y, int ns,
    camera** cam,
    hitable** world,
    curandState* rand_state
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0, 0, 0);

    for (int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;

    col /= float(ns);
    col.gamma_correction();
    fb[pixel_index] = col;
}