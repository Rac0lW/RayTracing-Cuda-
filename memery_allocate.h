#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "hitable.h"
#include "sphere.h"
#include "material.h"
#include "hitable_list.h"

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hitable** d_list, hitable** d_world, camera** d_camera, int nx, int ny, curandState* rand_state) {
    //Chapter 8
    /*if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_list[0] = new sphere(vec3(0, 0, -1), 0.5,
            new lambertian(vec3(0.8, 0.3, 0.3)));
        d_list[1] = new sphere(vec3(0, -100.5, -1), 100,
            new lambertian(vec3(0.8, 0.8, 0.0)));
        d_list[2] = new sphere(vec3(1, 0, -1), 0.5,
            new metal(vec3(0.8, 0.6, 0.2), 1.0));
        d_list[3] = new sphere(vec3(-1, 0, -1), 0.5,
            new metal(vec3(0.8, 0.8, 0.8), 0.3));
        *d_world = new hitable_list(d_list, 4);
        *d_camera = new camera();
    }*/

    //Chapter 9+
    /*if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_list[0] = new sphere(vec3(0, 0, -1), 0.5,
            new lambertian(vec3(1, 1, 1)));
        d_list[1] = new sphere(vec3(0, -100.5, -1), 100,
            new lambertian(vec3(0.8, 0.8, 0.0)));
        d_list[2] = new sphere(vec3(1, 0, -1), 0.5,
            new metal(vec3(0.8, 0.6, 0.2), 0.0));
        d_list[3] = new sphere(vec3(-1, 0, -1), 0.5,
            new dielectric(1.5));
        d_list[4] = new sphere(vec3(-1, 0, -1), -0.45,
            new dielectric(1.5));
        *d_world = new hitable_list(d_list, 5);
        vec3 lookfrom(3, 3, 2);
        vec3 lookat(0, 0, -1);
        float dist_to_focus = (lookfrom - lookat).length();
        float aperture = 2.0;
        *d_camera = new camera( lookfrom,
                                lookat,
                                vec3(0, 1, 0),
                                20.0,
                                float(nx) / float(ny),
                                aperture,
                                dist_to_focus);

    }*/

    //Chapter Final
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0, -1000.0, -1), 1000,
            new lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a + RND, 0.2, b + RND);
                if (choose_mat < 0.8f) {
                    vec3 center_1 = center + vec3(0, 0.5 * random_float(rand_state), 0);
                    d_list[i++] = new sphere(center, center_1,0.2,
                        new lambertian(vec3(RND * RND, RND * RND, RND * RND)));


                }
                else if (choose_mat < 0.95f) {
                    d_list[i++] = new sphere(center, 0.2,
                        new metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
                }
                else {
                    d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        d_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        *d_world = new hitable_list(d_list, 22 * 22 + 1 + 3);

        vec3 lookfrom(13, 2, 3);
        vec3 lookat(0, 0, 0);
        float dist_to_focus = 10.0; (lookfrom - lookat).length();
        float aperture = 0.1;
        *d_camera = new camera(lookfrom,
            lookat,
            vec3(0, 1, 0),
            30.0,
            float(nx) / float(ny),
            aperture,
            dist_to_focus);
    }
}

__global__ void free_world(hitable** d_list, hitable** d_world, camera** d_camera) {
    for (int i = 0; i < 5; i++) {
        delete ((sphere*)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete* d_world;
    delete* d_camera;
}

