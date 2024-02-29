#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>
#include <curand_kernel.h>
#include <time.h>
#include "vec3.h"
#include "color.h"
#include "ray.h"
#include "hitable_list.h"
#include "camera.h"

#include "error_check.h"
#include "memery_allocate.h"
#include "hitable.h"
#include "hitable_list.h"
#include "sphere.h"
#include "render.h"
#include "error_check.h"
#include "random_tools.h"



int main() {
    int nx = 1200;
    int ny = 600;
    int ns = 100;
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << nx << "x" << ny << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);

    vec3* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));
    curandState* d_rand_state2;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state2, 1 * sizeof(curandState)));

    // we need that 2nd random state to be initialized for the world creation
    rand_init << <1, 1 >> > (d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    hitable** d_list;
    int num_hitables = 22 * 22 + 1 + 3;
    checkCudaErrors(cudaMalloc((void**)&d_list, num_hitables * sizeof(hitable*)));//TODO:

    hitable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable*)));
    camera** d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));

    create_world << <1, 1 >> > (d_list, d_world, d_camera, nx, ny, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);

    render_init << <blocks, threads >> > (nx, ny,d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render << <blocks, threads >> > (fb, nx, ny,ns,d_camera,d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    // Output FB as Image
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * nx + i;
            int ir = int(255.99 * fb[pixel_index].r());
            int ig = int(255.99 * fb[pixel_index].g());
            int ib = int(255.99 * fb[pixel_index].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());

    free_world << <1, 1 >> > (d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(fb));

    // useful for cuda-memcheck --leak-check full
    cudaDeviceReset();
}