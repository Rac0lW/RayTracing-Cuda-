#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <limits>

const float infinity = std::numeric_limits<float>::infinity();

class interval {
public:
    float min, max;

    __device__ interval() : min(+infinity), max(-infinity) {} // Default interval is empty

    __device__ interval(double _min, double _max) : min(_min), max(_max) {}

    __device__ bool contains(double x) const {
        return min <= x && x <= max;
    }

    __device__ bool surrounds(double x) const {
        return min < x && x < max;
    }

    __device__ float clamp(double x) const {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }

    __device__ float size() const {
        return max - min;
    }

    __device__ interval expand(float delta) const {
        float padding = delta / 2;
        return interval(min - padding, max + padding);
    }
    
    static const interval empty, universe;
};

const static interval empty(+infinity, -infinity);
const static interval universe(-infinity, +infinity);

