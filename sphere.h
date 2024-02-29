
#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "hitable.h"



class sphere : public hitable {
public:
    __device__ sphere() {}
    // Stationary Sphere
    __device__ sphere(vec3 cen, float r, material* m) : center(cen), radius(r), mat_ptr(m), is_moving(false) {};

    // Moving Sphere
    __device__ sphere(vec3 cen, vec3 cen_1, float r, material* m) : center(cen), radius(r), mat_ptr(m), is_moving(true) {
        center_vec = cen_1 - cen;
    };

    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;


    vec3 center;
    vec3 center_vec;
    float radius;
    material* mat_ptr;
    bool is_moving;

    //The neet week
    __device__ vec3 t_center(float time) const {
        return center + time * center_vec;
    }
};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 ccenter = is_moving ? t_center(r.time()) : center;


    vec3 oc = r.origin() - ccenter;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - a * c;


    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - ccenter) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - ccenter) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
    }
    return false;
}


