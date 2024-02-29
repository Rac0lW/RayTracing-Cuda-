#pragma once
#include "random_tools.h"
#include "ray.h"
#include "hitable.h"
struct hit_record;


__device__ vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2.0f * dot(v, n) * n;
}


//cosine是光线击中表面时，光线方向和表面法线之间角度的余弦值。
//ref_idx是折射率（Refractive Index）的比值，通常是从一个介质进入另一个介质时的折射率之比，例如从空气进入水。

__device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    //r0​的平方操作是Schlick近似方法中的一个技巧，用于在保持计算效率的同时，尽可能地近似真实的物理现象
    r0 = r0 * r0;
    //其中pow((1 - cosine), 5)是一个权重因子,用于模拟光线以接近平行于表面的角度入射时反射率的增加
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

__device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    //这个值用于判断是否会发生全内反射。如果判别式大于0，说明没有全内反射，光线会发生折射；如果判别式小于等于0，则发生全内反射。
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
    if (discriminant > 0) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    }
    else
        return false;
}

class material {
public:
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const = 0;
};

class lambertian : public material {
public:
    __device__ lambertian(const vec3& a) : albedo(a) {}
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const {
        vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
        scattered = ray(rec.p, target - rec.p, r_in.time());
        attenuation = albedo;
        return true;
    }

    vec3 albedo;
};

class metal : public material {
public:
    __device__ metal(const vec3& a, float f) : albedo(a) { if (f < 1) fuzz = f; else fuzz = 1; }
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state), r_in.time());
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0.0f);
    }
    vec3 albedo;
    float fuzz;
};



//TODO;Ready to learn
class dielectric : public material {
public:
    __device__ dielectric(float ri) : ref_idx(ri) {}
    __device__ virtual bool scatter(const ray& r_in,
        const hit_record& rec,
        vec3& attenuation,
        ray& scattered,
        curandState* local_rand_state) const {

        vec3 outward_normal;
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        float ni_over_nt;
        attenuation = vec3(1.0, 1.0, 1.0);
        vec3 refracted;
        float reflect_prob;
        float cosine;
        if (dot(r_in.direction(), rec.normal) > 0.0f) {
            outward_normal = -rec.normal;
            ni_over_nt = ref_idx;
            cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
            cosine = sqrt(1.0f - ref_idx * ref_idx * (1 - cosine * cosine));
        }else {
            outward_normal = rec.normal;
            ni_over_nt = 1.0f / ref_idx;
            cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
        }

        if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
            reflect_prob = schlick(cosine, ref_idx);
        else
            reflect_prob = 1.0f;

        if (curand_uniform(local_rand_state) < reflect_prob)
            scattered = ray(rec.p, reflected, r_in.time());
        else
            scattered = ray(rec.p, refracted, r_in.time());

        return true;
    }

    float ref_idx;
};

//class diffuse_light : public material {
//public:
//    __device__ diffuse_light(const vec3& a) : emit(a) {}
//
//    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const override {
//        return false; // 光发射材质不散射光线
//    }
//
//    __device__ virtual vec3 emitted(float u, float v, const vec3& p) const {
//        return emit; // 返回材质发射的光线颜色
//    }
//
//    vec3 emit;
//};
