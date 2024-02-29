#pragma once
#include "ray.h"
#include "random_tools.h"

#define M_PI 3.14159

class camera {
public:
    //The default camera()
    /*__device__ camera() {
        lower_left_corner = vec3(-2.0, -1.0, -1.0);
        horizontal = vec3(4.0, 0.0, 0.0);
        vertical = vec3(0.0, 2.0, 0.0);
        origin = vec3(0.0, 0.0, 0.0);
    }*/

    //self-costumed camera()
    __device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, 
        float vfov, float aspect, float aperture, float focus_dist) {

        lens_radius = aperture / 2.0f;
        
        //将垂直视场角度从度转换为弧度
        float theta = vfov * float(M_PI) / 180.0f;
        //计算视图平面一半的高度。这是通过视场角度的一半的正切来获得的。
        float half_height = tan(theta / 2);
        float half_width = aspect * half_height;
        origin = lookfrom;
        //因为在计算机图形学中，通常摄像机是面向-z轴的
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        /*lower_left_corner = origin - half_width * u - half_height * v - w;
        horizontal = 2 * half_width * u;
        vertical = 2 * half_height * v;*/

        lower_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
        horizontal = 2.0f * half_width * focus_dist * u;
        vertical = 2.0f * half_height * focus_dist * v;
    }
    __device__ ray get_ray(float s, float t, curandState* local_rand_state) {
        vec3 rd = lens_radius * random_in_unit_disk(local_rand_state);
        vec3 offset = u * rd.x() + v * rd.y();

        //origin + offset 模拟光线从相机的某个位置出发
        
        //近视眼模拟
        //vec3 slight_offset = offset * 0.05;
        float ray_time = random_float(local_rand_state);
        return ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset - slight_offset, ray_time);
    }
    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;

    vec3 slight_offset{};
    float lens_radius;
};

