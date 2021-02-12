#include "Main.hpp"

#include <array>
#include <random>
#include <chrono>
#include <iostream>

// Math setup

#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <glm/vec3.hpp>
using glm::vec3;

std::default_random_engine generator;
std::uniform_real_distribution<float> distribution(0.0f, 100.0f);
inline float next_rand() { return distribution(generator); }

// Fibonacci Hashing
// https://probablydance.com/2018/06/16/

inline uint32_t calc_bucket_shift(const uint32_t bucket_count)
{
    return 32 - static_cast<uint32_t>(log2(bucket_count));
}

const float bucket_size = 0.5f;
const uint32_t bucket_count = 2048;
const uint32_t bucket_shift = calc_bucket_shift(bucket_count);

inline uint32_t hash_to_index(const uint32_t hash)
{
    const uint32_t hash2 = hash ^ (hash >> bucket_shift);
    return (2654435769u * hash2) >> bucket_shift;
}

// Hash functions
// Generate something to fib-hash.
const vec3 bounds = vec3(1024.0, 1024.0, 1024.0);
const uint32_t prime_1 = 73856093u;
const uint32_t prime_2 = 19349663u;
const uint32_t prime_3 = 83492791u;

inline float fract2(const float x)
{
    return x >= 0. ? x - std::floor(x) : x - std::ceil(x);
}

inline uint32_t lsh(const vec3 pos)
{
    const vec3 p = (pos + bounds) / bucket_size;
    const uint32_t x = static_cast<uint32_t>(p.x);
    const uint32_t y = static_cast<uint32_t>(p.y);
    const uint32_t z = static_cast<uint32_t>(p.z);
    return hash_to_index(prime_1 * x ^ prime_2 * y ^ prime_3 * z);
}

inline uint32_t lsh(const vec3 pos, const vec3 offset)
{
    const vec3 q = vec3(1024.0, 1024.0, 1024.0);
    const vec3 p0 = (pos + bounds) / bucket_size;

    const vec3 p1 = p0 + vec3(
        fract2(p0.x) < 0.5 ? -1 : 0,
        fract2(p0.y) < 0.5 ? -1 : 0,
        fract2(p0.z) < 0.5 ? -1 : 0);

    const vec3 p2 = p1 + offset;
    const uint32_t x = static_cast<uint32_t>(pos.x);
    const uint32_t y = static_cast<uint32_t>(pos.y);
    const uint32_t z = static_cast<uint32_t>(pos.z);
    return hash_to_index(prime_1 * x ^ prime_2 * y ^ prime_3 * z);
}

// Timing
std::chrono::high_resolution_clock::time_point timer_start_point;

inline void timer_start()
{
    timer_start_point = std::chrono::high_resolution_clock::now();
}

inline void timer_end()
{
    const auto end = std::chrono::high_resolution_clock::now();
    const auto time_span = std::chrono::duration_cast<std::chrono::duration<float>>(
        end - timer_start_point);
    std::cout << "RESOLVE TIME: " << (time_span.count() * 1000) << "ms" << std::endl;
}

// Point cloud
#define NUM_POINTS 1000000

struct Point
{
    vec3 position;
    uint32_t bucket_id = 0;
    bool found_nearest = false;
    size_t nearest_index = 0;
};

std::array<Point, NUM_POINTS> point_cloud_input;
std::array<Point, NUM_POINTS> point_cloud_output;

int main(int argc, char* argv[])
{
    // Create a random point cloud
    for (auto& point : point_cloud_input)
    {
        const vec3 position = vec3(next_rand(), next_rand(), next_rand());
        const uint32_t bucket_id = lsh(position);

        point = {
            position,
            bucket_id
        };
    }


}
