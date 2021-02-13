#include "Main.hpp"

#include "Worker.hpp"

#include <array>
#include <random>
#include <chrono>
#include <iostream>

// Parameters

#define CONCURRENT
#define NUM_POINTS 1000000
#define NUM_BUCKETS 16384
#define BUCKET_SIZE 0.5f

// Math setup

#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <glm/glm.hpp>
#include <glm/vec3.hpp>
using glm::vec3;

std::default_random_engine generator;
std::uniform_real_distribution<float> distribution(0.0f, 1000.0f);
inline float next_rand() { return distribution(generator); }

// Fibonacci Hashing
// https://probablydance.com/2018/06/16/

inline uint32_t calc_bucket_shift(const uint32_t bucket_count)
{
    return 32 - static_cast<uint32_t>(log2(bucket_count));
}

const uint32_t bucket_shift = calc_bucket_shift(NUM_BUCKETS);

inline uint32_t hash_to_index(const uint32_t hash)
{
    const uint32_t hash2 = hash ^ (hash >> bucket_shift);
    return (2654435769u * hash2) >> bucket_shift;
}

const vec3 bucket_offsets[8] = {
    vec3(0, 0, 0),
    vec3(1, 0, 0),
    vec3(0, 1, 0),
    vec3(1, 1, 0),
    vec3(0, 0, 1),
    vec3(1, 0, 1),
    vec3(0, 1, 1),
    vec3(1, 1, 1)
};

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

inline uint32_t generate_bucket_id(const vec3 pos)
{
    const vec3 p = (pos + bounds) / BUCKET_SIZE;
    const uint32_t x = static_cast<uint32_t>(p.x);
    const uint32_t y = static_cast<uint32_t>(p.y);
    const uint32_t z = static_cast<uint32_t>(p.z);
    return hash_to_index(prime_1 * x ^ prime_2 * y ^ prime_3 * z);
}

inline uint32_t generate_bucket_id(const vec3 pos, const vec3 offset)
{
    const vec3 q = vec3(1024.0, 1024.0, 1024.0);
    const vec3 p0 = (pos + bounds) / BUCKET_SIZE;

    const vec3 p1 = p0 + vec3(
        fract2(p0.x) < 0.5 ? -1 : 0,
        fract2(p0.y) < 0.5 ? -1 : 0,
        fract2(p0.z) < 0.5 ? -1 : 0);

    const vec3 p2 = p1 + offset;
    const uint32_t x = static_cast<uint32_t>(p2.x);
    const uint32_t y = static_cast<uint32_t>(p2.y);
    const uint32_t z = static_cast<uint32_t>(p2.z);
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

struct Point
{
    vec3 position;
    uint32_t bucket_id = 0;
    bool found_nearest = false;
    uint32_t nearest_index = 0;
};

std::array<Point, NUM_POINTS> point_cloud_input;
std::array<Point, NUM_POINTS> point_cloud_sorted;
std::array<Point, NUM_POINTS> point_cloud_final;

void NNApproxSearch(uint32_t start, uint32_t step);

// Sorting buckets

std::array<uint32_t, NUM_POINTS>  buckets_id;
std::array<uint32_t, NUM_BUCKETS> buckets_hash;
std::array<uint32_t, NUM_BUCKETS> buckets_boundary;

int main(int argc, char* argv[])
{
    // Create a random point cloud
    for (auto& point : point_cloud_input)
    {
        const vec3 position = vec3(next_rand(), next_rand(), next_rand());
        const uint32_t bucket_id = generate_bucket_id(position);

        point =
        {
            position,
            bucket_id
        };
    }

    // Create thread workers if using concurrency
#ifdef CONCURRENT
    int threads = std::thread::hardware_concurrency();
    WorkerGroup search_workers;

    for (int n = 0; n < threads; ++n)
    {
        search_workers.AddWorker(std::make_unique<Worker>([=]
        {
            NNApproxSearch(n, threads);
        }));
    }
#endif

    timer_start();

    // Sort points by buckets using O(n) sort.
    std::fill(buckets_hash.begin(), buckets_hash.end(), 0);

    for (auto& p : point_cloud_input)
    {
        buckets_hash[p.bucket_id]++;
    }

    for (uint32_t i = 1; i < NUM_BUCKETS; i++)
    {
        buckets_hash[i] += buckets_hash[i - 1];
    }

    for (auto& p : point_cloud_input)
    {
        buckets_hash[p.bucket_id] -= 1;
        point_cloud_sorted[buckets_hash[p.bucket_id]] = p;
    }

    // Calculate boundaries between buckets_ids of sorted points.
    uint32_t current = NUM_BUCKETS + 1;
    std::fill(buckets_boundary.begin(), buckets_boundary.end(), -1);

    for (uint32_t i = 0; i < NUM_POINTS; i++)
    {
        Point& p = point_cloud_sorted[i];
        if (p.bucket_id > current || current == NUM_BUCKETS + 1)
        {
            buckets_boundary[p.bucket_id] = i;
            current = p.bucket_id;
        }
    }

    // Points are now sorted by hash and we have a map of where groups of ids are,
    // now run search.

#ifdef CONCURRENT
    search_workers.Resolve();
#else
    NNApproxSearch();
#endif

    timer_end();

    // O(n) search for the closest that we found
    float nearest_found_dist = std::numeric_limits<float>::max();
    uint32_t nearest_found_index_0 = 0;
    uint32_t nearest_found_index_1 = 0;

    for (uint32_t i = 0; i < NUM_POINTS; i++)
    {
        Point& p0 = point_cloud_final[i];
        if (p0.found_nearest)
        {
            Point& p1 = point_cloud_final[p0.nearest_index];

            float dist = glm::length(p1.position - p0.position);
            if (dist < nearest_found_dist)
            {
                nearest_found_dist = dist;
                nearest_found_index_0 = i;
                nearest_found_index_1 = p0.nearest_index;
            }
        }
    }

    std::cout << "Nearest found points: ";
    std::cout << "#" << nearest_found_index_0;
    std::cout << ", ";
    std::cout << "#" << nearest_found_index_1;
    std::cout << " distance:";
    std::cout << nearest_found_dist;
    std::cout << " of " << NUM_POINTS;
    std::cout << std::endl;
}

void NNApproxSearch(uint32_t start = 0, uint32_t step = 1)
{
    // For each point
    for (uint32_t i = start; i < NUM_POINTS; i += step)
    {
        const Point& b0 = point_cloud_sorted[i];

        // search neighbor buckets
        float nearest_distance = std::numeric_limits<float>::max();
        bool nearest_found = false;
        uint32_t nearest_index = 0;

        // Iterate nearby buckets
        for (uint32_t j = 0; j < 8; j++)
        {
            const uint32_t bucket_index = generate_bucket_id(
                b0.position,
                bucket_offsets[j]);

            int32_t k = buckets_boundary[bucket_index];

            // Nothing in this bucket
            if (k == -1)
            {
                continue;
            }

            int32_t current_bucket_id = 0;

            do
            {
                const Point& b1 = point_cloud_sorted[k];
                current_bucket_id = b1.bucket_id;

                if (i != k && current_bucket_id == bucket_index)
                {
                    const float d = glm::length(b1.position - b0.position);
                    if (d < nearest_distance)
                    {
                        nearest_distance = d;
                        nearest_index = k;
                        nearest_found = true;
                    }
                }

                k++;
            }
            while (current_bucket_id == bucket_index && k < NUM_POINTS);
        }

        point_cloud_final[i] =
        {
            b0.position,
            b0.bucket_id,
            nearest_found,
            nearest_index
        };
    }
}
