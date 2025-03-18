#pragma once
#include <iostream>
#include <chrono>
#include <curand_kernel.h>
#include <cstdlib>
#include "type.h"
#include "api.h"
#include <stdlib.h>
#include <math.h>
using namespace std;
#define HE(err) (handleError(err, __FILE__, __LINE__))
inline __device__ void swAp(unsigned a, unsigned b) { a ^= b ^= a ^= b; }

class Timer
{
    std::chrono::time_point<std::chrono::system_clock> _start = std::chrono::system_clock::now();

public:
    void restart() { _start = std::chrono::system_clock::now(); }
    double duration()
    {
        std::chrono::duration<double> diff = std::chrono::system_clock::now() - _start;
        return diff.count();
    }
    static double current_time()
    {
        std::chrono::duration<double> val = std::chrono::system_clock::now().time_since_epoch();
        return val.count();
    }
};
class GPUTimer
{
private:
    cudaEvent_t start, end;

public:
    void init()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start);
    }
    float finish()
    {
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        float tmp;
        cudaEventElapsedTime(&tmp, start, end);
        cudaEventDestroy(start);
        cudaEventDestroy(end);
        return tmp;
    }
};

struct CPURand
{
    void init() { srand((unsigned long long)new char); }
    int rd(int x) { return (((1ll * rand()) << 15) + rand()) % x; }
};
__device__ static inline uint64_t rotl(const uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}
__device__ inline float next(uint64_t *s)
{
    const uint64_t result = rotl(s[0] + s[3], 23) + s[0];
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0], s[3] ^= s[1], s[1] ^= s[2], s[0] ^= s[3];
    s[2] ^= t, s[3] = rotl(s[3], 45);
    return (result & 65535) / 65535.;
}
__device__ float rand(uint64_t *s, int len) { return next(s) * len - EPS; }
static void handleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        cerr << cudaGetErrorString(err) << " in " << file << " at line " << line << ".\n";
        exit(EXIT_FAILURE);
    }
}
double generateGaussianNoise(double mu, double sigma)
{
    static int hasSpare = 0;
    static double spare;
    if (hasSpare)
    {
        hasSpare = 0;
        return mu + sigma * spare;
    }
    hasSpare = 1;
    double u1, u2;
    u1 = rand() / (RAND_MAX + 1.0);
    u2 = rand() / (RAND_MAX + 1.0);
    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * M_PI * u2;
    spare = r * sin(theta);
    return mu + sigma * (r * cos(theta));
}
double generatePowerLaw(double alpha, double xmin)
{
    double u = rand() / (RAND_MAX + 1.0);
    return xmin * pow((1 - u), -1.0 / (alpha - 1));
}