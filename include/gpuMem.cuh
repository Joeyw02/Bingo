#pragma once
#include "api.h"
#define MEMBLOCK 104857600
class MemPool
{
private:
    char *p[GPUS];
    int point[GPUS];
    void init(int g)
    {
        point[g] = 0;
        cudaMalloc((void **)&p[g], MEMBLOCK);
    }

public:
    void init()
    {
#pragma omp parallel for
        for (int g = 0; g < GPUS; ++g)
        {
            cudaSetDevice((g + 2) % GPUTOT);
            init(g);
        }
    }
    void apply(void **a, int sz, int g)
    {
        if (point[g] + sz > MEMBLOCK)
            init(g);
        (*a) = p[g] + point[g];
        point[g] = (((point[g] + sz + 3) >> 2) << 2);
    }
} mp;