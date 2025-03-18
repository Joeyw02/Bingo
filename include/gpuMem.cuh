#pragma once
<<<<<<< HEAD
#include "api.h"
#define MEMBLOCK 104857600
class MemPool
{
=======
#include"api.h"
#define MEMBLOCK 104857600
class MemPool{
>>>>>>> 981715a704c385ba213895fc06a00fd0792380b6
private:
    char *p[GPUS];
    int point[GPUS];
    void init(int g)
    {
        point[g] = 0;
        cudaMalloc((void **)&p[g], MEMBLOCK);
    }

public:
<<<<<<< HEAD
    void init()
    {
#pragma omp parallel for
        for (int g = 0; g < GPUS; ++g)
        {
            cudaSetDevice((g + 2) % GPUTOT);
=======
    void init(){
        #pragma omp parallel for
        for(int g=0;g<GPUS;++g){
            cudaSetDevice((g+1)%GPUTOT);
>>>>>>> 981715a704c385ba213895fc06a00fd0792380b6
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