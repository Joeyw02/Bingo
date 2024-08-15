#pragma once
#include<iostream>
#include<chrono>
#include<curand_kernel.h>
#include<cstdlib>
using namespace std;
#define HE(err) (handleError(err, __FILE__, __LINE__))
inline __device__ void swAp(unsigned a,unsigned b){a^=b^=a^=b;}
class Timer{
    std::chrono::time_point<std::chrono::system_clock> _start = std::chrono::system_clock::now();
public:
    void restart(){_start = std::chrono::system_clock::now();}
    double duration(){
        std::chrono::duration<double> diff = std::chrono::system_clock::now()-_start;
        return diff.count();
    }
    static double current_time(){
        std::chrono::duration<double> val = std::chrono::system_clock::now().time_since_epoch();
        return val.count();
    }
};
class GPUTimer{
private:
    cudaEvent_t start,end;
public:
    void init(){
        cudaEventCreate(&start);
	    cudaEventCreate(&end);
	    cudaEventRecord(start);
    }
    float finish(){
        cudaEventRecord(end);
	    cudaEventSynchronize(end);
        float tmp;
        cudaEventElapsedTime(&tmp,start,end);
        cudaEventDestroy(start);
        cudaEventDestroy(end);
        return tmp;
    }
};
struct RandomGenerator {
    curandState state;
    __device__ void init(ull seed,int tid) {
        curand_init(((seed*(tid+1))^tid), tid, 0, &state);
    }
    __device__ float getRandomNumber() {
        return curand_uniform(&state);
    }
};
static void handleError(cudaError_t err,const char *file,int line){
    if (err!=cudaSuccess){
        cerr<<cudaGetErrorString(err)<<" in "<<file<<" at line "<<line<<".\n";
        exit(EXIT_FAILURE);
    }
}
struct CPURand{
    void init(){srand((unsigned long long)new char);}
    int rd(int x){return (((1ll*rand())<<15)+rand())%x;}
};