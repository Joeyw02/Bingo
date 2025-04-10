#pragma once
#include <cstring>
using namespace std;
#include "type.h"
#define EPS (1e-3)
#include "utils.cuh"
#define BLKSZ 512
#define THDSZ 128
#define P 0.5
#define Q 2
const int MAXLOG = 16;
const int LEN = 80;
const int ORIGINAL = 16;
const int CPUTHD = 16;
const int GPUS = 1;
const int GPUTOT = 4;
const int SAMPLINGSIZE = 100000;
const bool BUFFER = 0;
const bool DETAIL = 0;
const float TP = 0.0125; // termination probability
const float IP = 1. / P;
const float IQ = 1. / Q;
const wtype tp = wtype::Int;
int PIECESIZE = 1;
float RATE = 0.4;
bool NODE2VEC = 0;
bool OUTPUT = 0;
int BATCHSIZE = 100000;
inline float bias(int d) { return d; }
inline float bias2(int d) { return pow(d, 0.3); }
// inline float biasN(int d){return (rand()&1023)+1;}
// inline float biasG(int d){return fabs(generateGaussianNoise(512, 512))+1;}
// inline float biasP(int d){return (((long long)generatePowerLaw(2,1))&8191)+1;}
