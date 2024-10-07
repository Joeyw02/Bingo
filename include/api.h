#pragma once
#include"type.h"
#define LOGT 4
#define EPS (1e-4)
#define BLKSZ 512
#define THDSZ 128
#define P 0.5
#define Q 2
const int BEG=LOGT*0.5;
const int BITS=LOGT-BEG;
const int LEN=80;
const int BATCHSIZE=100000;
const int CPUTHD=16;
const int GPUS=1;
const int GPUTOT=4;
const bool BUFFER=0;
const float TP=0.0125;//termination probability
const float IP=1./P;
const float IQ=1./Q;
const wtype tp=wtype::Int;
bool NODE2VEC=0;
bool OUTPUT=1;