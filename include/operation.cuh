#pragma once
#include "type.h"
#include "Bingo.cuh"
#include "utils.cuh"
#include "api.h"
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
Timer timer;
double xxx = 0;
__global__ void buildKernel(int n, int m, NodeData *ndD, Edges *edgesD, int *beginD, bool NODE2VEC)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x, wid = tid >> 5, totalW = (BLKSZ * THDSZ) >> 5;
    __shared__ NodeData::TEMP l[(MAXLOG * THDSZ) >> 5], s[(MAXLOG * THDSZ) >> 5];
    for (int i = wid; i < n; i += totalW)
    {
        int begin = beginD[i], end = (i == n - 1) ? m : beginD[i + 1];
        int u = edgesD[begin].u;
        if (!(tid & 31))
            for (int j = begin; j < end; ++j)
            {
                ndD[u].edgePush(edgesD[j].v, edgesD[j].weight);
                if (NODE2VEC)
                    ndD[u].hs.insert(edgesD[j].v);
            }
        __syncwarp();
        ndD[u].build(l + MAXLOG * (threadIdx.x >> 5), s + MAXLOG * (threadIdx.x >> 5));
    }
}
void requestBuffer(int n, NodeData **nd, NodeData **ndD);
int *groupNum[MAXLOG];
void build(Edges *edges, int amount, NodeData **nd, NodeData **ndD, int *d, int **sizeManager, int n)
{
    timer.restart();
    for (int g = 0; g < GPUS; ++g)
        nd[g] = new NodeData[n + 1];
    (*sizeManager) = new int[n + 1];
#pragma omp parallel for
    for (int i = 1; i <= n; ++i)
    {
        (*sizeManager)[i] = ORIGINAL;
        while ((*sizeManager)[i] < d[i])
            (*sizeManager)[i] <<= 1;
    }
    char *MXLOG = new char[n + 1];
    memset(MXLOG, 0, sizeof(MXLOG));
    int *LOG2 = new int[1 << 25 | 1];
    LOG2[0] = 0;
    for (int i = 1; i < (1 << 25); ++i)
        LOG2[i] = LOG2[i >> 1] + 1;
    for (int i = 0; i < amount; ++i)
    {
        int u = edges[i].u;
        MXLOG[u] = max(MXLOG[u], LOG2[(int)edges[i].weight]);
    }
    for (int i = 0; i < MAXLOG; ++i)
        groupNum[i] = new int[n + 1], memset(groupNum[i], 0, sizeof(groupNum[i]));
#pragma omp parallel for
    for (int j = 0; j < MAXLOG; ++j)
        for (int i = 0; i < amount; ++i)
            groupNum[j][edges[i].u] += ((((int)edges[i].weight) & (1 << j)) != 0);
#pragma omp parallel for
    for (int g = 0; g < GPUS; ++g)
    {
        cudaSetDevice((g + 2) % GPUTOT);
        for (int i = 1; i <= n; ++i)
        {
            char tp[MAXLOG] = {0};
            int bits = 0;
            for (int j = 0; j < MAXLOG; ++j)
                tp[j] = ((groupNum[j][i] == 0 || groupNum[j][i] >= (d[i] * RATE)) ? MAXLOG : bits++);
            nd[g][i].init((*sizeManager)[i], g, MXLOG[i], tp, bits);
        }
        HE(cudaGetLastError());
        cudaMalloc((void **)&ndD[g], (n + 1) * sizeof(NodeData));
        cudaMemcpyAsync(ndD[g], nd[g], (n + 1) * sizeof(NodeData), cudaMemcpyHostToDevice);
    }
    delete[] MXLOG;
    delete[] LOG2;
    // sort edges
    Edges *edgesD[GPUS];
    vector<Edges> *sortTMP = new vector<Edges>[n + 1];
    for (int i = 0; i < amount; ++i)
        sortTMP[edges[i].u].push_back(edges[i]);
    amount = 0;
    for (int i = 1; i <= n; ++i)
        for (unsigned j = 0; j < sortTMP[i].size(); ++j)
            edges[amount++] = sortTMP[i][j];
    delete[] sortTMP;
#pragma omp parallel for
    for (int g = 0; g < GPUS; ++g)
    {
        cudaSetDevice((g + 2) % GPUTOT);
        cudaMalloc((void **)&edgesD[g], amount * sizeof(Edges));
        cudaMemcpyAsync(edgesD[g], edges, amount * sizeof(Edges), cudaMemcpyHostToDevice);
    }
    vector<int> tmp;
    for (int i = 0; i < amount; ++i)
        if ((!i) || edges[i].u != edges[i - 1].u)
            tmp.push_back(i);
    int *beginD[GPUS];
    float time = 0;
#pragma omp parallel for
    for (int g = 0; g < GPUS; ++g)
    {
        cudaSetDevice((g + 2) % GPUTOT);
        cudaMalloc((void **)&beginD[g], tmp.size() * sizeof(int));
        cudaMemcpyAsync(beginD[g], tmp.data(), tmp.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        GPUTimer gT;
        gT.init();
        buildKernel<<<BLKSZ, THDSZ>>>(tmp.size(), amount, ndD[g], edgesD[g], beginD[g], NODE2VEC);
        cudaDeviceSynchronize();
        HE(cudaGetLastError());
        time = max(time, gT.finish());
        cudaFree(edgesD[g]), cudaFree(beginD[g]);
    } // if(BUFFER)requestBuffer(n,nd,ndD);
    if (DETAIL)
        cerr << "Load graph in " << time / 1000 << " s." << endl;
}
__global__ void insertKernel(int n, NodeData *ndD, Edges *edgesD, int *beginD)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x, wid = tid >> 5, totalW = (BLKSZ * THDSZ) >> 5;
    if (!tid)
        ndD[0].SZ = totalW;
    __shared__ NodeData::TEMP l[(MAXLOG * THDSZ) >> 5], s[(MAXLOG * THDSZ) >> 5];
    int pos = wid;
    while (pos < n)
    {
        int begin = beginD[pos], end = beginD[pos + 1];
        int u = edgesD[begin].u;
        ndD[u].copy();
        for (int j = begin; j < end; ++j)
            ndD[u].insert(edgesD[j].v, edgesD[j].weight);
        __syncwarp();
        // if(BUFFER)ndD[u].clearBuffer();
        ndD[u].buildAliasTable(l + MAXLOG * (threadIdx.x >> 5), s + MAXLOG * (threadIdx.x >> 5));
        if (!(tid & 31))
            pos = atomicAdd(&(ndD[0].SZ), 1);
        pos = __shfl_sync(0xffffffff, pos, 0);
    }
}
void insert(Edges *edges, int amount, NodeData **nd, NodeData **ndD, int *d, int *sizeManager)
{
    timer.restart();
    stable_sort(edges, edges + amount, cmpEdges);
    Edges *edgesD[GPUS];
#pragma omp parallel for
    for (int g = 0; g < GPUS; ++g)
    {
        cudaSetDevice((g + 2) % GPUTOT);
        cudaMalloc((void **)&edgesD[g], amount * sizeof(Edges));
        cudaMemcpyAsync(edgesD[g], edges, amount * sizeof(Edges), cudaMemcpyHostToDevice);
    }
    vector<int> tmp;
    timer.restart();
    for (int i = 0; i < amount; ++i)
        if ((!i) || edges[i].u != edges[i - 1].u)
        {
            tmp.push_back(i);
            int u = edges[i].u;
            if (sizeManager[u] >= d[u])
                continue;
            while (sizeManager[u] < d[u])
                sizeManager[u] <<= 1;
#pragma omp parallel for
            for (int g = 0; g < GPUS; ++g)
            {
                cudaSetDevice((g + 2) % GPUTOT);
                cudaMemcpy(nd[g] + u, ndD[g] + u, sizeof(NodeData), cudaMemcpyDeviceToHost);
                nd[g][u].reMalloc(sizeManager[u], g);
                cudaMemcpyAsync(ndD[g] + u, nd[g] + u, sizeof(NodeData), cudaMemcpyHostToDevice);
            }
        }
    xxx += timer.duration();
    tmp.push_back(amount);
    // cerr<<amount<<" "<<ttt.duration()<<endl;
    int *beginD[GPUS];
    // float time=0;
#pragma omp parallel for
    for (int g = 0; g < GPUS; ++g)
    {
        cudaSetDevice((g + 2) % GPUTOT);
        cudaMalloc((void **)&beginD[g], tmp.size() * sizeof(int));
        cudaMemcpy(beginD[g], tmp.data(), tmp.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        // GPUTimer gT;gT.init();
        insertKernel<<<BLKSZ, THDSZ>>>(tmp.size() - 1, ndD[g], edgesD[g], beginD[g]); // timer.restart();
        cudaDeviceSynchronize();
        HE(cudaGetLastError());
        // time=max(gT.finish(),time);
        cudaFree(edgesD[g]), cudaFree(beginD[g]);
    } // time/1000
    if (DETAIL)
        cerr << "insert edges in " << timer.duration() << " s." << endl;
}
__global__ void deleteKernel(int n, NodeData *ndD, Deleted *edgesD, int *beginD)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x, wid = tid >> 5, totalW = (BLKSZ * THDSZ) >> 5;
    __shared__ NodeData::TEMP l[(MAXLOG * THDSZ) >> 5], s[(MAXLOG * THDSZ) >> 5];
    for (int i = wid; i < n; i += totalW)
    {
        int begin = beginD[i], end = beginD[i + 1];
        int u = edgesD[begin].u;
        for (int j = begin; j < end; ++j)
            ndD[u].deleteE(edgesD[j].id);
        __syncwarp();
        for (int j = begin + (tid & 31); j < end; j += 32)
            ndD[u].edge[edgesD[j].id].weight = 0;
        // if(BUFFER)ndD[u].clearBuffer();
        ndD[u].buildAliasTable(l + MAXLOG * (threadIdx.x >> 5), s + MAXLOG * (threadIdx.x >> 5));
    }
}
void deleteE(Deleted *edges, int amount, NodeData **ndD)
{
    timer.restart();
    sort(edges, edges + amount, cmpDeleted);
    Deleted *edgesD[GPUS];
#pragma omp parallel for
    for (int g = 0; g < GPUS; ++g)
    {
        cudaSetDevice((g + 2) % GPUTOT);
        cudaMalloc((void **)&edgesD[g], amount * sizeof(Deleted));
        cudaMemcpyAsync(edgesD[g], edges, amount * sizeof(Deleted), cudaMemcpyHostToDevice);
    }
    vector<int> tmp;
    for (int i = 0; i < amount; ++i)
        if ((!i) || edges[i].u != edges[i - 1].u)
            tmp.push_back(i);
    tmp.push_back(amount);
    int *beginD[GPUS];
// float time=0;
#pragma omp parallel for
    for (int g = 0; g < GPUS; ++g)
    {
        cudaSetDevice((g + 2) % GPUTOT);
        cudaMalloc((void **)&beginD[g], tmp.size() * sizeof(int));
        cudaMemcpy(beginD[g], tmp.data(), tmp.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        timer.restart();
        // GPUTimer gT;gT.init();
        deleteKernel<<<BLKSZ, THDSZ>>>(tmp.size() - 1, ndD[g], edgesD[g], beginD[g]);
        cudaDeviceSynchronize();
        HE(cudaGetLastError());
        //  time=max(time,gT.finish());
        cudaFree(edgesD[g]), cudaFree(beginD[g]);
    } // time/1000
    if (DETAIL)
        cerr << "delete edges in " << timer.duration() << " s." << endl;
}
__global__ void deepwalkKernel(int gId, int pId, int n, NodeData *ndD, int *rwD, int PIECESIZE, ull seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bg = (n * gId / GPUS) / PIECESIZE + 1;
    ndD[0].edgeSZ = bg;
    uint64_t s[4] = {(unsigned long)tid, seed, (unsigned long)tid & 7, 127};
    int pos, ed = n * (gId + 1) / GPUS / PIECESIZE;
    if (!(tid & 31))
        pos = atomicAdd(&ndD[0].edgeSZ, 32);
    pos = __shfl_sync(0xffffffff, pos, 0);
    while (pos <= ed)
    {
        int u = pos + (tid & 31);
        if (u <= ed)
        {
            int *rw = rwD + (u - bg) * LEN;
            (*rw) = u;
            for (int j = 1; j < LEN && u != -1; ++j)
            {
                u = ndD[u].sample(s);
                (*(++rw)) = u;
            }
        }
        __syncwarp();
        if (!(tid & 31))
            pos = atomicAdd(&ndD[0].edgeSZ, 32);
        pos = __shfl_sync(0xffffffff, pos, 0);
    }
}
__global__ void node2vecKernel(int gId, int pId, int n, NodeData *ndD, int *rwD, int PIECESIZE, ull seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bg = (n * gId / GPUS) / PIECESIZE + 1;
    ndD[0].edgeSZ = bg;
    uint64_t s[4] = {(unsigned long)tid, seed, (unsigned long)tid & 7, 127};
    int pos, ed = n * (gId + 1) / PIECESIZE / GPUS;
    if (!(tid & 31))
        pos = atomicAdd(&ndD[0].edgeSZ, 32);
    pos = __shfl_sync(0xffffffff, pos, 0);
    float lim = max(IP, IQ);
    while (pos <= ed)
    {
        int u = pos + (tid & 31);
        if (u <= ed)
        {
            int *rw = rwD + (u - bg) * LEN, w = -1;
            (*rw) = u;
            for (int j = 1; j < LEN && u != -1; ++j)
            {
                int tmp;
                while (1)
                {
                    tmp = ndD[u].sample(s);
                    if (tmp == -1)
                        break;
                    if (tmp == w)
                    {
                        if (rand(s, lim) > IP)
                            continue;
                        break;
                    }
                    else if (ndD[u].hs.check(w))
                    {
                        if (rand(s, lim) > 1)
                            continue;
                        break;
                    }
                    else
                    {
                        if (rand(s, lim) > IQ)
                            continue;
                        break;
                    }
                }
                w = u;
                (*(++rw)) = u = tmp;
            }
        }
        __syncwarp();
        if (!(tid & 31))
            pos = atomicAdd(&ndD[0].edgeSZ, 32);
        pos = __shfl_sync(0xffffffff, pos, 0);
    }
}
__global__ void pprKernel(int gId, int pId, int n, NodeData *ndD, int *rwD, int PIECESIZE, ull seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bg = (n * gId / GPUS) / PIECESIZE + 1;
    ndD[0].edgeSZ = bg;
    uint64_t s[4] = {(unsigned long)tid, seed, (unsigned long)tid & 7, 127};
    int pos, ed = n * (gId + 1) / PIECESIZE / GPUS;
    if (!(tid & 31))
        pos = atomicAdd(&ndD[0].edgeSZ, 32);
    pos = __shfl_sync(0xffffffff, pos, 0);
    while (pos <= ed)
    {
        int u = pos + (tid & 31);
        if (u <= ed)
        {
            int *rw = rwD + (u - bg) * LEN;
            (*rw) = u;
            for (int j = 1; u != -1; ++j)
            {
                u = ndD[u].sample(s);
                if (j < LEN)
                    ++rw;
                (*(rw)) = u;
                if (next(s) < TP)
                    break;
            }
        }
        __syncwarp();
        if (!(tid & 31))
            pos = atomicAdd(&ndD[0].edgeSZ, 32);
        pos = __shfl_sync(0xffffffff, pos, 0);
    }
}
/*
// two step
#define KKK 5
__global__ void samplingKernel(int gId,int n, NodeData *ndD, int *rwD,ull seed){
    int tid = blockIdx.x * blockDim.x + threadIdx.x,tot=BLKSZ*THDSZ;
    int bg=(BATCHSIZE*gId/GPUS)+1,ed=BATCHSIZE*(gId+1)/GPUS;
    __shared__ int id[THDSZ*2*KKK];
    __shared__ short d[THDSZ*2*KKK];
    __shared__ int pr,pd;
    short tmp;
    uint64_t s[4]={tid,tid&7,122,321};
    for(int iii=bg+tid*KKK;iii<=ed;iii+=tot*KKK){
        if(threadIdx.x==0)pr=0,pd=0;
        for(int j=0;j<KKK;++j){
            int u=iii+j,pos;
            char tp=ndD[u].sample_step_one(s,pos);
            if(tp==1)tmp=atomicAdd(&pr,1);
            else if(tp==0)tmp=atomicAdd(&pd,1);
            if(tp==2)continue;
            id[tmp+THDSZ*KKK*tp]=u;
            d[tmp+THDSZ*KKK*tp]=pos;
        }
        __syncthreads();
        int nr=pr,nd=pd;
        //for(int i=threadIdx.x;i<nr;i+=THDSZ)id[nd+i]=id[i+THDSZ*2],d[nd+i]=d[i+THDSZ*2];
        for(int i=threadIdx.x;i<nd;i+=THDSZ){
            int u=id[i],*rw=rwD+(u-bg);
            int pos=d[i];
            (*(rw))=ndD[u].sample_step_two(s,pos);
        }
        for(int i=THDSZ-1-threadIdx.x;i<nr;i+=THDSZ){
            int u=id[i+THDSZ*KKK],*rw=rwD+(u-bg);
            int pos=d[i+THDSZ*KKK];
            (*(rw))=ndD[u].sample_step_two(s,pos);
        }
        __syncthreads();
    }
}*/
__global__ void samplingKernel(int gId, int pId, int n, NodeData *ndD, int *rwD, int PIECESIZE, ull seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bg = (SAMPLINGSIZE * gId / GPUS / PIECESIZE) + 1;
    uint64_t s[4] = {(unsigned long)tid, seed, (unsigned long)tid & 7, 127};
    int pos, ed = SAMPLINGSIZE * (gId + 1) / GPUS;
    if (!(tid & 31))
        pos = atomicAdd(&ndD[0].edgeSZ, 32);
    pos = __shfl_sync(0xffffffff, pos, 0);
    while (pos <= ed)
    {
        int u = pos + (tid & 31);
        int i = pos + (tid & 31);
        if (i <= ed)
        {
            int *rw = rwD + (i - bg);
            (*(rw)) = ndD[u].sample(s);
        }
        __syncwarp();
        if (!(tid & 31))
            pos = atomicAdd(&ndD[0].edgeSZ, 32);
        pos = __shfl_sync(0xffffffff, pos, 0);
    }
} /*
 __global__ void naive_samplingKernel(int gId,int n, NodeData *ndD, int *rwD,ull seed){
     int tid = blockIdx.x * blockDim.x + threadIdx.x,tot=BLKSZ*THDSZ;
     int bg=(BATCHSIZE*gId/GPUS)+1,ed=BATCHSIZE*(gId+1)/GPUS;
     //RandomGenerator rdG;
     //rdG.init((19260817^seed)+tid,tid);
     uint64_t s[4]={tid,123,tid&7,233};
     for(int i=bg+tid;i<ed;i+=tot){
         int *rw=rwD+(i-bg);
         (*(rw))=ndD[i].sample(s);
     }
 }*/
/*
__global__ void step_one_samplingKernel(int gId,int n, NodeData *ndD, int *rwD,ull seed,int *id,short *d){
 int tid = blockIdx.x * blockDim.x + threadIdx.x;
 int bg=(BATCHSIZE*gId/GPUS)+1;
 if(!tid)ndD[0].edgeSZ=bg,ndD[0].oldSZ=ndD[0].SZ=0;
 RandomGenerator rdG;
 rdG.init((19260817^seed)+tid,tid);
 int pos,ed=BATCHSIZE*(gId+1)/GPUS;
 if(!(tid&31))pos=atomicAdd(&ndD[0].edgeSZ,32);
 pos= __shfl_sync(0xffffffff,pos,0);
 while(pos<=ed){
     int u=pos+(tid&31);
     if(u<=ed){
         int p;short tmp;
         char tp=ndD[u].sample_step_one(rdG,p);
         if(tp==2){
             int *rw=rwD+(u-bg);
             (*(rw))=p;
         }
         else{
             if(tp==1)tmp=atomicAdd(&ndD[0].oldSZ,1);
             else if(tp==0)tmp=atomicAdd(&ndD[0].SZ,1);
             id[tmp+BATCHSIZE*tp]=u;
             d[tmp+BATCHSIZE*tp]=p;
         }
     }
     __syncwarp();
     if(!(tid&31))pos=atomicAdd(&ndD[0].edgeSZ,32);
     pos= __shfl_sync(0xffffffff,pos,0);
 }
}
__global__ void step_two_samplingKernel(int gId,int n,NodeData *ndD,int *rwD,ull seed,int *id,short *d){
 int tid = blockIdx.x * blockDim.x + threadIdx.x;
 if(!tid)ndD[0].edgeSZ=ndD[0].aliasTableSZ=0;
 int nr=ndD[0].oldSZ,nd=ndD[0].SZ;
 int bg=(BATCHSIZE*gId/GPUS)+1;
 int p1,p2;
 if(!(tid&31))p1=atomicAdd(&ndD[0].edgeSZ,32);
 p1= __shfl_sync(0xffffffff,p1,0);
 RandomGenerator rdG;
 rdG.init((19260817^seed)+tid,tid);
 while(p1<=nr){
     int i=p1+(tid&31);
     if(i<=nr){
         int u=id[BATCHSIZE+i],*rw=rwD+(u-bg);
         (*(rw))=ndD[u].sample_step_two(rdG,d[BATCHSIZE+i]);
     }
     __syncwarp();
     if(!(tid&31))p1=atomicAdd(&ndD[0].edgeSZ,32);
     p1= __shfl_sync(0xffffffff,p1,0);
 }
 if(!(tid&31))p2=atomicAdd(&ndD[0].aliasTableSZ,32);
 p2= __shfl_sync(0xffffffff,p2,0);
 while(p2<=nd){
     int i=p2+(tid&31);
     if(i<=nd){
         int u=id[i],*rw=rwD+(u-bg);
         (*(rw))=ndD[u].sample_step_two(rdG,d[i]);
     }
     __syncwarp();
     if(!(tid&31))p2=atomicAdd(&ndD[0].aliasTableSZ,32);
     p2= __shfl_sync(0xffffffff,p2,0);
 }
}*/
/*
__global__ void resetKernel(int n,NodeData *ndD){
    int tid = blockIdx.x * blockDim.x + threadIdx.x,totalT=blockDim.x*gridDim.x;
    for(int i=tid+1;i<=n;i+=totalT)ndD[i].resetBuffer();
}
void requestBuffer(int n, NodeData **nd,NodeData **ndD){
    int *rw=new int[LEN*n/2];
    int *rwD[GPUS];
    #pragma omp parallel for
    for(int g=0;g<GPUS;++g){
        cudaSetDevice(g);
        cudaMalloc((void **)&rwD[g],(LEN*(n/GPUS+1))*sizeof(int));
        randomWalkKernel<<<BLKSZ,THDSZ>>>(g,n,ndD[g],rwD[g],((ull)new char)+g*13);
        cudaDeviceSynchronize();
        cudaMemcpy(nd[g],ndD[g],(n+1)*sizeof(NodeData),cudaMemcpyDeviceToHost);
        HE(cudaGetLastError());
        for(int t=0;t<2;++t){
            int lPos=1+n*t/GPUS/2,rPos=n*(t+1)/GPUS/2;
            cudaMemcpy(rw,rwD[g]+(lPos-1)*LEN,(LEN*(rPos-lPos+1))*sizeof(int),cudaMemcpyDeviceToHost);
            for(int i=0;i<LEN*n/2;++i)if(rw[i]!=-1&&rw[i]!=0)++nd[g][rw[i]].BUFFERSZ;
        }
    }
    #pragma omp parallel for
    for(int g=0;g<GPUS;++g){
        cudaSetDevice(g);
        for(int i=1;i<=n;++i)nd[g][i].initBuffer(g);
        cudaMemcpy(ndD[g],nd[g],(n+1)*sizeof(NodeData),cudaMemcpyHostToDevice);
    }
    delete[] rw;
}*/