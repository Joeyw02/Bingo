#pragma once
#include"type.h"
#include"Bingo.cuh"
#include"utils.cuh"
#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;
Timer ttt;
__global__ void buildKernel(int n,int m,NodeData *ndD,Edges *edgesD,int *beginD){
    int tid = blockIdx.x * blockDim.x + threadIdx.x,wid=tid>>5,totalW=(BLKSZ*THDSZ)>>5;
    __shared__ NodeData::TEMP l[(LOGT*THDSZ)>>5],s[(LOGT*THDSZ)>>5];
    for(int i=wid;i<n;i+=totalW){
        int begin=beginD[i],end=(i==n-1)?m:beginD[i+1];
        int u=edgesD[begin].u;
        if(!(tid&31))for(int j=begin;j<end;++j)ndD[u].edgePush(edgesD[j].v,edgesD[j].weight);
        __syncwarp();
        ndD[u].build(l+LOGT*(threadIdx.x>>5),s+LOGT*(threadIdx.x>>5));
    }
}
void requestBuffer(int n, NodeData **nd,NodeData **ndD);
void build(Edges *edges,int amount,NodeData **nd,NodeData **ndD,int *d,int **sizeManager,int n){
    ttt.restart();
    for(int g=0;g<GPUS;++g)nd[g]=new NodeData[n+1];
    (*sizeManager)=new int[n+1];
    #pragma omp parallel for
    for(int i=1;i<=n;++i){
        (*sizeManager)[i]=16;
        while((*sizeManager)[i]<d[i])(*sizeManager)[i]<<=1;
    }
    #pragma omp parallel for
    for(int g=0;g<GPUS;++g){
        cudaSetDevice(g);
        //cudaMalloc((void**)&LD[g],128*sizeof(int));
        //cudaMemcpyAsync(LD[g],L,128*sizeof(int),cudaMemcpyHostToDevice);
        for(int i=1;i<=n;++i)nd[g][i].init((*sizeManager)[i],g);HE(cudaGetLastError());
        cudaMalloc((void**)&ndD[g],(n+1)*sizeof(NodeData));
        cudaMemcpyAsync(ndD[g],nd[g],(n+1)*sizeof(NodeData),cudaMemcpyHostToDevice);
    }
    Edges *edgesD[GPUS];
    vector<Edges> *sortTMP=new vector<Edges>[n+1];
    for(int i=0;i<amount;++i)sortTMP[edges[i].u].push_back(edges[i]);
    amount=0;
    for(int i=1;i<=n;++i)
    for(unsigned j=0;j<sortTMP[i].size();++j)
    edges[amount++]=sortTMP[i][j];
    delete[] sortTMP;
    #pragma omp parallel for
    for(int g=0;g<GPUS;++g){
       cudaSetDevice(g);
       cudaMalloc((void **)&edgesD[g],amount*sizeof(Edges));
       cudaMemcpyAsync(edgesD[g],edges,amount*sizeof(Edges),cudaMemcpyHostToDevice);
    }
    vector<int>tmp;
    for(int i=0;i<amount;++i)if((!i)||edges[i].u!=edges[i-1].u)tmp.push_back(i);
    
    int *beginD[GPUS];float time=0;
    #pragma omp parallel for
    for(int g=0;g<GPUS;++g){
        cudaSetDevice(g);
        cudaMalloc((void **)&beginD[g],tmp.size()*sizeof(int));
        cudaMemcpyAsync(beginD[g],tmp.data(),tmp.size()*sizeof(int),cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        GPUTimer gT;gT.init();
        buildKernel<<<BLKSZ,THDSZ>>>(tmp.size(),amount,ndD[g],edgesD[g],beginD[g]);
        cudaDeviceSynchronize();
        HE(cudaGetLastError());
        time=max(time,gT.finish());
        cudaFree(edgesD[g]),cudaFree(beginD[g]);
    }
   // if(BUFFER)requestBuffer(n,nd,ndD);
    cerr<<"Load graph in "<<time/1000<<" s."<<endl;
}
__global__ void insertKernel(int n,NodeData *ndD,Edges *edgesD,int *beginD){
    int tid = blockIdx.x * blockDim.x + threadIdx.x,wid=tid>>5,totalW=(BLKSZ*THDSZ)>>5;
    if(!tid)ndD[0].SZ=totalW;
    __shared__ NodeData::TEMP l[(LOGT*THDSZ)>>5],s[(LOGT*THDSZ)>>5];
    int pos=wid;
    while(pos<n){
        int begin=beginD[pos],end=beginD[pos+1];
        int u=edgesD[begin].u;//if(!(tid&31))printf("%d %d %d %d %d\n",pos,begin,end,u,tid);
        ndD[u].copy();
        for(int j=begin;j<end;++j)ndD[u].insert(edgesD[j].v,edgesD[j].weight);
        __syncwarp();
        //if(BUFFER)ndD[u].clearBuffer();
        ndD[u].buildAliasTable(l+LOGT*(threadIdx.x>>5),s+LOGT*(threadIdx.x>>5));
      //  pos+=totalW;
        if(!(tid&31))pos=atomicAdd(&(ndD[0].SZ),1);
        pos= __shfl_sync(0xffffffff,pos,0);
    }
}
void insert(Edges *edges,int amount,NodeData **nd,NodeData **ndD,int *d,int *sizeManager){
    ttt.restart();
    stable_sort(edges,edges+amount,cmpEdges);
    Edges *edgesD[GPUS];
    #pragma omp parallel for
    for(int g=0;g<GPUS;++g){
        cudaSetDevice(g);
        cudaMalloc((void **)&edgesD[g],amount*sizeof(Edges));
        cudaMemcpyAsync(edgesD[g],edges,amount*sizeof(Edges),cudaMemcpyHostToDevice);
    }
    vector<int>tmp;
    for(int i=0;i<amount;++i)
    if((!i)||edges[i].u!=edges[i-1].u){
        tmp.push_back(i);
        int u=edges[i].u;
        if(sizeManager[u]>=d[u])continue;
        while(sizeManager[u]<d[u])sizeManager[u]<<=1;
        #pragma omp parallel for
        for(int g=0;g<GPUS;++g){
            cudaSetDevice(g);
            cudaMemcpy(nd[g]+u,ndD[g]+u,sizeof(NodeData),cudaMemcpyDeviceToHost);
            nd[g][u].reMalloc(sizeManager[u],g);
            cudaMemcpyAsync(ndD[g]+u,nd[g]+u,sizeof(NodeData),cudaMemcpyHostToDevice);
        }
    }
    tmp.push_back(amount);
    //cerr<<amount<<" "<<ttt.duration()<<endl;
    int *beginD[GPUS];
   // float time=0;
   ttt.restart();
    #pragma omp parallel for
    for(int g=0;g<GPUS;++g){
        cudaSetDevice(g);
        cudaMalloc((void **)&beginD[g],tmp.size()*sizeof(int));
        cudaMemcpy(beginD[g],tmp.data(),tmp.size()*sizeof(int),cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        //GPUTimer gT;gT.init();
    	insertKernel<<<BLKSZ,THDSZ>>>(tmp.size()-1,ndD[g],edgesD[g],beginD[g]);
        cudaDeviceSynchronize();
        HE(cudaGetLastError());
        //time=max(gT.finish(),time);
        cudaFree(edgesD[g]),cudaFree(beginD[g]);
    }//time/1000
    cerr<<"insert edges in "<<ttt.duration()<<" s."<<endl;
}
__global__ void deleteKernel(int n,NodeData *ndD,Deleted *edgesD,int *beginD){
    int tid = blockIdx.x * blockDim.x + threadIdx.x,wid=tid>>5,totalW=(BLKSZ*THDSZ)>>5;
    __shared__ NodeData::TEMP l[(LOGT*THDSZ)>>5],s[(LOGT*THDSZ)>>5];
    for(int i=wid;i<n;i+=totalW){
        int begin=beginD[i],end=beginD[i+1];
        int u=edgesD[begin].u;
        for(int j=begin;j<end;++j)ndD[u].deleteE(edgesD[j].id);
        __syncwarp();
        for(int j=begin+(tid&31);j<end;j+=32)ndD[u].edge[edgesD[j].id].weight=0;
        //if(BUFFER)ndD[u].clearBuffer();
        ndD[u].buildAliasTable(l+LOGT*(threadIdx.x>>5),s+LOGT*(threadIdx.x>>5));
    }
}
void deleteE(Deleted *edges,int amount,NodeData **ndD){
    ttt.restart();
    sort(edges,edges+amount,cmpDeleted);
    Deleted *edgesD[GPUS];
    #pragma omp parallel for
    for(int g=0;g<GPUS;++g){
        cudaSetDevice(g);
        cudaMalloc((void **)&edgesD[g],amount*sizeof(Deleted));
        cudaMemcpyAsync(edgesD[g],edges,amount*sizeof(Deleted),cudaMemcpyHostToDevice);
    }
    vector<int>tmp;
    for(int i=0;i<amount;++i)if((!i)||edges[i].u!=edges[i-1].u)tmp.push_back(i);
    tmp.push_back(amount);
    int *beginD[GPUS];
    //float time=0;
    #pragma omp parallel for
    for(int g=0;g<GPUS;++g){
        cudaSetDevice(g);
        cudaMalloc((void **)&beginD[g],tmp.size()*sizeof(int));
        cudaMemcpy(beginD[g],tmp.data(),tmp.size()*sizeof(int),cudaMemcpyHostToDevice);  ttt.restart();
        cudaDeviceSynchronize();
       // GPUTimer gT;gT.init();
        deleteKernel<<<BLKSZ,THDSZ>>>(tmp.size()-1,ndD[g],edgesD[g],beginD[g]);
        cudaDeviceSynchronize();
        HE(cudaGetLastError());
      //  time=max(time,gT.finish());
        cudaFree(edgesD[g]),cudaFree(beginD[g]);
    }//time/1000
    cerr<<"delete edges in "<<ttt.duration()<<" s."<<endl;
}
__global__ void deepwalkKernel(int gId,int n, NodeData *ndD, int *rwD,ull seed){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bg=(n*gId/GPUS)+1;
    ndD[0].edgeSZ=bg;
 //   __shared__ NodeData::AliasElement AliasSM[LOGT*128];
   // for(int i=threadIdx.x/32;i<=128;i+=blockDim.x/32)
   // for(int j=threadIdx.x&31;j<LOGT;j+=32)
   // AliasSM[i*LOGT+j]=ndD[L[i]].aliasTable[j];
    //__syncthreads();
    RandomGenerator rdG;
    rdG.init((19260817^seed)+tid,tid);
    int pos,ed=n*(gId+1)/GPUS;
    if(!(tid&31))pos=atomicAdd(&ndD[0].edgeSZ,32);
    pos= __shfl_sync(0xffffffff,pos,0);
    while(pos<=ed){
        int u=pos+(tid&31);
        if(u<=ed){
            int *rw=rwD+(u-bg)*LEN;
            (*rw)=u;
            for(int j=1;j<LEN&&u!=-1;++j){
                u=ndD[u].sample(rdG);
                (*(++rw))=u;
            }
        }
        __syncwarp();
        if(!(tid&31))pos=atomicAdd(&ndD[0].edgeSZ,32);
        pos= __shfl_sync(0xffffffff,pos,0);
    }
}
__global__ void pprKernel(int gId,int n, NodeData *ndD, int *rwD,ull seed){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bg=(n*gId/GPUS)+1;
    ndD[0].edgeSZ=bg;
    RandomGenerator rdG;
    rdG.init((19260817^seed)+tid,tid);
    int pos,ed=n*(gId+1)/GPUS;
    if(!(tid&31))pos=atomicAdd(&ndD[0].edgeSZ,32);
    pos= __shfl_sync(0xffffffff,pos,0);
    while(pos<=ed){
        int u=pos+(tid&31);
        if(u<=ed){
            int *rw=rwD+(u-bg)*LEN;
            (*rw)=u;
           for(int j=1;u!=-1;++j){
                u=ndD[u].sample(rdG);
                if(j<LEN)++rw;
                (*(rw))=u;
                if(rdG.getRandomNumber()<TP)break;
            }
        }
        __syncwarp();
        if(!(tid&31))pos=atomicAdd(&ndD[0].edgeSZ,32);
        pos= __shfl_sync(0xffffffff,pos,0);
    }
}
__global__ void samplingKernel(int gId,int n, NodeData *ndD, int *rwD,ull seed){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bg=(BATCHSIZE*gId/GPUS)+1;
    ndD[0].edgeSZ=bg;
    RandomGenerator rdG;
    rdG.init((19260817^seed)+tid,tid);
    int pos,ed=BATCHSIZE*(gId+1)/GPUS;
    if(!(tid&31))pos=atomicAdd(&ndD[0].edgeSZ,32);
    pos= __shfl_sync(0xffffffff,pos,0);
    while(pos<=ed){
        int u=rand(rdG,n);
        if(u<=ed){
            int *rw=rwD+(u-bg);
            (*(rw))=ndD[u].sample(rdG);
        }
        __syncwarp();
        if(!(tid&31))pos=atomicAdd(&ndD[0].edgeSZ,32);
        pos= __shfl_sync(0xffffffff,pos,0);
    }
}
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