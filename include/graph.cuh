#pragma once
#include<vector>
#include"api.h"
#include"graph.cuh"
#include"operation.cuh"
#include"Bingo.cuh"
#include"test.cuh"
using namespace std;
NodeData *nd[GPUS],*ndD[GPUS];
int *d,*sizeManager;
int n,batchSize=100000;
vector<EdgeData>edgeData;
CPURand r;
int *rwD[GPUS];
double totalTime=0;
void loadGraph(){    
    r.init();mp.init();
    int u,v;n=0;
    while(scanf("%d%d",&u,&v)!=EOF){
        EdgeData tmp;tmp.u=u+1,tmp.v=v+1;
        edgeData.push_back(tmp);
    }   
    cerr<<"Edge number: "<<edgeData.size()<<"."<<endl;
    for(int i=0;i<edgeData.size();++i)n=max(n,max(edgeData[i].u,edgeData[i].v));
    int amount=edgeData.size();
    Edges *edges=new Edges[amount];
    d=new int[n+1];
    memset(d,0,sizeof(int)*(n+1));
    for(int i=0;i<amount;++i){
        EdgeData *tmp=&edgeData[i];
        tmp->nodeIdu=d[tmp->u]++;
    }
    #pragma omp parallel for
    for(int i=0;i<amount;++i){
        int u=edgeData[i].u,v=edgeData[i].v;
        edges[i]=(Edges){u,v,(unsigned)d[v]};
    }
    build(edges,amount,nd,ndD,d,&sizeManager,n);
    delete[] edges;
}
void insertGraph(){
    Edges *edges=new Edges[batchSize];
    int lastAmount=edgeData.size();
    for(int i=0;i<batchSize;++i){
        int pos=r.rd(lastAmount);
        int u=edgeData[pos].u,v=edgeData[pos].v;
        edgeData.push_back((EdgeData){u,v,d[u]++});
        edges[i]=(Edges){u,v,(unsigned)d[u]};
    }
    insert(edges,batchSize,nd,ndD,d,sizeManager);
    delete[] edges;
}
void deleteGraph(){
    Deleted *edges=new Deleted[batchSize];
    for(int i=0;i<batchSize;++i){
        int pos=r.rd(edgeData.size());
        while(edgeData[pos].nodeIdu==-1)pos=r.rd(edgeData.size());
        int u=edgeData[pos].u,idu=edgeData[pos].nodeIdu;
        edges[i]=(Deleted){u,idu};
        edgeData[pos].nodeIdu=-1;
    }
    deleteE(edges,batchSize,ndD);
    delete[] edges;
}
void randomWalk(app a){
    //int *rw=new int[LEN*n];
    Timer tt;
  //  float time=0;;
    tt.restart();
    #pragma omp parallel for
    for(int g=0;g<GPUS;++g){
        cudaSetDevice(g);
        //if(BUFFER)resetKernel<<<BLKSZ,THDSZ>>>(n,ndD[g]);
        if(rwD[g]==NULL)cudaMalloc((void **)&rwD[g],(LEN*(n/GPUS+1))*sizeof(int));
        HE(cudaGetLastError());
   //     for(int t=0;t<2;++t){
           // GPUTimer gT;gT.init();//cerr<<g<<" "<<n<<endl;
            cudaDeviceSynchronize();
            if(a==app::deepwalk)deepwalkKernel<<<BLKSZ,THDSZ>>>(g,n,ndD[g],rwD[g],((ull)new char)+g*13/*abc,bbc*/);
            else pprKernel<<<BLKSZ,THDSZ>>>(g,n,ndD[g],rwD[g],((ull)new char)+g*13);
            cudaDeviceSynchronize();
            HE(cudaGetLastError());
          //  time=max(time,gT.finish());
     //   }
    }
    totalTime+=tt.duration();
     #pragma omp parallel for
    for(int g=0;g<GPUS;++g){
        cudaSetDevice(g);
        //int lPos=1+n*g/GPUS,rPos=n*(g+1)/GPUS;//(rPos-lPos+1)
       // cudaMemcpy(rw+(lPos-1)*LEN,rwD[g],(LEN*n/4)*sizeof(int),cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }
    /*
   for(int i=0;i<=10;++i){
    for(int j=1;j<=80;++j)cout<<rw[i*80+j-1]<<" ";cout<<endl;}
    */
   // delete[] rw;
}