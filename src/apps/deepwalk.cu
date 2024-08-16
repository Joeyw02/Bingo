#include<iostream>
#include<cstdio>
#include<cstring>
#include<omp.h>
#include"api.h"
#include"apps.h"
#include"gpuMem.cuh"
#include"operation.cuh"
#include"Bingo.cuh"
#include"utils.cuh"
#include"test.cuh"
using namespace std;
CPURand r;
int n,batchSize=100000;
int *d,*sizeManager;
vector<EdgeData>edgeData;
bool buffer=0;
NodeData *nd[GPUS],*ndD[GPUS];
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
double totalTime=0;

int *rwD[GPUS];

void randomWalk(){
    //int *rw=new int[LEN*n];
    Timer tt;//tt.restart();
  //  float time=0;
//    cerr<<tt.duration()<<endl;
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
            randomWalkKernel<<<BLKSZ,THDSZ>>>(g,n,ndD[g],rwD[g],((ull)new char)+g*13/*abc,bbc*/);
            cudaDeviceSynchronize();
            HE(cudaGetLastError());
         
          //  time=max(time,gT.finish());
     //   }
    }
    totalTime+=tt.duration();//time/1000;
  /*   #pragma omp parallel for
    for(int g=0;g<GPUS;++g){
        cudaSetDevice(g);
        int lPos=1+n*g/GPUS,rPos=n*(g+1)/GPUS;//(rPos-lPos+1)
        cudaMemcpy(rw+(lPos-1)*LEN,rwD[g],(LEN*n/4)*sizeof(int),cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }
   for(int i=0;i<=10;++i){
    for(int j=1;j<=80;++j)cout<<rw[i*80+j-1]<<" ";cout<<endl;}


*/

   // delete[] rw;
}
int deepwalk(){
    freopen("../dataset/AM","r",stdin);
    Timer TT;TT.restart();
    omp_set_num_threads(CPUTHD);
    loadGraph();
    randomWalk();
    Timer T;T.restart();
 //   freopen("out.txt","w",stdout);
    for(int i=0;i<10;++i){
        insertGraph();
        deleteGraph();
        randomWalk();
    }
    cerr<<"Evaluation time: "<<T.duration()<<" s."<<endl;
    cerr<<"Random walk in "<<totalTime<<" s."<<endl;
    cerr<<"Total time: "<<TT.duration()<<" s."<<endl;
    return 0;
}