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
int n;
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
    for(int i=0;i<edgeData.size();++i)n=max(n,max(edgeData[i].u,edgeData[i].v));    //    initTrans(n);
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
        if(tp==wtype::Int)edges[i]=(Edges){u,v,(float)d[v]};
        else edges[i]=(Edges){u,v,(float)(d[v]+r.rd(100)*.01)};
    }
    build(edges,amount,nd,ndD,d,&sizeManager,n);
    delete[] edges;
}
void insertGraph(){//recordOld(n,d,edgeData);
    Edges *edges=new Edges[BATCHSIZE];
    int lastAmount=edgeData.size();
    for(int i=0;i<BATCHSIZE;++i){
        int pos=r.rd(lastAmount);
        int u=edgeData[pos].u,v=edgeData[pos].v;
        edgeData.push_back((EdgeData){u,v,d[u]++});
        if(tp==wtype::Int)edges[i]=(Edges){u,v,(float)d[v]};
        else edges[i]=(Edges){u,v,(float)(d[v]+.01)};
    }
    insert(edges,BATCHSIZE,nd,ndD,d,sizeManager);
    delete[] edges;
}
void deleteGraph(){
    Deleted *edges=new Deleted[BATCHSIZE];
    for(int i=0;i<BATCHSIZE;++i){
        int pos=r.rd(edgeData.size());
        while(edgeData[pos].nodeIdu==-1)pos=r.rd(edgeData.size());
        int u=edgeData[pos].u,idu=edgeData[pos].nodeIdu;
        edges[i]=(Deleted){u,idu};
        edgeData[pos].nodeIdu=-1;
    }//recordNew(n,d,edgeData);
    deleteE(edges,BATCHSIZE,ndD);
    delete[] edges;
}
void randomWalk(app a){
    ll len=LEN*(n/GPUS+1);
    if(a==app::sampling)len=BATCHSIZE;
    Timer tt;
    //float time=0;;
    tt.restart();
    #pragma omp parallel for
    for(int g=0;g<GPUS;++g){
        cudaSetDevice((g+1)%GPUTOT);
        //if(BUFFER)resetKernel<<<BLKSZ,THDSZ>>>(n,ndD[g]);
        if(rwD[g]==NULL)cudaMalloc((void **)&rwD[g],len*sizeof(int));
        HE(cudaGetLastError());
//        for(int t=0;t<2;++t){
            GPUTimer gT;gT.init();//cerr<<g<<" "<<n<<endl;
            cudaDeviceSynchronize();
            switch (a){
                case app::deepwalk:
                    deepwalkKernel<<<BLKSZ,THDSZ>>>(g,n,ndD[g],rwD[g],((ull)new char)+g*13/*abc,bbc*/);
                    break;
                case app::node2vec:
                    node2vecKernel<<<BLKSZ,THDSZ>>>(g,n,ndD[g],rwD[g],((ull)new char)+g*13);
                    break;
                case app::ppr:
                    pprKernel<<<BLKSZ,THDSZ>>>(g,n,ndD[g],rwD[g],((ull)new char)+g*13);
                    break;
                case app::sampling:
                    samplingKernel<<<BLKSZ,THDSZ>>>(g,n,ndD[g],rwD[g],((ull)new char)+g+13);
                    break;
                default:
                    deepwalkKernel<<<BLKSZ,THDSZ>>>(g,n,ndD[g],rwD[g],((ull)new char)+g*13);
                    break;
            }
            cudaDeviceSynchronize();
            HE(cudaGetLastError());
          //  time=max(time,gT.finish());
       // }
    }
    totalTime+=tt.duration();
    if(OUTPUT){
        //freopen("","w",stdout);
        int *rw=new int[LEN*10];
        #pragma omp parallel for
        for(int g=0;g<GPUS;++g){
            cudaSetDevice((g+1)%GPUTOT);
            int lPos=1+n*g/GPUS;//,rPos=n*(g+1)/GPUS;//(rPos-lPos+1)
            cudaMemcpy(rw+(lPos-1)*LEN,rwD[g],(LEN*10)*sizeof(int),cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
        }
        for(int i=0;i<=10;++i){
            for(int j=1;j<=80;++j)
            if(rw[i*80+j-1]!=0)cout<<rw[i*80+j-1]<<" ";
            else break;
            cout<<endl;
        }
        delete[] rw;
        //fclose(stdout);
    }
}