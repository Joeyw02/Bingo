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
    int amount=edgeData.size();cerr<<n<<endl;
    //batchSize=amount*ChangeR;
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

__global__ void countGraph(int n,NodeData *ndD,int *a,double *b,double *c,double *d,double *e,int *log2){
    if(threadIdx.x!=0)return;
    a[0]=a[1]=a[2]=a[3]=0;
    log2[0]=0;
    log2[1]=1;
    for(int i=2;i<5000000;++i){log2[i]=log2[i>>1]<<1;if(log2[i]<i)log2[i]<<=1;}
    float b11=0,b12=0;
    float b31=0,b32=0;
    for(int i=1;i<=n;++i){
        int x=4;
        if(ndD[i].edgeSZ<256)x=1;
        else if(ndD[i].edgeSZ<65536)x=2;
        for(int j=0;j<LOGT;++j){
            if(ndD[i].num[j]==0)continue;if((100ll*i/n)!=(100ll*(i-1)/n)&&j==0)printf("%d %d %d\n",i,j,n);
            ++a[0];
            if(ndD[i].num[j]==1){
                ++a[1];b11+=1;b12+=1+ndD[i].edgeSZ;
                c[1]+=4+1;
                d[1]+=x+1;
                e[1]+=(log2[ndD[i].num[j]]+log2[ndD[i].edgeSZ])*4;
            }
            else if(ndD[i].num[j]*5<ndD[i].edgeSZ){
                ++a[2];b31+=ndD[i].num[j]*2+1;b32+=ndD[i].num[j]+ndD[i].edgeSZ;
                c[2]+=(log2[ndD[i].num[j]]+log2[ndD[i].num[j]+1])*4+1;
                d[2]+=(log2[ndD[i].num[j]]+log2[ndD[i].num[j]+1])*x+1;
                e[2]+=(log2[ndD[i].num[j]]+log2[ndD[i].edgeSZ])*4;
            }
            else if(ndD[i].num[j]>ndD[i].edgeSZ*0.5){
                ++a[3];
                c[3]+=1;
                d[3]+=1;
                e[3]+=(log2[ndD[i].num[j]]+log2[ndD[i].edgeSZ])*4;
            }
            else{
                c[0]+=(log2[ndD[i].num[j]]+log2[ndD[i].edgeSZ])*4+1;
                d[0]+=(log2[ndD[i].num[j]]+log2[ndD[i].edgeSZ])*x+1;
                e[0]+=(log2[ndD[i].num[j]]+log2[ndD[i].edgeSZ])*4;
            }
        }
    }
    b[1]=b11/b12;
    b[2]=b31/b32;
    for(int i=0;i<4;++i)(c[i]/=1048576)/=1024,(d[i]/=1048576)/=1024,(e[i]/=1048576)/=1024;
    
}
void countGraph(){
    int a[4]={0,0,0,0};//total one sparse dense
    int *aD;cudaMalloc((void**)&aD,4*sizeof(int));
    double *bD;cudaMalloc((void**)&bD,4*sizeof(double));
    double *cD;cudaMalloc((void**)&cD,4*sizeof(double));
    double *dD;cudaMalloc((void**)&dD,4*sizeof(double));
    double *eD;cudaMalloc((void**)&eD,4*sizeof(double));
    int *log2;cudaMalloc((void**)&log2,5000000*sizeof(int));
    countGraph<<<1,32>>>(n,ndD[0],aD,bD,cD,dD,eD,log2);
    cudaDeviceSynchronize();
    HE(cudaGetLastError());
    cudaMemcpy(a,aD,4*sizeof(int),cudaMemcpyDeviceToHost);
    double b[4],c[4],d[4],e[4];
    cudaMemcpy(b,bD,4*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(c,cD,4*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(d,dD,4*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(e,eD,4*sizeof(double),cudaMemcpyDeviceToHost);
    cout<<a[0]<<" "<<a[1]<<" "<<a[2]<<" "<<a[3]<<endl;
    cout<<"one"<<" "<<"sparse "<<"dense"<<endl;
    cout<<1.*a[1]/a[0]<<" "<<1.*a[2]/a[0]<<" "<<1.*a[3]/a[0]<<endl;
    cout<<b[1]<<" "<<b[2]<<" "<<b[3]<<endl;
    cout<<c[0]<<" "<<c[1]<<" "<<c[2]<<" "<<c[3]<<" "<<c[0]+c[1]+c[2]+c[3]<<endl;
    cout<<d[0]<<" "<<d[1]<<" "<<d[2]<<" "<<d[3]<<" "<<d[0]+d[1]+d[2]+d[3]<<endl;
    cout<<e[0]<<" "<<e[1]<<" "<<e[2]<<" "<<e[3]<<" "<<e[0]+e[1]+e[2]+e[3]<<endl;
}

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
            //randomWalkKernel<<<BLKSZ,THDSZ>>>(1,n,ndD[g],rwD[g],((ull)new char)+g*13/*abc,bbc*/);
            //cudaDeviceSynchronize();
            //HE(cudaGetLastError());
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

    long long a=0,mx=0;
    for(int iii=1;iii<=1000000;++iii){
        int pos=r.rd(edgeData.size());
        //while(edgeData[pos].u<=n/2)pos=r.rd(edgeData.size());
        int u=edgeData[pos].u;
        long long sum=0;
        for(int i=0;i<LEN*n/10;++i)
        if((rw[i])==u)++sum;
        a+=sum;mx=max(mx,sum);
        if(iii%10==0)cout<<1.*a/iii<<" "<<mx<<endl;
    }
*/
    //int *xxx=new int[n];
   // memset(xxx,0,sizeof(int)*n);
//    for(int i=0;i<n*80;++i)if(rw[i]!=-1&&rw[i]!=0)++xxx[rw[i]];
    //sort(xxx,xxx+n);
   // for(int i=1;i<=3000;++i)cerr<<xxx[n-i]<<" "<<100.*xxx[n-i]/(80*n)<<"% "<<endl;
    //for(int i=1;i<=10;++i){
      //  for(int j=1;j<=LEN;++j)printf("%d ",rw[(i-1)*LEN+j-1]);
        //puts("");
   // }
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