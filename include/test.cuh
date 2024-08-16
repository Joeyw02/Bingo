#pragma once
#include<cstdio>
#include"type.h"
#include"Bingo.cuh"
using namespace std;
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
void countGraph(int n,NodeData *ndD){
    int a[4]={0,0,0,0};//total one sparse dense
    int *aD;cudaMalloc((void**)&aD,4*sizeof(int));
    double *bD;cudaMalloc((void**)&bD,4*sizeof(double));
    double *cD;cudaMalloc((void**)&cD,4*sizeof(double));
    double *dD;cudaMalloc((void**)&dD,4*sizeof(double));
    double *eD;cudaMalloc((void**)&eD,4*sizeof(double));
    int *log2;cudaMalloc((void**)&log2,5000000*sizeof(int));
    countGraph<<<1,32>>>(n,ndD,aD,bD,cD,dD,eD,log2);
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
