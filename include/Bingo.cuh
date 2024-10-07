#pragma once
#include"api.h"
#include"type.h"
#include"utils.cuh"
#include"gpuMem.cuh"
struct NodeData{
    int oldSZ,SZ;  float sum=0;//,LP;
    struct AliasElement {float p;char a,b;}__attribute__((packed));
    int aliasTableSZ=0;AliasElement *aliasTable;
    struct TEMP{float p; char id;}__attribute__((packed));
    struct Edge{int v;float weight;}mx;
    int edgeSZ=0;Edge *edge,*oldEdge;
    __device__ inline void edgePush(int v,float w){edge[edgeSZ++]=(Edge){v,w};}
    int *group,*idx,*groupSZ,*oldGroup,*oldIdx;
    __device__ inline void groupPush(int i,int id){group[i*SZ+(groupSZ[i]++)]=id;}
    __device__ inline void idxSet(int i,int id,int pos){idx[i*SZ+id]=pos;}
    __device__ inline int groupGet(int i,int pos){return group[i*SZ+pos];}
    __device__ inline int groupBack(int i){return group[i*SZ+groupSZ[i]-1];}
    __device__ inline int idxGet(int i,int id){return idx[i*SZ+id];}
    float *num;
    struct HASH{
       int *a,*s,SZ;
        void init(int SZ,int g){
            this->SZ=SZ;
            mp.apply(reinterpret_cast<void**>(&a),SZ*sizeof(int),g);
            mp.apply(reinterpret_cast<void**>(&s),SZ*sizeof(int),g);
        }
        __device__ bool check(ull v){
            ((v<<=2)^=19260817)%=SZ;
            return 0;
        }
    }hs;
    void init(int SZ,int g){
        oldSZ=-1;
        this->SZ=SZ;
        mx.weight=0;
        //this->LP=LP;
        mp.apply(reinterpret_cast<void**>(&aliasTable),(LOGT+2)*sizeof(AliasElement),g);
        if(tp==wtype::Float)mp.apply(reinterpret_cast<void**>(&num),(LOGT+2)*sizeof(float),g);
        else mp.apply(reinterpret_cast<void**>(&num),LOGT*sizeof(float),g);
        mp.apply(reinterpret_cast<void**>(&groupSZ),BITS*sizeof(int),g);
        mp.apply(reinterpret_cast<void**>(&group),SZ*BITS*sizeof(int),g);
        mp.apply(reinterpret_cast<void**>(&idx),SZ*BITS*sizeof(int),g);
        mp.apply(reinterpret_cast<void**>(&edge),SZ*sizeof(Edge),g);
       // initBuffer(g);
        if(NODE2VEC)hs.init(SZ,g);
    }
    void reMalloc(int SZ,int g){
        oldSZ=this->SZ;this->SZ=SZ;
        oldGroup=group,oldIdx=idx;
        oldEdge=edge;
        mp.apply(reinterpret_cast<void**>(&group),SZ*BITS*sizeof(int),g);
        mp.apply(reinterpret_cast<void**>(&idx),SZ*BITS*sizeof(int),g);
        mp.apply(reinterpret_cast<void**>(&edge),SZ*sizeof(Edge),g);
    }
    __device__ void copy(){
        if(oldSZ==-1)return;
        int tid=threadIdx.x&31;
        for(int i=tid;i<oldSZ;i+=32)edge[i]=oldEdge[i];
        for(int i=0;i<BITS;++i)
        for(int j=tid;j<oldSZ;j+=32){
            group[i*SZ+j]=oldGroup[i*oldSZ+j];
            idx[i*SZ+j]=oldIdx[i*oldSZ+j];
        }
        __syncwarp();
        oldSZ=-1;
    }
    __device__ void buildAliasTable(TEMP *l,TEMP *s){
        int tid=threadIdx.x&31;
        if(tid)return;
        aliasTableSZ=0;
        int lSZ=0,sSZ=0;
        int nonZ=0; double sum=0;
        for(int i=0;i<LOGT+(tp==wtype::Float);++i)if(num[i]>EPS)++nonZ,sum+=num[i];
        if(sum<EPS)return;
        float pFactor=1.*nonZ/sum;
        for(int i=0;i<LOGT+(tp==wtype::Float);++i){
            float numi=num[i];
            if(numi<EPS)continue;
            TEMP tmp;tmp.p=numi*pFactor,tmp.id=i;
            if(tmp.p+EPS>=1)l[++lSZ]=tmp;
            else s[++sSZ]=tmp;
        }
        while(lSZ){
            TEMP a = l[lSZ--];
            AliasElement tmp;
            while(a.p + EPS >= 1){
                if (sSZ) { 
                    TEMP b = s[sSZ--];
                    tmp.p = b.p, tmp.a = b.id, tmp.b = a.id;
                    a.p += b.p;
                }
                else tmp.p = 1, tmp.a = a.id, tmp.b = -1;
                aliasTable[aliasTableSZ++]=tmp;
                a.p -= 1;
            }
            if (a.p > 0)s[++sSZ]=a;
        }
    }
    __device__ void build(TEMP *l, TEMP *s) {
        int tid = threadIdx.x&31;
        for(int i=tid;i<LOGT+(tp==wtype::Float);i+=32)num[i]=0;
        for(int i=tid;i<BITS;i+=32)groupSZ[i]=0;
        for(int i=0;i<edgeSZ;++i){
            unsigned w=edge[i].weight;
            for(int j=tid;j<LOGT;j+=32)
            if(w&(1ull<<j)){
                num[j]+=(1ull<<j);
                if(j>=BEG)idxSet(j-BEG,i,groupSZ[j-BEG]),groupPush(j-BEG,i);
            }
            if(tid==31&&tp==wtype::Float)num[LOGT]+=edge[i].weight-w;
        }
        __syncwarp();
        buildAliasTable(l,s);
    }
    __device__ void insert(int v,float weight){
        int tid=threadIdx.x&31;
        unsigned w=weight;
        for (int i = tid; i < LOGT;i+=32)
        if (w & (1ull<<i)) {
            num[i]+=(1ull<<i);
            if(i>=BEG)idxSet(i-BEG,edgeSZ,groupSZ[i-BEG]),groupPush(i-BEG,edgeSZ);
        }
        if(tid==31&&tp==wtype::Float)num[LOGT]+=weight-w;
        if(((tid+w)&31))return;
        edgePush(v,w);
    }
    __device__ void deleteE(int id){
        int tid=threadIdx.x&31;
        float weight=edge[id].weight;
        unsigned w=weight;
        for (int i = tid; i < LOGT; i+=32) 
        if (w & (1ull<<i)) {
            num[i]-=(1ull<<i);
            if(i<BEG)continue;
            int backid=groupBack(i-BEG),pos=idxGet(i-BEG,id);
            int *a=&group[(i-BEG)*SZ+pos],*b=&group[(i-BEG)*SZ+groupSZ[i-BEG]-1];
            (*a)^=(*b)^=(*a)^=(*b);
            idxSet(i-BEG,backid,pos);
            --groupSZ[i-BEG];
        }
        if(tid==31&&tp==wtype::Float)num[LOGT]-=(weight-w);
    }
    __device__ int sample(RandomGenerator &rdG){
        if(!aliasTableSZ)return -1;
        if(edgeSZ==1)return edge[0].v;
        int ans=-1;
       // if(BUFFER)ans=fetchingBuffer();
        //if(ans!=-1)return ans;
        float p = rand(rdG,aliasTableSZ);
        int pos = p;
        AliasElement tmp=aliasTable[pos];
        int i=((p-pos)>tmp.p?tmp.b:tmp.a);
        if(i>=BEG&&(!(tp==wtype::Float&&i==LOGT))){
            int res=rand(rdG,groupSZ[i-BEG]);
            if(res==groupSZ[i-BEG])--res;
            ans=edge[groupGet(i-BEG,res)].v;
        }
        else if(i<BEG){
            int id=rand(rdG,edgeSZ);id=rand(rdG,edgeSZ);
//            while(!(((unsigned)edge[id].weight)&(1ull<<i))){id=rand(rdG,edgeSZ);if(id>=edgeSZ)--id;}
            ans=edge[id].v;
        }
        else{
            int id=rand(rdG,edgeSZ);float rej=rand(rdG,1);
            while(((edge[id].weight-((unsigned)edge[id].weight)))>rej){id=rand(rdG,edgeSZ);rej=rand(rdG,1);if(id>=edgeSZ)--id;}
            ans=edge[id].v;
        }
       // if(BUFFER)addingBuffer(ans);
        return ans;
    }
    //Buffer
    /*
        bool emptyBuffer=1,fullBuffer=1;
        int peBuffer,pfBuffer,*bBuffer,BUFFERSZ=0;
        void initBuffer(int g){
            emptyBuffer=1;fullBuffer=peBuffer=pfBuffer=0;
            mp.apply(reinterpret_cast<void**>(&bBuffer),BUFFERSZ*sizeof(int),g);
        }
        __device__ void clearBuffer(){
            fullBuffer=pfBuffer=0;
        }
        __device__ void resetBuffer(){
            emptyBuffer=((bool)(peBuffer=pfBuffer))^1;
        }
        __device__ int fetchingBuffer(){
           if(emptyBuffer)return -1;
           int x=atomicAdd(&peBuffer,-1)-1;
           if(x<0){emptyBuffer=1;return -1;}
           return bBuffer[x];
        }
        __device__ void addingBuffer(int a){
            if(fullBuffer)return;
            int x=atomicAdd(&pfBuffer,1);
            if(x>=BUFFERSZ){fullBuffer=1;pfBuffer=BUFFERSZ;return;}
            bBuffer[x]=a;
        }
        */
};