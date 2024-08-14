#pragma once
#define ull unsigned long long
struct Edges{int u,v;unsigned weight;};
inline bool cmpEdges(const Edges &a,const Edges &b){return a.u<b.u;}
struct EdgeData{int u,v,nodeIdu;};
struct Deleted{int u,id;};
inline bool cmpDeleted(const Deleted &a, const Deleted &b){return a.u<b.u;}
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