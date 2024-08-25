#pragma once
#define ull unsigned long long
enum class app{
    deepwalk,
    ppr
};
struct Edges{int u,v;unsigned weight;};
inline bool cmpEdges(const Edges &a,const Edges &b){return a.u<b.u;}
struct EdgeData{int u,v,nodeIdu;};
struct Deleted{int u,id;};
inline bool cmpDeleted(const Deleted &a, const Deleted &b){return a.u<b.u;}