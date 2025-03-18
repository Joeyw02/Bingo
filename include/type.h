#pragma once
#define ll long long
#define ull unsigned long long
enum class app
{
    deepwalk,
    node2vec,
    ppr,
    sampling
};
enum class wtype
{
    Int,
    Float
};
enum class utype
{
    I,
    D,
    M
};
struct Edges
{
    int u, v;
    float weight;
};
inline bool cmpEdges(const Edges &a, const Edges &b) { return a.u < b.u; }
struct EdgeData
{
    int u, v, nodeIdu;
};
struct Deleted
{
    int u, id;
};
inline bool cmpDeleted(const Deleted &a, const Deleted &b) { return a.u < b.u; }