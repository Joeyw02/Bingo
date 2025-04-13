#pragma once
#include <vector>
#include "api.h"
#include "graph.cuh"
#include "operation.cuh"
#include "Bingo.cuh"
#include "test.cuh"
using namespace std;
NodeData *nd[GPUS], *ndD[GPUS];
int *d, *sizeManager;
int n;
vector<EdgeData> edgeData;
CPURand r;
int *rwD[GPUS];
double totalTime = 0;
void loadGraph()
{
    r.init();
    mp.init();
    int u, v;
    n = 0;
    while (scanf("%d%d", &u, &v) != EOF)
    {
        EdgeData tmp;
        tmp.u = u + 1, tmp.v = v + 1;
        edgeData.push_back(tmp);
    }
    cerr << "Edge number: " << edgeData.size() << "." << endl;
    for (int i = 0; i < edgeData.size(); ++i)
        n = max(n, max(edgeData[i].u, edgeData[i].v)); // initTrans(n);
    bool L = (n > 1e7);
    if (L)
        RATE = 0.1, PIECESIZE = 2;
    int amount = edgeData.size();
    Edges *edges = new Edges[amount];
    d = new int[n + 1]; // int mx=0;
    memset(d, 0, sizeof(int) * (n + 1));
    for (int i = 0; i < amount; ++i)
    {
        EdgeData *tmp = &edgeData[i];
        tmp->nodeIdu = d[tmp->u]++; // mx=max(mx,d[tmp->u]);
    }
#pragma omp parallel for
    for (int i = 0; i < amount; ++i)
    {
        int u = edgeData[i].u, v = edgeData[i].v;
        if (tp == wtype::Int)
        {
            int w = (L ? bias2(d[v]) : bias(d[v])) + 1;
            edges[i] = (Edges){u, v, (float)w};
        }
        else
            edges[i] = (Edges){u, v, (float)(d[v] + r.rd(100) * .01)};
    }

    // todo:debug
    build(edges, amount, nd, ndD, d, &sizeManager, n); // countGraph(n,ndD[0]);
    delete[] edges;
}
void insertGraph()
{ // recordOld(n,d,edgeData);
    Edges *edges = new Edges[BATCHSIZE];
    int lastAmount = edgeData.size();
    for (int i = 0; i < BATCHSIZE; ++i)
    {
        int pos = r.rd(lastAmount);
        int u = edgeData[pos].u, v = edgeData[pos].v;
        edgeData.push_back((EdgeData){u, v, d[u]++});
        if (tp == wtype::Int)
        {
            int w = ((n > 1e7) ? bias2(d[v]) : bias(d[v])) + 1;
            edges[i] = (Edges){u, v, bias(d[v])};
        }
        else
            edges[i] = (Edges){u, v, (float)(d[v] + r.rd(100) * .01)};
    }
    insert(edges, BATCHSIZE, nd, ndD, d, sizeManager);
    delete[] edges;
}
void deleteGraph()
{
    Deleted *edges = new Deleted[BATCHSIZE];
    for (int i = 0; i < BATCHSIZE; ++i)
    {
        int pos = r.rd(edgeData.size());
        while (edgeData[pos].nodeIdu == -1)
            pos = r.rd(edgeData.size());
        int u = edgeData[pos].u, idu = edgeData[pos].nodeIdu;
        edges[i] = (Deleted){u, idu};
        edgeData[pos].nodeIdu = -1;
    } // recordNew(n,d,edgeData);
    deleteE(edges, BATCHSIZE, ndD);
    delete[] edges;
}

void randomWalk(app a)
{
    ll len = LEN * (n / GPUS / PIECESIZE + 1);
    if (a == app::sampling)
        len = BATCHSIZE;
    Timer tt; // float time=0;
    tt.restart();

#pragma omp parallel for
    for (int g = 0; g < GPUS; ++g)
    {
        cudaSetDevice((g + 2) % GPUTOT);
        // if(BUFFER)resetKernel<<<BLKSZ,THDSZ>>>(n,ndD[g]);
        if (rwD[g] == NULL)
            cudaMalloc((void **)&rwD[g], (len * sizeof(int)));
        HE(cudaGetLastError());
        GPUTimer gT;
        gT.init();
        for (int p = 0; p < PIECESIZE; ++p)
        {
            cudaDeviceSynchronize();
            switch (a)
            {
            case app::deepwalk:
                deepwalkKernel<<<BLKSZ, THDSZ>>>(g, p, n, ndD[g], rwD[g], PIECESIZE, ((ull) new char)+g * 13);
                break;
            case app::node2vec:
                node2vecKernel<<<BLKSZ, THDSZ>>>(g, p, n, ndD[g], rwD[g], PIECESIZE, ((ull) new char)+g * 13);
                break;
            case app::ppr:
                pprKernel<<<BLKSZ, THDSZ>>>(g, p, n, ndD[g], rwD[g], PIECESIZE, ((ull) new char)+g * 13);
                break;
            case app::sampling:
                samplingKernel<<<BLKSZ, THDSZ>>>(g, p, n, ndD[g], rwD[g], PIECESIZE, ((ull) new char)+g * 13);
                break;
            }
            cudaDeviceSynchronize();
            HE(cudaGetLastError());
        } // time=gT.finish();
    }
    totalTime += tt.duration();
    if (OUTPUT)
    {
        // freopen("paths.txt","w",stdout);
        int PATH = 10;
        int *rw = new int[LEN * PATH];
        cudaSetDevice(1);
        cudaMemcpy(rw, rwD[0], (LEN * PATH) * sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        for (int i = 0; i < PATH; ++i)
        {
            for (int j = 1; j <= LEN; ++j)
                cout << rw[i * LEN + j - 1] << " ";
            cout << endl;
        }
        delete[] rw;
        // fclose(stdout);
    }
}