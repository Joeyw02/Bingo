#include <iostream>
#include <cstdio>
#include <cstring>
#include <omp.h>
#include "apps.h"
#include "utils.cuh"
#include "test.cuh"
#include "graph.cuh"
using namespace std;
int instance(app a, utype utp, string DATASET)
{
    // freopen(("../dataset/" + DATASET).c_str(), "r", stdin);
    FILE* fd = freopen(("../dataset/" + DATASET).c_str(), "r", stdin);
    if (!fd) {
        perror(("Failed to open file: ../dataset/" + DATASET).c_str());
        exit(EXIT_FAILURE);
    }
    
    cerr << "Graph dataset: " << DATASET << "." << endl;
    if (a == app::node2vec)
        NODE2VEC = 1;
    Timer TT;
    TT.restart();
    omp_set_num_threads(CPUTHD);
    loadGraph();
    Timer T;
    T.restart();
    if (utp == utype::M)
        BATCHSIZE >>= 1;
    for (int i = 0; i < 10; ++i)
    {
        if (utp == utype::M || utp == utype::I)
            insertGraph();
        if (utp == utype::M || utp == utype::D)
            deleteGraph();
        randomWalk(a);
    }
    double tt = T.duration();
    cerr << "Random walk in " << totalTime << " s." << endl;
    cerr << "Evaluation time: " << tt << " s." << endl;
    if (DETAIL)
        cerr << "Total time: " << TT.duration() << " s." << endl;
    
    // todo: 计算真实采样索引空间大小
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    cerr << "Memory Consumption: " << ((total_mem - free_mem) / 1024. / 1024. - 4) / 1024. << " GB." << endl; // cerr<<xxx<<" "<<tt-totalTime-xxx<<endl;
    fclose(stdin);
    return 0;
}