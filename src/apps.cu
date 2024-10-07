#include<iostream>
#include<cstdio>
#include<cstring>
#include<omp.h>
#include"apps.h"
#include"utils.cuh"
#include"test.cuh"
#include"graph.cuh"
using namespace std;
int instance(app a){
    if(a==app::node2vec)NODE2VEC=1;
    Timer TT;TT.restart();
    omp_set_num_threads(CPUTHD);
    loadGraph();
    Timer T;T.restart();
    for(int i=0;i<10;++i){
        insertGraph();
        deleteGraph();
        randomWalk(a);
    }//    printTrans();
    cerr<<"Evaluation time: "<<T.duration()<<" s."<<endl;
    cerr<<"Random walk in "<<totalTime<<" s."<<endl;
    cerr<<"Total time: "<<TT.duration()<<" s."<<endl;//while(1);
    return 0;
}