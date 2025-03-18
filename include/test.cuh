#pragma once
#include <cstdio>
#include <cstring>
#include <vector>
#include "type.h"
#include "Bingo.cuh"
using namespace std;
// estimate group memory
__global__ void countGraph(int n, NodeData *ndD, int *a, double *b, double *c, double *d, double *e, int *log2)
{
    if (threadIdx.x != 0)
        return;
    a[0] = a[1] = a[2] = a[3] = a[4] = 0;
    log2[0] = 0;
    log2[1] = 1;
    for (int i = 2; i < 5000000; ++i)
    {
        log2[i] = log2[i >> 1] << 1;
        if (log2[i] < i)
            log2[i] <<= 1;
    }
    float b11 = 0, b12 = 0;
    float b31 = 0, b32 = 0;
    for (int i = 1; i <= n; ++i)
    {
        int x = 4;
        if (ndD[i].edgeSZ < 256)
            x = 1;
        else if (ndD[i].edgeSZ < 65536)
            x = 2;
        for (int j = 0; j < ndD[i].LOGT; ++j)
        {
            int number = ndD[i].num[j] / (1 << j);
            if (number == 0)
                continue;
            if ((100ll * i / n) != (100ll * (i - 1) / n) && j == 0)
                printf("%d %d %d %d\n", number, i, j, n); //
            ++a[0];                                       // break;
            if (number == 1)
            {
                ++a[1];
                b11 += 1;
                b12 += 1 + ndD[i].edgeSZ;
                c[1] += 4 + 1;
                d[1] += x + 1;
                e[1] += (log2[number] + log2[(int)ndD[i].edgeSZ]) * 4;
            }
            else if (number * 5 < ndD[i].edgeSZ)
            {
                ++a[2];
                b31 += number * 2 + 1;
                b32 += number + ndD[i].edgeSZ;
                c[2] += (log2[number] + log2[number + 1]) * 4 + 1;
                d[2] += (log2[number] + log2[number + 1]) * x + 1;
                e[2] += (log2[number] + log2[(int)ndD[i].edgeSZ]) * 4;
            }
            else if (number > ndD[i].edgeSZ * 0.5)
            {
                ++a[3];
                c[3] += 1;
                d[3] += 1;
                e[3] += (log2[number] + log2[ndD[i].edgeSZ]) * 4;
            }
            else
            {
                ++a[4];
                c[0] += (log2[number] + log2[ndD[i].edgeSZ]) * 4 + 1;
                d[0] += (log2[number] + log2[ndD[i].edgeSZ]) * x + 1;
                e[0] += (log2[number] + log2[ndD[i].edgeSZ]) * 4;
            }
        }
    }
    // b[1]=b11/b12;
    // b[2]=b31/b32;
    for (int i = 0; i < 4; ++i)
        (c[i] /= 1048576) /= 1024, (d[i] /= 1048576) /= 1024, (e[i] /= 1048576) /= 1024;
}
void countGraph(int n, NodeData *ndD)
{
    cudaSetDevice(1);
    int a[5] = {0, 0, 0, 0, 0}; // total one sparse dense
    int *aD;
    cudaMalloc((void **)&aD, 5 * sizeof(int));
    double *bD;
    cudaMalloc((void **)&bD, 4 * sizeof(double));
    double *cD;
    cudaMalloc((void **)&cD, 4 * sizeof(double));
    double *dD;
    cudaMalloc((void **)&dD, 4 * sizeof(double));
    double *eD;
    cudaMalloc((void **)&eD, 4 * sizeof(double));
    int *log2;
    cudaMalloc((void **)&log2, 5000000 * sizeof(int));
    HE(cudaGetLastError());
    countGraph<<<1, 32>>>(n, ndD, aD, bD, cD, dD, eD, log2);
    cudaDeviceSynchronize();
    HE(cudaGetLastError());
    cudaMemcpy(a, aD, 5 * sizeof(int), cudaMemcpyDeviceToHost);
    double b[4], c[4], d[4], e[4];
    cudaMemcpy(b, bD, 4 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(c, cD, 4 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(d, dD, 4 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(e, eD, 4 * sizeof(double), cudaMemcpyDeviceToHost);
    cout << a[0] << " " << a[1] << " " << a[2] << " " << a[3] << " " << a[4] << endl;
    cout << "one" << " " << "sparse " << "dense" << endl;
    cout << 1. * a[1] / a[0] << " " << 1. * a[2] / a[0] << " " << 1. * a[3] / a[0] << endl;
    cout << b[1] << " " << b[2] << " " << b[3] << endl;
    cout << c[0] << " " << c[1] << " " << c[2] << " " << c[3] << " " << c[0] + c[1] + c[2] + c[3] << endl; // group
    cout << d[0] << " " << d[1] << " " << d[2] << " " << d[3] << " " << d[0] + d[1] + d[2] + d[3] << endl; // type
    cout << e[0] << " " << e[1] << " " << e[2] << " " << e[3] << " " << e[0] + e[1] + e[2] + e[3] << endl; // total
}
// count trans
// 0: dense 1: r 2:s 3: one
int *oldcnt[MAXLOG + 1], *cnt[MAXLOG + 1];
int ans[4][4] = {0}, ct[4] = {0};
void initTrans(int n)
{
    for (int i = 0; i <= MAXLOG; ++i)
        oldcnt[i] = new int[n + 1], cnt[i] = new int[n + 1];
}
void recordOld(int n, int *d, vector<EdgeData> &edge)
{
    for (int i = 0; i <= MAXLOG; ++i)
        memset(oldcnt[i], 0, sizeof(oldcnt[i])), memset(cnt[i], 0, sizeof(cnt[i]));
    for (int i = 0; i < edge.size(); ++i)
        if (edge[i].nodeIdu != -1)
        {
            int u = edge[i].u;
            ++oldcnt[MAXLOG][u];
            int w = d[edge[i].v];
            for (int j = 0; j < MAXLOG; ++j)
                if (w & (1 << j))
                    ++oldcnt[j][u];
        }
    puts("-1");
    for (int j = 0; j < MAXLOG; ++j)
        for (int i = 0; i <= n; ++i)
            // if(oldcnt[j][i]==0)oldcnt[j][i]=-1;
            // else
            if (oldcnt[j][i] <= 1)
                oldcnt[j][i] = 3, ++ct[3];
            else if (oldcnt[j][i] > 0.45 * oldcnt[MAXLOG][i])
                oldcnt[j][i] = 0, ++ct[0];
            else if (oldcnt[j][i] < 0.2 * oldcnt[MAXLOG][i])
                oldcnt[j][i] = 2, ++ct[2];
            else
                oldcnt[j][i] = 1, ++ct[1];
}
void recordNew(int n, int *d, vector<EdgeData> &edge)
{
    puts("1");
    for (int i = 0; i < edge.size(); ++i)
        if (edge[i].nodeIdu != -1)
        {
            int u = edge[i].u;
            ++cnt[MAXLOG][u];
            int w = d[edge[i].v];
            for (int j = 0; j < MAXLOG; ++j)
                if (w & (1 << j))
                    ++cnt[j][u];
        }
    for (int j = 0; j < MAXLOG; ++j)
        for (int i = 0; i <= n; ++i)
            if (oldcnt[j][i] == 0 && cnt[j][i] > 0.225 * cnt[MAXLOG][i])
                cnt[j][i] = 0;
            else if (oldcnt[j][i] == 2 && cnt[j][i] < 0.4 * cnt[MAXLOG][i] && cnt[j][i] != 1)
                cnt[j][i] = 2;
            else if (oldcnt[j][i] == 1 && cnt[j][i] > 0.1 * cnt[MAXLOG][i] && cnt[j][i] < 0.675 * cnt[MAXLOG][i])
                cnt[j][i] = 1;
            else if (cnt[j][i] <= 1)
                cnt[j][i] = 3;
            else if (cnt[j][i] > 0.45 * cnt[MAXLOG][i])
                cnt[j][i] = 0;
            else if (cnt[j][i] < 0.2 * cnt[MAXLOG][i])
                cnt[j][i] = 2;
            else
                cnt[j][i] = 1;
    for (int j = 0; j < MAXLOG; ++j)
        for (int i = 0; i <= n; ++i)
        {
            // if(oldcnt[MAXLOG][i]<1&&cnt[MAXLOG][i]<1)continue;
            if (oldcnt[j][i] == -1)
                continue;
            ++ans[oldcnt[j][i]][cnt[j][i]];
        }
}
void printTrans()
{
    puts("-------------------------");
    for (int i = 0; i <= 3; ++i)
    {
        for (int j = 0; j <= 3; ++j)
            printf("%lf ", ans[i][j] / (1. * ct[i]));
        puts("");
    }
    puts("-------------------------");
}