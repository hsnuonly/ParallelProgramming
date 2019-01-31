#include <stdlib.h>
#include <string.h>
#include <climits>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <queue>
#include <thread>
#include "mpi.h"

#define MAX_SIZE 100000000
#define MAX_DIST 1000000000

MPI_Comm comm;
int rank, size;
int w[MAX_SIZE];
int d[MAX_SIZE];
int buf[MAX_SIZE];
int v, e;
std::queue<int> q;

int main(int argc, char **argv) {
    std::ifstream ifs(argv[1], std::ios::binary);

    ifs.read((char *)&v, sizeof(int));
    ifs.read((char *)&e, sizeof(int));

    for (int i = 0; i < v * v; i++) w[i] = MAX_DIST;
    for (int i = 0; i < v; i++) d[i] = MAX_DIST;

    ifs.read((char *)buf, sizeof(int) * e * 3);
    ifs.close();

    for (int i = 0; i < e; i++) {
        int s = buf[i * 3];
        int r = buf[i * 3 + 1];
        int w_e = buf[i * 3 + 2];
        w[s * v + r] = w_e;
    }

    for (int i = 0; i < v; i++) w[i * v + i] = 0;


    q.push(0);
    d[0] = 0;

    bool inqueue[v];
    bzero(inqueue, sizeof(bool) * v);
    
    while (!q.empty()) {
        int s = q.front();
        q.pop();
        inqueue[s] = 0;

        for (int i = 0; i < v; i++) {
            if (d[i] > d[s] + w[s * v + i]) {
                d[i] = d[s] + w[s * v + i];
                if (!inqueue[i]) {
                    inqueue[i] = 1;
                    q.push(i);
                }
            }
        }
    }

    std::ofstream ofs(argv[2], std::ios::binary);
    ofs.write((char *)d, sizeof(int) * v);
    ofs.close();
}