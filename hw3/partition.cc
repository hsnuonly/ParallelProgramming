#include <iostream>
#include <fstream>
#include <string.h>
#include <stdlib.h>
#include <math.h>
using namespace std;

int buf[200000000];

int main(int argc, char **argv)
{
    ifstream ifs(argv[1], ios::in | ios::binary);
    int v, e;
    int size = atoi(argv[3]);
    ifs.read((char *)&v, sizeof(int));
    ifs.read((char *)&e, sizeof(int));

    ifs.read((char *)buf, sizeof(int) * e);
    ifs.close();

    int slice = e / size;
    int slice_v = v / size;

    int edges[v];
    memset(edges, 0, sizeof(int) * v);

    for (int i = 0; i < e; i++)
    {
        int s = buf[e * 3];
        edges[s]++;
    }

    int begin[size + 1];
    int edge_per_rank[v];
    memset(edge_per_rank, 0, sizeof(int) * v);

    begin[size] = v;
    for (int i = 0; i < size; i++)
        begin[i] = slice_v * i;

    const int MAX_ITER = 10;

    for (int i = 0; i < size; i++)
    {
        for (int j = begin[i]; j < begin[i + 1]; i++)
        {
            edge_per_rank[i] += edges[j];
        }
    }

    for (int _ = 0; _ < MAX_ITER; _++)
    {
        bool stop = 1;
        for (int i = 0; i < size - 1; i++)
        {
            while (1)
            {
                int a = abs(edge_per_rank[i] + edges[begin[i + 1]] - slice);
                int b = abs(edge_per_rank[i + 1] - edges[begin[i + 1]] - slice);

                int c = abs(edge_per_rank[i] - slice);
                int d = abs(edge_per_rank[i + 1] - slice);

                if (a + b < c + d)
                {
                    edge_per_rank[i] += edges[begin[i + 1]];
                    edge_per_rank[i + 1] -= edges[begin[i + 1]];
                    begin[i + 1]++;
                    stop = 0;
                }
                else
                    break;
            }
        }
        if (stop)
            break;
    }

    fstream ofs(argv[2], ios::out | ios::binary);
    for (int i = 0; i < size; i++)
    {
        for (int j = begin[i]; j < begin[i + 1]; j++)
            // ofs.write((char *)&i, sizeof(int));
            ofs << i << '\n';
    }
    ofs.close();
}