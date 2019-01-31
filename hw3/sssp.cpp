#include <pthread.h>
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
#define MAX_DIST 100000000

#define DATA 0
#define TOKEN 1
#define TERM 2

#define NONE 0
#define WHITE 1
#define BLACK 2

MPI_Comm comm;
int rank, size;
int v, e;
int *w;
int *d;
int buf[MAX_SIZE];

std::queue<int> q;
std::map<int, bool> inqueue;

std::mutex queue_lock;
std::mutex token_lock;
std::mutex d_lock;
std::condition_variable gcv;
std::mutex cv_lock;

bool token_received = 0;
int color = NONE;

int term = 0;
bool token_sent = 0;

void task_recv() {
    int recvbuf[2];
    MPI_Status status;
    while (1) {
        MPI_Recv(recvbuf, 2, MPI_INT, MPI_ANY_SOURCE, DATA, comm, &status);
        if (recvbuf[0] == -1) break;
        {
            std::lock_guard<std::mutex> lg(token_lock);
            color = BLACK;
        }

        int s = recvbuf[0];
        int d_s = recvbuf[1];
        if (d[s] >= d_s) {
            {
                std::lock_guard<std::mutex> lg(d_lock);
                d[s] = d_s;
            }
            if (!inqueue[s]) {
                std::lock_guard<std::mutex> lg(queue_lock);
                inqueue[s] = 1;
                q.push(s);
                gcv.notify_all();
            }
        }
    }
}

void token_recv() {
    int recvbuf[1];
    MPI_Status status;
    while (1) {
        MPI_Recv(recvbuf, 1, MPI_INT, MPI_ANY_SOURCE, TOKEN, comm, &status);
        std::lock_guard<std::mutex> lg(token_lock);
        if (recvbuf[0] == -1) break;
        token_received = 1;
        if (color != BLACK) color = recvbuf[0];
        gcv.notify_all();
    }
}

void term_recv() {
    MPI_Status status;
    MPI_Recv(&term, 1, MPI_INT, MPI_ANY_SOURCE, TERM, comm, &status);
    gcv.notify_all();
}

int main(int argc, char **argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    comm = MPI_COMM_WORLD;

    MPI_Status status;
    MPI_File ifs;
    MPI_File_open(comm, argv[1], MPI_MODE_RDONLY, MPI_INFO_NULL, &ifs);
    MPI_File_read_at(ifs, 0, buf, 2, MPI_INT, &status);
    v = buf[0];
    e = buf[1];

    w = new int[v * v];
    d = new int[v];

    for (int i = 0; i < v * v; i++) w[i] = MAX_DIST;
    for (int i = 0; i < v; i++) d[i] = MAX_DIST;
    MPI_File_read_at(ifs, sizeof(int) * 2, buf, e * 3, MPI_INT, &status);

    MPI_File_close(&ifs);
    for (int i = 0; i < e; i++) {
        int s = buf[i * 3];
        int r = buf[i * 3 + 1];
        int w_e = buf[i * 3 + 2];
        w[s * v + r] = w_e;
    }

    for (int i = 0; i < v; i++) w[i * v + i] = 0;

    // start threads
    std::thread t1(task_recv);
    std::thread t2(token_recv);
    std::thread t3(term_recv);

    // init queue
    {
        std::lock_guard<std::mutex> lg(d_lock);
        d[0] = 0;
    }
    if (rank == 0) {
        std::lock_guard<std::mutex> lg(queue_lock);
        q.push(0);
    }
    if (rank == 0) {
        std::lock_guard<std::mutex> lg(token_lock);
        token_received = 1;
    }

    while (1) {
        // terminate
        if (term) break;
        // state: passive
        else if (q.empty()) {
            if (token_received) {
                // terminate detected
                if (rank == 0 && color == WHITE) {
                    MPI_Request reqs[size];
                    int flag = 1;
                    for (int i = 0; i < size; i++)
                        MPI_Isend(&flag, 1, MPI_INT, i, TERM, comm, reqs + i);
                    MPI_Status status3[size];
                    MPI_Waitall(size, reqs, status3);
                    break;
                    // forwarding
                } else {
                    if (rank == 0) color = WHITE;
                    MPI_Send(&color, 1, MPI_INT, (rank + 1) % size, TOKEN,
                             comm);
                    color = NONE;
                }
                if (term) break;
                // critical section
                std::lock_guard<std::mutex> lg(token_lock);
                token_received = 0;
            } else {
                // critical section
                std::unique_lock<std::mutex> glock(cv_lock);
                gcv.wait(glock);
            }
            // state: active
        } else {
            std::map<int, bool> inq;
            std::vector<int> send_list;
            while (!q.empty()) {
                int s;
                {
                    // critical section
                    std::lock_guard<std::mutex> lg(queue_lock);
                    s = q.front();
                    q.pop();
                    inqueue[s] = 0;
                }

                // main algorithm
                for (int i = 0; i < v; i++) {
                    if (d[i] > d[s] + w[s * v + i]) {
                        {
                            std::lock_guard<std::mutex> lg(d_lock);
                            d[i] = d[s] + w[s * v + i];
                        }
                        if (i % size == rank) {
                            if (!inqueue[i]) {
                                // critical section
                                std::lock_guard<std::mutex> lg(queue_lock);
                                inqueue[i] = 1;
                                q.push(i);
                            }
                        } else {
                            if (!inq[i]) {
                                inq[i] = 1;
                                send_list.push_back(i);
                            }
                        }
                    }
                }
            }
            MPI_Request req;
            std::vector<MPI_Request> reqs;
            for (auto i : send_list) {
                buf[i * 2] = i;
                buf[i * 2 + 1] = d[i];
                MPI_Isend(buf + i * 2, 2, MPI_INT, i % size, DATA, comm, &req);
                reqs.push_back(req);
            }
            MPI_Status status2[reqs.size()];
            MPI_Waitall(reqs.size(), &reqs[0], status2);
        }
    }

    buf[0] = -1;
    MPI_Send(buf, 1, MPI_INT, rank, DATA, comm);
    MPI_Send(buf, 1, MPI_INT, rank, TOKEN, comm);

    t1.join();
    t2.join();
    t3.join();

    // for (int i = 0; i < v; i++)
    // {
    //     if (rank == i % size)
    //         std::cout << d[i] << "\n";
    //     MPI_Barrier(comm);
    // }

    int serial = 0;
    if (!serial) {
        MPI_File ofs;
        MPI_File_open(comm, argv[2], MPI_MODE_WRONLY | MPI_MODE_CREATE,
                      MPI_INFO_NULL, &ofs);
        for (int i = rank; i < v; i += size)
            MPI_File_write_at(ofs, sizeof(int) * i, d + i, 1, MPI_INT, &status);
        MPI_File_close(&ofs);
    } else {
        std::ofstream ofs(argv[2], std::ios::binary);
        ofs.write((char *)buf, sizeof(int) * v);
        ofs.close();
    }

    MPI_Finalize();
    delete w, d;
}