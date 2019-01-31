#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <float.h>
#include <thread>
#include "mpi.h"
#include <chrono>
#include <iostream>
using namespace std::chrono;

MPI_Comm comm;
int rank, size;
float buf[100000000];
float left[100000000];
float right[100000000];
float c[200000000];

inline int my_merge(float *a, float *b, int m, int n, int d)
{
    int i = 0, j = 0;
    int count;
    if (d == 0)
    {
        while (i < m && j < n)
        {
            if (a[i] <= b[j])
            {
                c[i + j] = a[i];
                i++;
            }
            else
            {
                c[i + j] = b[j];
                j++;
            }
        }
        count = i == m && j == 0;
        while (i < m)
        {
            c[i + j] = a[i];
            i++;
        }
        while (j < n)
        {
            c[i + j] = b[j];
            j++;
        }
        if (i < m)
            memcpy(c + i + j, c + i, (m - i) * sizeof(float));
        if (j < n)
            memcpy(c + i + j, b + j, (n - j) * sizeof(float));
        memcpy(a, c, m * sizeof(float));
        memcpy(b, c + m, n * sizeof(float));
    }
    else if (d == 1)
    {
        while (i + j < m && i < m && j < n)
        {
            if (a[i] <= b[j])
            {
                c[i + j] = a[i];
                i++;
            }
            else
            {
                c[i + j] = b[j];
                j++;
            }
        }
        if (i + j < m)
            memcpy(c + i + j, a + i, sizeof(float) * (m - i - j));
        count = i == m && j == 0;
        memcpy(a, c, m * sizeof(float));
    }
    else if (d == 2)
    {
        i = m - 1, j = n - 1;
        while (i + j + 1 >= m && i >= 0 && j >= 0)
        {
            if (a[i] <= b[j])
            {
                c[i + j + 1] = b[j];
                j--;
            }
            else
            {
                c[i + j + 1] = a[i];
                i--;
            }
        }
        count = i == m - 1 && j == -1;
        memcpy(b, c + m, n * sizeof(float));
    }

    return count;
}

inline int my_search(float *a, int len, float val, int type)
{
    if (len == 1)
        return 0;
    int l = 0, r = len;
    while (l < r && l < len && r > 0)
    {
        int m = (l + r) / 2;
        if (val < a[m])
            r = m - 1;
        else
            l = m + 1;
    }
    int i;
    if (type == 0)
    {
        for (i = std::min(l, r); i < len; i++)
        {
            if (val < a[i])
                return i;
        }
    }
    else
    {
        for (i = std::max(l, r); i >= 0; i--)
        {
            if (a[i] < val)
                return i;
        }
    }
    return std::max(i, 0);
}

int main(int argc, char **argv)
{
    double io_time = 0;
    double commu_time = 0;
    double total_time = 0;
    auto program_start = high_resolution_clock::now();
    auto commu_start = high_resolution_clock::now();
    auto commu_end = high_resolution_clock::now();
    if (argc < 4)
    {
        return 1;
    }

    int rc = MPI_Init(&argc, &argv);

    if (rc != MPI_SUCCESS)
    {
        printf("Error starting MPI program. Terminating.\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    // initialize rank and size
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int slices = atoi(argv[1]);
    int slice_proc = slices / size;

    comm = MPI_COMM_WORLD;

    if (slice_proc <= 0)
    {
        slice_proc = 1;
        size = slices;
        MPI_Group orig_group, new_group;
        MPI_Comm_group(MPI_COMM_WORLD, &orig_group);
        int ranks1[slices], ranks2[size - slices];
        for (int i = 0; i < slices; i++)
            ranks1[i] = i;
        for (int i = 0; i < size - slices; i++)
            ranks2[i] = i + slices;
        if (rank < slices)
        {
            MPI_Group_incl(orig_group, slices, ranks1, &new_group);
            MPI_Comm_create(MPI_COMM_WORLD, new_group, &comm);
        }
        else
        {
            MPI_Group_incl(orig_group, slices, ranks2, &new_group);
            MPI_Comm_create(MPI_COMM_WORLD, new_group, &comm);
            MPI_Finalize();
            return 0;
        }
    }

    int n = slices / size;
    if (rank == size - 1)
    {
        slice_proc = slices - slice_proc * (size - 1);
    }

    MPI_File input_file;
    MPI_Status status;

    auto input_start = high_resolution_clock::now();
    MPI_File_open(comm, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL,
                  &input_file);

    // MPI_File_read_ordered(input_file, buf, slice_proc, MPI_FLOAT, &status);
    MPI_File_read_at(input_file, n * rank * sizeof(float), buf, slice_proc, MPI_FLOAT, &status);
    MPI_File_close(&input_file);
    auto input_end = high_resolution_clock::now();
    io_time += duration_cast<duration<double>>(input_end - input_start).count();

    std::sort(buf, buf + slice_proc);

    int sorted = 0;
    std::thread t;

    for (int j = 0;; j++)
    {
        MPI_Request req[4];
        float leftmost, rightmost;
        // if (t.joinable())
        //     t.join();

        if (rank % 2 == j % 2 && rank + 1 < size)
        {
            commu_start = high_resolution_clock::now();
            MPI_Isend(buf + slice_proc - 1, 1, MPI_FLOAT, rank + 1, 0, comm, &req[0]);
            MPI_Recv(&rightmost, 1, MPI_FLOAT, rank + 1, 0, comm, &status);
            commu_end = high_resolution_clock::now();
            commu_time += duration_cast<duration<double>>(commu_end - commu_start).count();

            if (rightmost < buf[n - 1])
            {
                int pos = my_search(buf, n, rightmost, 1);
                int send_size = n - pos, recv_size;
                commu_start = high_resolution_clock::now();
                MPI_Sendrecv(&send_size, 1, MPI_INT, rank + 1, 0,
                             &recv_size, 1, MPI_INT, rank + 1, 0,
                             comm, &status);
                MPI_Sendrecv(buf + pos, send_size, MPI_INT, rank + 1, 0,
                             right, recv_size, MPI_INT, rank + 1, 0,
                             comm, &status);
                commu_end = high_resolution_clock::now();
                commu_time += duration_cast<duration<double>>(commu_end - commu_start).count();
                sorted = 0;
                sorted &= my_merge(buf + pos, right, send_size, recv_size, 1);
                // t = std::thread(my_merge, buf + pos, right, send_size, recv_size, 1);
            }
        }
        if (rank % 2 != j % 2 && rank > 0)
        {
            commu_start = high_resolution_clock::now();
            MPI_Isend(buf, 1, MPI_FLOAT, rank - 1, 0, comm, &req[2]);
            MPI_Recv(&leftmost, 1, MPI_FLOAT, rank - 1, 0, comm, &status);
            commu_end = high_resolution_clock::now();
            commu_time += duration_cast<duration<double>>(commu_end - commu_start).count();
            if (leftmost > buf[0])
            {
                int pos = my_search(buf, n, leftmost, 0);
                int send_size = std::max(pos, 1), recv_size;
                commu_start = high_resolution_clock::now();
                MPI_Sendrecv(&send_size, 1, MPI_INT, rank - 1, 0,
                             &recv_size, 1, MPI_INT, rank - 1, 0,
                             comm, &status);
                MPI_Sendrecv(buf, send_size, MPI_INT, rank - 1, 0,
                             left, recv_size, MPI_INT, rank - 1, 0,
                             comm, &status);
                commu_end = high_resolution_clock::now();
                commu_time += duration_cast<duration<double>>(commu_end - commu_start).count();
                sorted = 0;
                sorted &= my_merge(left, buf, recv_size, send_size, 2);
                // t = std::thread(my_merge, left, buf, recv_size, send_size, 2);
            }
        }

        int total_sorted = 0;
        commu_start = high_resolution_clock::now();
        MPI_Allreduce(&sorted, &total_sorted, 1, MPI_INT, MPI_LAND, comm);
        commu_end = high_resolution_clock::now();
        commu_time += duration_cast<duration<double>>(commu_end - commu_start).count();
        if (rank == size - 1 && slice_proc > n && !sorted && n > 1 && buf[n] < buf[n - 1])
        {
            // if (t.joinable())
            //     t.join();
            int pos = my_search(buf, n, buf[n], 1);
            my_merge(buf + pos, buf + n, n - pos, slice_proc - n, 0);
            // my_merge(buf, buf + n, n, slice_proc - n, 0);
        }

        sorted = 1;
        if (total_sorted)
            break;
    }

    MPI_File output_file;

    auto output_start = high_resolution_clock::now();
    MPI_File_open(comm, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL,
                  &output_file);
    // MPI_File_write_ordered(output_file, buf, slice_proc, MPI_FLOAT, &status);
    MPI_File_write_at(output_file, n * rank * sizeof(float), buf, slice_proc, MPI_FLOAT, &status);

    MPI_File_close(&output_file);
    auto output_end = high_resolution_clock::now();
    io_time += duration_cast<duration<double>>(output_end - output_start).count();

    auto program_end = high_resolution_clock::now();
    total_time += duration_cast<duration<double>>(program_end - program_start).count();
    if (rank == 0)
    {
        std::cout << "IO Time: " << io_time << "\n";
        std::cout << "Communication Time: " << commu_time << "\n";
        std::cout << "Computing Time: " << total_time - io_time - commu_time << "\n";
    }
    MPI_Finalize();
    return 0;
}
