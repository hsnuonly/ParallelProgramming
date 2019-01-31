#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <float.h>
#include "mpi.h"

MPI_Comm comm;
int rank, size;
float buf[45000000];
float left[45000000];
float right[45000000];
float c[90000000];

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

        memcpy(a, c, m * sizeof(float));
        memcpy(b, c + m, n * sizeof(float));
    }
    else if (d == 1)
    {
        while (i + j < m)
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
        memcpy(a, c, m * sizeof(float));
    }
    else if (d == 2)
    {
        i = m - 1, j = n - 1;
        while (i + j + 1 >= m)
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

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        return 1;
    }

    // initialize mpi
    int rc = MPI_Init(&argc, &argv);

    // error handling for mpi
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
    //float left[slice_proc], right[slice_proc];

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
    /*
    if (n > 1 && n % 2 && size > 1)
    {
        n--;
        slice_proc = n;
    }
    */
    if (rank == size - 1)
    {
        slice_proc = slices - slice_proc * (size - 1);
    }

    MPI_File input_file;
    MPI_Status status;

    MPI_File_open(comm, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL,
                  &input_file);

    MPI_File_read_ordered(input_file, buf, slice_proc, MPI_FLOAT, &status);
    MPI_File_close(&input_file);

    std::sort(buf, buf + slice_proc);

    int sorted = 0;

    for (int j = 0;; j++)
    {
        MPI_Request req[4];
        float leftmost, rightmost;

        if (rank % 2 == j % 2 && rank + 1 < size)
        {
            MPI_Isend(buf + slice_proc - 1, 1, MPI_FLOAT, rank + 1, 0, comm, &req[0]);
            MPI_Recv(&rightmost, 1, MPI_FLOAT, rank + 1, 0, comm, &status);
            if (rightmost < buf[n - 1])
            {
                MPI_Isend(buf + slice_proc - n, n, MPI_FLOAT, rank + 1, 0, comm, &req[0]);
                MPI_Irecv(right, n, MPI_FLOAT, rank + 1, 0, comm, &req[1]);
                MPI_Wait(&req[0], &status);
                MPI_Wait(&req[1], &status);
                sorted &= my_merge(buf + slice_proc - n, right, n, n, 1);
            }
        }
        if (rank % 2 != j % 2 && rank > 0)
        {
            MPI_Isend(buf, 1, MPI_FLOAT, rank - 1, 0, comm, &req[2]);
            MPI_Recv(&leftmost, 1, MPI_FLOAT, rank - 1, 0, comm, &status);
            if (leftmost > buf[0])
            {
                MPI_Isend(buf, n, MPI_FLOAT, rank - 1, 0, comm, &req[2]);
                MPI_Irecv(left, n, MPI_FLOAT, rank - 1, 0, comm, &req[3]);
                MPI_Wait(&req[2], &status);
                MPI_Wait(&req[3], &status);
                sorted &= my_merge(left, buf, n, n, 2);
            }
        }

        if (rank == size - 1 && slice_proc > n && !sorted && n > 1 && buf[n] < buf[n - 1])
            my_merge(buf, buf + n, n, slice_proc - n, 0);

        int total_sorted = 0;
        MPI_Allreduce(&sorted, &total_sorted, 1, MPI_INT, MPI_LAND, comm);
        sorted = 1;
        if (total_sorted)
            break;
    }

    MPI_File output_file;

    MPI_File_open(comm, argv[3], MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL,
                  &output_file);
    MPI_File_write_ordered(output_file, buf, slice_proc, MPI_FLOAT, &status);

    MPI_File_close(&output_file);

    MPI_Finalize();
    //delete buf;
    return 0;
}
