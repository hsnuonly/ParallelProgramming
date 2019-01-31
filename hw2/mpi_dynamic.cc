#define PNG_NO_SETJMP

#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#include <thread>
#include <mutex>
#include <iostream>
#include <vector>
#include <map>
#include <pthread.h>
#include <emmintrin.h>

#define MAX_ITER 10000
#define REQ_DATA 1
#define TO_COD 3
#define IMAGE_HEAD 4
#define IMAGE_DATA 6
#define FROM_COD 5
#define MAX_BUF 100000000
#define SLICE 12

MPI_Comm comm;
int rank_id, size;
int width, height;
std::mutex m_lock;

inline void write_png(const char *filename, const int width, const int height, const int *buffer)
{
    FILE *fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png_ptr, info_ptr);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y)
    {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x)
        {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != MAX_ITER)
            {
                if (p & 16)
                {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                }
                else
                {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

inline void in_set(double *x0, double *y0, int *ret_val)
{
    int repeats = 0;
    bool finished[2] = {0};
    // double length_squared = 0;

    __m128d xm = _mm_set_pd(0, 0);
    __m128d ym = _mm_set_pd(0, 0);
    __m128d x2m = _mm_set_pd(0, 0);
    __m128d y2m = _mm_set_pd(0, 0);
    __m128d x0m = _mm_load_pd(x0);
    __m128d y0m = _mm_load_pd(y0);

    while (repeats < MAX_ITER)
    {
        ym = _mm_mul_pd(xm, ym);
        ym = _mm_add_pd(ym, ym);
        ym = _mm_add_pd(ym, y0m);
        xm = _mm_sub_pd(x2m, y2m);
        xm = _mm_add_pd(xm, x0m);
        x2m = _mm_mul_pd(xm, xm);
        y2m = _mm_mul_pd(ym, ym);
        repeats++;
        __m128d lenm = _mm_add_pd(x2m, y2m);
        double len[2];
        _mm_store_pd(len, lenm);
        for (int i = 0; i < 2; i++)
        {
            if (!finished[i])
            {
                ret_val[i] = repeats;
                if (len[i] > 4)
                    finished[i] = 1;
            }
        }
        if (finished[0] && finished[1])
            break;
    }
}

void *thread_coordinator(void *arg)
{
    int *image = (int *)arg;
    int buf;
    MPI_Status status;
    MPI_Request req;
    std::vector<MPI_Request> reqs;
    int count = 0;
    while (1)
    {
        MPI_Recv(&buf, 1, MPI_INT, MPI_ANY_SOURCE, TO_COD, comm, &status);
        if (buf < 0)
            break;
        else if (buf == REQ_DATA)
        {
            if (count < height)
            {
                buf = count;
                count += SLICE;
            }
            else
            {
                buf = -1;
            }
            MPI_Send(&buf, 1, MPI_INT, status.MPI_SOURCE, FROM_COD, comm);
            if (buf >= 0)
            {
                MPI_Irecv(image + buf * width, width * SLICE, MPI_INT, status.MPI_SOURCE, buf * width, comm, &req);
                reqs.push_back(req);
            }
        }
    }
    MPI_Status statuses[reqs.size()];
    MPI_Waitall(reqs.size(), &reqs[0], statuses);
    pthread_exit(0);
}

void *image_data_receive(void *arg)
{
    int *image = (int *)arg;
    int buf[2];
    std::vector<MPI_Request> reqs;
    while (1)
    {
        MPI_Status status;
        MPI_Request req;
        MPI_Recv(buf, 1, MPI_INT, MPI_ANY_SOURCE, IMAGE_HEAD, comm, &status);
        if (buf[0] <= -1)
            break;
        MPI_Irecv(image + buf[0], width, MPI_INT, status.MPI_SOURCE, IMAGE_DATA, comm, &req);
        reqs.push_back(req);
    }
    MPI_Status statuses[reqs.size()];
    MPI_Waitall(reqs.size(), &reqs[0], statuses);
    pthread_exit(0);
}

int main(int argc, char **argv)
{ /* argument parsing */
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    assert(argc == 9);
    int num_threads = atoi(argv[1]);
    double left = atof(argv[2]);
    double right = atof(argv[3]);
    double lower = atof(argv[4]);
    double upper = atof(argv[5]);
    width = atoi(argv[6]);
    height = atoi(argv[7]);
    const char *filename = argv[8];

    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    comm = MPI_COMM_WORLD;

    int *image = new int[width * height * 2];
    std::vector<int> buf;
    buf.resize(width * height);

    pthread_t t1, t2;
    if (rank_id == 0)
    {
        // pthread_create(&t1, 0, image_data_receive, image);
        pthread_create(&t2, 0, thread_coordinator, image);
    }

    MPI_Status status;
    MPI_Request req;

    std::vector<MPI_Request> reqs;
    int begin = 0;

    double unit_y = ((upper - lower) / height);
    double unit_x = ((right - left) / width);
    double *y0 = new double[2];
    double *x0 = new double[2];
    while (1)
    {
        int query = REQ_DATA;
        int pos;
        MPI_Send(&query, 1, MPI_INT, 0, TO_COD, comm);
        MPI_Recv(&pos, 1, MPI_INT, 0, FROM_COD, comm, &status);
        if (pos < 0)
            break;
        for (int j = 0; j < SLICE; j++)
        {
            if (pos + j >= height)
                break;
            y0[0] = y0[1] = (pos + j) * unit_y + lower;
            for (int i = 0; i < width; i += 2)
            {
                x0[0] = i * unit_x + left;
                x0[1] = (i + 1) * unit_x + left;
                in_set(x0, y0, &buf[begin + j * width + i]);
            }
        }
        // buf[begin] = width * pos;
        // MPI_Isend(&buf[begin], 1, MPI_INT, 0, IMAGE_HEAD, comm, &req);
        // reqs.push_back(req);
        MPI_Isend(&buf[begin], width * SLICE, MPI_INT, 0, pos * width, comm, &req);
        reqs.push_back(req);
        begin += width * SLICE;
    }
    delete y0;
    delete x0;

    MPI_Status statuses[reqs.size()];
    MPI_Waitall(reqs.size(), &reqs[0], statuses);

    MPI_Barrier(comm);

    buf[0] = buf[1] = -1;
    if (rank_id == 0)
    {
        // MPI_Send(&buf[0], 1, MPI_INT, rank_id, IMAGE_HEAD, comm);
        MPI_Send(&buf[1], 1, MPI_INT, rank_id, TO_COD, comm);
        // pthread_join(t1, 0);
        pthread_join(t2, 0);
    }

    MPI_Finalize();
    if (rank_id == 0)
        write_png(filename, width, height, image);
    delete image;
    return 0;
}
