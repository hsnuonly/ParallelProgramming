#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "mpi.h"

#define BlockSize 32

const int INF = 1000000000;
void input(char *inFileName);
void output(char *outFileName);

void block_FW(int B,char*);
int ceil(int a, int b);
void cal(char* d,size_t pitch,int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height,cudaStream_t stream);
void cpu(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height);

int n, m;	
int* d;

int rank,size;
MPI_Comm comm = MPI_COMM_WORLD;

double io_time = 0;
double comp_time = 0;
double mem_time = 0;
double comm_time = 0;
double total_time = 0;

int main(int argc, char* argv[]) {
    auto total_beg = std::chrono::high_resolution_clock::now();
    MPI_Init(&argc,&argv);
    MPI_Comm_size(comm,&size);
    MPI_Comm_rank(comm,&rank);

    auto io_beg = std::chrono::high_resolution_clock::now();
    input(argv[1]);
    auto io_end = std::chrono::high_resolution_clock::now();
    io_time += std::chrono::duration<double>(io_end-io_beg).count();

    int B = BlockSize;
    block_FW(B,argv[2]);

    io_beg = std::chrono::high_resolution_clock::now();
    // if(rank==0) output(argv[2]);
    io_end = std::chrono::high_resolution_clock::now();
    io_time += std::chrono::duration<double>(io_end-io_beg).count();

    auto total_end = std::chrono::high_resolution_clock::now();
    total_time += std::chrono::duration<double>(total_end-total_beg).count();
    comp_time = total_time - mem_time - io_time - comm_time;
    if(rank==0)
        std::cout<< comp_time <<" "<<mem_time<<" "<<io_time<<" "<<comm_time<<"\n";
    MPI_Finalize();
    // delete d;
	return 0;
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    // MPI_File ifs;
    // MPI_File_open(comm, infile, MPI_MODE_RDONLY, MPI_INFO_NULL, &ifs);
    // MPI_Status status;
    // MPI_File_read(ifs,&n,1,MPI_INT,&status);
    // MPI_File_read(ifs,&m,1,MPI_INT,&status);
    // int *buf = new int[m*3];
    d = new int[n*n*2];
    // fread(buf, sizeof(int), 3*m, file);

    #pragma omp parallel for
    for (int i = 0; i < n; ++ i) {
        for (int j = 0; j < n; ++ j) {
            if (i == j) {
                d[i*n+j] = 0;
            } else {
                d[i*n+j] = INF;
            }
        }
    }

    for (int i = 0; i < m; ++ i) {
        int pair[3];
        fread(pair, sizeof(int), 3, file);
        // MPI_File_read(ifs,pair,3,MPI_INT,&status);
        // for(int j=0;j<3;j++)
        //     pair[j]=buf[i*3+j];
        d[pair[0]*n+pair[1]] = pair[2];
    }
    fclose(file);
    // MPI_File_close(&ifs);
    // delete buf;
}

void output(char *outFileName) {
	FILE *outfile = fopen(outFileName, "w");
	fwrite(d, sizeof(int), n*n, outfile);
    fclose(outfile);
}

int ceil(int a, int b) {
	return (a + b - 1) / b;
}

__global__ void kernel_I(char* d,size_t pitch,int block_x,
    int block_y,int n,int B,int r);
__global__ void kernel_II(char* d,size_t pitch,int block_x,
    int block_y,int n,int B,int r);
__global__ void kernel_III(char* d,size_t pitch,int block_x,
    int block_y,int n,int B,int r);

void block_FW(int B, char* outFileName) {
	int round = ceil(n, B);
    char *device_d;
    size_t pitch;
    // cudaMalloc(&device_d,sizeof(int)*n*n);
    // cudaMemcpy(device_d,d,sizeof(int)*n*n,cudaMemcpyHostToDevice);

    auto mem_beg = std::chrono::high_resolution_clock::now();    
    cudaMallocPitch(&device_d,&pitch,sizeof(int)*round*B,round*B);
    cudaMemcpy2DAsync(device_d,pitch,d,sizeof(int)*n,sizeof(int)*n,n,cudaMemcpyHostToDevice);
    auto mem_end = std::chrono::high_resolution_clock::now();
    mem_time += std::chrono::duration<double>(mem_end-mem_beg).count();
    MPI_Request reqs[round];
    if(rank==0){
        for(int r=round/2;r<round;r++){
            MPI_Irecv(d+(n*r*B), min(n*B,n*n-(n*r*B)), MPI_INT, !rank, 0, comm, &reqs[r]);
        }
    }
    else{
        for(int r=0;r<round/2;r++){
            MPI_Irecv(d+(n*r*B), min(n*B,n*n-(n*r*B)), MPI_INT, !rank, 0, comm, &reqs[r]);
        }
    }
    
    auto comp_beg = std::chrono::high_resolution_clock::now();
	for (int r = 0; r < round; ++r) {
        dim3 dimBlock(B,B);
        dim3 dimGrid(1,1);
        
        /* Phase 1*/
        // if(r==0)cudaStreamSynchronize(streams[0]);

        // for (int k = r * B; k < (r+1) * B && k < n; ++k) 
        int dev = r>=(round/2);
        MPI_Request req;
        if(rank==dev){
            kernel_I  <<<dimGrid,dimBlock,0>>>(device_d,pitch,r,r,n,B,r);
            dimGrid = dim3(1,round);
            kernel_III<<<dimGrid,dimBlock,0>>>(device_d,pitch,r,0,n,B,r);
            auto mem_beg = std::chrono::high_resolution_clock::now();
            cudaMemcpy2D(d+n*r*B,sizeof(int)*n,
                device_d+pitch*r*B,pitch,
                sizeof(int)*n,min(B,n),cudaMemcpyDeviceToHost);
            auto mem_end = std::chrono::high_resolution_clock::now();
            mem_time += std::chrono::duration<double>(mem_end-mem_beg).count();
            MPI_Isend(d+n*r*B, min(n*B,n*n-n*r*B), MPI_INT, !rank, 0, comm, &req);
        }
        else{
            MPI_Status status;
            // MPI_Recv(d+n*r*B, min(n*B,n*n-n*r*B), MPI_INT, !rank, 0, comm, &status);
            auto comm_beg = std::chrono::high_resolution_clock::now();
            MPI_Wait(&reqs[r],&status);
            auto comm_end = std::chrono::high_resolution_clock::now();
            comm_time += std::chrono::duration<double>(comm_end-comm_beg).count();
            
            auto mem_beg = std::chrono::high_resolution_clock::now();
            cudaMemcpy2D(device_d+pitch*r*B,pitch,
                d+n*r*B,sizeof(int)*n,
                sizeof(int)*n,min(B,n),cudaMemcpyHostToDevice);
            auto mem_end = std::chrono::high_resolution_clock::now();
            mem_time += std::chrono::duration<double>(mem_end-mem_beg).count();
        }
        if(rank==0){
            dimGrid = dim3(round/2,1);
            kernel_III<<<dimGrid,dimBlock,0>>>(device_d,pitch,0,r,n,B,r);

            dimGrid = dim3(round/2,round);
            kernel_III<<<dimGrid,dimBlock,0>>>(device_d,pitch,0,0,n,B,r);
        }
        else{
            dimGrid = dim3(round-(round/2),1);
            kernel_III<<<dimGrid,dimBlock,0>>>(device_d,pitch,round/2,r,n,B,r);

            dimGrid = dim3(round-(round/2),round);
            kernel_III<<<dimGrid,dimBlock,0>>>(device_d,pitch,round/2,0,n,B,r);
        }
        // MPI_Status status;
        // if(rank==dev)
        //     MPI_Wait(&req,&status);
    }
    cudaStreamSynchronize(0);
    
    auto comp_end = std::chrono::high_resolution_clock::now();
    comp_time += std::chrono::duration<double>(comp_end-comp_beg).count();
    if(rank==0){
        MPI_Request req;
        MPI_Irecv(d+n*B*(round/2),(n-(round/2)*B)*n,MPI_INT,1,0,comm,&req);
        mem_beg = std::chrono::high_resolution_clock::now();
        cudaMemcpy2D(d,sizeof(int)*n,
            device_d,pitch,
            sizeof(int)*n,(round/2)*B,cudaMemcpyDeviceToHost);
        mem_end = std::chrono::high_resolution_clock::now();
        mem_time += std::chrono::duration<double>(mem_end-mem_beg).count();
        FILE *outfile = fopen(outFileName, "w");
        fwrite(d, sizeof(int), n*(round/2)*B, outfile);
        MPI_Status status;
        auto comm_beg = std::chrono::high_resolution_clock::now();
        MPI_Wait(&req,&status);
        auto comm_end = std::chrono::high_resolution_clock::now();
        comm_time += std::chrono::duration<double>(comm_end-comm_beg).count();
        fwrite(d+n*B*(round/2), sizeof(int), (n-(round/2)*B)*n, outfile);
        fclose(outfile);
    }
    else{
        cudaMemcpy2D(d+n*(round/2)*B,sizeof(int)*n,
            device_d+pitch*(round/2)*B,pitch, 
            sizeof(int)*n,n-B*(round/2),cudaMemcpyDeviceToHost);
        MPI_Send(d+n*B*(round/2),(n-(round/2)*B)*n,MPI_INT,0,0,comm);
    }
    
    // cudaMemcpy(d,device_d,sizeof(int)*n*n,cudaMemcpyDeviceToHost);
}

__device__ inline int gmin(int a,int b){
    return (a>b)*b+(a<=b)*a;
}


__global__ void kernel_I(char* d,size_t pitch,int block_x,
    int block_y,int n,int B, int r){
    __shared__ int d_i_j[BlockSize][BlockSize+1];

    const int i = block_x*B+threadIdx.x;
    const int j = block_y*B+threadIdx.y;
    // const int idx = threadIdx.y*blockDim.x*threadIdx.x;

    int* d_i = (int*)(d+pitch*i);

    int origin_path = i<n&&j<n? __ldg(&d_i[j]) : INF;
    d_i_j[threadIdx.x][threadIdx.y] = origin_path;

    // int* d_k_j = (int*)(d+pitch*k);
    const int k_max = gmin((r+1) * B,n);
    #pragma unroll
    for (int k = r * B; k < k_max; ++k) {
        __syncthreads();
        int new_d = d_i_j[threadIdx.x][k-r*B]+d_i_j[k-r*B][threadIdx.y];
        if(d_i_j[threadIdx.x][threadIdx.y]>new_d){
            d_i_j[threadIdx.x][threadIdx.y]=new_d;
        }
    }

    if(origin_path>d_i_j[threadIdx.x][threadIdx.y] && i<n && j < n){
        d_i[j]=d_i_j[threadIdx.x][threadIdx.y];
    }
}

__global__ void kernel_III(char* d,size_t pitch,int block_x,
    int block_y,int n,int B,int r){
    __shared__ int d_i_k[BlockSize][BlockSize+1];
    __shared__ int d_k_j[BlockSize][BlockSize+1];

    // int i = (block_x+blockIdx.x)>=r?
    //     (block_x+blockIdx.x+1)*B+threadIdx.x:(block_x+blockIdx.x)*B+threadIdx.x;
    // int j = (block_y+blockIdx.y)>=r?
    //     (block_y+blockIdx.y+1)*B+threadIdx.y:(block_y+blockIdx.y)*B+threadIdx.y;
    int i = (block_x+blockIdx.x)*BlockSize+threadIdx.x;
    int j = (block_y+blockIdx.y)*BlockSize+threadIdx.y;


    int* d_i = ((int*)(d+pitch*i));
    // int path = i<n&&j<n ? __ldg(&d_i[j]) : INF;
    int path = __ldg(&d_i[j]);
    int origin_path = path;
    // if(r*B+threadIdx.y < n && i < n)
    //     d_i_k[threadIdx.x][threadIdx.y] = __ldg(&d_i[r*B+threadIdx.y]);
    // else
    //     d_i_k[threadIdx.x][threadIdx.y] = INF;
    // if(r*B+threadIdx.x < n && j < n)
    //     d_k_j[threadIdx.x][threadIdx.y] = __ldg(&((int*)(d+pitch*(r*B+threadIdx.x)))[j]);
    // else 
    //     d_k_j[threadIdx.x][threadIdx.y] = INF;
    d_i_k[threadIdx.x][threadIdx.y] = __ldg(&d_i[r*BlockSize+threadIdx.y]);
    d_k_j[threadIdx.x][threadIdx.y] = __ldg(&((int*)(d+pitch*(r*BlockSize+threadIdx.x)))[j]);

    __syncthreads();
    // const int k_max = gmin((r+1) * B , n);
    const int k_max = gmin((r+1) * BlockSize,n);
    #pragma unroll
    for (int k = r * BlockSize; k < k_max; ++k) {
        int new_path = d_i_k[threadIdx.x][k-r*BlockSize]+d_k_j[k-r*BlockSize][threadIdx.y];
        if(path>new_path) path = new_path;
    }
    if(origin_path>path&&i<n&&j<n){
        d_i[j]=path;
    }
}
__global__ void kernel_II(char* d,size_t pitch,int block_x,
    int block_y,int n,int B,int r){
    __shared__ int d_i_k[BlockSize][BlockSize+1];
    __shared__ int d_k_j[BlockSize][BlockSize+1];

    int i, j;
    if(blockIdx.x==0){
        i = r*B + threadIdx.x;
        j = blockIdx.y>=r ? 
            (blockIdx.y+1) * B + threadIdx.y : blockIdx.y * B + threadIdx.y ;
    }
    else{
        i = blockIdx.y>=r ? 
            (blockIdx.y+1) * B + threadIdx.x : blockIdx.y * B + threadIdx.x ;
        j = r*B + threadIdx.y;
    }
    // int i = (block_x+blockIdx.x)>=r?
    //     (block_x+blockIdx.x+1)*B+threadIdx.x:(block_x+blockIdx.x)*B+threadIdx.x;
    // int j = (block_y+blockIdx.y)>=r?
    //     (block_y+blockIdx.y+1)*B+threadIdx.y:(block_y+blockIdx.y)*B+threadIdx.y;
    // int j = (block_y+blockIdx.y)*B+threadIdx.y;


    int* d_i = (int*)(d+pitch*i);
    int path = i<n && j<n ? d_i[j] : INF;
    int origin_path = path;
    d_i_k[threadIdx.x][threadIdx.y] = i<n&&r*B+threadIdx.y<n ? __ldg(&d_i[r*B+threadIdx.y]) : INF;
    d_k_j[threadIdx.x][threadIdx.y] = r*B+threadIdx.x<n&&j<n ? __ldg(&((int*)(d+pitch*(r*B+threadIdx.x)))[j]) : INF;

    __syncthreads();
    const int k_max = gmin((r+1) * B , n);
    #pragma unroll
    for (int k = r * B; k < k_max; ++k) {
        // int* d_k = (int*)(d+pitch*k);
        int new_path = d_i_k[threadIdx.x][k-r*B]+d_k_j[k-r*B][threadIdx.y];
        if(path>new_path) path = new_path;
    }
    if(origin_path>path && i<n && j < n){
        d_i[j]=path;
    }
}

