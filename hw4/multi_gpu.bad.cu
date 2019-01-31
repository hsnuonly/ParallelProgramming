#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

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


double io_time = 0;
double comp_time = 0;
double mem_time = 0;

int main(int argc, char* argv[]) {

    auto io_beg = std::chrono::high_resolution_clock::now();
    input(argv[1]);
    auto io_end = std::chrono::high_resolution_clock::now();
    io_time += std::chrono::duration<double>(io_end-io_beg).count();

    int B = BlockSize;
    block_FW(B,argv[2]);

    io_beg = std::chrono::high_resolution_clock::now();
    output(argv[2]);
    io_end = std::chrono::high_resolution_clock::now();
    io_time += std::chrono::duration<double>(io_end-io_beg).count();

    std::cout<< comp_time <<" "<<mem_time<<" "<<io_time;
    delete d;
	return 0;
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    int *buf = new int[m*3];
    d = new int[n*n];
    fread(buf, sizeof(int), 3*m, file);

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

    #pragma omp parallel for
    for (int i = 0; i < m; ++ i) {
        int pair[3];
        // fread(pair, sizeof(int), 3, file);
        for(int j=0;j<3;j++)
            pair[j]=buf[i*3+j];
        d[pair[0]*n+pair[1]] = pair[2];
    }
    fclose(file);
    delete buf;
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

inline void moveBlock(char** ptr,int dst, int src,int x,int y,int B,size_t pitch){
    for(int k=B*x;k<B*(x+1);k++)
        cudaMemcpyPeer(ptr[dst]+pitch*B*k+sizeof(int)*B*y,dst,
                        ptr[src]+pitch*B*k+sizeof(int)*B*y,src,
                        sizeof(int)*B);
}

void block_FW(int B, char* outFileName) {
	int round = ceil(n, B);
    char *device_d[2];
    size_t pitch;
    // cudaMalloc(&device_d,sizeof(int)*n*n);
    // cudaMemcpy(device_d,d,sizeof(int)*n*n,cudaMemcpyHostToDevice);
    auto mem_beg = std::chrono::high_resolution_clock::now();
    for(int dev=0;dev<2;dev++){
        cudaSetDevice(dev);
        cudaDeviceEnablePeerAccess(!dev,0);
        cudaMallocPitch(&device_d[dev],&pitch,sizeof(int)*round*B,round*B);
        cudaMemcpy2DAsync(device_d[dev],pitch,
                            d,sizeof(int)*n,
                            sizeof(int)*n,n,cudaMemcpyHostToDevice);
    }
    auto mem_end = std::chrono::high_resolution_clock::now();
    mem_time += std::chrono::duration<double>(mem_end-mem_beg).count();

    for(int dev=0;dev<2;dev++){
        cudaSetDevice(dev);
        cudaDeviceSynchronize();
    }
    
    auto comp_beg = std::chrono::high_resolution_clock::now();
	for (int r = 0; r <= round; ++r) {
        dim3 dimBlock(B,B);
        dim3 dimGrid(1,1);

        for(int dev=0;dev<2;dev++){
            cudaSetDevice(dev);
            kernel_I  <<<dimGrid,dimBlock,0>>>(device_d[dev],pitch,r,r,n,B,r);
        }

        int dev = r>=round/2;
        moveBlock(device_d,!dev,dev,r,r,B,pitch);
        
        dimGrid = dim3(1,round-1);
        kernel_II <<<dimGrid,dimBlock,0>>>(device_d[dev],pitch,0,0,n,B,r);

        for(int i=0;i<round;i++){
            moveBlock(device_d,!dev,dev,r,i,B,pitch);
        }
        for(int dev=0;dev<2;dev++){
            cudaSetDevice(dev);
            cudaDeviceSynchronize();
        }

        cudaSetDevice(0);
        dimGrid = dim3(round/2,1);
        kernel_III<<<dimGrid,dimBlock,0>>>(device_d[0],pitch,0,r,n,B,r);
        cudaSetDevice(1);
        dimGrid = dim3(round-round/2,1);
        kernel_III<<<dimGrid,dimBlock,0>>>(device_d[1],pitch,round/2,r,n,B,r);

        
        
        cudaSetDevice(0);
        dimGrid = dim3(round/2,round);
        kernel_III<<<dimGrid,dimBlock,0>>>(device_d[0],pitch,0,0,n,B,r);
        cudaSetDevice(1);
        dimGrid = dim3(round-round/2,round);
        kernel_III<<<dimGrid,dimBlock,0>>>(device_d[1],pitch,round/2,0,n,B,r);
        
        for(int dev=0;dev<2;dev++){
            cudaSetDevice(dev);
            cudaDeviceSynchronize();
        }
    }

    for(int dev=0;dev<2;dev++){
        cudaSetDevice(dev);
        cudaDeviceSynchronize();
    }
    auto comp_end = std::chrono::high_resolution_clock::now();
    comp_time += std::chrono::duration<double>(comp_end-comp_beg).count();
    mem_beg = std::chrono::high_resolution_clock::now();

    cudaSetDevice(0);
    cudaMemcpy2DAsync(d,sizeof(int)*n,
        device_d[0],pitch,
        sizeof(int)*n,round/2*B,cudaMemcpyDeviceToHost);
    cudaSetDevice(1);
    cudaMemcpy2DAsync(d+round/2*B*n,sizeof(int)*n,
        device_d[1]+round/2*B*pitch,pitch,
        sizeof(int)*n,n-round/2*B,cudaMemcpyDeviceToHost);
    
    for(int dev=0;dev<2;dev++){
        cudaSetDevice(dev);
        cudaDeviceSynchronize();
    }
    mem_end = std::chrono::high_resolution_clock::now();
    mem_time += std::chrono::duration<double>(mem_end-mem_beg).count();
    // cudaMemcpy(d,device_d,sizeof(int)*n*n,cudaMemcpyDeviceToHost);
    for(int dev=0;dev<2;dev++)
        cudaFree(device_d[dev]);
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

    int origin_path = __ldg(&d_i[j]);
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

    if(origin_path>d_i_j[threadIdx.x][threadIdx.y]){
        d_i[j]=d_i_j[threadIdx.x][threadIdx.y];
    }
}

__global__ void kernel_III(char* d,size_t pitch,int block_x,
    int block_y,int n,int B,int r){
    __shared__ int d_i_k[BlockSize][BlockSize+1];
    __shared__ int d_k_j[BlockSize][BlockSize+1];

    int i = (block_x+blockIdx.x)*B+threadIdx.x;
    int j = (block_y+blockIdx.y)*B+threadIdx.y;

    int* d_i = ((int*)(d+pitch*i));
    int path;
    if(i<n&&j<n)
        path = __ldg(&d_i[j]);
    else
        path = INF;
    int origin_path = path;
    if(r*B+threadIdx.y < n)
        d_i_k[threadIdx.x][threadIdx.y] = __ldg(&d_i[r*B+threadIdx.y]);
    else
        d_i_k[threadIdx.x][threadIdx.y] = INF;
    if(r*B+threadIdx.x < n)
        d_k_j[threadIdx.x][threadIdx.y] = __ldg(&((int*)(d+pitch*(r*B+threadIdx.x)))[j]);
    else 
        d_k_j[threadIdx.x][threadIdx.y] = INF;

    __syncthreads();
    // const int k_max = gmin((r+1) * B , n);
    const int k_max = gmin((r+1) * B,n);
    #pragma unroll
    for (int k = r * B; k < k_max; ++k) {
        int new_path = d_i_k[threadIdx.x][k-r*B]+d_k_j[k-r*B][threadIdx.y];
        if(path>new_path) path = new_path;
    }
    if(origin_path>path){
        d_i[j]=path;
    }
}
__global__ void kernel_II(char* d,size_t pitch,int block_x,
    int block_y,int n,int B,int r){
    __shared__ int d_i_k[BlockSize][BlockSize+1];
    __shared__ int d_k_j[BlockSize][BlockSize+1];

    int i, j;
    if(block_x==0){
        i = r*B + threadIdx.x;
        j = blockIdx.y>=r?
            (blockIdx.y * B + 1) + threadIdx.y : blockIdx.y * B + threadIdx.y ;
    }
    else{
        i = blockIdx.y * B + threadIdx.x ;
        j = r*B + threadIdx.y;
    }

    int* d_i = (int*)(d+pitch*i);
    int path;
    path = d_i[j];
    int origin_path = path;
    d_i_k[threadIdx.x][threadIdx.y] = __ldg(&d_i[r*B+threadIdx.y]);
    d_k_j[threadIdx.x][threadIdx.y] = __ldg(&((int*)(d+pitch*(r*B+threadIdx.x)))[j]);

    __syncthreads();
    const int k_max = gmin((r+1) * B , n);
    #pragma unroll
    for (int k = r * B; k < k_max; ++k) {
        // int* d_k = (int*)(d+pitch*k);
        int new_path = d_i_k[threadIdx.x][k-r*B]+d_k_j[k-r*B][threadIdx.y];
        if(path>new_path) path = new_path;
    }
    if(origin_path>path){
        d_i[j]=path;
    }
}

