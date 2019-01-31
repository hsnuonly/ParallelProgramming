#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define BlockSize 28

const int INF = 1000000000;
void input(char *inFileName);
void output(char *outFileName);

void block_FW(int B);
int ceil(int a, int b);
void cal(char* d,size_t pitch,int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height,cudaStream_t stream);
void cpu(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height);

int n, m;	
static int* d;

double io_time = 0;
double comp_time = 0;
double mem_time = 0;

int main(int argc, char* argv[]) {
    auto io_beg = std::chrono::high_resolution_clock::now();
	input(argv[1]);
    auto io_end = std::chrono::high_resolution_clock::now();
    io_time += std::chrono::duration<double>(io_end-io_beg).count();

	int B = BlockSize;
	block_FW(B);

    io_beg = std::chrono::high_resolution_clock::now();
	output(argv[2]);
    io_end = std::chrono::high_resolution_clock::now();
    io_time += std::chrono::duration<double>(io_end-io_beg).count();

    std::cout<< comp_time <<" "<<mem_time<<" "<<io_time;
	return 0;
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    d = new int[n*n];

    for (int i = 0; i < n; ++ i) {
        for (int j = 0; j < n; ++ j) {
            if (i == j) {
                d[i*n+j] = 0;
            } else {
                d[i*n+j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++ i) {
        fread(pair, sizeof(int), 3, file);
        d[pair[0]*n+pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char *outFileName) {
	FILE *outfile = fopen(outFileName, "w");
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
            if (d[i*n+j] >= INF)
                d[i*n+j] = INF;
		}
	}
	fwrite(d, sizeof(int), n*n, outfile);
    fclose(outfile);
}

int ceil(int a, int b) {
	return (a + b - 1) / b;
}
__global__ void kernel(char* d,size_t pitch,int block_x,
    int block_y,int k,int n,int B,int round);
__global__ void kernel_simple(char* d,size_t pitch,int block_x,
    int block_y,int k,int n,int B);

void block_FW(int B) {
	int round = ceil(n, B);
    char *device_d;
    size_t pitch;
    // cudaMalloc(&device_d,sizeof(int)*n*n);
    // cudaMemcpy(device_d,d,sizeof(int)*n*n,cudaMemcpyHostToDevice);
    cudaMallocPitch(&device_d,&pitch,sizeof(int)*round*B,round*B);
    auto mem_beg = std::chrono::high_resolution_clock::now();
    cudaMemcpy2D(device_d,pitch,d,sizeof(int)*n,sizeof(int)*n,n,cudaMemcpyHostToDevice);
    auto mem_end = std::chrono::high_resolution_clock::now();

    // cudaStream_t streams[4];
    // for(int i=0;i<4;i++)
    //     cudaStreamCreate(streams+i);

    auto comp_beg = std::chrono::high_resolution_clock::now();
	for (int r = 0; r < round; ++r) {
        dim3 dimBlock(B,B);
        dim3 dimGrid(1,1);
        
        /* Phase 1*/
        for (int k = r * B; k < (r+1) * B && k < n; ++k) 
            kernel_simple<<<dimGrid,dimBlock>>>(device_d,pitch,r,r,k,n,B);
        if(r==0){
            auto mem_end = std::chrono::high_resolution_clock::now();
            mem_time += std::chrono::duration<double>(mem_end-mem_beg).count();
        }
        // for (int k = r * B; k < (r+1) * B && k < n; ++k){
            // for(int i=0;i<round;i++){
            //     if(i!=r){ 
            //         kernel<<<dimGrid,dimBlock,0>>>(device_d,pitch,i,r,0,n,B,r);
            //     }
            // }
            // for(int j=0;j<round;j++){
            //     if(j!=r){ 
            //         kernel<<<dimGrid,dimBlock,0>>>(device_d,pitch,r,j,0,n,B,r);
            //     }
            // }
        // }
        dimGrid = dim3(1,r);
        kernel<<<dimGrid,dimBlock,0>>>(device_d,pitch,r,0,0,n,B,r);
        dimGrid = dim3(1,round-r-1);
        kernel<<<dimGrid,dimBlock,0>>>(device_d,pitch,r,r+1,0,n,B,r);
        dimGrid = dim3(r,1);
        kernel<<<dimGrid,dimBlock,0>>>(device_d,pitch,0,r,0,n,B,r);
        dimGrid = dim3(round-r-1,1);
        kernel<<<dimGrid,dimBlock,0>>>(device_d,pitch,r+1,r,0,n,B,r);
        
        // for (int k = r * B; k < (r+1) * B && k < n; ++k) {
            // for(int i=0;i<round;i++){
            //     for(int j=0;j<round;j++){
            //         if(i!=r&&j!=r){
            //             kernel<<<dimGrid,dimBlock,0>>>(device_d,pitch,i,j,0,n,B,r);
            //         }
            //     }
            // }
        // }
        dimGrid = dim3(r,r);
        kernel<<<dimGrid,dimBlock,0>>>(device_d,pitch,0,0,0,n,B,r);
        dimGrid = dim3(round-r-1,r);
        kernel<<<dimGrid,dimBlock,0>>>(device_d,pitch,r+1,0,0,n,B,r);
        dimGrid = dim3(r,round-r-1);
        kernel<<<dimGrid,dimBlock,0>>>(device_d,pitch,0,r+1,0,n,B,r);
        dimGrid = dim3(round-r-1,round-r-1);
        kernel<<<dimGrid,dimBlock,0>>>(device_d,pitch,r+1,r+1,0,n,B,r);
        // dimGrid = dim3(round,round);
        // kernel<<<dimGrid,dimBlock,0>>>(device_d,pitch,0,0,0,n,B,r);
        
        // std::cout<< std::chrono::duration<double>(comp_end-comp_beg).count()<<"\n";
    }
        auto comp_end = std::chrono::high_resolution_clock::now();
        comp_time += std::chrono::duration<double>(comp_end-comp_beg).count();
    // cudaStreamSynchronize(0);
    mem_beg = std::chrono::high_resolution_clock::now();
    cudaMemcpy2D(d,sizeof(int)*n,device_d,pitch,sizeof(int)*n,n,cudaMemcpyDeviceToHost);
    mem_end = std::chrono::high_resolution_clock::now();
    mem_time += std::chrono::duration<double>(mem_end-mem_beg).count();
    // cudaMemcpy(d,device_d,sizeof(int)*n*n,cudaMemcpyDeviceToHost);
}

__device__ inline int gmin(int a,int b){
    return (a>b)*b+(a<=b)*a;
}

__global__ void kernel(char* d,size_t pitch,int block_x,
        int block_y,int k,int n,int B,int r){
   
    const int i = (block_x+blockIdx.x)*B+threadIdx.x;
    const int j = (block_y+blockIdx.y)*B+threadIdx.y;
    // const int idx = threadIdx.y*blockDim.x*threadIdx.x;

    __shared__ int p[BlockSize][BlockSize];
    __shared__ int d_i_k[BlockSize][BlockSize];
    __shared__ int d_k_j[BlockSize][BlockSize];
    __shared__ int d_i_j[BlockSize][BlockSize];

    int* d_i = (int*)(d+pitch*i);
    p[threadIdx.x][threadIdx.y] = d_i[j];
    d_i_j[threadIdx.x][threadIdx.y] = p[threadIdx.x][threadIdx.y];
    d_i_k[threadIdx.x][threadIdx.y] = d_i[r*B+threadIdx.y];
    d_k_j[threadIdx.x][threadIdx.y] = ((int*)(d+pitch*(r*B+threadIdx.x)))[j];

    const int k_max = gmin((r+1) * B , n);
    __syncthreads();
    #pragma unroll
    for (int k = r * B; k < k_max; ++k) {
        // int* d_k = (int*)(d+pitch*k);
        p[threadIdx.x][threadIdx.y] = gmin( p[threadIdx.x][threadIdx.y],
                    d_i_k[threadIdx.x][k-r*B]+d_k_j[k-r*B][threadIdx.y] );
    }
    if(d_i_j[threadIdx.x][threadIdx.y]>p[threadIdx.x][threadIdx.y]){
        d_i[j]=p[threadIdx.x][threadIdx.y];
    }
    
    // int new_d = ((int*)(d+pitch*(i)))[k]+((int*)(d+pitch*k))[j];
    // if(((int*)(d+pitch*i))[j]>new_d && j<n && i<n){
    //     ((int*)(d+pitch*i))[j]=new_d;
    // }
}
__global__ void kernel_simple(char* d,size_t pitch,int block_x,
    int block_y,int k,int n,int B){

    const int i = block_x*B+threadIdx.x;
    const int j = block_y*B+threadIdx.y;
    // const int idx = threadIdx.y*blockDim.x*threadIdx.x;

    __shared__ int p[BlockSize][BlockSize];
    __shared__ int d_k_j[BlockSize];
    __shared__ int d_i_k[BlockSize];
    if(threadIdx.x==0)
        d_k_j[threadIdx.y] = ((int*)(d+pitch*k))[j];
    else if(threadIdx.x==1)
        d_i_k[threadIdx.y] = ((int*)(d+pitch*(block_x*B+threadIdx.y)))[k];

    int* d_i = (int*)(d+pitch*i);
    p[threadIdx.x][threadIdx.y] = d_i[j];

    __syncthreads();
    // int* d_k_j = (int*)(d+pitch*k);
    int new_d = d_i_k[threadIdx.x]+d_k_j[threadIdx.y];

    if(p[threadIdx.x][threadIdx.y]>new_d){
        d_i[j]=new_d;
    }
}