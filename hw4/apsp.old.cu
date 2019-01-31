#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

const int INF = 1000000000;
void input(char *inFileName);
void output(char *outFileName);

void block_FW(int B);
int ceil(int a, int b);
void cal(char* d,size_t pitch,int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height);
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

	int B = 192;
	block_FW(B);

    io_beg = std::chrono::high_resolution_clock::now();
	output(argv[2]);
    io_end = std::chrono::high_resolution_clock::now();
    io_time += std::chrono::duration<double>(io_end-io_beg).count();

    // std::cout<< comp_time <<" "<<mem_time<<" "<<io_time;
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

void block_FW(int B) {
	int round = ceil(n, B);
    char *device_d;
    size_t pitch = sizeof(int)*n;
    // cudaMalloc(&device_d,sizeof(int)*n*n);
    // cudaMemcpy(device_d,d,sizeof(int)*n*n,cudaMemcpyHostToDevice);
    cudaMallocPitch(&device_d,&pitch,sizeof(int)*n,n);
    auto mem_beg = std::chrono::high_resolution_clock::now();
    cudaMemcpy2D(device_d,pitch,d,sizeof(int)*n,sizeof(int)*n,n,cudaMemcpyHostToDevice);
    auto mem_end = std::chrono::high_resolution_clock::now();
    mem_time += std::chrono::duration<double>(mem_end-mem_beg).count();


	for (int r = 0; r < round; ++r) {
    auto comp_beg = std::chrono::high_resolution_clock::now();
        /* Phase 1*/
		cal(device_d,pitch, B,	r,	r,	r,	1,	1);
        // cpu(B,	r,	r,	r,	1,	1);
        // cudaMemcpy2D(device_d+r*B*pitch+r*B*sizeof(int),pitch,
        //             d+r*B*n+r*B,sizeof(int)*n,
        //             sizeof(int)*B,B,
        //             cudaMemcpyHostToDevice);
        
        /* Phase 2*/
        // #pragma omp parallel
        //#pragma omp sections
        {
        //#pragma omp section
		cal(device_d,pitch, B, r,     r,     0,             r,             1);
        //#pragma omp section
		cal(device_d,pitch, B, r,     r,  r +1,  round - r -1,             1);
        //#pragma omp section
		cal(device_d,pitch, B, r,     0,     r,             1,             r);
        //#pragma omp section
		cal(device_d,pitch, B, r,  r +1,     r,             1,  round - r -1);
        }

		/* Phase 3*/
        // #pragma omp parallel
        //#pragma omp sections
        {
        //#pragma omp section
		cal(device_d,pitch, B, r,     0,     0,            r,             r);
        //#pragma omp section
		cal(device_d,pitch, B, r,     0,  r +1,  round -r -1,             r);
        //#pragma omp section
		cal(device_d,pitch, B, r,  r +1,     0,            r,  round - r -1);
        //#pragma omp section
        cal(device_d,pitch, B, r,  r +1,  r +1,  round -r -1,  round - r -1);
        }
    auto comp_end = std::chrono::high_resolution_clock::now();
    std::cout<< std::chrono::duration<double>(comp_end-comp_beg).count()<<"\n";
    }
    // cudaStreamSynchronize(0);
    mem_beg = std::chrono::high_resolution_clock::now();
    cudaMemcpy2DAsync(d,sizeof(int)*n,device_d,pitch,sizeof(int)*n,n,cudaMemcpyDeviceToHost);
    mem_end = std::chrono::high_resolution_clock::now();
    mem_time += std::chrono::duration<double>(mem_end-mem_beg).count();
    // cudaMemcpy(d,device_d,sizeof(int)*n*n,cudaMemcpyDeviceToHost);
}

__device__ inline int gmin(int a,int b){
    return (a>b)*b+(a<=b)*a;
}

__global__ void gpu(char* d,size_t pitch,int block_start_x,
        int block_start_y,int k,int n){
   
    int B = blockDim.y;
    int j = gmin( (block_start_y+blockIdx.y)*B+threadIdx.y , n-1 );
    // int* d_k = (int*)(d+pitch*k);
    int x_beg = (block_start_x + blockIdx.x)*B;
    int x_end = gmin( x_beg+B , n );

    int j_offset = (block_start_y+blockIdx.y)*B;

    int slice = (B+blockDim.y-1)/blockDim.y;
    extern __shared__ int d_k[];
    #pragma unroll 
    for(int i=(threadIdx.y)*slice;i<(threadIdx.y+1)*slice&&i<n;i++){
        d_k[i]   = ((int*)(d+pitch*k))[i+j_offset];
        // d_k[i+B] = ((int*)(d+pitch*(i+x_beg)))[k];
    }
    __syncthreads();

    // int d_i[n];
    // for(int i=0;i<n;i++)
    //     d_i[i] = ((int*)(d+pitch*i))[i];

    #pragma unroll 
    for(int i=x_beg;i<x_end;i++){
        int* d_i = ((int*)(d+pitch*i));
        __syncthreads();
        int new_d = d_i[k]+d_k[j-j_offset];
        if(d_i[j]>new_d){
            d_i[j]=new_d;
        }
    }
}

void cal(char* d,size_t pitch,int B, int Round, int block_start_x, 
        int block_start_y, int block_width, int block_height) {
    // To calculate B*B elements in the block (b_i, b_j)
    // For each block, it need to compute B times
    for (int k = Round * B; k < (Round +1) * B && k < n; ++k) {
        // To calculate original index of elements in the block (b_i, b_j)
        // For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2

        dim3 dimBlock(1,B);
        dim3 dimGrid(block_height,block_width);
        gpu<<<dimGrid,dimBlock,sizeof(int)*(B)>>>(
            d,pitch,
            block_start_x,
            block_start_y,
            k,n
        );
    }
}

void cpu(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height) {
	int block_end_x = block_start_x + block_height;
	int block_end_y = block_start_y + block_width;

	for (int b_i =  block_start_x; b_i < block_end_x; ++b_i) {
		for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
			// To calculate B*B elements in the block (b_i, b_j)
			// For each block, it need to compute B times
			for (int k = Round * B; k < (Round +1) * B && k < n; ++k) {
				// To calculate original index of elements in the block (b_i, b_j)
				// For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2
				int block_internal_start_x 	= b_i * B;
				int block_internal_end_x 	= (b_i +1) * B;
				int block_internal_start_y  = b_j * B;
				int block_internal_end_y 	= (b_j +1) * B;

				if (block_internal_end_x > n)	block_internal_end_x = n;
				if (block_internal_end_y > n)	block_internal_end_y = n;

				for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
					for (int j = block_internal_start_y; j < block_internal_end_y; ++j) {
						if (d[i*n+k] + d[k*n+j] < d[i*n+j]) {
							d[i*n+j] = d[i*n+k] + d[k*n+j];
                        }
					}
				}
			}
		}
	}
}
