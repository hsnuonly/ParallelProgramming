#include <iostream>
#include <cstdlib>
#include <cassert>
#include <zlib.h>
#include <png.h>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
using namespace std;

#define MASK_N 2
#define MASK_X 5
#define MASK_Y 5
#define SCALE 8
#define NUM_THREAD 512

__constant__ int _mask[MASK_N][MASK_X][MASK_Y] = { 
    {{ -1, -4, -6, -4, -1},
     { -2, -8,-12, -8, -2},
     {  0,  0,  0,  0,  0}, 
     {  2,  8, 12,  8,  2}, 
     {  1,  4,  6,  4,  1}},
    {{ -1, -2,  0,  2,  1}, 
     { -4, -8,  0,  8,  4}, 
     { -6,-12,  0, 12,  6}, 
     { -4, -8,  0,  8,  4}, 
     { -1, -2,  0,  2,  1}} 
};

int read_png(const char* filename, unsigned char** image, unsigned* height, 
			 unsigned* width, unsigned* channels) {

	unsigned char sig[8];
	FILE* infile;
	infile = fopen(filename, "rb");

    fread(sig, 1, 8, infile);
    if (!png_check_sig(sig, 8))
        return 1;   /* bad signature */

	png_structp png_ptr;
	png_infop info_ptr;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
        return 4;   /* out of memory */
  
    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return 4;   /* out of memory */
    }

	png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
	int bit_depth, color_type;
	png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);

	png_uint_32  i, rowbytes;
    png_bytep  row_pointers[*height];
    png_read_update_info(png_ptr, info_ptr);
    rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *channels = (int) png_get_channels(png_ptr, info_ptr);

    if ((*image = (unsigned char *) malloc(rowbytes * *height)) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return 3;
    }

    for (i = 0;  i < *height;  ++i)
        row_pointers[i] = *image + i * rowbytes;
	png_read_image(png_ptr, row_pointers);
	png_read_end(png_ptr, NULL);
	return 0;
}

void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width, 
			   const unsigned channels) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
	png_set_compression_level(png_ptr, 1);

	png_bytep row_ptr[height];
	for (int i = 0; i < height; ++ i) {
		row_ptr[i] = image + i * width * channels * sizeof(unsigned char);
	}
	png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

__global__ void sobel (unsigned char* s, unsigned char* t, unsigned height, unsigned width, unsigned channels, size_t pitch, int y) {
	int  x, i, v, u;
	int  R, G, B;
	double val[MASK_N*3] = {0.0};
    const int adjustX=1, adjustY=1, xBound=2, yBound=2;
    x = (blockIdx.x * NUM_THREAD + threadIdx.x)%width;

    __shared__ int mask[MASK_N][MASK_X][MASK_Y];
    mask[(x/25)%2][(x/5)%5][x%5] = _mask[(x/25)%2][(x/5)%5][x%5];
    __syncthreads();

    for (i = 0; i < MASK_N; ++i) {

        val[i*3+2] = 0.0;
        val[i*3+1] = 0.0;
        val[i*3] = 0.0;
        for (v = -yBound; v < yBound + adjustY; ++v) {
            #pragma unroll 5
            for (u = -xBound; u < xBound + adjustX; ++u) {
                int valid = (x + u) >= 0 && (x + u) < width && y + v >= 0 && y + v < height;
                R = s[((pitch * (y+v) + (x+u)*channels) + 2)*valid]*valid;
                G = s[((pitch * (y+v) + (x+u)*channels) + 1)*valid]*valid;
                B = s[((pitch * (y+v) + (x+u)*channels) + 0)*valid]*valid;
                val[i*3+2] += R * mask[i][u + xBound][v + yBound];
                val[i*3+1] += G * mask[i][u + xBound][v + yBound];
                val[i*3+0] += B * mask[i][u + xBound][v + yBound];
            }
        }
    }

    double totalR = 0.0;
    double totalG = 0.0;
    double totalB = 0.0;
    
    #pragma unroll 5
    for (i = 0; i < MASK_N; ++i) {
        totalR += val[i * 3 + 2] * val[i * 3 + 2];
        totalG += val[i * 3 + 1] * val[i * 3 + 1];
        totalB += val[i * 3 + 0] * val[i * 3 + 0];
    }

    totalR = sqrt(totalR) / SCALE;
    totalG = sqrt(totalG) / SCALE;
    totalB = sqrt(totalB) / SCALE;
    const unsigned char cR = min(255.0,totalR);
    const unsigned char cG = min(255.0,totalG);
    const unsigned char cB = min(255.0,totalB);
    t[(pitch * y + x*channels) + 2] = cR;
    t[(pitch * y + x*channels) + 1] = cG;
    t[(pitch * y + x*channels) + 0] = cB;
}

queue<int> q;
mutex qLock;
mutex cvLock;
condition_variable cv;

void png_writer(int n,png_structp png_ptr,png_bytep* row_ptr,unsigned char* host_t,unsigned char* device_t,size_t pitch,int width,int height,int channels){
    while(n--){
        if(q.empty()){
            unique_lock<mutex> ul(cvLock);
            cv.wait(ul);
        }
        qLock.lock();
        int i = q.front();
        q.pop();
        qLock.unlock();
        
        // cudaMemcpy2D(host_t+sizeof(char)*width*channels*i,sizeof(char)*width*channels,device_t+pitch*i,pitch,width*channels,1,cudaMemcpyDeviceToHost);
        png_write_row(png_ptr,host_t + i * width * channels * sizeof(unsigned char));
    }
}

int main(int argc, char** argv) {

    assert(argc == 3);
	unsigned height, width, channels;
	unsigned char* host_s = NULL;
	read_png(argv[1], &host_s, &height, &width, &channels);
    unsigned char* host_t = (unsigned char*) malloc(height * width * channels * sizeof(unsigned char));

    unsigned char* device_s;
    unsigned char* device_t;
    size_t pitch;
    cudaMallocPitch(&device_s,&pitch,sizeof(unsigned char)*width*channels,height);
    cudaMallocPitch(&device_t,&pitch,sizeof(unsigned char)*width*channels,height);
    FILE* fp = fopen(argv[2], "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
	png_set_compression_level(png_ptr, 1);

	png_bytep* row_ptr = new png_bytep[height];
	for (int i = 0; i < height; ++ i) {
		row_ptr[i] = host_t + i * width * channels * sizeof(unsigned char);
    } 
    
    thread t(png_writer,height,png_ptr,row_ptr,host_t,device_t,pitch,width,height,channels);

    cudaStream_t streams[height];
    cudaMemcpy2D(device_s,pitch,host_s,sizeof(char)*width*channels,width*channels,height,cudaMemcpyHostToDevice);
    for(int i=0;i<height;i++){
        cudaStreamCreate(&streams[i]);
        sobel<<<width/NUM_THREAD+1,NUM_THREAD,sizeof(int)*50,streams[i]>>> (device_s, device_t, height, width, channels, pitch, i);
        cudaMemcpy2DAsync(host_t+sizeof(char)*width*channels*i,sizeof(char)*width*channels,device_t+pitch*i,pitch,width*channels,1,cudaMemcpyDeviceToHost,streams[i]);        // lock_guard<mutex> lg(qLock);
        // q.push(i);
        // cv.notify_all();
    }
    for(int i=0;i<height;i++){
        cudaStreamSynchronize(streams[i]);
        q.push(i);
        cv.notify_all();
    }
    t.join();
    
    cudaFree(device_s);
    cudaFree(device_t);

    
    // png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);

    return 0;
}
