// MIT License

// Copyright (c) 2019 - Daniel Peter Playne

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Headers
#include <stdio.h>
#include <sys/time.h>
#include <assert.h>

// Project Headers
#include "utils.h"
#include "cuda_utils.cuh"
#include "reduction.cuh"

// Device Functions
__global__ void resolve_labels   (unsigned int* g_labels);
__global__ void block_label      (unsigned int *g_labels, unsigned char *g_image);
__global__ void x_label_reduction(unsigned int *g_labels, unsigned char *g_image);
__global__ void y_label_reduction(unsigned int *g_labels, unsigned char *g_image);

// Image Size (Host)
unsigned int X, Y;

// Image Size (Device Constants)
__constant__ unsigned int cX, cY, cXY;

// Shift (Constants)
__constant__ unsigned int sX, sY, sZ;
__constant__ unsigned int mX, mY, mZ;

// Block Size (Block X should always be 32 for __shfl to work correctly)
#define BLOCK_X 32
#define BLOCK_Y 4

// Remove if BLOCK dimensions are not a power-of-two
#define BLOCK_IS_POW2

// Main Method
int main(int argc,char **argv) {
	// Check Arguments
	if(argc < 3) {
		printf("Usage: ./playne_equivalence_block_2D_clamp <gpu-device> <file0> <file1> ...\n");
		exit(1);
	}

	// Check block size
	assert(BLOCK_X == 32);

	// Initialise device
	cudaSetDevice(atoi(argv[1]));

	// CUDA Streams
	cudaStream_t stream1, stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);

	// For each input
	for(int f = 2; f < argc; f++) {
		// Read Image from file
		unsigned int X, Y;
		unsigned char *h_image = readPGM(argv[f], X, Y);

		// Calculate Image Mean
		unsigned char image_mean = mean(h_image, X*Y);

		// Convert Image to Binary
		threshold(h_image, image_mean, X*Y);

		// Number of Pixels
		unsigned int XY = X*Y;
	
		// Create Grid/Block
		dim3 block(BLOCK_X, BLOCK_Y);
		dim3 grid(ceil(X/(float)block.x), ceil(Y/(float)block.y)); 

		// Create Resolve Grid/Block
		dim3 resolve_block(32, 4);
		dim3 resolve_grid(ceil(X/(float)resolve_block.x), ceil(Y/(float)resolve_block.y)); 
		
		// Create Border X/Y Grid
		dim3 border_grid_x(ceil(Y/(float)block.x), ceil(grid.x/(float)block.y));
		dim3 border_grid_y(ceil(X/(float)block.x), ceil(grid.y/(float)block.y));

		// Calculate shift constants
		unsigned int SX = 0;
		unsigned int SY = (int)(log2((float)block.x));
		unsigned int MX = (block.x-1);
		unsigned int MY = (block.y-1);

		// If Block Sizes are a Power-of-Two
		#ifdef BLOCK_IS_POW2
			// Check shift constants
			assert(pow(2, SY) == block.x);
			assert(pow(2, (int)(log2((float)block.x))) == block.x);
			assert(pow(2, (int)(log2((float)block.y))) == block.y);
		#endif

		// Allocate host memory
		unsigned int  *h_labels = (unsigned int*)malloc(Y * X * sizeof(unsigned int));
		unsigned int  *d_labels;
		unsigned char *d_image;
		bool          *d_changed;
		
		// Allocate device memory
		cudaMalloc((void**) &d_labels, Y * X * sizeof(unsigned int));
		cudaMalloc((void**) &d_image,  Y * X * sizeof(unsigned char));
		cudaMalloc((void**) &d_changed,        sizeof(bool));
		
		// Copy host to device memory
		cudaMemcpyToSymbol(cX,  &X,  sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(cY,  &Y,  sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(cXY, &XY, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(sX,  &SX, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(sY,  &SY, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(mX,  &MX, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(mY,  &MY, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);

		cudaMemcpy(d_image, h_image, X*Y*sizeof(unsigned char), cudaMemcpyHostToDevice);

		// Timing
		const int N = 100;
		float *times = new float[N];
		cudaEvent_t time_event[2];
		cudaEventCreate(&time_event[0]);
		cudaEventCreate(&time_event[1]);

		// Run N times
		for(int i = 0; i < N; i++) {
			// Record start time
			cudaEventRecord(time_event[0]);
		
			// Label Blocks
			block_label <<< grid, block, block.x*block.y*sizeof(unsigned int) >>>(d_labels, d_image);
			cudaDeviceSynchronize();

			// Label Reduction
			y_label_reduction <<< border_grid_y, block, 0, stream1 >>> (d_labels, d_image);
			x_label_reduction <<< border_grid_x, block, 0, stream2 >>> (d_labels, d_image);
			cudaDeviceSynchronize();

			// Analysis
			resolve_labels <<< resolve_grid, resolve_block >>>(d_labels);
			
			// Record end event
			cudaEventRecord(time_event[1]);
			cudaDeviceSynchronize();

			// Calculate Elapsed Time
			cudaEventElapsedTime(&times[i], time_event[0], time_event[1]);
		}

		// Copy labels back to host
		cudaMemcpy(h_labels, d_labels, Y*X*sizeof(unsigned int), cudaMemcpyDeviceToHost);

		// Print Number of Components
		printf("Number of Components (%s): %u\n", argv[f], count_components(h_labels, X*Y));

		// Check for any errors
		checkCUDAErrors();

		// Measure the time the algorithm took
		print_mean_sd(times, N, X);

		// Delete memory
		delete[] h_image;
		delete[] h_labels;
		delete[] times;
		cudaFree(d_labels);
		cudaFree(d_image);
		cudaFree(d_changed);

		// Delete Events
		cudaEventDestroy(time_event[0]);
		cudaEventDestroy(time_event[1]);
	}

	// Destroy CUDA Streams
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
}

// ----------------------------------------
// Device Kernels
// ----------------------------------------

// Resolve Kernel
__global__ void resolve_labels(unsigned int *g_labels) {
	// Calculate index
	const unsigned int id = ((blockIdx.y * blockDim.y) + threadIdx.y) * cX +
							((blockIdx.x * blockDim.x) + threadIdx.x);
	
	// Check Thread Range
	if(id < cXY) {
		// Resolve Label
		g_labels[id] = find_root(g_labels, g_labels[id]);
	}
}

// Playne-Equivalence Block-Label method
__global__ void block_label(unsigned int *g_labels, unsigned char *g_image) {
	// Shared Memory Label Cache
	extern __shared__ unsigned int s_labels[];

	// Calculate the index inside the grid
	const unsigned int ix = ((blockIdx.x * blockDim.x) + threadIdx.x);
	const unsigned int iy = ((blockIdx.y * blockDim.y) + threadIdx.y);

	// Check Thread Range
	if((ix < cX) && (iy < cY)) {
		// Calculate the index inside the block
		const unsigned int bx = threadIdx.x;
		const unsigned int by = threadIdx.y;

		// ----------------------------------------
		// Global Memory - Neighbour Connections
		// ----------------------------------------
		// Load pixels
		const unsigned char pyx = g_image[iy*cX + ix];

		// Load neighbour from global memory
		const unsigned char pym1x = (by > 0) ? g_image[(iy-1)*cX + ix] : 0;

		// Shuffle Pixels
		const unsigned char pyxm1   = __shfl_up_sync(__activemask(), pyx, 1);
		const unsigned char pym1xm1 = __shfl_up_sync(__activemask(), pym1x, 1);

		// Neighbour Connections
		const bool nym1x   = (by > 0)           ? (pyx == pym1x)   : false;
		const bool nyxm1   = (bx > 0)           ? (pyx == pyxm1)   : false;
		const bool nym1xm1 = (by > 0 && bx > 0) ? (pyx == pym1xm1) : false;

		// Label
		unsigned int label1;

		// ---------- Initialisation ----------
		label1 = (nyxm1) ?  by   *blockDim.x + (bx-1) : by*blockDim.x + bx;
		label1 = (nym1x) ? (by-1)*blockDim.x +  bx    : label1;

		// Write label to shared memory
		s_labels[by*blockDim.x + bx] = label1;

		// Synchronise Threads
		__syncthreads();
		
		// ---------- Analysis ----------
		// Resolve Label
		s_labels[by*blockDim.x + bx] = find_root(s_labels, label1);

		// Synchronise Threads
		__syncthreads();

		// ---------- Reduction ----------
		// Check critical
		if(nym1x && nyxm1 && !nym1xm1) {
			// Get neighbouring label
			unsigned int label2 = s_labels[by*blockDim.x + bx-1];
			
			// Reduction
			label1 = reduction(s_labels, label1, label2);
		}

		// Synchronise Threads
		__syncthreads();

		// ---------- Analysis ----------
		// Resolve Label
		label1 = find_root(s_labels, label1);
		
		#ifdef BLOCK_IS_POW2
			// Extract label components
			const unsigned int lx =  label1        & mX;
			const unsigned int ly = (label1 >> sY) & mY;
		#else
			// Extract label components
			const unsigned int lx =  label1               % blockDim.x;
			const unsigned int ly = (label1 / blockDim.x) % blockDim.y;
		#endif

		// Write to Global
		g_labels[iy*cX + ix] = ((blockIdx.y * blockDim.y) + ly) * cX + ((blockIdx.x * blockDim.x) + lx);
	}
}

// X - Reduction
__global__ void x_label_reduction(unsigned int *g_labels, unsigned char *g_image) {
	// Calculate Index
	unsigned int ix = ((blockIdx.y * blockDim.y) + threadIdx.y) * BLOCK_X + BLOCK_X;
	unsigned int iy = ((blockIdx.x * blockDim.x) + threadIdx.x);
	
	// Check Range
	if(ix < cX && iy < cY) {
		// Get image and label values
		const unsigned char pyx   = g_image[iy*cX + ix];

		// Neighbour Values
		const unsigned char pyxm1 = g_image[iy*cX + ix-1];

		// Edge of block flag
		#ifdef BLOCK_IS_POW2
			const bool thread_y = (iy & mY) == 0;
		#else
			const bool thread_y = (iy % BLOCK_Y) == 0;
		#endif

		// Fetch Neighbours
		const unsigned char pym1x   = __shfl_up_sync(__activemask(), pyx,   1);
		const unsigned char pym1xm1 = __shfl_up_sync(__activemask(), pyxm1, 1);

		// If connected to left neighbour
		if((pyx == pyxm1) && (thread_y || (pyx != pym1x) || (pyx != pym1xm1))) {
			// Get Labels
			unsigned int label1 = g_labels[iy*cX + ix  ];
			unsigned int label2 = g_labels[iy*cX + ix-1];

			// Reduction
			reduction(g_labels, label1, label2);
		}
	}
}

// Y - Reduction
__global__ void y_label_reduction(unsigned int *g_labels, unsigned char *g_image) {
	// Calculate Index
	unsigned int ix = ((blockIdx.x * blockDim.x) + threadIdx.x);
	unsigned int iy = ((blockIdx.y * blockDim.y) + threadIdx.y) * BLOCK_Y + BLOCK_Y;

	// Check Range
	if(ix < cX && iy < cY) {
		// Get image and label values
		const unsigned char pyx   = g_image[ iy   *cX + ix];
		const unsigned char pym1x = g_image[(iy-1)*cX + ix];

		// Neighbour Connections
		const unsigned char pyxm1   = __shfl_up_sync(__activemask(), pyx,   1);
		const unsigned char pym1xm1 = __shfl_up_sync(__activemask(), pym1x, 1);

		// If connected to neighbour
		if((pyx == pym1x) && ((threadIdx.x == 0) || (pyx != pyxm1) || (pyx != pym1xm1))) {
			// Get labels
			unsigned int label1 = g_labels[ iy   *cX + ix];
			unsigned int label2 = g_labels[(iy-1)*cX + ix];

			// Reduction
			reduction(g_labels, label1, label2);
		}
	}
}

 