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
__global__ void block_label      (unsigned int *g_labels, const unsigned char *g_image);
__global__ void x_label_reduction(unsigned int *g_labels, unsigned char *g_image);
__global__ void y_label_reduction(unsigned int *g_labels, unsigned char *g_image);
__global__ void z_label_reduction(unsigned int *g_labels, unsigned char *g_image);

// Image Size (Constants)
__constant__ unsigned int cX, cY, cZ, cXYZ;
__constant__ unsigned int pX, pY;

// Shift (Constants)
__constant__ unsigned int sX, sY, sZ;
__constant__ unsigned int mX, mY, mZ;

// Block Size - (Block X should always be 32 for __shfl to work correctly)
#define BLOCK_X 32
#define BLOCK_Y 4
#define BLOCK_Z 2

// Remove if BLOCK dimensions are not power-of-two
#define BLOCK_IS_POW2

// Main Method
int main(int argc,char **argv) {
	// Check Arguments
	if(argc < 3) {
		printf("Usage: ./playne_equivalence_block_3D_clamp <gpu-device> <file0> <file1> ...\n");
		exit(1);
	}

	// Check block size
	assert(BLOCK_X == 32);

	// Initialise device
	cudaSetDevice(atoi(argv[1]));

	// CUDA Streams
	cudaStream_t stream1, stream2, stream3;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);

	// For each input
	for(int f = 2; f < argc; f++) {
		// Read Data from file
		unsigned int X, Y, Z;
		unsigned char *h_image = readPG3D(argv[f], X, Y, Z);

		// Calculate Data Mean
		unsigned char image_mean = mean(h_image, X*Y*Z);

		// Convert Data to Binary
		threshold(h_image, image_mean, X*Y*Z);
	
		// Number of Voxels
		unsigned int XYZ = X*Y*Z;

		// Calculate Pitch
		unsigned int PX = X;
		unsigned int PY = X*Y;

		// Block Label - Block, Grid
		dim3 block(min(BLOCK_X, X), BLOCK_Y, BLOCK_Z);
		dim3 grid(ceil(X/(float)block.x), ceil(Y/(float)block.y), ceil(Z/(float)block.z)); 

		// Label Reduction X - Block, Grid
		dim3 block_x(min(BLOCK_X, X), BLOCK_Y, BLOCK_Z);
		dim3 grid_x(ceil(Z/(float)block_x.x), Y/block_x.y, ceil((grid.x)/(float)block_x.z));

		// Label Reduction Y - Block, Grid
		dim3 block_y(min(BLOCK_X, X), BLOCK_Y, BLOCK_Z);
		dim3 grid_y(ceil(X/(float)block_y.x), Z/block_y.y, ceil((grid.y)/(float)block_y.z));

		// Label Reduction Z - Block, Grid
		dim3 block_z(min(BLOCK_X, X), BLOCK_Y, BLOCK_Z);
		dim3 grid_z(ceil(X/(float)block_z.x), Y/block_z.y, ceil((grid.z)/(float)block_z.z));

		// Calculate shift constants
		unsigned int SX = 0;
		unsigned int SY = (int)(log2((float)block.x));
		unsigned int SZ = (int)(log2((float)block.x) + log2((float)block.y));
		unsigned int MX = (block.x-1);
		unsigned int MY = (block.y-1);
		unsigned int MZ = (block.z-1);

		#ifdef BLOCK_IS_POW2
			// Check shift constants
			assert(pow(2, SY) == block.x);
			assert(pow(2, SZ) == (block.x*block.y));

			assert(pow(2, (int)(log2((float)block.x))) == block.x);
			assert(pow(2, (int)(log2((float)block.y))) == block.y);
			assert(pow(2, (int)(log2((float)block.z))) == block.z);
		#endif

		// Allocate host memory
		unsigned int  *h_labels = (unsigned int*)malloc(X*Y*Z*sizeof(unsigned int));
		unsigned int  *d_labels;
		unsigned char *d_image;
		bool          *d_changed;
		
		// Allocate device memory
		cudaMalloc((void**) &d_labels, X * Y * Z * sizeof(unsigned int));
		cudaMalloc((void**) &d_image,  X * Y * Z * sizeof(unsigned char));
		cudaMalloc((void**) &d_changed,            sizeof(bool));
		
		// Initialise constant memory
		cudaMemcpyToSymbol(cX,   &X,   sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(cY,   &Y,   sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(cZ,   &Z,   sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(cXYZ, &XYZ, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(pX,   &PX,  sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(pY,   &PY,  sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(sX,   &SX,  sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(sY,   &SY,  sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(sZ,   &SZ,  sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(mX,   &MX,  sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(mY,   &MY,  sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(mZ,   &MZ,  sizeof(unsigned int), 0, cudaMemcpyHostToDevice);

		// Copy image to device memory
		cudaMemcpy(d_image, h_image, X*Y*Z*sizeof(unsigned char), cudaMemcpyHostToDevice);

		// Arrays to keep track of times, iterations and critical points
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
			block_label <<< grid, block, block.x*block.y*block.z*sizeof(unsigned int) >>>(d_labels, d_image);
			cudaDeviceSynchronize();

			// Label Reduction
			z_label_reduction <<< grid_z, block_z, 0, stream3 >>> (d_labels, d_image);
			y_label_reduction <<< grid_y, block_y, 0, stream2 >>> (d_labels, d_image);
			x_label_reduction <<< grid_x, block_x, 0, stream1 >>> (d_labels, d_image);
			cudaDeviceSynchronize();

			// Analysis
			resolve_labels <<< grid, block >>>(d_labels);
			cudaDeviceSynchronize();

			// Record end event
			cudaEventRecord(time_event[1]);
			cudaDeviceSynchronize();

			// Calculate Elapsed Time
			cudaEventElapsedTime(&times[i], time_event[0], time_event[1]);
		}

		// Copy labels back to host
		cudaMemcpy(h_labels, d_labels, X*Y*Z*sizeof(unsigned int), cudaMemcpyDeviceToHost);

		// Print Number of Components
		printf("Number of Components (%s): %u\n", argv[f], count_components(h_labels, X*Y*Z));

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
	cudaStreamDestroy(stream3);
}

//------------------------------------------------------------------------------------------------------------------------
// Device Functions
//------------------------------------------------------------------------------------------------------------------------

// Resolve Kernel
__global__ void resolve_labels(unsigned int *g_labels) {
	// Calculate index
	const unsigned int id = ((blockIdx.z * blockDim.z) + threadIdx.z) * pY +
							((blockIdx.y * blockDim.y) + threadIdx.y) * pX +
							((blockIdx.x * blockDim.x) + threadIdx.x);

	// Check Range
	if(id < cXYZ) {
		// Resolve Label
		g_labels[id] = find_root(g_labels, g_labels[id]);
	}
}

// Label the block using the Playne method
__global__ void block_label(unsigned int *g_labels, const unsigned char *g_image) {
	// Shared Memory Label Cache
	extern __shared__ unsigned int s_labels[];

	// Calculate the index inside the grid
	const unsigned int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int iy = (blockIdx.y * blockDim.y) + threadIdx.y;
	const unsigned int iz = (blockIdx.z * blockDim.z) + threadIdx.z;

	// Check Range
	if((ix < cX) && (iy < cY) && (iz < cZ)) {
		// Calculate the index inside the block
		const unsigned int bx = threadIdx.x;
		const unsigned int by = threadIdx.y;
		const unsigned int bz = threadIdx.z;

		// Block Dimensions
		const unsigned int bX = blockDim.x;
		const unsigned int bY = blockDim.x * blockDim.y;

		// Calculate neighbouring indexes
		const unsigned int xm1 = bx-1;
		const unsigned int ym1 = by-1;
		const unsigned int zm1 = bz-1;

		// ----------------------------------------
		// Global Memory - Neighbour Connetions
		// ----------------------------------------
		// Load pixel
		const unsigned char pzyx = __ldg(&g_image[iz*pY + iy*pX + ix]);

		// Load pixels
		const unsigned char pzym1x = (by > 0) ? __ldg(&g_image[ iz   *pY + (iy-1)*pX + ix]) : 0;
		const unsigned char pzm1yx = (bz > 0) ? __ldg(&g_image[(iz-1)*pY +  iy   *pX + ix]) : 0;
		
		// Shuffle pixels
		const unsigned char pzyxm1   = __shfl_up_sync(0xffffffff, pzyx,   1);
		const unsigned char pzym1xm1 = __shfl_up_sync(0xffffffff, pzym1x, 1);
		const unsigned char pzm1yxm1 = __shfl_up_sync(0xffffffff, pzm1yx, 1);

		// Neighbour Connections
		const bool nzm1yx = (bz > 0) && (pzyx == pzm1yx);
		const bool nzym1x = (by > 0) && (pzyx == pzym1x);
		const bool nzyxm1 = (bx > 0) && (pzyx == pzyxm1);

		const bool nzym1xm1 = ((by > 0) && (bx > 0) && (pzyx == pzym1xm1));
		const bool nzm1yxm1 = ((bz > 0) && (bx > 0) && (pzyx == pzm1yxm1));
		const bool nzm1ym1x = ((bz > 0) && (by > 0) && (pzyx == __ldg(&g_image[(iz-1)*pY + (iy-1)*pX + ix])));

		// Label
		unsigned int label;//, next;

		// ---------- Initialisation ----------
		label = (nzyxm1) ? ( bz*bY +  by*bX + xm1) : (bz*bY + by*bX + bx);
		label = (nzym1x) ? ( bz*bY + ym1*bX +  bx) : label;
		label = (nzm1yx) ? (zm1*bY +  by*bX +  bx) : label;

		// Write label to shared memory
		s_labels[bz*bY + by*bX + bx] = label;

		// Synchronise Threads
		__syncthreads();
		
		// ---------- Analysis ----------
		// Resolve Label
		s_labels[bz*bY + by*bX + bx] = find_root(s_labels, label);

		// Synchronise Threads
		__syncthreads();

		// ---------- Reduction ----------
		// Check critical - y-neighbour
		if(nzym1x && nzm1yx && !nzm1ym1x) {
			// Reduction
			reduction(s_labels, label, s_labels[bz*bY + ym1*bX + bx]);
		}

		// Check critical - x-neighbour
		if(nzyxm1 && ((nzm1yx && !nzm1yxm1) || (nzym1x && !nzym1xm1))) {
			// Reduction
			label = reduction(s_labels, label, s_labels[bz*bY + by*bX + xm1]);
		}

		// Synchronise Threads
		__syncthreads();

		// ---------- Analysis ----------
		// Get Label
		label = s_labels[bz*bY + by*bX + bx];

		// Resolve Label
		label = find_root(s_labels, label);

		#ifdef BLOCK_IS_POW2
			// Extract label parts
			const unsigned int lx =  label        & mX;
			const unsigned int ly = (label >> sY) & mY;
			const unsigned int lz = (label >> sZ) & mZ;
		#else
			// Extract label parts
			const unsigned int lx =  label       % blockDim.x;
			const unsigned int ly = (label / bX) % blockDim.y;
			const unsigned int lz = (label / bY) % blockDim.z;
		#endif
		
		// Write to global label
		g_labels[iz*pY + iy*pX + ix] = ((blockIdx.z * blockDim.z) + lz) * pY +
		                               ((blockIdx.y * blockDim.y) + ly) * pX +
		                               ((blockIdx.x * blockDim.x) + lx);
	}
}

// X Label - Reduction
__global__ void x_label_reduction(unsigned int *g_labels, unsigned char *g_image) {
	// Calculate Index
	const unsigned int ix = ((blockIdx.z * blockDim.z) + threadIdx.z) * BLOCK_X + BLOCK_X;
	const unsigned int iy = ((blockIdx.y * blockDim.y) + threadIdx.y);
	const unsigned int iz = ((blockIdx.x * blockDim.x) + threadIdx.x);

	// Check Range
	if((ix < cX) && (iy < cY) && (iz < cZ)) {
		// Block Borders - X
		#ifdef BLOCK_IS_POW2
			const bool thread_y = (iy & mY) == 0;
			const bool thread_z = (iz & mZ) == 0;
		#else
			const bool thread_y = (iy % BLOCK_Y) == 0;
			const bool thread_z = (iz % BLOCK_Z) == 0;
		#endif

		// Load image
		const unsigned char pzyx = g_image[iz*pY + iy*pX + ix];

		// Load neighbours
		const unsigned char pzyxm1 = (g_image[ iz   *pY +  iy   *pX + ix-1]);

		// Shuffle neighours
		const unsigned char pzm1yx   = __shfl_up_sync(0xffffffff, pzyx,   1);
		const unsigned char pzm1yxm1 = __shfl_up_sync(0xffffffff, pzyxm1, 1);

		// Connected to neighbour
		if(pzyx == pzyxm1) {
			// Load neighbours
			const bool nzym1x   = (!thread_y) ? (pzyx == g_image[ iz   *pY + (iy-1)*pX + ix  ]) : false;
			const bool nzym1xm1 = (!thread_y) ? (pzyx == g_image[ iz   *pY + (iy-1)*pX + ix-1]) : false;

			// Check Critical
			if((thread_z || (pzyx != pzm1yx) || (pzyx != pzm1yxm1)) && (!nzym1x || !nzym1xm1)) {
				// Get labels
				unsigned int label1 = g_labels[ iz   *pY +  iy   *pX + ix  ];
				unsigned int label2 = g_labels[ iz   *pY +  iy   *pX + ix-1];

				// Reduction
				reduction(g_labels, label1, label2);
			}
		}
	}
}

// Y Label - Reduction
__global__ void y_label_reduction(unsigned int *g_labels, unsigned char *g_image) {
	// Calculate Index
	const unsigned int ix = ((blockIdx.x * blockDim.x) + threadIdx.x);
	const unsigned int iy = ((blockIdx.z * blockDim.z) + threadIdx.z) * BLOCK_Y + BLOCK_Y;
	const unsigned int iz = ((blockIdx.y * blockDim.y) + threadIdx.y);

	// Check Range
	if((ix < cX) && (iy < cY) && (iz < cZ)) {
		// Block Borders - Y
		#ifdef BLOCK_IS_POW2
			const bool thread_x = (ix & mX) == 0;
			const bool thread_z = (iz & mZ) == 0;
		#else
			const bool thread_x = (ix % BLOCK_X) == 0;
			const bool thread_z = (iz % BLOCK_Z) == 0;
		#endif

		// Load image
		const unsigned char pzyx     = g_image[iz*pY +  iy   *pX + ix];

		// Load neighbours
		const unsigned char pzym1x   = g_image[iz*pY + (iy-1)*pX + ix];
		
		// Shuffle neighbours
		const unsigned char pzyxm1   = __shfl_up_sync(0xffffffff, pzyx,   1);
		const unsigned char pzym1xm1 = __shfl_up_sync(0xffffffff, pzym1x, 1);

		// Connected to neighbour
		if(pzyx == pzym1x) {
			// Load neighbours
			const bool nzm1yx   = (!thread_z) ? (pzyx == g_image[(iz-1)*pY +  iy   *pX +  ix   ]) : false;
			const bool nzm1ym1x = (!thread_z) ? (pzyx == g_image[(iz-1)*pY + (iy-1)*pX +  ix   ]) : false;

			// Check Critical
			if((!nzm1yx || !nzm1ym1x) && (thread_x || (pzyx != pzyxm1) || (pzyx != pzym1xm1))) {
				// Get labels
				unsigned int label1 = g_labels[iz*pY +  iy   *pX + ix];
				unsigned int label2 = g_labels[iz*pY + (iy-1)*pX + ix];

				// Reduction
				reduction(g_labels, label1, label2);
			}
		}
	}
}

// Z Label
__global__ void z_label_reduction(unsigned int *g_labels, unsigned char *g_image) {
	// Calculate Index
	const unsigned int ix = ((blockIdx.x * blockDim.x) + threadIdx.x);
	const unsigned int iy = ((blockIdx.y * blockDim.y) + threadIdx.y);
	const unsigned int iz = ((blockIdx.z * blockDim.z) + threadIdx.z) * BLOCK_Z + BLOCK_Z;

	// Check Range
	if((ix < cX) && (iy < cY) && (iz < cZ)) {
		// Block Borders - Z
		#ifdef BLOCK_IS_POW2
			const bool thread_x = (ix & mX) == 0;
			const bool thread_y = (iy & mY) == 0;
		#else
			const bool thread_x = (ix % BLOCK_X) == 0;
			const bool thread_y = (iy % BLOCK_Y) == 0;
		#endif

		// Load image
		const unsigned char pzyx     = g_image[ iz   *pY + iy*pX + ix];

		// Load neighbours
		const unsigned char pzm1yx   = g_image[(iz-1)*pY + iy*pX + ix];

		// Shuffle neighbours
		const unsigned char pzyxm1   = __shfl_up_sync(0xffffffff, pzyx,   1);
		const unsigned char pzm1yxm1 = __shfl_up_sync(0xffffffff, pzm1yx, 1);

		// Connected to neighbour
		if(pzyx == pzm1yx) {
			// Load neighbours
			const bool nzym1x   = (!thread_y) ? (pzyx == g_image[ iz   *pY + (iy-1)*pX + ix]) : false;
			const bool nzm1ym1x = (!thread_y) ? (pzyx == g_image[(iz-1)*pY + (iy-1)*pX + ix]) : false;

			// Check Critical
			if((thread_x || (pzyx != pzyxm1) || (pzyx != pzm1yxm1)) && (!nzym1x || !nzm1ym1x)) {
				// Get labels
				unsigned int label1 = g_labels[ iz   *pY +  iy   *pX + ix];
				unsigned int label2 = g_labels[(iz-1)*pY +  iy   *pX + ix];

				// Reduction
				reduction(g_labels, label1, label2);
			}
		}
	}
}
