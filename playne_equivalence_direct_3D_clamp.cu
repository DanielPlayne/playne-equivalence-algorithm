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

// Project Headers
#include "utils.h"
#include "cuda_utils.cuh"
#include "reduction.cuh"

// Device Functions
__global__ void init_labels    (unsigned int *g_labels, const unsigned char *g_image);
__global__ void resolve_labels (unsigned int *g_labels);
__global__ void label_reduction(unsigned int *g_labels, unsigned char *g_image);

// Image Size (Device Constant)
__constant__ unsigned int cX, cY, cZ, cXYZ;
__constant__ unsigned int pX, pY;

// Main Function
int main(int argc,char **argv) {
	// Check Arguments
	if(argc < 3) {
		printf("Usage: ./playne_equivalence_direct_3D_clamp <gpu-device> <file0> <file1> ...\n");
		exit(1);
	}

	// Initialise device
	cudaSetDevice(atoi(argv[1]));

	// For each input1
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

		// Allocate host memory
		unsigned int  *h_labels = (unsigned int*)malloc(X*Y*Z*sizeof(unsigned int));
		unsigned int  *d_labels;
		unsigned char *d_image;
		bool          *d_changed;
		
		// Allocate device memory
		cudaMalloc((void**) &d_labels, X*Y*Z*sizeof(unsigned int));
		cudaMalloc((void**) &d_image,  X*Y*Z*sizeof(unsigned char));
		cudaMalloc((void**) &d_changed, sizeof(bool));
		
		// Copy host to device memory
		cudaMemcpyToSymbol(cX,   &X,   sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(cY,   &Y,   sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(cZ,   &Z,   sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(cXYZ, &XYZ, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(pX,   &PX,  sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(pY,   &PY,  sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpy(d_image, h_image, X*Y*Z*sizeof(unsigned char), cudaMemcpyHostToDevice);

		// Set block size
		dim3 block(32, 4, 4);
		dim3 grid(ceil(X/(float)block.x), ceil(Y/(float)block.y), ceil(Z/(float)block.z));

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
			
			// Initialise labels
			init_labels <<< grid, block >>>(d_labels, d_image);
			cudaDeviceSynchronize();
			
			// Analysis
			resolve_labels <<< grid, block >>>(d_labels);
			cudaDeviceSynchronize();

			// Label Reduction
			label_reduction <<< grid, block >>>(d_labels, d_image);
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
}

//------------------------------------------------------------------------------------------------------------------------
// Device Functions
//------------------------------------------------------------------------------------------------------------------------

// Initialise Kernel
__global__ void init_labels(unsigned int *g_labels, const unsigned char *g_image) {
	// Calculate index
	const unsigned int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int iy = (blockIdx.y * blockDim.y) + threadIdx.y;
	const unsigned int iz = (blockIdx.z * blockDim.z) + threadIdx.z;

	// Check Range
	if((ix < cX) && (iy < cY) && (iz < cZ)) {
		// Load image
		const unsigned char pzyx = g_image[iz*pY + iy*pX + ix];

		// Neighbour Connections
		const bool nzm1yx   = (iz > 0) ? (pzyx == __ldg(&g_image[(iz-1)*pY +  iy   *pX + ix  ])) : false;
		const bool nzym1x   = (iy > 0) ? (pzyx == __ldg(&g_image[ iz   *pY + (iy-1)*pX + ix  ])) : false;
		const bool nzyxm1   = (ix > 0) ? (pzyx == __ldg(&g_image[ iz   *pY +  iy   *pX + ix-1])) : false;

		// Label
		unsigned int label;

		// Initialise Label
		label = (nzyxm1) ? ( iz   *pY +  iy   *pX + ix-1) : (iz*pY + iy*pX + ix);
		label = (nzym1x) ? ( iz   *pY + (iy-1)*pX + ix  ) : label;
		label = (nzm1yx) ? ((iz-1)*pY +  iy   *pX + ix  ) : label;

		// Write to Global Memory
		g_labels[iz*pY + iy*pX + ix] = label;
	}
}

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


// Label Reduction
__global__ void label_reduction(unsigned int *g_labels, unsigned char *g_image) {
	// Calculate index
	const unsigned int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int iy = (blockIdx.y * blockDim.y) + threadIdx.y;
	const unsigned int iz = (blockIdx.z * blockDim.z) + threadIdx.z;

	// Check Range
	if((ix < cX) && (iy < cY) && (iz < cZ)) {
		// Fetch Image
		const unsigned char pzyx = __ldg(&g_image[iz*pY + iy*pX + ix]);

		// Compare Image Values
		const bool nzm1yx = (iz > 0) ? (pzyx == __ldg(&g_image[(iz-1)*pY +  iy   *pX + ix  ])) : false;
		const bool nzym1x = (iy > 0) ? (pzyx == __ldg(&g_image[ iz   *pY + (iy-1)*pX + ix  ])) : false;
		const bool nzyxm1 = (ix > 0) ? (pzyx == __ldg(&g_image[ iz   *pY +  iy   *pX + ix-1])) : false;

		const bool nzym1xm1 = ((iy > 0) && (ix > 0)) ? (pzyx == __ldg(&g_image[ iz   *pY + (iy-1)*pX + ix-1])) : false;
		const bool nzm1yxm1 = ((iz > 0) && (ix > 0)) ? (pzyx == __ldg(&g_image[(iz-1)*pY +  iy   *pX + ix-1])) : false;
		const bool nzm1ym1x = ((iz > 0) && (iy > 0)) ? (pzyx == __ldg(&g_image[(iz-1)*pY + (iy-1)*pX + ix  ])) : false;

		// Critical point conditions
		const bool cond_x = nzyxm1 && ((nzm1yx && !nzm1yxm1) || (nzym1x && !nzym1xm1));
		const bool cond_y = nzym1x &&   nzm1yx && !nzm1ym1x;

		// Get label
		unsigned int label1 = (cond_x || cond_y) ? g_labels[iz*pY + iy*pX + ix] : 0;

		// Y - Neighbour
		if(cond_y) {
			// Get neighbouring label
			unsigned int label2 = g_labels[iz*pY + (iy-1)*pX + ix];

			// Reduction
			label1 = reduction(g_labels, label1, label2);
		}

		// X - Neighbour
		if(cond_x) {
			// Get neighbouring label
			unsigned int label2 = g_labels[iz*pY + iy*pX + ix-1];

			// Reduction
			label1 = reduction(g_labels, label1, label2);
		}
	}
}