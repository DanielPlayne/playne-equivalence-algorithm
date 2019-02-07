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
__constant__ unsigned int cX, cY, cXY;

// Main Function
int main(int argc,char **argv) {
	// Check Arguments
	if(argc < 3) {
		printf("Usage: ./playne_equivalence_direct_2D_clamp <gpu> <file0> <file1> ...\n");
		exit(1);
	}

	// Initialise device
	cudaSetDevice(atoi(argv[1]));

	// For each input file
	for(int f = 2; f < argc; f++) {
		// Read Image from file
		unsigned int X, Y;
		unsigned char *h_image = readPGM(argv[f], X, Y);

		// Calculate Image Mean
		unsigned char image_mean = mean(h_image, X*Y);

		// Convert to Binary Image based on Threshold
		threshold(h_image, image_mean, X*Y);
		
		// Number of Pixels
		unsigned int XY = X*Y;
		
		// Allocate host memory
		unsigned int  *h_labels = (unsigned int*)malloc(Y * X * sizeof(unsigned int));
		unsigned int  *d_labels;
		unsigned char *d_image;
		bool          *d_changed;
		
		// Allocate device memory
		cudaMalloc((void**) &d_labels, Y * X * sizeof(unsigned int));
		cudaMalloc((void**) &d_image,  Y * X * sizeof(unsigned char));
		cudaMalloc((void**) &d_changed, sizeof(bool));
		
		// Copy host to device memory
		cudaMemcpyToSymbol(cX,  &X,  sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(cY,  &Y,  sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(cXY, &XY, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpy(d_image, h_image, X*Y*sizeof(unsigned char), cudaMemcpyHostToDevice);

		// Create Grid/Block
		dim3 block(32, 4);
		dim3 grid(ceil(X/(float)block.x), ceil(Y/(float)block.y)); 

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

			// Record end event
			cudaEventRecord(time_event[1]);
			cudaDeviceSynchronize();

			// Calculate Elapsed Time
			cudaEventElapsedTime(&times[i], time_event[0], time_event[1]);
		}

		// Copy labels back to host
		cudaMemcpy(h_labels, d_labels, X*Y*sizeof(unsigned int), cudaMemcpyDeviceToHost);

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
}

// ----------------------------------------
// Device Kernels
// ----------------------------------------

// Initialise Kernel
__global__ void init_labels(unsigned int* g_labels, const unsigned char *g_image) {
	// Calculate index
	const unsigned int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int iy = (blockIdx.y * blockDim.y) + threadIdx.y;

	// Check Thread Range
	if((ix < cX) && (iy < cY)) {
		// Fetch three image values
		const unsigned char pyx = g_image[iy*cX + ix];

		// Neighbour Connections
		const bool nym1x =  (iy > 0) ? (pyx == g_image[(iy-1)*cX + ix  ]) : false;
		const bool nyxm1 =  (ix > 0) ? (pyx == g_image[ iy   *cX + ix-1]) : false;

		// Label
		unsigned int label;

		// Initialise Label
		label = (nyxm1) ?  iy   *cX + ix-1 : iy*cX + ix;
		label = (nym1x) ? (iy-1)*cX + ix   : label;

		// Write to Global Memory
		g_labels[iy*cX + ix] = label;
	}
}

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

// Label Reduction
__global__ void label_reduction(unsigned int *g_labels, unsigned char *g_image) {
	// Calculate index
	const unsigned int iy = ((blockIdx.y * blockDim.y) + threadIdx.y);
	const unsigned int ix = ((blockIdx.x * blockDim.x) + threadIdx.x);

	// Check Thread Range
	if((ix < cX) && (iy < cY)) {
		// Compare Image Values
		const unsigned char pyx = g_image[iy*cX + ix];
		const bool nyxm1 = (ix > 0) ? (pyx == g_image[iy*cX + ix-1]) : false;

		// If connected to neighbour
		if(nyxm1) {
			// Neighbouring values
			const bool nym1xm1 = ((iy > 0) && (ix > 0)) ? (pyx == g_image[(iy-1)*cX + ix-1]) : false;
			const bool nym1x   =  (iy > 0)              ? (pyx == g_image[(iy-1)*cX + ix  ]) : false;

			// Check Critical
			if(nym1x && !nym1xm1) {
				// Get labels
				unsigned int label1 = g_labels[iy*cX + ix  ];
				unsigned int label2 = g_labels[iy*cX + ix-1];

				// Reduction
				reduction(g_labels, label1, label2);
			}
		}
	}
}
