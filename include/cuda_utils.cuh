#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

// ---------- Check for errors in the CUDA runtime ----------
void checkCUDAErrors() {
	// Get Last Error
	cudaError_t error = cudaGetLastError();

	// If not Success
	if(error != cudaSuccess) {
		// Print Error Message
		printf("Error: %s\n", cudaGetErrorString(error));
	}
}

#endif // CUDA_UTILS_H
