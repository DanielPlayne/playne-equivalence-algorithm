#ifndef CCL_UTILS_H
#define CCL_UTILS_H

// ---------- Read image from PGM file ----------
unsigned char* readPGM(const char *name, unsigned int &X, unsigned int &Y) {
	// Open File
	FILE *input = fopen(name, "rb");

	// Check file
	if(!input) {
		// Error Message
		printf("Error: Cannot open %s\n", name);
		exit(1);
	}

	// Consume Header
	unsigned char c = getc(input);
	while(c != '\n') {
		c = getc(input);
	}
	
	// Consume Contents
	c = getc(input);
	if(c == '#') {
		while(c != '\n') {
			c = getc(input);
		}
	} else {
		ungetc(c, input);
	}
	
	// Read Dimensions
	int result = fscanf(input, "%i %i\n", &X, &Y);
	
	// Consume line
	c = getc(input);
	while(c != '\n') {
		c = getc(input);
	}
	
	// Allocate Memory
	unsigned char *image = new unsigned char[X*Y];

	// Read Data
	for(int k = 0; k < X*Y; k++) {
		// Read Pixel
		image[k] = getc(input);
	}
	
	// Close File
	fclose(input);

	// Return Data
	return image;
}


// ---------- Read image from PG3D file ----------
unsigned char* readPG3D(const char *name, unsigned int &X, unsigned int &Y, unsigned int &Z) {
	// Open File
	FILE *input = fopen(name,"rb");

	// Check File
	if(!input) {
		// Error Message
		printf("Error: Cannot open %s\n", name);
		exit(1);
	}

	// Consume Header
	unsigned char c = getc(input);
	while(c != '\n') {
		c = getc(input);
	}
	
	// Consume Comments
	c = getc(input);
	if(c == '#') {
		while(c != '\n') {
			c = getc(input);
		}
	} else {
		ungetc(c, input);
	}
	
	// Read Dimensions
	int result = fscanf(input, "%i %i %i\n", &X, &Y, &Z);
	
	// Consume line
	c = getc(input);
	while(c != '\n') {
		c = getc(input);
	}
	
	// Allocate Memory
	unsigned char *image = new unsigned char[X*Y*Z];

	// Read Data
	for(int k = 0; k < X*Y*Z; k++) {
		// Read Voxel
		image[k] = getc(input);
	}

	// Close File
	fclose(input);

	// Return Data
	return image;
}


// ---------- Converts Data to Binary (0,255) based on threshold value ----------
void threshold(unsigned char *image, unsigned char value, const unsigned int K) {
	for(int k = 0; k < K; k++) {
		image[k] = (image[k] <= value) ? 0 : 255;
	}
}


// ---------- CPU - count components ----------
unsigned int count_components(const unsigned int *labels, const unsigned int K) {
	unsigned int count = 0;
	for(int k = 0; k < K; k++) {
		if(labels[k] == k) {
			count++;
		}
	}
	return count;
}


// ---------- Calculate time from (start - end) ----------
double gettime_seconds(timeval start, timeval end) {
	return ((end.tv_sec-start.tv_sec) + (end.tv_usec-start.tv_usec)/1000000.0);
}


// ---------- Calculate Mean ----------
template <class T>
T mean(const T *a, const int N) {
	// Mean
	T mean;
	mean = T();

	// Calculate Mean
	for(int i = 0; i < N; i++) {
		mean += a[i];
	}
	mean /= N;

	// Return Mean
	return mean;
}


// ---------- Calculate Standard Deviation ----------
template <class T>
T sd(const T *a, const int N) {
	// Mean & SD
	T mean, sd;
	mean = sd = T();

	// Calculate Mean
	for(int i = 0; i < N; i++) {
		mean += a[i];
	}
	mean /= N;

	// Calculate SD
	for(int i = 0; i < N; i++) {
		sd += (a[i] - mean) * (a[i] - mean);
	}
	sd /= N;
	sd = sqrt(sd);

	// Return SD
	return sd;
}


// ---------- Calculate Mean and Standard Deviation ----------
template <class T>
void print_mean_sd(const T *a, const int N, const unsigned int X) {
	// Mean & SD
	T mean, sd;
	mean = sd = T();

	// Calculate Mean
	for(int i = 0; i < N; i++) {
		mean += a[i];
	}
	mean /= N;

	// Calculate SD
	for(int i = 0; i < N; i++) {
		sd += (a[i] - mean) * (a[i] - mean);
	}
	sd /= N;
	sd = sqrt(sd);

	// Output
	printf("Time (ms): %i %0.16f %0.16f\n", X, mean, sd);
}

#endif // CCL_UTILS_H