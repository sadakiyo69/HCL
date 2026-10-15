#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h> // Für CPU-Zeitmessung (clock)

// CUDA kernel function to add two vectors (GPU Version)
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

// CPU version
void vectorAddCPU(const float *A, const float *B, float *C, int numElements) {
    for (int i = 0; i < numElements; ++i) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // Initalize event variables
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    long int numElements = 500000000;
    size_t size = numElements * sizeof(float);
    
    printf("Vector addition of %ld elements\n", numElements);
    
    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C_GPU = (float *)malloc(size); // GPU result
    float *h_C_CPU = (float *)malloc(size); // CPU result
    
    // Initialize host arrays with random data
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    // Allocate device memory
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);
    
    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // Launch the CUDA kernel
    int threadsPerBlock = 1024;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    // Measure execution time GPU (Start)
    cudaEventRecord(start);

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    
    // Measure execution time GPU (Stop)
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_milliseconds = 0;
    cudaEventElapsedTime(&gpu_milliseconds, start, stop);

    // Check for errors in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch kernel: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Copy result back to host
    cudaMemcpy(h_C_GPU, d_C, size, cudaMemcpyDeviceToHost);
    
    // Measure execution time CPU
    clock_t cpu_start = clock();
    
    vectorAddCPU(h_A, h_B, h_C_CPU, numElements);
    
    clock_t cpu_stop = clock();
    double cpu_milliseconds = ((double)(cpu_stop - cpu_start) / CLOCKS_PER_SEC) * 1000.0;

    // Verify the result (Vergleich GPU vs CPU)
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_C_CPU[i] - h_C_GPU[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("Test PASSED (GPU and CPU results match)\n");
    
    // Print timnes
    printf("GPU Execution Time: %f ms\n", gpu_milliseconds);
    printf("CPU Execution Time: %f ms\n", cpu_milliseconds);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C_GPU);
    free(h_C_CPU);
    
    // Destroy Events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("Done\n");
    return 0;
}
