
#include <stdio.h>
#include <stdlib.h>

// Number of runs to demonstrate the race condition
#define NUM_RUNS 5

// CUDA kernel: count elements of A greater than a defined threshold
__global__ void countAboveThreshold(const float *A, int *count, float threshold,
                           int numElements) {
    // Get the unique thread ID, which is the index in the vector
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // Make sure we don't go out of bounds & count elements greater than threshold
    if (i < numElements && A[i] > threshold) {
        // (*count)++;
        atomicAdd(count, 1);

    }
}

int main() {
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    float threshold = 0.5f;

    printf("Counting elements > %.2f in a vector of %d elements\n\n",
           threshold, numElements);

    // Allocate and initialise host memory
    float *h_A = (float *)malloc(size);
    if (!h_A) {
        fprintf(stderr, "Host allocation failed\n");
        return EXIT_FAILURE;
    }
    srand(42);
    // Fill the array with random values between 0 and 1
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
    }

    // CPU reference result
    int cpuCount = 0;
    for (int i = 0; i < numElements; ++i) {
        if (h_A[i] > threshold) cpuCount++;
    }

    // Allocate device memory and copy input
    float *d_A = NULL;
    int *d_count = NULL;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_count, sizeof(int));
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n",
           blocksPerGrid, threadsPerBlock);
    printf("CPU reference count: %d\n\n", cpuCount);

    int allPassed = 1;

    // Run several times to notice the fact that race conditions give different results at each run
    for (int run = 1; run <= NUM_RUNS; ++run) {
        cudaMemset(d_count, 0, sizeof(int));

        countAboveThreshold<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_count,
                                                       threshold, numElements);
        cudaDeviceSynchronize();

        int gpuCount = 0;
        cudaMemcpy(&gpuCount, d_count, sizeof(int), cudaMemcpyDeviceToHost);

        int ok = (gpuCount == cpuCount);
        printf("Run %d: GPU count = %6d (%s)\n", run, gpuCount,
               ok ? "ok" : "WRONG");
        if (!ok) allPassed = 0;
    }

    printf("\n%s\n", allPassed ? "PASSED" : "FAILED: a race condition is present!");

    // Free memory
    cudaFree(d_A);
    cudaFree(d_count);
    free(h_A);

    return allPassed ? EXIT_SUCCESS : EXIT_FAILURE;
}
