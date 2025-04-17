#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024  // Vector size
#define THREADS_PER_BLOCK 256

// global arrays
__device__ __managed__ int A[N];
__device__ __managed__ int B[N];
__device__ __managed__ int C[N];

// Kernel for vector addition
__global__ void vectorAdd() {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Fill input vectors
    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = 2 * i;
    }

    // Launch kernel and time it
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaEventRecord(start, 0);
    vectorAdd<<<blocks, THREADS_PER_BLOCK>>>();
    cudaEventRecord(stop, 0);

    // Wait and compute elapsed time
    cudaEventSynchronize(stop);
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    // Display some results
    printf("Sample result: C[10] = %d\n", C[10]);

    // Query device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    float memClockMHz = (float)prop.memoryClockRate;  // kHz
    int memBusWidthBits = prop.memoryBusWidth;        // bits

    // Theoretical BW = 2 * memClock * memBusWidth
    float theoreticalBW = 2.0 * memClockMHz * memBusWidthBits / 8.0 / 1e6; // GB/s
    printf("Theoretical Bandwidth = %.2f GB/s\n", theoreticalBW);

    // Measured bandwidth
    float elapsed_sec = elapsed_ms / 1000.0f;
    int totalBytes = 3 * N * sizeof(int);  // A, B read + C write
    float measuredBW = (float)totalBytes / elapsed_sec / (1 << 30); // GB/s
    printf("Measured Bandwidth = %.6f GB/s\n", measuredBW);
  
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
