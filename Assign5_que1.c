#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024  // Vector size

// CUDA kernel for vector addition
__global__ void vectorAdd(int *A, int *B, int *C) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int A[N], B[N], C[N];
    int *d_A, *d_B, *d_C;

    // Fill host arrays
    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = i * 2;
    }

    // Allocate device memory
    cudaMalloc((void **)&d_A, N * sizeof(int));
    cudaMalloc((void **)&d_B, N * sizeof(int));
    cudaMalloc((void **)&d_C, N * sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_A, A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(int), cudaMemcpyHostToDevice);

    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start time
    cudaEventRecord(start, 0);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);

    // Record end time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop); // in ms

    // Copy result back to host
    cudaMemcpy(C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print a sample result
    printf("Sample result: C[10] = %d\n", C[10]);

    // Device property query
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    float memClock = (float)prop.memoryClockRate;  
    int busWidth = prop.memoryBusWidth;            // in bits

    float theoreticalBW = 2.0 * memClock * busWidth / 8 / 1e6; // in GB/s
    printf("Theoretical Bandwidth: %.2f GB/s\n", theoreticalBW);

    // Measured bandwidth
    int totalBytes = 3 * N * sizeof(int); // A and B read, C written
    float measuredBW = totalBytes / (elapsedTime / 1000.0f) / (1 << 30);
    printf("Measured Bandwidth: %.6f GB/s\n", measuredBW);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
