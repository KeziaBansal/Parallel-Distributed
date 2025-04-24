#include <stdio.h>
#include <cuda_runtime.h>

#define N  (1 << 24)        
#define TPB 256             // threads per block

__global__ void vecAdd(const float *A, const float *B, float *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

__global__ void vecMul(const float *A, const float *B, float *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] * B[i];
}

__global__ void vecSqrt(const float *A, float *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = sqrtf(A[i]);
}


float timeKernel(void(*kernel)(const float*, const float*, float*, int),
                 const float *A, const float *B, float *C, int n)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);  cudaEventCreate(&stop);

    int blocks = (n + TPB - 1) / TPB;
    cudaEventRecord(start);
    kernel<<<blocks, TPB>>>(A, B, C, n);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);  cudaEventDestroy(stop);
    return ms;
}

float timeKernelSqrt(void(*kernel)(const float*, float*, int),
                     const float *A, float *C, int n)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);  cudaEventCreate(&stop);

    int blocks = (n + TPB - 1) / TPB;
    cudaEventRecord(start);
    kernel<<<blocks, TPB>>>(A, C, n);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);  cudaEventDestroy(stop);
    return ms;
}

int main()
{
    printf("Vector length: %d (%.1f MiB per vector)\n", N, N*sizeof(float)/1048576.0);

    // Allocate host memory
    float *hA = (float*)malloc(N*sizeof(float));
    float *hB = (float*)malloc(N*sizeof(float));

    // Init host data
    for (int i=0;i<N;++i){
        hA[i] = 1.0f*i;          // some non‑trivial numbers
        hB[i] = 0.5f*i + 1.0f;
    }

    // Allocate device memory
    float *dA, *dB, *dC;
    cudaMalloc(&dA, N*sizeof(float));
    cudaMalloc(&dB, N*sizeof(float));
    cudaMalloc(&dC, N*sizeof(float));

    cudaMemcpy(dA, hA, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, N*sizeof(float), cudaMemcpyHostToDevice);

    vecAdd<<<(N+TPB-1)/TPB, TPB>>>(dA, dB, dC, N);
    cudaDeviceSynchronize();

    float add_ms  = timeKernel(vecAdd, dA, dB, dC, N);
    float mul_ms  = timeKernel(vecMul, dA, dB, dC, N);
    float sqrt_ms = timeKernelSqrt(vecSqrt, dA, dC, N);

    printf("Addition   : %7.3f ms\n", add_ms);
    printf("Multiplication: %7.3f ms\n", mul_ms);
    printf("Square root: %7.3f ms\n", sqrt_ms);

    // Cleanup
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB);
    return 0;
}
