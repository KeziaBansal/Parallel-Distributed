#include <stdio.h>
#include <cuda_runtime.h>

#define N   (1 << 20)      
#define TPB 256            // threads‑per‑block

__global__ void vecSqrt(const float *A, float *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)               
        C[i] = sqrtf(A[i]);
}

int main(void)
{
    size_t bytes = N * sizeof(float);
    float *hA = (float*)malloc(bytes);
    float *hC = (float*)malloc(bytes);

    for (int i = 0; i < N; ++i)
        hA[i] = (float)i;

    float *dA, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dC, bytes);

    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);

    int blocks = (N + TPB - 1) / TPB;   // ceil(N/TPB)
    vecSqrt<<<blocks, TPB>>>(dA, dC, N);

    cudaDeviceSynchronize();            

    cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);

    printf("First 10 results:\n");
    for (int i = 0; i < 10; ++i)
        printf("sqrt(%d) = %.4f\n", i, hC[i]);

    cudaFree(dA);  cudaFree(dC);
    free(hA);      free(hC);

    return 0;
}
