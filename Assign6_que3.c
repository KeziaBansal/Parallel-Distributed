#include <stdio.h>
#include <cuda_runtime.h>

#define TPB 256                       

__global__ void vecSqrt(const float *A, float *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = sqrtf(A[i]);
}

float run_once(int N)
{
    size_t bytes = (size_t)N * sizeof(float);

    float *hA = (float*)malloc(bytes);
    for (int i=0;i<N;++i) hA[i] = (float)i;

    /* device buffers */
    float *dA, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dC, bytes);
    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);

    /* create events for timing */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);  cudaEventCreate(&stop);

    /* launch kernel */
    int blocks = (N + TPB - 1) / TPB;
    cudaEventRecord(start);
    vecSqrt<<<blocks, TPB>>>(dA, dC, N);
    cudaEventRecord(stop);

    /* wait & read elapsed time */
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    /* cleanup */
    cudaFree(dA);  cudaFree(dC);
    free(hA);
    cudaEventDestroy(start);  cudaEventDestroy(stop);

    return ms;   /* milliseconds */
}

int main(void)
{
    const int Ns[] = { 50000, 500000, 5000000, 50000000 };

    printf("N\tTime(ms)\n");
    for (int k=0;k<4;++k)
    {
        float t = run_once(Ns[k]);
        printf("%d\t%.3f\n", Ns[k], t);
    }
    return 0;
}
