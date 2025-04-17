#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024  // Number of integers

// CUDA Kernel
__global__ void perform_tasks(int *input, int *output) {
    int tid = threadIdx.x;

    if (tid == 0) {
        // Task a: Iterative sum
        int sum = 0;
        for (int i = 0; i < N; i++) {
            sum += input[i];
        }
        output[0] = sum;
    } else if (tid == 1) {
        // Task b: Formula-based sum
        output[1] = N * (N + 1) / 2;
    }
}

int main() {
    int h_input[N];         // Host input array
    int h_output[2];        // Host output array: index 0 = iterative sum, index 1 = formula sum
    int *d_input, *d_output;

    // Step 1: Fill input array with first N integers
    for (int i = 0; i < N; i++) {
        h_input[i] = i + 1;
    }

    // Step 2: Allocate device memory
    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMalloc((void**)&d_output, 2 * sizeof(int));

    // Step 3: Copy data from host to device
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // Step 4: Launch kernel with 1 block and 2 threads
    perform_tasks<<<1, 2>>>(d_input, d_output);

    // Step 5: Copy result from device to host
    cudaMemcpy(h_output, d_output, 2 * sizeof(int), cudaMemcpyDeviceToHost);

    // Step 6: Print the results
    printf("Sum using iterative approach: %d\n", h_output[0]);
    printf("Sum using formula approach: %d\n", h_output[1]);

    // Step 7: Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

