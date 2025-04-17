#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1000  // Size of the array 
#define THREADS_PER_BLOCK 512

// CUDA kernel to merge two sorted subarrays
__global__ void merge_kernel(int *input, int *output, int width, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start = tid * 2 * width;

    if (start >= size) return;

    int mid = min(start + width, size);
    int end = min(start + 2 * width, size);

    int i = start;
    int j = mid;
    int k = start;

    while (i < mid && j < end) {
        if (input[i] <= input[j]) {
            output[k++] = input[i++];
        } else {
            output[k++] = input[j++];
        }
    }
    while (i < mid) output[k++] = input[i++];
    while (j < end) output[k++] = input[j++];
}

// Host function to manage parallel merge sort
void parallel_merge_sort(int *h_array) {
    int *d_input, *d_output;
    size_t size = N * sizeof(int);

    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, size);
    cudaMemcpy(d_input, h_array, size, cudaMemcpyHostToDevice);

    int width;
    for (width = 1; width < N; width *= 2) {
        int numBlocks = (N / (2 * width) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        merge_kernel<<<numBlocks, THREADS_PER_BLOCK>>>(d_input, d_output, width, N);

        // Swap input/output for next iteration
        int *temp = d_input;
        d_input = d_output;
        d_output = temp;
    }

    cudaMemcpy(h_array, d_input, size, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}

// Function to generate random array
void generate_array(int *arr) {
    for (int i = 0; i < N; i++) {
        arr[i] = rand() % 1000;
    }
}

// Function to display the array (optional)
void print_array(int *arr) {
    for (int i = 0; i < N; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main() {
    int h_array[N];

    generate_array(h_array);

    printf("Unsorted Array:\n");
    print_array(h_array);

    parallel_merge_sort(h_array);

    printf("Sorted Array:\n");
    print_array(h_array);

    return 0;
}
