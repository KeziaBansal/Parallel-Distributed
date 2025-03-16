 //  3. Parallel Sorting using MPI (Odd-Even Sort)

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 12 

void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void odd_even_sort(int arr[], int n) {
    for (int phase = 0; phase < n; phase++) 
{
        if (phase % 2 == 0) 
{  // Even phase
            for (int i = 0; i < n - 1; i += 2)
                if (arr[i] > arr[i + 1])
                    swap(&arr[i], &arr[i + 1]);
        } 
else 
{  // Odd phase
            for (int i = 1; i < n - 1; i += 2)
{
                if (arr[i] > arr[i + 1])
{
                    swap(&arr[i], &arr[i + 1]);
        }
    }
}
}


int main(int argc, char* argv[]) {
    int rank, size;
    int local_array[N];

    MPI_Init(&argc, &argv);                 // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // Get process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);   // Get total processes

    int local_n = N / size; // Elements per process

    // Initialize array randomly (only in rank 0)
    if (rank == 0) {
        int arr[N];
        printf("Unsorted array: ");
        for (int i = 0; i < N; i++) {
            arr[i] = rand() % 100;
            printf("%d ", arr[i]);
        }
        printf("\n");

        // Distribute data to other processes
        MPI_Scatter(arr, local_n, MPI_INT, local_array, local_n, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        // Other processes receive data
        MPI_Scatter(NULL, local_n, MPI_INT, local_array, local_n, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Perform local sorting
    odd_even_sort(local_array, local_n);

    // Gather sorted subarrays at rank 0
    int sorted_array[N];
    MPI_Gather(local_array, local_n, MPI_INT, sorted_array, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    // Final sorting at rank 0
    if (rank == 0) {
        odd_even_sort(sorted_array, N);
        printf("Sorted array: ");
        for (int i = 0; i < N; i++) {
            printf("%d ", sorted_array[i]);
        }
        printf("\n");
    }

    MPI_Finalize();  // Finalize MPI
    return 0;
}
