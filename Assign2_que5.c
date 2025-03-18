// 5.	Parallel Reduction using MPI 

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int n = 100; 
    int *array = NULL;
    int local_sum = 0;
    int global_sum = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        array = (int *)malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) {
            array[i] = i + 1;
        }
    }

    int local_n = n / size;
    int *local_array = (int *)malloc(local_n * sizeof(int));

    MPI_Scatter(array, local_n, MPI_INT, local_array, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < local_n; i++) {
        local_sum += local_array[i];
    }

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Total Sum: %d\n", global_sum);
        free(array);
    }

    free(local_array);
    MPI_Finalize();
    return 0;
}
