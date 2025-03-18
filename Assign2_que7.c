// 7.	Parallel Prefix Sum (Scan) using MPI

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void prefix_sum(int *local_array, int *local_prefix_sum, int local_n) {
    local_prefix_sum[0] = local_array[0];
    for (int i = 1; i < local_n; i++) {
        local_prefix_sum[i] = local_prefix_sum[i - 1] + local_array[i];
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int n = 16; 
    int *array = NULL;
    int *local_array = NULL;
    int *local_prefix_sum = NULL;
    int *global_prefix_sum = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_n = n / size;

    if (rank == 0) {
        array = (int *)malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) {
            array[i] = i + 1;
        }
    }

    local_array = (int *)malloc(local_n * sizeof(int));
    MPI_Scatter(array, local_n, MPI_INT, local_array, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    local_prefix_sum = (int *)malloc(local_n * sizeof(int));
    prefix_sum(local_array, local_prefix_sum, local_n);

    if (rank > 0) {
        MPI_Send(&local_prefix_sum[0], 1, MPI_INT, rank, 0, MPI_COMM_WORLD);
    }

    int offset = 0;
    if (rank > 0) {
        MPI_Recv(&offset, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    for (int i = 0; i < local_n; i++) {
        local_prefix_sum[i] += offset;
    }

    if (rank < size - 1) {
        MPI_Send(&local_prefix_sum[local_n - 1], 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        global_prefix_sum = (int *)malloc(n * sizeof(int));
    }

    MPI_Gather(local_prefix_sum, local_n, MPI_INT, global_prefix_sum, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Prefix Sum: ");
        for (int i = 0; i < n; i++) {
            printf("%d ", global_prefix_sum[i]);
        }
        printf("\n");
        free(array);
        free(global_prefix_sum);
    }

    free(local_array);
    free(local_prefix_sum);
    MPI_Finalize();
    return 0;
}

