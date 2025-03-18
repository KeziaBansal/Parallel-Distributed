// 8.	Parallel Matrix Transposition using MPI

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 4 // Size of the matrix (N x N)

void transpose(int *matrix, int *transposed, int n, int rows) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < n; j++) {
            transposed[j * N + i] = matrix[i * N + j];
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int matrix[N][N];
    int local_rows;
    int *local_matrix = NULL;
    int *local_transposed = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    local_rows = N / size;

    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                matrix[i][j] = i * N + j + 1;
            }
        }
    }

    local_matrix = (int *)malloc(local_rows * N * sizeof(int));
    MPI_Scatter(matrix, local_rows * N, MPI_INT, local_matrix, local_rows * N, MPI_INT, 0, MPI_COMM_WORLD);

    local_transposed = (int *)malloc(N * local_rows * sizeof(int));
    transpose(local_matrix, local_transposed, N, local_rows);

    int *transposed_matrix = NULL;
    if (rank == 0) {
        transposed_matrix = (int *)malloc(N * N * sizeof(int));
    }

    MPI_Gather(local_transposed, N * local_rows, MPI_INT, transposed_matrix, N * local_rows, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Original Matrix:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%d ", matrix[i][j]);
            }
            printf("\n");
        }

        printf("Transposed Matrix:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%d ", transposed_matrix[i * N + j]);
            }
            printf("\n");
        }
        free(transposed_matrix);
    }

    free(local_matrix);
    free(local_transposed);
    MPI_Finalize();
    return 0;
}

