// 4. Heat Distribution Simulation using MPI 

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 100
#define T 100
#define ALPHA 0.1

void initialize_grid(double grid[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            grid[i][j] = 0.0;
        }
    }
    grid[N/2][N/2] = 100.0;
}

void update_grid(double grid[N][N], double new_grid[N][N]) {
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            new_grid[i][j] = grid[i][j] + ALPHA * (
                grid[i-1][j] + grid[i+1][j] + grid[i][j-1] + grid[i][j+1] - 4 * grid[i][j]);
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    double grid[N][N], new_grid[N][N];
    int rows_per_process = N / size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        initialize_grid(grid);
    }

    MPI_Bcast(grid, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int t = 0; t < T; t++) {
        MPI_Scatter(grid, rows_per_process * N, MPI_DOUBLE, grid + rank * rows_per_process * N, rows_per_process * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        update_grid(grid + rank * rows_per_process, new_grid + rank * rows_per_process);
        MPI_Gather(new_grid + rank * rows_per_process, rows_per_process * N, MPI_DOUBLE, grid, rows_per_process * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        printf("Heat distribution after %d iterations:\n", T);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%.2f ", grid[i][j]);
            }
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
}
