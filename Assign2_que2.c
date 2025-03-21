//2.	Matrix Multiplication using MPI. Consider 70X70 matrix compute using serial sequential order 
//and compare the time. For computing the time use double start_time, run_time; run_time = omp_get_wtime() start_time; Time in second

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>  // For omp_get_wtime()

#define N 70  

void serial_matrix_multiply(double A[N][N], double B[N][N], double C[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void parallel_matrix_multiply(double A[N][N], double B[N][N], double C[N][N], int rank, int size) {
    int rows_per_process = N / size;
    double local_A[rows_per_process][N];
    double local_C[rows_per_process][N];

    // Scatter matrix A among processes
    MPI_Scatter(A, rows_per_process * N, MPI_DOUBLE, local_A, 
                rows_per_process * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Broadcast matrix B to all processes
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Matrix multiplication 
    for (int i = 0; i < rows_per_process; i++) {
        for (int j = 0; j < N; j++) {
            local_C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                local_C[i][j] += local_A[i][k] * B[k][j];
            }
        }
    }

    // Gather the results back in matrix C
    MPI_Gather(local_C, rows_per_process * N, MPI_DOUBLE, C, 
               rows_per_process * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

int main(int argc, char* argv[]) {
    int rank, size;
    double A[N][N], B[N][N], C[N][N];
    double start_time, run_time;

    MPI_Init(&argc, &argv);                  
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);   

    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] = rand() % 10;
                B[i][j] = rand() % 10;
            }
        }
        start_time = omp_get_wtime();
        serial_matrix_multiply(A, B, C);
        run_time = omp_get_wtime() - start_time;
        printf("Serial Execution Time: %f seconds\n", run_time);
    }

    // Synchronize all processes before parallel execution
    MPI_Barrier(MPI_COMM_WORLD);

    start_time = omp_get_wtime();
    parallel_matrix_multiply(A, B, C, rank, size);
    run_time = omp_get_wtime() - start_time;

    if (rank == 0) {
        printf("Parallel Execution Time: %f seconds\n", run_time);
    }

    MPI_Finalize();  
    return 0;
}
