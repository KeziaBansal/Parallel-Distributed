//  1.	MPI DAXPY Implementation
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N (1 << 16)

void daxpy(double a, double *x, double *y, double *result, int n) {
    for (int i = 0; i < n; i++) {
        result[i] = a * x[i] + y[i];
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    double a = 2.0;
    double *x = NULL;
    double *y = NULL;
    double *result = NULL;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        x = (double *)malloc(N * sizeof(double));
        y = (double *)malloc(N * sizeof(double));
        result = (double *)malloc(N * sizeof(double));
        for (int i = 0; i < N; i++) {
            x[i] = 1.0; // Initialize X
            y[i] = 2.0; // Initialize Y
        }
    }

    double start_time, end_time;
    start_time = MPI_Wtime();

    int local_n = N / size;
    double *local_x = (double *)malloc(local_n * sizeof(double));
    double *local_y = (double *)malloc(local_n * sizeof(double));
    double *local_result = (double *)malloc(local_n * sizeof(double));

    MPI_Scatter(x, local_n, MPI_DOUBLE, local_x, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(y, local_n, MPI_DOUBLE, local_y, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    daxpy(a, local_x, local_y, local_result, local_n);

    MPI_Gather(local_result, local_n, MPI_DOUBLE, result, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    end_time = MPI_Wtime();

    if (rank == 0) {
        printf("DAXPY operation completed.\n");
        printf("Elapsed time: %f seconds\n", end_time - start_time);
        free(x);
        free(y);
        free(result);
    }

    free(local_x);
    free(local_y);
    free(local_result);
    MPI_Finalize();
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N (1 << 16)

void daxpy(double a, double *x, double *y, double *result, int n) {
    for (int i = 0; i < n; i++) {
        result[i] = a * x[i] + y[i];
    }
}

int main() {
    double a = 2.0;
    double *x = (double *)malloc(N * sizeof(double));
    double *y = (double *)malloc(N * sizeof(double));
    double *result = (double *)malloc(N * sizeof(double));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0; // Initialize X
        y[i] = 2.0; // Initialize Y
    }

    clock_t start_time = clock();
    daxpy(a, x, y, result, N);
    clock_t end_time = clock();

    printf("Uniprocessor DAXPY operation completed.\n");
    printf("Elapsed time: %f seconds\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);

    free(x);
    free(y);
    free(result);
    return 0;
}

