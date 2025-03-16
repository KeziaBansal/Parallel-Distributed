#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

static long num_steps = 100000; // Total number of steps
double step;

int main(int argc, char *argv[]) {
    int rank, size;
    double x, pi, sum = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Bcast(&num_steps, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    step = 1.0 / (double)num_steps;

    for (long i = rank; i < num_steps; i += size) {
        x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    double total_sum;
    MPI_Reduce(&sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Calculate π only in the root process
    if (rank == 0) {
        pi = step * total_sum;
        printf("Approximation of π: %f\n", pi);
    }

    MPI_Finalize();
    return 0;
}
