// 1.	Estimate the value of Pi using the Monte Carlo method and demonstrate basic MPI functions

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

double estimate_pi(int num_samples) {
    int count = 0;
    for (int i = 0; i < num_samples; i++) {
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;
        if (x * x + y * y <= 1.0) {
            count++;
        }
    }
    return (double)count / num_samples * 4.0;
}

int main(int argc, char *argv[]) {
    int rank, size;
    int num_samples = 1000000;
    double local_pi, global_pi;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(time(NULL) + rank);
    int samples_per_process = num_samples / size;
    local_pi = estimate_pi(samples_per_process);

    MPI_Reduce(&local_pi, &global_pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        global_pi /= size;
        printf("Estimated value of Pi: %f\n", global_pi);
    }

    MPI_Finalize();
    return 0;
}
