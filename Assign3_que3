#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int is_prime(int n) {
    if (n <= 1) return 0;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {
    int rank, size;
    int max_value = 100;
    int number;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        for (int i = 2; i <= max_value; i++) {
            MPI_Recv(&number, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_SOURCE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&i, 1, MPI_INT, number, 0, MPI_COMM_WORLD);
        }
        for (int j = 1; j < size; j++) {
            int end_signal = 0;
            MPI_Send(&end_signal, 1, MPI_INT, j, 0, MPI_COMM_WORLD);
        }
    } else {
        while (1) {
            int request = rank;
            MPI_Send(&request, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (number == 0) {
                break;
            }
            if (is_prime(number)) {
                MPI_Send(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            } else {
                int negative_number = -number;
                MPI_Send(&negative_number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            }
        }
    }

    MPI_Finalize();
    return 0;
}

