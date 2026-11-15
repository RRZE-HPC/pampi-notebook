#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 10000000;

    int start = (long)rank       * N / size;
    int end   = (long)(rank + 1) * N / size;

    float sum = 0.0f;
    for (int i = start; i < end; ++i)
        sum += 1.0f / (i + 1);                      // single precision

    float total;
    MPI_Reduce(&sum, &total, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double ref = 0.0;
        for (int i = N - 1; i >= 0; --i)
            ref += 1.0 / (i + 1);                   // double precision, small terms first
        printf("%d processes: sum = %.7f, double precision: %.7f\n", size, total, ref);
    }

    MPI_Finalize();
    return 0;
}
