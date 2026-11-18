#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0)
            fprintf(stderr, "This benchmark requires exactly 2 processes, got %d\n", size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0)
        printf("%12s %12s %12s\n", "bytes", "time [us]", "BW [MB/s]");

    for (int n = 1; n <= (1 << 22); n *= 4) {
        char *buf = calloc(n, 1);
        int reps = n <= (1 << 18) ? 10000 : 1000;

        // the first reps / 10 round trips are a warm-up and not timed
        double start = 0.0;
        for (int r = -reps / 10; r < reps; ++r) {
            if (r == 0)
                start = MPI_Wtime();

            if (rank == 0) {
                MPI_Send(buf, n, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(buf, n, MPI_BYTE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(buf, n, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buf, n, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
            }
        }
        double time = (MPI_Wtime() - start) / reps / 2;     // one-way time per message

        if (rank == 0)
            printf("%12d %12.2f %12.1f\n", n, 1e6 * time, 1e-6 * n / time);

        free(buf);
    }

    MPI_Finalize();
    return 0;
}
