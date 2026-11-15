#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // largest message size in bytes per rank (command line argument, default 512 KiB)
    const int maxn = argc > 1 ? atoi(argv[1]) : 1 << 19;

    const int ncoll = 8;
    const char *names[] = { "Bcast", "Scatter", "Gather", "Reduce", "Allreduce", "Allgather", "Gather+Bcast", "Alltoall" };

    char *sendbuf = calloc((size_t)maxn * size, 1);
    char *recvbuf = calloc((size_t)maxn * size, 1);

    if (rank == 0) {
        printf("%10s", "bytes");
        for (int k = 0; k < ncoll; ++k)
            printf(" %12s", names[k]);
        printf("\n");
    }

    for (int n = 8; n <= maxn; n *= 4) {
        int reps = n <= (1 << 12) ? 1000 : n <= (1 << 16) ? 100 : 20;
        if (rank == 0)
            printf("%10d", n);

        for (int k = 0; k < ncoll; ++k) {
            double t0 = 0.0;
            for (int r = -reps / 10 - 1; r < reps; ++r) {   // negative r: warm-up, not timed
                if (r == 0) {
                    MPI_Barrier(MPI_COMM_WORLD);
                    t0 = MPI_Wtime();
                }
                switch (k) {
                case 0: MPI_Bcast(sendbuf, n, MPI_BYTE, 0, MPI_COMM_WORLD); break;
                case 1: MPI_Scatter(sendbuf, n, MPI_BYTE, recvbuf, n, MPI_BYTE, 0, MPI_COMM_WORLD); break;
                case 2: MPI_Gather(sendbuf, n, MPI_BYTE, recvbuf, n, MPI_BYTE, 0, MPI_COMM_WORLD); break;
                // reductions work on floats: MPI_SUM is not defined for MPI_BYTE
                case 3: MPI_Reduce(sendbuf, recvbuf, n / 4, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD); break;
                case 4: MPI_Allreduce(sendbuf, recvbuf, n / 4, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD); break;
                case 5: MPI_Allgather(sendbuf, n, MPI_BYTE, recvbuf, n, MPI_BYTE, MPI_COMM_WORLD); break;
                case 6: MPI_Gather(sendbuf, n, MPI_BYTE, recvbuf, n, MPI_BYTE, 0, MPI_COMM_WORLD);
                        MPI_Bcast(recvbuf, n * size, MPI_BYTE, 0, MPI_COMM_WORLD); break;
                case 7: MPI_Alltoall(sendbuf, n, MPI_BYTE, recvbuf, n, MPI_BYTE, MPI_COMM_WORLD); break;
                }
            }
            double t = (MPI_Wtime() - t0) / reps, tmax;
            MPI_Reduce(&t, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0)
                printf(" %12.2f", 1e6 * tmax);
        }
        if (rank == 0)
            printf("\n");
    }

    free(sendbuf);
    free(recvbuf);

    MPI_Finalize();
    return 0;
}
