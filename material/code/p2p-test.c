#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int n = 1 << 24;      // 16 MiB, far above the eager limit
    char *buf = calloc(n, 1);

    if (rank == 0) {
        MPI_Request request;
        int flag = 0;
        long iterations = 0;
        double t0 = MPI_Wtime();

        MPI_Isend(buf, n, MPI_BYTE, 1, 0, MPI_COMM_WORLD, &request);
        do {
            ++iterations;                           // "useful work", which must not modify buf
            MPI_Test(&request, &flag, MPI_STATUS_IGNORE);
        } while (!flag);

        printf("rank 0: send completed after %.3f s and %ld iterations\n", MPI_Wtime() - t0, iterations);
    } else if (rank == 1) {
        double t0 = MPI_Wtime();
        while (MPI_Wtime() - t0 < 0.5)
            ;                                       // busy for 0.5 s before posting the receive
        MPI_Recv(buf, n, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("rank 1: message received after %.3f s\n", MPI_Wtime() - t0);
    }

    free(buf);

    MPI_Finalize();
    return 0;
}
