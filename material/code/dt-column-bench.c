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

    const int maxn = argc > 1 ? atoi(argv[1]) : 2048;      // largest matrix dimension
    const int nmethods = 5;
    const char *names[] = { "row", "element-wise", "manual packing", "MPI_Type_vector", "MPI_Pack" };
    const int other = 1 - rank;

    double *a   = calloc((size_t)maxn * maxn, sizeof(double)); // the matrix, n x n with n <= maxn
    double *buf = malloc(maxn * sizeof(double));

    int packmax;
    MPI_Datatype col;
    MPI_Type_vector(maxn, 1, maxn, MPI_DOUBLE, &col);
    MPI_Type_commit(&col);
    MPI_Pack_size(1, col, MPI_COMM_WORLD, &packmax);
    MPI_Type_free(&col);
    char *packbuf = malloc(packmax);

    if (rank == 0) {
        printf("%8s", "n");
        for (int k = 0; k < nmethods; ++k)
            printf(" %16s", names[k]);
        printf("\n");
    }

    for (int n = 16; n <= maxn; n *= 2) {
        MPI_Datatype column;
        MPI_Type_vector(n, 1, n, MPI_DOUBLE, &column);      // column 0 of the n x n matrix
        MPI_Type_commit(&column);
        int reps = n <= 256 ? 200 : 50;

        if (rank == 0)
            printf("%8d", n);

        for (int k = 0; k < nmethods; ++k) {
            double t0 = 0.0;
            for (int r = -reps / 10 - 1; r < reps; ++r) {      // negative r: warm-up, not timed
                if (r == 0) {
                    MPI_Barrier(MPI_COMM_WORLD);
                    t0 = MPI_Wtime();
                }
                // ping-pong: rank 0 sends first, then receives; rank 1 the other way round
                for (int turn = 0; turn < 2; ++turn) {
                    const int sending = (turn == 0) == (rank == 0);
                    int pos = 0;
                    switch (k) {
                    case 0:     // a row: contiguous
                        if (sending) MPI_Send(a, n, MPI_DOUBLE, other, 0, MPI_COMM_WORLD);
                        else         MPI_Recv(a, n, MPI_DOUBLE, other, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        break;
                    case 1:     // element by element
                        for (int i = 0; i < n; ++i) {
                            if (sending) MPI_Send(&a[(size_t)i * n], 1, MPI_DOUBLE, other, 0, MPI_COMM_WORLD);
                            else         MPI_Recv(&a[(size_t)i * n], 1, MPI_DOUBLE, other, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        }
                        break;
                    case 2:     // manual packing and unpacking
                        if (sending) {
                            for (int i = 0; i < n; ++i)
                                buf[i] = a[(size_t)i * n];
                            MPI_Send(buf, n, MPI_DOUBLE, other, 0, MPI_COMM_WORLD);
                        } else {
                            MPI_Recv(buf, n, MPI_DOUBLE, other, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            for (int i = 0; i < n; ++i)
                                a[(size_t)i * n] = buf[i];
                        }
                        break;
                    case 3:     // derived type
                        if (sending) MPI_Send(a, 1, column, other, 0, MPI_COMM_WORLD);
                        else         MPI_Recv(a, 1, column, other, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        break;
                    case 4:     // MPI_Pack with the derived type
                        if (sending) {
                            MPI_Pack(a, 1, column, packbuf, packmax, &pos, MPI_COMM_WORLD);
                            MPI_Send(packbuf, pos, MPI_PACKED, other, 0, MPI_COMM_WORLD);
                        } else {
                            MPI_Recv(packbuf, packmax, MPI_PACKED, other, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            MPI_Unpack(packbuf, packmax, &pos, a, 1, column, MPI_COMM_WORLD);
                        }
                        break;
                    }
                }
            }
            double t = (MPI_Wtime() - t0) / reps / 2;          // one-way time per message
            if (rank == 0)
                printf(" %16.2f", 1e6 * t);
        }
        if (rank == 0)
            printf("\n");
        MPI_Type_free(&column);
    }

    free(a);
    free(buf);
    free(packbuf);

    MPI_Finalize();
    return 0;
}
