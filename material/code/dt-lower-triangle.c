#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    #define N 6

    double a[N * N], b[N * N];
    for (int i = 0; i < N * N; ++i) {
        a[i] = i;
        b[i] = 0;
    }

    int lens[N], displs[N];
    for (int i = 0; i < N; ++i) {
        displs[i] = i * N;
        lens[i]   = i + 1;
    }

    MPI_Datatype tril;
    MPI_Type_indexed(N, lens, displs, MPI_DOUBLE, &tril);
    MPI_Type_commit(&tril);

    MPI_Sendrecv(a, 1, tril, rank, 0, b, 1, tril, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Type_free(&tril);

    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c)
            printf(" %3.0f", b[r * N + c]);
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
