#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    struct particle { double x[3]; double v[3]; double m; int id; };
    #define NP 4

    // the position: 3 doubles at the start of the struct ...
    MPI_Datatype pos, pos_resized;
    MPI_Type_contiguous(3, MPI_DOUBLE, &pos);
    MPI_Type_commit(&pos);
    // ... with the extent of a whole particle, so that consecutive copies are consecutive particles
    MPI_Type_create_resized(pos, 0, sizeof(struct particle), &pos_resized);
    MPI_Type_commit(&pos_resized);

    if (rank == 0) {
        struct particle p[NP] = { 0 };                              // zeroed, including the padding
        for (int k = 0; k < NP; ++k) {
            for (int d = 0; d < 3; ++d) {
                p[k].x[d] = 10 * k + d;
                p[k].v[d] = -1.0;
            }
            p[k].m  = -1.0;
            p[k].id = k;
        }
        MPI_Send(p, NP, pos_resized, 1, 0, MPI_COMM_WORLD);         // the positions of all particles
    } else if (rank == 1) {
        double x[NP][3];
        MPI_Recv(x, 3 * NP, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int k = 0; k < NP; ++k)
            printf("rank 1: particle %d at (%4.1f, %4.1f, %4.1f)\n", k, x[k][0], x[k][1], x[k][2]);
    }

    MPI_Type_free(&pos);
    MPI_Type_free(&pos_resized);

    MPI_Finalize();
    return 0;
}
