#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    struct config { int nx, ny; double du; int it; };
    struct config cfg = { 0, 0, 0.0, 0 };

    if (rank == 0) {
        cfg.nx = 100;
        cfg.ny = 200;
        cfg.du = 0.01;
        cfg.it = 500;
    }

    // describe the struct: nx and ny are adjacent, so they form one block
    int          blocklengths[3] = { 2, 1, 1 };
    MPI_Datatype types[3]        = { MPI_INT, MPI_DOUBLE, MPI_INT };
    MPI_Aint     displs[3], base;

    MPI_Get_address(&cfg,    &base);
    MPI_Get_address(&cfg.nx, &displs[0]);
    MPI_Get_address(&cfg.du, &displs[1]);
    MPI_Get_address(&cfg.it, &displs[2]);
    for (int i = 0; i < 3; ++i)
        displs[i] = MPI_Aint_diff(displs[i], base);         // relative to the start of the struct

    MPI_Datatype cfgtype;
    MPI_Type_create_struct(3, blocklengths, displs, types, &cfgtype);
    MPI_Type_commit(&cfgtype);

    MPI_Bcast(&cfg, 1, cfgtype, 0, MPI_COMM_WORLD);          // the whole struct in one call

    if (rank == 0) {
        MPI_Aint lb, extent;
        MPI_Type_get_extent(cfgtype, &lb, &extent);
        printf("extent of cfgtype: %ld bytes, sizeof(struct config): %zu bytes\n", (long)extent, sizeof(struct config));
    }
    printf("rank %d: nx = %d, ny = %d, du = %g, it = %d\n", rank, cfg.nx, cfg.ny, cfg.du, cfg.it);

    MPI_Type_free(&cfgtype);

    MPI_Finalize();
    return 0;
}
