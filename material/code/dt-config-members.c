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
        // in a real code: read from a file
        cfg.nx = 100;
        cfg.ny = 200;
        cfg.du = 0.01;
        cfg.it = 500;
    }

    // one broadcast per member
    MPI_Bcast(&cfg.nx, 1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&cfg.ny, 1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&cfg.du, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cfg.it, 1, MPI_INT,    0, MPI_COMM_WORLD);

    printf("rank %d: nx = %d, ny = %d, du = %g, it = %d\n", rank, cfg.nx, cfg.ny, cfg.du, cfg.it);

    MPI_Finalize();
    return 0;
}
