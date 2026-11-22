#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 12) {
        if (rank == 0)
            fprintf(stderr, "This example requires exactly 12 processes, got %d\n", size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // the dimensions of the process grid: dimension 0 runs along i, dimension 1 along j
    enum { DIM_I, DIM_J, NUM_DIMS };
    int dims[NUM_DIMS] = { [DIM_I] = 4, [DIM_J] = 3 }, periods[NUM_DIMS] = { [DIM_I] = 1, [DIM_J] = 0 };
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, NUM_DIMS, dims, periods, 0, &cart);   // reorder = 0

    int crank, coords[NUM_DIMS];
    MPI_Comm_rank(cart, &crank);
    MPI_Cart_coords(cart, crank, NUM_DIMS, coords);

    // the right neighbor of the last column: dimension 0 is periodic, so (4, c1) wraps around to (0, c1)
    int beyond[NUM_DIMS] = { [DIM_I] = dims[DIM_I], [DIM_J] = coords[DIM_J] }, wrapped;
    MPI_Cart_rank(cart, beyond, &wrapped);

    // the fields every rank sends to rank 0, in this order
    enum { RANK, COORD_I, COORD_J, WRAPPED, NUM_FIELDS };

    int mine[NUM_FIELDS] = { [RANK] = crank, [COORD_I] = coords[DIM_I], [COORD_J] = coords[DIM_J], [WRAPPED] = wrapped };
    int *all = crank == 0 ? malloc(NUM_FIELDS * size * sizeof(int)) : NULL;
    MPI_Gather(mine, NUM_FIELDS, MPI_INT, all, NUM_FIELDS, MPI_INT, 0, cart);

    if (crank == 0) {
        for (int c1 = dims[DIM_J] - 1; c1 >= 0; --c1) {
            for (int c0 = 0; c0 < dims[DIM_I]; ++c0) {
                int c[NUM_DIMS] = { [DIM_I] = c0, [DIM_J] = c1 }, r;
                MPI_Cart_rank(cart, c, &r);
                const int *f = &all[NUM_FIELDS * r];        // the fields of rank r
                printf("   %2d (%d,%d)", f[RANK], f[COORD_I], f[COORD_J]);
            }
            printf("        coordinates (4,%d) wrap around to rank %d\n", c1, all[NUM_FIELDS * c1 + WRAPPED]);
        }
        free(all);
    }

    MPI_Comm_free(&cart);

    MPI_Finalize();
    return 0;
}
