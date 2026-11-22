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
    MPI_Cart_create(MPI_COMM_WORLD, NUM_DIMS, dims, periods, 0, &cart);
    int crank;
    MPI_Comm_rank(cart, &crank);

    // the neighbor order of a Cartesian topology: for each dimension the - and then the + neighbor,
    // i.e. the order of the results of MPI_Cart_shift() for dimension 0, then for dimension 1
    enum { LEFT, RIGHT, BOTTOM, TOP, NUM_NEIGHBORS };

    // every rank sends its rank to all neighbors; entries of MPI_PROC_NULL neighbors stay -1
    int neighbors[NUM_NEIGHBORS] = { [LEFT] = -1, [RIGHT] = -1, [BOTTOM] = -1, [TOP] = -1 };
    MPI_Neighbor_allgather(&crank, 1, MPI_INT, neighbors, 1, MPI_INT, cart);

    int *all = crank == 0 ? malloc(NUM_NEIGHBORS * size * sizeof(int)) : NULL;
    MPI_Gather(neighbors, NUM_NEIGHBORS, MPI_INT, all, NUM_NEIGHBORS, MPI_INT, 0, cart);
    if (crank == 0) {
        for (int r = 0; r < size; ++r) {
            int c[NUM_DIMS];
            MPI_Cart_coords(cart, r, NUM_DIMS, c);
            const int *f = &all[NUM_NEIGHBORS * r];     // the neighbors of rank r
            printf("rank %2d (%d,%d): left %2d, right %2d, bottom %2d, top %2d\n",
                   r, c[DIM_I], c[DIM_J], f[LEFT], f[RIGHT], f[BOTTOM], f[TOP]);
        }
        free(all);
    }

    MPI_Comm_free(&cart);

    MPI_Finalize();
    return 0;
}
