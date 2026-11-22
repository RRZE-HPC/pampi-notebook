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
    int dims[NUM_DIMS] = { [DIM_I] = 4, [DIM_J] = 3 }, periods[NUM_DIMS] = { 0, 0 };
    MPI_Comm cart, rowcomm, colcomm;
    MPI_Cart_create(MPI_COMM_WORLD, NUM_DIMS, dims, periods, 0, &cart);
    int crank, coords[NUM_DIMS];
    MPI_Comm_rank(cart, &crank);
    MPI_Cart_coords(cart, crank, NUM_DIMS, coords);

    int keep_dim0[NUM_DIMS] = { [DIM_I] = 1, [DIM_J] = 0 }, keep_dim1[NUM_DIMS] = { [DIM_I] = 0, [DIM_J] = 1 };
    MPI_Cart_sub(cart, keep_dim0, &rowcomm);    // the processes with the same coordinate in dimension 1
    MPI_Cart_sub(cart, keep_dim1, &colcomm);    // the processes with the same coordinate in dimension 0

    int rowrank, rowsize, colrank, colsize, rowsum, colsum;
    MPI_Comm_rank(rowcomm, &rowrank);
    MPI_Comm_size(rowcomm, &rowsize);
    MPI_Comm_rank(colcomm, &colrank);
    MPI_Comm_size(colcomm, &colsize);
    MPI_Allreduce(&crank, &rowsum, 1, MPI_INT, MPI_SUM, rowcomm);  // sum of the ranks in my row
    MPI_Allreduce(&crank, &colsum, 1, MPI_INT, MPI_SUM, colcomm);  // sum of the ranks in my column

    // the fields every rank sends to rank 0, in this order
    enum { COORD_I, COORD_J, ROWRANK, ROWSIZE, ROWSUM, COLRANK, COLSIZE, COLSUM, NUM_FIELDS };

    int mine[NUM_FIELDS] = { [COORD_I]  = coords[DIM_I], [COORD_J]  = coords[DIM_J],
                             [ROWRANK] = rowrank,   [ROWSIZE] = rowsize,   [ROWSUM] = rowsum,
                             [COLRANK] = colrank,   [COLSIZE] = colsize,   [COLSUM] = colsum };
    int *all = crank == 0 ? malloc(NUM_FIELDS * size * sizeof(int)) : NULL;
    MPI_Gather(mine, NUM_FIELDS, MPI_INT, all, NUM_FIELDS, MPI_INT, 0, cart);
    if (crank == 0) {
        for (int r = 0; r < size; ++r) {
            const int *f = &all[NUM_FIELDS * r];    // the fields of rank r
            printf("rank %2d (%d,%d): rank %d of %d in its row (sum %2d), rank %d of %d in its column (sum %2d)\n",
                   r, f[COORD_I], f[COORD_J], f[ROWRANK], f[ROWSIZE], f[ROWSUM], f[COLRANK], f[COLSIZE], f[COLSUM]);
        }
        free(all);

        // the sub-communicators are Cartesian again: 1D, with the kept dims and periods
        MPI_Comm sub[2] = { rowcomm, colcomm };
        const char *name[2] = { "rowcomm", "colcomm" };
        for (int k = 0; k < 2; ++k) {
            int ndims, d, p, c;
            MPI_Cartdim_get(sub[k], &ndims);
            MPI_Cart_get(sub[k], 1, &d, &p, &c);
            printf("%s of rank 0: %dD Cartesian, dims %d, periods %d, my coordinate %d\n", name[k], ndims, d, p, c);
        }
    }

    MPI_Comm_free(&rowcomm);
    MPI_Comm_free(&colcomm);
    MPI_Comm_free(&cart);

    MPI_Finalize();
    return 0;
}
