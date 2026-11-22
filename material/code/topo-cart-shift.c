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

    // a rank as text, "-" for MPI_PROC_NULL
    #define NAME(r, buf) ((r) == MPI_PROC_NULL ? "-" : (snprintf(buf, sizeof(buf), "%d", (r)), buf))
    char s[16], d[16];

    // 1. the query: src and dst of every rank for disp = 1, collected on rank 0
    enum { SRC, DST, NUM_PER_DIM };                 // the two results of MPI_Cart_shift() for one dimension
    int mine[NUM_DIMS][NUM_PER_DIM];                       // mine[dim][SRC or DST]
    MPI_Cart_shift(cart, DIM_I, 1, &mine[DIM_I][SRC], &mine[DIM_I][DST]);
    MPI_Cart_shift(cart, DIM_J, 1, &mine[DIM_J][SRC], &mine[DIM_J][DST]);
    int *all = crank == 0 ? malloc(NUM_DIMS * NUM_PER_DIM * size * sizeof(int)) : NULL;
    MPI_Gather(mine, NUM_DIMS * NUM_PER_DIM, MPI_INT, all, NUM_DIMS * NUM_PER_DIM, MPI_INT, 0, cart);

    if (crank == 0) {
        for (int dim = 0; dim < NUM_DIMS; ++dim) {
            printf("src > rank > dst in dimension %d (%s), top row first:\n", dim, periods[dim] ? "periodic" : "not periodic");
            for (int c1 = dims[DIM_J] - 1; c1 >= 0; --c1) {
                for (int c0 = 0; c0 < dims[DIM_I]; ++c0) {
                    int c[NUM_DIMS] = { [DIM_I] = c0, [DIM_J] = c1 }, r;
                    MPI_Cart_rank(cart, c, &r);
                    const int *f = &all[(NUM_DIMS * r + dim) * NUM_PER_DIM];   // src and dst of rank r in dimension dim
                    printf("   %2s >%2d > %-2s", NAME(f[SRC], s), r, NAME(f[DST], d));
                }
                printf("\n");
            }
        }

        // MPI_Cart_shift() is local: rank 0 alone can query other displacements
        int disps[4] = { -1, 1, 2, 4 }, src, dst;
        printf("rank 0 = (0,0) with other displacements:\n");
        for (int dim = 0; dim < NUM_DIMS; ++dim)
            for (int k = 0; k < 4; ++k) {
                MPI_Cart_shift(cart, dim, disps[k], &src, &dst);
                printf("  dim %d, disp %2d: src %s, dst %s\n", dim, disps[k], NAME(src, s), NAME(dst, d));
            }
    }

    // 2. using src and dst for a shift: every rank sends its value to dst and receives from src
    int src, dst, value = crank;
    MPI_Cart_shift(cart, DIM_J, 1, &src, &dst);
    MPI_Sendrecv_replace(&value, 1, MPI_INT, dst, 0, src, 0, cart, MPI_STATUS_IGNORE);
    MPI_Gather(&value, 1, MPI_INT, all, 1, MPI_INT, 0, cart);

    if (crank == 0) {
        printf("values after the shift in dimension 1 (before: value = rank), top row first:\n");
        for (int c1 = dims[DIM_J] - 1; c1 >= 0; --c1) {
            for (int c0 = 0; c0 < dims[DIM_I]; ++c0) {
                int c[NUM_DIMS] = { [DIM_I] = c0, [DIM_J] = c1 }, r;
                MPI_Cart_rank(cart, c, &r);
                printf(" %3d", all[r]);
            }
            printf("\n");
        }
        free(all);
    }

    MPI_Comm_free(&cart);

    MPI_Finalize();
    return 0;
}
