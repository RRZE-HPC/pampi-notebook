#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // some local value between 0 and 1, different on every rank
    unsigned int hash = (rank + 1) * 2654435761u;
    double local = (hash % 1000) / 1000.0;

    // the pair (value, index) - its layout matches MPI_DOUBLE_INT
    struct { double value; int rank; } in = { local, rank }, out;

    MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

    printf("rank %d: local value %.3f, global maximum %.3f on rank %d\n", rank, local, out.value, out.rank);

    MPI_Finalize();
    return 0;
}
