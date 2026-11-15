#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // MPI_Allreduce: input and result in the same buffer, on every rank
    int values[2] = { rank, 10 * rank };
    MPI_Allreduce(MPI_IN_PLACE, values, 2, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    printf("rank %d: values = { %d, %d }\n", rank, values[0], values[1]);

    // MPI_Reduce: only the root uses MPI_IN_PLACE, the others have no receive buffer
    int partial = rank + 1;
    if (rank == 0) {
        int total = partial;
        MPI_Reduce(MPI_IN_PLACE, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        printf("rank 0: total = %d\n", total);
    } else {
        MPI_Reduce(&partial, NULL, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
