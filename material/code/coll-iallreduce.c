#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double local = rank + 1.0, global;
    MPI_Request request;

    MPI_Iallreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request);

    // independent work, which must not touch local or global;
    // MPI_Test() checks for completion and gives the library a chance to make progress
    double work = 0.0;
    long iterations = 0;
    int flag = 0;
    do {
        for (int i = 0; i < 1000; ++i)
            work += 1e-3 * i;                       // some computation
        ++iterations;
        MPI_Test(&request, &flag, MPI_STATUS_IGNORE);
    } while (!flag);

    printf("rank %d: global = %g after %ld iterations of work (%g)\n", rank, global, iterations, work);

    MPI_Finalize();
    return 0;
}
