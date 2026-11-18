#include <stdio.h>
#include <mpi.h>
#include <omp.h>

int main(int argc, char **argv) {
    int provided, rank, size;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (provided < MPI_THREAD_MULTIPLE || size != 2) {
        if (rank == 0)
            fprintf(stderr, "requires MPI_THREAD_MULTIPLE and exactly 2 processes\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int other = 1 - rank;
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int sendval = 100 * rank + tid, recvval;

        // the thread number as tag pairs thread t of one process with thread t of the other
        MPI_Sendrecv(&sendval, 1, MPI_INT, other, tid,
                     &recvval, 1, MPI_INT, other, tid, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("rank %d, thread %d: received %3d\n", rank, tid, recvval);
    }

    MPI_Finalize();
    return 0;
}
