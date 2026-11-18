#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#define NPART 8                                 // number of partitions
#define COUNT 100000                            // elements per partition

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

    double *buf = malloc(NPART * COUNT * sizeof(double));
    MPI_Request req;
    if (rank == 0)                              // one message of NPART partitions, set up once
        MPI_Psend_init(buf, NPART, COUNT, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_INFO_NULL, &req);
    else
        MPI_Precv_init(buf, NPART, COUNT, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_INFO_NULL, &req);

    for (int iter = 0; iter < 3; ++iter) {
        MPI_Start(&req);                        // start the operation: no data is sent yet

        if (rank == 0) {
            // producer: fill the partitions, mark each one ready as soon as it is complete
            #pragma omp parallel for schedule(dynamic)
            for (int p = 0; p < NPART; ++p) {
                for (int i = 0; i < COUNT; ++i)
                    buf[p * COUNT + i] = iter + p;
                MPI_Pready(p, req);             // partition p may be sent now
            }
        } else {
            // consumer: every thread waits only for the partition it works on
            double sum = 0.0;
            #pragma omp parallel for schedule(dynamic) reduction(+:sum)
            for (int p = 0; p < NPART; ++p) {
                int arrived = 0;
                while (!arrived)
                    MPI_Parrived(req, p, &arrived);
                for (int i = 0; i < COUNT; ++i)
                    sum += buf[p * COUNT + i];
            }
            double expected = (double)COUNT * (NPART * iter + NPART * (NPART - 1) / 2);
            printf("iteration %d: sum = %.0f, expected %.0f\n", iter, sum, expected);
        }

        MPI_Wait(&req, MPI_STATUS_IGNORE);      // complete the whole message
    }

    MPI_Request_free(&req);
    free(buf);
    MPI_Finalize();
    return 0;
}
