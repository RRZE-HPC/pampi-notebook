#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int npart = 8;                        // number of partitions
    const int count = 100000;                   // elements per partition
    if (size != 2) {
        if (rank == 0) fprintf(stderr, "run with exactly 2 processes\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double *buf = malloc(npart * count * sizeof(double));
    MPI_Request req;
    if (rank == 0)                              // one message of npart partitions, set up once
        MPI_Psend_init(buf, npart, count, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_INFO_NULL, &req);
    else
        MPI_Precv_init(buf, npart, count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_INFO_NULL, &req);

    for (int iter = 0; iter < 3; ++iter) {
        MPI_Start(&req);                        // start the operation: no data is sent yet

        if (rank == 0) {
            // compute the buffer block by block, release each block as soon as it is complete
            for (int p = 0; p < npart; ++p) {
                for (int i = 0; i < count; ++i)
                    buf[p * count + i] = iter + p;
                MPI_Pready(p, req);             // partition p may be sent now
            }
        } else {
            // process each block as soon as it has arrived
            double sum = 0.0;
            for (int p = 0; p < npart; ++p) {
                int arrived = 0;
                while (!arrived)
                    MPI_Parrived(req, p, &arrived);
                for (int i = 0; i < count; ++i)
                    sum += buf[p * count + i];
            }
            double expected = (double)count * (npart * iter + npart * (npart - 1) / 2);
            printf("iteration %d: sum = %.0f, expected %.0f\n", iter, sum, expected);
        }

        MPI_Wait(&req, MPI_STATUS_IGNORE);      // complete the whole message
    }

    MPI_Request_free(&req);
    free(buf);

    MPI_Finalize();
    return 0;
}
