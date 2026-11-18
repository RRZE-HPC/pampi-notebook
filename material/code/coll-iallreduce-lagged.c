#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // toy solver: the local residual halves in every sweep, the initial residual of rank r is r + 1
    const double tol = 1e-3;
    double res_local = rank + 1.0;
    double sendbuf, res_global = 1e30;              // separate send buffer: res_local changes while the check is pending
    MPI_Request request = MPI_REQUEST_NULL;
    int sweeps = 0;

    while (1) {
        // sweep: a real solver would update its part of the solution here (1 ms of work)
        double t0 = MPI_Wtime();
        while (MPI_Wtime() - t0 < 1e-3)
            ;
        res_local *= 0.5;
        ++sweeps;

        // complete the check started in the previous sweep (MPI_Wait() returns at once for MPI_REQUEST_NULL)
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        if (res_global < tol)
            break;

        // start the check of this sweep; it completes while the next sweep is computed
        sendbuf = res_local;
        MPI_Iallreduce(&sendbuf, &res_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD, &request);
    }

    if (rank == 0)
        printf("rank %d: stopped after %d sweeps, global residual %g, current local residual %g\n",
               rank, sweeps, res_global, res_local);

    MPI_Finalize();
    return 0;
}
