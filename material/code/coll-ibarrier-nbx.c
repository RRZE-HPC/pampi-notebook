#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // every rank sends its rank number to one or two targets that it chooses itself;
    // the targets do not know who sends them something, or how many messages they get
    srand(rank + 1);
    int ntargets = 1 + rand() % 2;
    int targets[2];
    MPI_Request sreq[2];
    for (int k = 0; k < ntargets; ++k) {
        targets[k] = (rank + 1 + rand() % (size - 1)) % size;  // any rank except myself
        MPI_Issend(&rank, 1, MPI_INT, targets[k], 0, MPI_COMM_WORLD, &sreq[k]);
    }

    MPI_Request barrier;
    int in_barrier = 0, done = 0, nrecv = 0;
    char received[256] = "";
    int len = 0;

    while (!done) {
        // receive whatever has arrived, from anybody
        int flag;
        MPI_Status status;
        MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
        if (flag) {
            int value;
            MPI_Recv(&value, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            len += snprintf(received + len, sizeof received - len, " %d", value);
            ++nrecv;
        }

        if (!in_barrier) {
            // all my messages have been received → tell the others with the nonblocking barrier
            int sent;
            MPI_Testall(ntargets, sreq, &sent, MPI_STATUSES_IGNORE);
            if (sent) {
                MPI_Ibarrier(MPI_COMM_WORLD, &barrier);
                in_barrier = 1;
            }
        } else {
            // barrier complete → all messages of all ranks have been received
            MPI_Test(&barrier, &done, MPI_STATUS_IGNORE);
        }
    }

    if (ntargets == 1)
        printf("rank %d: sent to %d,    received from%s\n", rank, targets[0], received);
    else
        printf("rank %d: sent to %d, %d, received from%s\n", rank, targets[0], targets[1], received);

    int counts[2] = { ntargets, nrecv }, totals[2];
    MPI_Reduce(counts, totals, 2, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("rank 0: %d messages sent, %d received in total\n", totals[0], totals[1]);

    MPI_Finalize();
    return 0;
}
