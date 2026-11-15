#include <stdio.h>
#include <mpi.h>

// element-wise maximum of the absolute values: inoutvec[i] = max(|invec[i]|, |inoutvec[i]|)
void absmax(void *invec, void *inoutvec, int *len, MPI_Datatype *datatype) {
    double *in = invec, *inout = inoutvec;
    for (int i = 0; i < *len; ++i) {
        double a = in[i]    < 0.0 ? -in[i]    : in[i];
        double b = inout[i] < 0.0 ? -inout[i] : inout[i];
        inout[i] = a > b ? a : b;
    }
}

int main(int argc, char **argv) {
    int rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double value = (rank % 2 ? -1.0 : 1.0) * (rank + 0.5);     // 0.5, -1.5, 2.5, -3.5, ...
    double result;

    MPI_Op op;
    MPI_Op_create(absmax, 1, &op);                              // 1: commutative
    MPI_Reduce(&value, &result, 1, MPI_DOUBLE, op, 0, MPI_COMM_WORLD);
    MPI_Op_free(&op);

    printf("rank %d: value = %4.1f\n", rank, value);
    if (rank == 0)
        printf("rank 0: largest absolute value = %.1f\n", result);

    MPI_Finalize();
    return 0;
}
