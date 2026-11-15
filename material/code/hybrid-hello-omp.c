#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    printf("serial part: %d thread\n", omp_get_num_threads());

    #pragma omp parallel
    {
        int id = omp_get_thread_num();          // declared in the region: private to each thread
        printf("hello from thread %d of %d\n", id, omp_get_num_threads());
    }

    printf("serial part again\n");
    return 0;
}
