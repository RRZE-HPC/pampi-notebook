#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    const int N = 3000, ROUNDS = 20;

    double *mat = malloc((size_t)N * N * sizeof(double));
    double *b = malloc(N * sizeof(double)), *c = malloc(N * sizeof(double));
    for (size_t k = 0; k < (size_t)N * N; ++k)
        mat[k] = 1.0 / (1 + k % 7);
    for (int i = 0; i < N; ++i)
        b[i] = 1.0;

    const char *names[]  = { "static", "static,16", "dynamic,16", "guided" };
    omp_sched_t kinds[]  = { omp_sched_static, omp_sched_static, omp_sched_dynamic, omp_sched_guided };
    int chunks[]         = { 0, 16, 16, 0 };    // 0: the default chunk size
    const int nthreads   = omp_get_max_threads();
    double *busy = malloc(nthreads * sizeof(double));

    printf("%-12s %10s", "schedule", "total[ms]");
    for (int t = 0; t < nthreads; ++t)
        printf("  thread%d[ms]", t);
    printf("\n");

    for (int s = 0; s < 4; ++s) {
        omp_set_schedule(kinds[s], chunks[s]);  // used by schedule(runtime)
        double t0 = omp_get_wtime();

        #pragma omp parallel
        {
            double mine = 0.0;
            for (int k = 0; k < ROUNDS; ++k) {
                double start = omp_get_wtime();
                #pragma omp for schedule(runtime) nowait
                for (int i = 0; i < N; ++i) {
                    double sum = 0.0;
                    for (int j = i; j < N; ++j)
                        sum += mat[(size_t)i * N + j] * b[j];
                    c[i] = sum;
                }
                mine += omp_get_wtime() - start;    // the time this thread spent working
                #pragma omp barrier
            }
            busy[omp_get_thread_num()] = mine;
        }

        double total = omp_get_wtime() - t0, checksum = 0.0;
        for (int i = 0; i < N; ++i)
            checksum += c[i];                       // use the result, so the compiler can't drop the loop
        printf("%-12s %10.1f", names[s], 1e3 * total);
        for (int t = 0; t < nthreads; ++t)
            printf(" %12.1f", 1e3 * busy[t]);
        printf("   (checksum %.6e)\n", checksum);
    }

    free(mat);
    free(b);
    free(c);
    free(busy);
    return 0;
}
