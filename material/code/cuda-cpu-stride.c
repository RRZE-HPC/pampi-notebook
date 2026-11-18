#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

// wall-clock time in seconds; timespec_get() is standard C11, unlike POSIX clock_gettime()
static double seconds(void) {
    struct timespec t;
    timespec_get(&t, TIME_UTC);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

int main(void) {
    const int64_t n = 1 << 25;                  // 256 MiB
    int64_t *a = malloc(n * sizeof(int64_t));
    for (int64_t i = 0; i < n; ++i)
        a[i] = i;

    printf("stride   elements  time [ms]  useful GB/s\n");
    for (int stride = 1; stride <= 64; stride *= 2) {
        int64_t sum = 0;
        double best = 1e30;
        for (int r = 0; r < 5; ++r) {           // the fastest of 5 runs
            double t0 = seconds();
            if (stride == 1)                    // a separate loop with a constant stride can be vectorized
                for (int64_t i = 0; i < n; ++i)
                    sum += a[i];
            else
                for (int64_t i = 0; i < n; i += stride)
                    sum += a[i];
            double t = seconds() - t0;
            best = t < best ? t : best;
        }
        const int64_t count = n / stride;       // elements read
        printf("%6d  %9lld  %9.2f  %11.2f   (sum %lld)\n", stride, (long long)count, 1e3 * best,
               count * sizeof(int64_t) / best / 1e9, (long long)sum);
    }
    free(a);
    return 0;
}
