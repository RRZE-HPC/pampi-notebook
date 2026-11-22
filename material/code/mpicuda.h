// Helpers for the MPI+CUDA examples of the PAMPI notebooks.
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#if defined(OPEN_MPI) && OPEN_MPI
#include <mpi-ext.h>                // Open MPI extensions, among them MPIX_Query_cuda_support()
#endif
#include "cuda-check.h"             // CUDA_CHECK() and GpuTimer from the CUDA primer

// The rank of this process among the processes on its node: its rank in a shared-memory communicator.
static inline int node_rank(void) {
    MPI_Comm node;
    int local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
    MPI_Comm_rank(node, &local);
    MPI_Comm_free(&node);
    return local;
}

// One GPU per process: process k of a node uses GPU k modulo the number of GPUs it can see.
// Call it before any other CUDA call, so that the CUDA context is created on the right GPU.
// Returns the number of the GPU.
static inline int select_gpu(void) {
    int ngpus;
    CUDA_CHECK(cudaGetDeviceCount(&ngpus));         // does not create a context
    const int gpu = node_rank() % ngpus;
    CUDA_CHECK(cudaSetDevice(gpu));
    return gpu;
}

// Can the MPI library work on device pointers? 1: yes, 0: no, -1: the library can't tell.
// *how describes where the answer comes from. The environment variable PAMPI_CUDA_AWARE=1 or 0
// overrides it - set it to 1 only if you know that your library is CUDA-aware (e.g. Cray MPICH
// with MPICH_GPU_SUPPORT_ENABLED=1): a library that is not crashes on device pointers.
static inline int mpi_cuda_aware(const char **how) {
    const char *env = getenv("PAMPI_CUDA_AWARE");
    if (env && *env) {
        *how = "set by PAMPI_CUDA_AWARE";
        return atoi(env) != 0;
    }
#if defined(MPIX_CUDA_AWARE_SUPPORT)
    *how = "Open MPI: MPIX_Query_cuda_support()";
    return MPIX_Query_cuda_support();
#elif defined(MPICH_NUMVERSION) && MPICH_NUMVERSION >= 40000000 && !defined(I_MPI_VERSION)
    int supported = 0;
    MPIX_GPU_query_support(MPIX_GPU_SUPPORT_CUDA, &supported);
    *how = "MPICH: MPIX_GPU_query_support()";
    return supported;
#else
    *how = "no query available for this MPI library";
    return -1;
#endif
}

// The memory kinds the library reports for MPI_COMM_WORLD (MPI 4.1, info key mpi_memory_alloc_kinds),
// e.g. "system,mpi,cuda:device,cuda:host,cuda:managed"
static inline void mpi_memory_kinds(char *kinds, int len) {
#if MPI_VERSION > 4 || (MPI_VERSION == 4 && MPI_SUBVERSION >= 1)
    MPI_Info info;
    int flag = 0, buflen = len;
    MPI_Comm_get_info(MPI_COMM_WORLD, &info);
    MPI_Info_get_string(info, "mpi_memory_alloc_kinds", &buflen, kinds, &flag);
    if (!flag)
        snprintf(kinds, len, "not reported");
    MPI_Info_free(&info);
#else
    snprintf(kinds, len, "n/a - the library implements MPI %d.%d, memory kinds came with MPI 4.1", MPI_VERSION, MPI_SUBVERSION);
#endif
}
