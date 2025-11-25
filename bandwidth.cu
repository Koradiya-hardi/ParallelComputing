// bandwidth.cu
// Measures Host->Device and Device->Host bandwidth for pageable and pinned host memory.
// Produces CSV to stdout or to file when run with --output filename

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <chrono>

#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

int main(int argc, char** argv) {
    // settings
    size_t min_size = 1 << 20;   // 1 MB
    size_t max_size = 1 << 30;   // 1 GB
    int iterations = 10;         // iterations to average (first is warmup)
    const char* out_filename = nullptr;

    // parse optional args: --min, --max, --iter, --output
    for (int i=1;i<argc;i++){
        if (!strcmp(argv[i],"--min") && i+1<argc) { min_size = (size_t)atoll(argv[++i]); }
        else if (!strcmp(argv[i],"--max") && i+1<argc) { max_size = (size_t)atoll(argv[++i]); }
        else if (!strcmp(argv[i],"--iter") && i+1<argc) { iterations = atoi(argv[++i]); }
        else if (!strcmp(argv[i],"--output") && i+1<argc) { out_filename = argv[++i]; }
        else {
            fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            return 1;
        }
    }

    // CSV header
    FILE* out = stdout;
    if (out_filename) {
        out = fopen(out_filename,"w");
        if (!out) { perror("fopen"); out = stdout; }
    }
    fprintf(out, "size_bytes,pageable_h2d_gb_s,pageable_d2h_gb_s,pinned_h2d_gb_s,pinned_d2h_gb_s\n");
    fflush(out);

    // device check
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return 1;
    }
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "Using device 0: %s\n", prop.name);

    // sizes: powers of two from min_size up to max_size
    for (size_t size = min_size; size <= max_size; size <<= 1) {
        // align size to floats (use float array)
        size_t nFloats = (size + sizeof(float)-1) / sizeof(float);

        // --- Pageable host memory measurement ---
        float* h_page = (float*)malloc(nFloats * sizeof(float));
        if (!h_page) {
            fprintf(stderr, "malloc failed for size %zu bytes (pageable). Skipping.\n", nFloats * sizeof(float));
            fprintf(out, "%zu,NaN,NaN,NaN,NaN\n", nFloats * sizeof(float));
            fflush(out);
            continue;
        }
        // init
        for (size_t i=0;i<nFloats;i++) h_page[i] = (float)(i & 0xFFFF);

        float *d_buf = nullptr;
        cudaError_t e = cudaMalloc((void**)&d_buf, nFloats * sizeof(float));
        if (e != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed for size %zu bytes: %s. Skipping size.\n", nFloats * sizeof(float), cudaGetErrorString(e));
            free(h_page);
            fprintf(out, "%zu,NaN,NaN,NaN,NaN\n", nFloats * sizeof(float));
            fflush(out);
            continue;
        }

        // CUDA events for timing
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        double sum_pageable_h2d_ms = 0.0;
        double sum_pageable_d2h_ms = 0.0;

        // iterations (first warmup)
        for (int it=0; it<iterations; ++it) {
            // H2D
            CHECK_CUDA(cudaEventRecord(start, 0));
            CHECK_CUDA(cudaMemcpy(d_buf, h_page, nFloats * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaEventRecord(stop, 0));
            CHECK_CUDA(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
            if (it > 0) sum_pageable_h2d_ms += ms; // skip first
            // D2H
            CHECK_CUDA(cudaEventRecord(start, 0));
            CHECK_CUDA(cudaMemcpy(h_page, d_buf, nFloats * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaEventRecord(stop, 0));
            CHECK_CUDA(cudaEventSynchronize(stop));
            ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
            if (it > 0) sum_pageable_d2h_ms += ms;
        }

        int measured = iterations - 1;
        double avg_pageable_h2d_ms = (measured>0) ? (sum_pageable_h2d_ms / measured) : NAN;
        double avg_pageable_d2h_ms = (measured>0) ? (sum_pageable_d2h_ms / measured) : NAN;

        // compute GB/s: bytes / seconds / 1e9
        double bytes = (double)nFloats * sizeof(float);
        double pageable_h2d_gb_s = bytes / (avg_pageable_h2d_ms / 1000.0) / 1e9;
        double pageable_d2h_gb_s = bytes / (avg_pageable_d2h_ms / 1000.0) / 1e9;

        // cleanup pageable
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(d_buf));
        free(h_page);

        // --- Pinned (page-locked) host memory measurement ---
        float* h_pinned = nullptr;
        cudaError_t cerr = cudaHostAlloc((void**)&h_pinned, nFloats * sizeof(float), cudaHostAllocDefault);
        if (cerr != cudaSuccess) {
            fprintf(stderr, "cudaHostAlloc failed for size %zu bytes: %s. Skipping pinned.\n", nFloats * sizeof(float), cudaGetErrorString(cerr));
            fprintf(out, "%zu,%.6f,%.6f,NaN,NaN\n", nFloats * sizeof(float),
                    pageable_h2d_gb_s, pageable_d2h_gb_s);
            fflush(out);
            continue;
        }
        // init
        for (size_t i=0;i<nFloats;i++) h_pinned[i] = (float)(i & 0xFFFF);

        // allocate device buffer for pinned test
        float* d_buf2 = nullptr;
        cerr = cudaMalloc((void**)&d_buf2, nFloats * sizeof(float));
        if (cerr != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed for pinned test size %zu bytes: %s. Skipping pinned.\n", nFloats * sizeof(float), cudaGetErrorString(cerr));
            cudaFreeHost(h_pinned);
            fprintf(out, "%zu,%.6f,%.6f,NaN,NaN\n", nFloats * sizeof(float),
                    pageable_h2d_gb_s, pageable_d2h_gb_s);
            fflush(out);
            continue;
        }

        // events
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        double sum_pinned_h2d_ms = 0.0;
        double sum_pinned_d2h_ms = 0.0;

        for (int it=0; it<iterations; ++it) {
            CHECK_CUDA(cudaEventRecord(start, 0));
            CHECK_CUDA(cudaMemcpy(d_buf2, h_pinned, nFloats * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaEventRecord(stop, 0));
            CHECK_CUDA(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
            if (it>0) sum_pinned_h2d_ms += ms;

            CHECK_CUDA(cudaEventRecord(start, 0));
            CHECK_CUDA(cudaMemcpy(h_pinned, d_buf2, nFloats * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaEventRecord(stop, 0));
            CHECK_CUDA(cudaEventSynchronize(stop));
            ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
            if (it>0) sum_pinned_d2h_ms += ms;
        }

        double avg_pinned_h2d_ms = (measured>0) ? (sum_pinned_h2d_ms / measured) : NAN;
        double avg_pinned_d2h_ms = (measured>0) ? (sum_pinned_d2h_ms / measured) : NAN;
        double pinned_h2d_gb_s = bytes / (avg_pinned_h2d_ms / 1000.0) / 1e9;
        double pinned_d2h_gb_s = bytes / (avg_pinned_d2h_ms / 1000.0) / 1e9;

        // free pinned and device
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(d_buf2));
        CHECK_CUDA(cudaFreeHost(h_pinned));

        // Output CSV row and stdout
        fprintf(stderr, "Size %10zu bytes: pageable H2D=%.3f GB/s D2H=%.3f GB/s | pinned H2D=%.3f GB/s D2H=%.3f GB/s\n",
                nFloats * sizeof(float),
                pageable_h2d_gb_s, pageable_d2h_gb_s,
                pinned_h2d_gb_s, pinned_d2h_gb_s);
        fprintf(out, "%zu,%.6f,%.6f,%.6f,%.6f\n",
                nFloats * sizeof(float),
                pageable_h2d_gb_s, pageable_d2h_gb_s, pinned_h2d_gb_s, pinned_d2h_gb_s);
        fflush(out);
    }

    if (out != stdout) fclose(out);
    return 0;
}

