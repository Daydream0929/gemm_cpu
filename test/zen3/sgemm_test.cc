#include <benchmark/benchmark.h>
#include <cblas.h>
#include "zen3/gemm.h"

static void BM_zne3_sgemm(benchmark::State& state) {
    int N = state.range(0);
    float *A = new float[N * N];
    float *B = new float[N * N];
    float *C = new float[N * N];

    // 初始化 A 和 B
    // ...

    for (auto _ : state) {
        gemm::zen3::sgemm('N', 'N', N, N, N, 1.0f, A, N, B, N, 0.0f, C, N);
    }

    delete[] A;
    delete[] B;
    delete[] C;
}
BENCHMARK(BM_zne3_sgemm)->Range(8, 8<<10);

static void BM_OpenBLAS_sgemm(benchmark::State& state) {
    int N = state.range(0);
    float *A = new float[N * N];
    float *B = new float[N * N];
    float *C = new float[N * N];

    // 初始化 A 和 B
    // ...

    for (auto _ : state) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0f, A, N, B, N, 0.0f, C, N);
    }

    delete[] A;
    delete[] B;
    delete[] C;
}
BENCHMARK(BM_OpenBLAS_sgemm)->Range(8, 8<<10);

BENCHMARK_MAIN();
