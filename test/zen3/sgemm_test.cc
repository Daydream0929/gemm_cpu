#include <benchmark/benchmark.h>
#include <cblas.h>
#include "zen3/gemm.h"
#include <chrono>
#include <iostream>

// 定义 OpenBLAS_GEMM 操作的基准测试函数
static void openblas_gemm(benchmark::State &state)
{
    const int M = state.range(0);
    const int N = M;
    const int K = M;

    std::vector<float> A(M * K, 1.0f); // 使用 1.0f 初始化矩阵 A
    std::vector<float> B(K * N, 1.0f); // 使用 1.0f 初始化矩阵 B
    std::vector<float> C(M * N, 0.0f); // 使用 0.0f 初始化矩阵 C

    for (auto _ : state)
    {
        // 执行 GEMM
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f,
                    A.data(), K, B.data(), N, 0.0f, C.data(), N);   
    }

    // 计算执行的总浮点运算次数
    auto ops = 2.0 * M * N * K;

    // 计算 GFlops
    state.SetComplexityN(state.range(0) * state.range(0) * state.range(0));
    state.counters["GFlops"] =
        benchmark::Counter(ops / 1e9, benchmark::Counter::kIsRate);
}

// 注册基准测试函数
BENCHMARK(openblas_gemm)
    ->RangeMultiplier(2)  // 设置增长倍数
    ->Ranges({{2, 2048}}) // K 的范围
    ->UseRealTime();


// 定义 MY_GEMM 操作的基准测试函数
static void zen3_gemm(benchmark::State &state)
{
    const int M = state.range(0);
    const int N = M;
    const int K = M;

    std::vector<float> A(M * K, 1.0f); // 使用 1.0f 初始化矩阵 A
    std::vector<float> B(K * N, 1.0f); // 使用 1.0f 初始化矩阵 B
    std::vector<float> C(M * N, 0.0f); // 使用 0.0f 初始化矩阵 C

    for (auto _ : state)
    {
        // 执行 GEMM
        gemm::zen3::sgemm(CblasNoTrans, CblasNoTrans, M, N, K, 1.0f,
                    A.data(), K, B.data(), N, 0.0f, C.data(), N);
    }

    // 计算执行的总浮点运算次数
    auto ops = 2.0 * M * N * K;

    // 计算 GFlops
    state.SetComplexityN(state.range(0) * state.range(0) * state.range(0));
    state.counters["GFlops"] =
        benchmark::Counter(ops / 1e9, benchmark::Counter::kIsRate);
}

// 注册基准测试函数
BENCHMARK(zen3_gemm)
    ->RangeMultiplier(2)  // 设置增长倍数
    ->Ranges({{2, 2048}}) // K 的范围
    ->UseRealTime();

BENCHMARK_MAIN();
