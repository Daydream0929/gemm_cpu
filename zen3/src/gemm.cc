#include "zen3/gemm.h"
#include <iostream>

#define A(i_, j_) A[ (i_) + (j_) * lda ]
#define B(i_, j_) B[ (i_) + (j_) * ldb ]
#define C(i_, j_) C[ (i_) + (j_) * ldc ]

void gemm::zen3::AddDot1x4(int k, int alpha, int beta, const float* A, int lda, const float* B, int ldb, float *C, int ldc)
{
    int p;
    
    *C *= beta;
    for (p = 0; p < k; p ++) {
        C(0, 0) += alpha * A(0, p) * B(p, 0);
    }

    *C *= beta;
    for (p = 0; p < k; p ++) {
        C(0, 1) += alpha * A(0, p) * B(p, 1);
    }

    *C *= beta;
    for (p = 0; p < k; p ++) {
        C(0, 2) += alpha * A(0, p) * B(p, 2);
    }

    *C *= beta;
    for (p = 0; p < k; p ++) {
        C(0, 3) += alpha * A(0, p) * B(p, 3);
    }

}

void gemm::zen3::sgemm(
    const char transa,
    const char transb,
    const int m,
    const int n,
    const int k,
    const float alpha,
    const float *A,
    const int lda,
    const float *B,
    const int ldb,
    const float beta,
    float *C,
    const int ldc
)
{
    int i, j, p;
    for (j = 0; j < n; j += 4) {
        for (i = 0; i < m; i ++) {
            // AddDot1x4(k, alpha, beta, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
            C(i, j)     += beta * C(i, j);
            C(i, j + 1) += beta * C(i, j + 1);
            C(i, j + 2) += beta * C(i, j + 2);
            C(i, j + 3) += beta * C(i, j + 3);
            for (p = 0; p < k; p ++) {
                C(i, j) += alpha * A(i, p) * B(p, j);
                C(i, j + 1) += alpha * A(i, p) * B(p, j + 1);
                C(i, j + 2) += alpha * A(i, p) * B(p, j + 2);
                C(i, j + 3) += alpha * A(i, p) * B(p, j + 3);
            }
        }
    }
}
