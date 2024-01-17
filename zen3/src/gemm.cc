#include "zen3/gemm.h"
#include <iostream>

#define A(i_, j_) A[ (i_) + (j_) * lda ]
#define B(i_, j_) B[ (i_) + (j_) * ldb ]
#define C(i_, j_) C[ (i_) + (j_) * ldc ]

void gemm::zen3::printf()
{
    std::cout << "This is zen3 gemm implement~" << std::endl;
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
    for (j = 0; j < n; j ++) {
        for (i = 0; i < m; i ++) {
            C(i, j) *= beta;
            for (p = 0; p < k; p ++) {
                C(i, j) += alpha * A(i, p) * B(p, j);
            }
        }
    }
}
