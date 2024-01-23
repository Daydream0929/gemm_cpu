#include "zen3/gemm.h"
#include <iostream>

#define A(i_, j_) A[ (i_) + (j_) * lda ]
#define B(i_, j_) B[ (i_) + (j_) * ldb ]
#define C(i_, j_) C[ (i_) + (j_) * ldc ]

void gemm::zen3::AddDot1x4(int k, int alpha, int beta, const float* A, int lda, const float* B, int ldb, float *C, int ldc)
{
    C(0, 0) *= beta;
    C(0, 1) *= beta;
    C(0, 2) *= beta;
    C(0, 3) *= beta;

    int p;
    register float c_00_reg, c_01_reg, c_02_reg, c_03_reg, a_0p_reg, alpha_reg;
    c_00_reg = 0.0f;
    c_01_reg = 0.0f;
    c_02_reg = 0.0f;
    c_03_reg = 0.0f;
    alpha_reg = alpha;

    const float *bp0_pntr, *bp1_pntr, *bp2_pntr, *bp3_pntr;
    bp0_pntr = &B(0, 0);
    bp1_pntr = &B(0, 1);
    bp2_pntr = &B(0, 2);
    bp3_pntr = &B(0, 3);
    
    for (p = 0; p < k; p ++) {
        a_0p_reg = A(0, p);
        c_00_reg += alpha_reg * a_0p_reg * (*bp0_pntr++);
        c_01_reg += alpha_reg * a_0p_reg * (*bp1_pntr++);
        c_02_reg += alpha_reg * a_0p_reg * (*bp2_pntr++);
        c_03_reg += alpha_reg * a_0p_reg * (*bp3_pntr++);
    }

    C(0, 0) += c_00_reg;
    C(0, 1) += c_01_reg;
    C(0, 2) += c_02_reg;
    C(0, 3) += c_03_reg;
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
            AddDot1x4(k, alpha, beta, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
        }
    }
}
