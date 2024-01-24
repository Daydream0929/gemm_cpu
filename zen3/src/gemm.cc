#include "zen3/gemm.h"
#include <iostream>

#define A(i_, j_) A[(i_) + (j_) * lda]
#define B(i_, j_) B[(i_) + (j_) * ldb]
#define C(i_, j_) C[(i_) + (j_) * ldc]

void gemm::zen3::AddDot1x4(int k, int alpha, int beta, const float *A, int lda,
                           const float *B, int ldb, float *C, int ldc)
{
    C(0, 0) += beta * C(0, 0);
    C(0, 1) += beta * C(0, 1);
    C(0, 2) += beta * C(0, 2);
    C(0, 3) += beta * C(0, 3);

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

    for (p = 0; p < k; p += 4)
    {
        a_0p_reg = A(0, p);

        c_00_reg += alpha_reg * a_0p_reg * *bp0_pntr++;
        c_01_reg += alpha_reg * a_0p_reg * *bp1_pntr++;
        c_02_reg += alpha_reg * a_0p_reg * *bp2_pntr++;
        c_03_reg += alpha_reg * a_0p_reg * *bp3_pntr++;

        a_0p_reg = A(0, p + 1);

        c_00_reg += alpha_reg * a_0p_reg * *bp0_pntr++;
        c_01_reg += alpha_reg * a_0p_reg * *bp1_pntr++;
        c_02_reg += alpha_reg * a_0p_reg * *bp2_pntr++;
        c_03_reg += alpha_reg * a_0p_reg * *bp3_pntr++;

        a_0p_reg = A(0, p + 2);

        c_00_reg += alpha_reg * a_0p_reg * *bp0_pntr++;
        c_01_reg += alpha_reg * a_0p_reg * *bp1_pntr++;
        c_02_reg += alpha_reg * a_0p_reg * *bp2_pntr++;
        c_03_reg += alpha_reg * a_0p_reg * *bp3_pntr++;

        a_0p_reg = A(0, p + 3);

        c_00_reg += alpha_reg * a_0p_reg * *bp0_pntr++;
        c_01_reg += alpha_reg * a_0p_reg * *bp1_pntr++;
        c_02_reg += alpha_reg * a_0p_reg * *bp2_pntr++;
        c_03_reg += alpha_reg * a_0p_reg * *bp3_pntr++;
    }

    C(0, 0) += c_00_reg;
    C(0, 1) += c_01_reg;
    C(0, 2) += c_02_reg;
    C(0, 3) += c_03_reg;
}

void gemm::zen3::AddDot4x4(int k, int alpha, int beta, const float *A, int lda,
                           const float *B, int ldb, float *C, int ldc)
{
    int p;

    register float 
        c_00_reg, c_01_reg, c_02_reg, c_03_reg,
        c_10_reg, c_11_reg, c_12_reg, c_13_reg,
        c_20_reg, c_21_reg, c_22_reg, c_23_reg,
        c_30_reg, c_31_reg, c_32_reg, c_33_reg;

    register float 
        a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg,
        b_p0_reg, b_p1_reg, b_p2_reg, b_p3_reg;
    
    const float 
        *bp0_pntr, *bp1_pntr, *bp2_pntr, *bp3_pntr;

    c_00_reg = 0.0f, c_01_reg = 0.0f, c_02_reg = 0.0f, c_03_reg = 0.0f;
    c_10_reg = 0.0f, c_11_reg = 0.0f, c_12_reg = 0.0f, c_13_reg = 0.0f;
    c_20_reg = 0.0f, c_21_reg = 0.0f, c_22_reg = 0.0f, c_23_reg = 0.0f;
    c_30_reg = 0.0f, c_31_reg = 0.0f, c_32_reg = 0.0f, c_33_reg = 0.0f;

    bp0_pntr = &B(0, 0);
    bp1_pntr = &B(0, 1);
    bp2_pntr = &B(0, 2);
    bp3_pntr = &B(0, 3);

    for (p = 0; p < k; p ++ ) {
        a_0p_reg = A(0, p);
        a_1p_reg = A(1, p);
        a_2p_reg = A(2, p);
        a_3p_reg = A(3, p);

        b_p0_reg = *bp0_pntr++;
        b_p1_reg = *bp1_pntr++;
        b_p2_reg = *bp2_pntr++;
        b_p3_reg = *bp3_pntr++;

        c_00_reg += a_0p_reg * b_p0_reg;
        c_01_reg += a_0p_reg * b_p1_reg;    
        c_02_reg += a_0p_reg * b_p2_reg;    
        c_03_reg += a_0p_reg * b_p3_reg; 

        c_10_reg += a_1p_reg * b_p0_reg;
        c_11_reg += a_1p_reg * b_p1_reg;    
        c_12_reg += a_1p_reg * b_p2_reg;     
        c_13_reg += a_1p_reg * b_p3_reg; 

        c_20_reg += a_2p_reg * b_p0_reg;
        c_21_reg += a_2p_reg * b_p1_reg;    
        c_22_reg += a_2p_reg * b_p2_reg;     
        c_23_reg += a_2p_reg * b_p3_reg; 

        c_30_reg += a_3p_reg * b_p0_reg;
        c_31_reg += a_3p_reg * b_p1_reg;    
        c_32_reg += a_3p_reg * b_p2_reg;     
        c_33_reg += a_3p_reg * b_p3_reg;
    }

    C(0, 0) += c_00_reg, C(0, 1) += c_01_reg, C(0, 2) += c_02_reg, C(0, 3) += c_03_reg;
    C(1, 0) += c_10_reg, C(1, 1) += c_11_reg, C(1, 2) += c_12_reg, C(1, 3) += c_13_reg;
    C(2, 0) += c_20_reg, C(2, 1) += c_21_reg, C(2, 2) += c_22_reg, C(2, 3) += c_23_reg;
    C(3, 0) += c_30_reg, C(3, 1) += c_31_reg, C(3, 2) += c_32_reg, C(3, 3) += c_33_reg;
}

void gemm::zen3::sgemm(const char transa, const char transb, const int m,
                       const int n, const int k, const float alpha,
                       const float *A, const int lda, const float *B,
                       const int ldb, const float beta, float *C, const int ldc)
{
    int i, j;
    for (j = 0; j < n; j += 4)
    {
        for (i = 0; i < m; i += 4)
        {
            AddDot4x4(k, alpha, beta, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
        }
    }
    // for (int j = 0; j < n; j += 4) {
    //     for (int i = 0; i < m; i ++) {
    //         AddDot1x4(k, alpha, beta, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
    //     } 
    // }
}
