#ifndef ZEN3_GEMM_H_
#define ZEN3_GEMM_H_

namespace gemm {

namespace zen3 {
    void printf();

    /*
    C = α × op(A) × op(B) + β × C
    */
    void sgemm (
        const char transa,
        const char transb,
        const int m,
        const int n,
        const int k,
        const float alpha,
        const float *a,
        const int lda,
        const float *b,
        const int ldb,
        const float beta,
        float *c,
        const int ldc
    ) ;


}

}



#endif // ZEN3_GEMM_H_