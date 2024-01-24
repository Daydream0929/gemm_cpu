#include <iostream>
#include <zen3/gemm.h>
#include <cblas.h>

int main()
{
    std::cout << "This is my optimized_gemm for zen3 ...." << std::endl;
    int m = 12, k = 12, n = 12;
    float alpha = 1.0f, beta = 0.0f;
    float *A = (float*)malloc(m * k);
    float *B = (float*)malloc(k * n);
    
    for (int i = 0; i < m * k; i ++) A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < k * n; i ++) B[i] = rand() / (float)RAND_MAX;
    
    float C[m * n] = {0.0};
    float CC[m * n] = {0.0};

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, m, B, k, beta, CC, m);
    gemm::zen3::sgemm('N', 'N', m, n, k, alpha, A, m, B, k, beta, C, m);
    

    std::cout << "My_Gemm result " << std::endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%f  ", C[i + j * m]);
        }
        printf("\n");
    }

    std::cout << "Openblas_Gemm result" << std::endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%f  ", CC[i + j * m]);
        }
        printf("\n");
    }

    return 0;
}