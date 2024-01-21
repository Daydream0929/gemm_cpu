#include <iostream>
#include <zen3/gemm.h>

int main()
{
    std::cout << "This is my optimized_gemm for zen3 ...." << std::endl;
    int m = 3, k = 2, n = 4;
    float alpha = 1.0f, beta = 0.0f;
    float A[m * k] = {1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0};
    float B[k * n] = {13.0, 17.0, 14.0, 18.0, 15.0, 19.0, 16.0, 20.0};
    float C[m * n];

    float CC[m * n];

    gemm::zen3::sgemm('N', 'N', m, n, k, alpha, A, m, B, k, beta, C, m);
    

     // Display the result
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%f  ", C[i + j * m]);
        }
        printf("\n");
    }

    return 0;
}