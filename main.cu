#include <cstdio>
#include <nvtx3/nvToolsExt.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <vector>

#include <cublas_v2.h>

#include "mult_cublas.cuh"
#include "mult_one_thread_per_tile_in_c_mat.cuh"
#include "mult_naive.cuh"
#include "mult_tiled_32x32_conventional.cuh"
#include "mult_one_warp_per_tile.cuh"
#include "mult_one_warp_per_tile_con_switch.cuh"
#include "mult_multiple_warps.cuh"
#include "mult_multiple_warps_con_switch.cuh"

#include "util.cuh"

#ifndef MATRIX_SIZE_A
#define MATRIX_SIZE_A 1024
#endif

#ifndef MATRIX_SIZE_B
#define MATRIX_SIZE_B 1024
#endif

struct Algorithm {
    const char *name;
    void (*function)(const float* mat_a, const float* mat_b, float* mat_c, int N);
};

constexpr int ITERATIONS = 1;

const std::vector<Algorithm> algorithms = {
    { "mult_cublas", &mult_cublas },
    { "mult_tiled_32x32_conventional", &mult_tiled_32x32_conventional },
    { "mult_one_thread_per_tile_in_c_mat", &mult_one_thread_per_tile_in_c_mat },
    { "mult_naive", &mult_naive },
    { "mult_multiple_warps", &mult_multiple_warps },
    { "mult_multiple_warps_con_switch", &mult_multiple_warps_con_switch },
    { "mult_one_warp_per_tile", &mult_one_warp_per_tile },
    { "mult_one_warp_per_tile_con_switch", &mult_one_warp_per_tile_con_switch },
};

// Takes device pointers
bool verify(const float *correct_mat, const float *to_verify_mat, size_t length)
{
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    float *mat_cpy;
    CUDA_CHK(cudaMalloc((void **)&mat_cpy, length * sizeof(float)));
    CUDA_CHK(cudaMemcpy(mat_cpy, to_verify_mat, length * sizeof(float), cudaMemcpyHostToDevice));

    /* step 1: create cublas handle, bind a stream */
    cublasCreate(&cublasH);

    CUDA_CHK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cublasSetStream(cublasH, stream);

    float alpha = -1.0f;
    cublasSaxpy_v2(cublasH, length, &alpha, correct_mat, 1, mat_cpy, 1);
    float res = 0.0f;
    cublasSnrm2_v2(cublasH, length, mat_cpy, 1, &res);

    CUDA_CHK(cudaStreamSynchronize(stream));
    cublasDestroy(cublasH);
    CUDA_CHK(cudaStreamDestroy(stream));

    printf("NORM %f", res);

    return res < 0.0001f;
}

// mat_c has to be a device pointer
void print_matrix(const char *label, float *mat_c, int M, int N)
{
    float *h_mat_c = (float *)malloc(M*N * sizeof(float));
    CUDA_CHK(cudaMemcpy(h_mat_c, mat_c, M*N * sizeof(float), cudaMemcpyDeviceToHost));
    printf("============== %s =================\n", label);
    for (int y = 0; y < M; y++)
    {
        for (int x = 0; x < N; x++)
        {
            printf("%1.1f ", h_mat_c[y * N + x]);
        }
        printf("\n");
    }
    printf("==================================\n");
}

int main(int argc, char *argv[])
{
    int N = 5*32*3*10;
    //int N = 160;
    int array_size = N * N;
    float *h_mat_a = (float *)malloc(array_size * sizeof(float));
    float *h_mat_b = (float *)malloc(array_size * sizeof(float));
    float *h_mat_c = (float *)malloc(array_size * sizeof(float));

    srand(1231323);
    for (int i = 0; i < array_size; i++)
    {
        //h_mat_a[i] = 1.0f; //rand() % 11;
        h_mat_a[i] = rand() % 11;
        //h_mat_b[i] = 1.0f; //rand() % 11;
        h_mat_b[i] = rand() % 11;
        h_mat_c[i] = 0.0f;
    }

    float *d_mat_a, *d_mat_b, *d_mat_c_correct;
    CUDA_CHK(cudaMalloc((void **)&d_mat_a, array_size * sizeof(float)));
    CUDA_CHK(cudaMalloc((void **)&d_mat_b, array_size * sizeof(float)));
    CUDA_CHK(cudaMalloc((void **)&d_mat_c_correct, array_size * sizeof(float)));
    
    // Copio matrices al device
    CUDA_CHK(cudaMemcpy(d_mat_a, h_mat_a, array_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_mat_b, h_mat_b, array_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_mat_c_correct, h_mat_c, array_size * sizeof(float), cudaMemcpyHostToDevice));

    mult_cublas(d_mat_a, d_mat_b, d_mat_c_correct, N);

    //print_matrix("A", d_mat_a, N, N);
    //print_matrix("B", d_mat_b, N, N);

    float *d_mat_c_algorithm;
    CUDA_CHK(cudaMalloc((void **)&d_mat_c_algorithm, array_size * sizeof(float)));
    for (auto &algorithm : algorithms)
    {
        printf("%s:", algorithm.name);
        for (int i = 0; i < ITERATIONS; i++)
        {
            CUDA_CHK(cudaMemcpy(d_mat_c_algorithm, h_mat_c, array_size * sizeof(float), cudaMemcpyHostToDevice));
            nvtxRangePush(algorithm.name);
            algorithm.function(d_mat_a, d_mat_b, d_mat_c_algorithm, N);
            nvtxRangePop();
        }

        bool correct = verify(d_mat_c_correct, d_mat_c_algorithm, array_size);
        printf(" %s\n", correct ? "correct" : "incorrect");
    }
    CUDA_CHK(cudaFree(d_mat_c_algorithm));

    CUDA_CHK(cudaFree(d_mat_a));
    CUDA_CHK(cudaFree(d_mat_b));
    CUDA_CHK(cudaFree(d_mat_c_correct));

#ifdef PRINT
    CUDA_CHK(cudaMemcpy(h_mat_c, d_mat_c_correct, array_size * sizeof(float), cudaMemcpyDeviceToHost));
    printf("============== C =================\n");
    for (int y = 0; y < N; y++)
    {
        for (int x = 0; x < N; x++)
        {
            printf("%1.0f ", h_mat_c[y * N + x]);
        }
        printf("\n");
    }
    printf("==================================\n");
#endif

    free(h_mat_a);
    free(h_mat_b);
    free(h_mat_c);
}
