#include <cstdio>
#include <nvtx3/nvToolsExt.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>

#include <stdio.h>

#include "util.cuh"

#include "mult_one_thread_per_tile_in_c_mat.cuh"
#include "mult_naive.cuh"
#include "mult_2.cuh"

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

constexpr int ITERATIONS = 3;

const std::vector<Algorithm> algorithms = {
    { "mult_naive", &mult_naive },
    { "mult_one_thread_per_tile_in_c_mat", &mult_one_thread_per_tile_in_c_mat },
    { "mult_2", &mult_2 },
};

// Takes device pointers
bool verify(const float *correct_mat, const float *to_verify_mat, size_t length)
{
    return thrust::equal(thrust::device, correct_mat, correct_mat + length, to_verify_mat);
}

int main(int argc, char *argv[])
{
    int N = 5*32*20;
    int array_size = N * N;
    float *h_mat_a = (float *)malloc(array_size * sizeof(float));
    float *h_mat_b = (float *)malloc(array_size * sizeof(float));
    float *h_mat_c = (float *)malloc(array_size * sizeof(float));

    // srand(1231323);
    for (int i = 0; i < array_size; i++)
    {
        h_mat_a[i] = 1.0f; //rand() % 11;
        h_mat_b[i] = 1.0f; //rand() % 11;
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

    mult_naive(d_mat_a, d_mat_b, d_mat_c_correct, N);
  
    float *d_mat_c_algorithm;
    CUDA_CHK(cudaMalloc((void **)&d_mat_c_algorithm, array_size * sizeof(float)));
    for (auto &algorithm : algorithms)
    {
        for (int i = 0; i < ITERATIONS; i++)
        {
            CUDA_CHK(cudaMemcpy(d_mat_c_algorithm, h_mat_c, array_size * sizeof(float), cudaMemcpyHostToDevice));
            nvtxRangePush(algorithm.name);
            algorithm.function(d_mat_a, d_mat_b, d_mat_c_algorithm, N);
            nvtxRangePop();
        }

        bool correct = verify(d_mat_c_correct, d_mat_c_algorithm, array_size);
        printf("%s: %s\n", algorithm.name, correct ? "correct" : "incorrect");
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
