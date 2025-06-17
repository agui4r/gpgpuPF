#include "mult_naive.cuh"

#include "util.cuh"

__global__ void mult_naive_kernel(const float* mat_a, const float* mat_b, float* mat_c, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= N || y >= N) return;

    mat_c[y * N + x] = 0.0f;
    for (int o = 0; o < N; o++)
    {
        mat_c[y * N + x] += mat_a[y * N + o] * mat_b[o * N + x];
    }
}

void mult_naive(const float* mat_a, const float* mat_b, float* mat_c, int N)
{
    constexpr int BLOCK_SIZE_X = 32;
    constexpr int BLOCK_SIZE_Y = 32;
    dim3 gridDim(N / BLOCK_SIZE_X, N / BLOCK_SIZE_Y, 1);
    dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    mult_naive_kernel<<<gridDim, blockDim>>>(mat_a, mat_b, mat_c, N);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());
}

