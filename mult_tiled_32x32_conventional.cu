#include "mult_tiled_32x32_conventional.cuh"

#include "util.cuh"

constexpr int BLOCK_SIZE_X = 32;
constexpr int BLOCK_SIZE_Y = 32;

constexpr int TILE_LENGTH = 32;

__global__ void mult_tiled_32x32_conventional_kernel(const float* mat_a, const float* mat_b, float* mat_c, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float tile_a[BLOCK_SIZE_Y][TILE_LENGTH];
    __shared__ float tile_b[TILE_LENGTH][BLOCK_SIZE_X];

    float value = 0.0f;
    for (int tile_i = 0; tile_i < N / TILE_LENGTH; tile_i++)
    {
        tile_a[threadIdx.y][threadIdx.x] = mat_a[(y)*N + (tile_i*TILE_LENGTH+threadIdx.x)];
        tile_b[threadIdx.y][threadIdx.x] = mat_b[(tile_i*TILE_LENGTH+threadIdx.y)*N + (x)];

        __syncthreads();

        for (int i = 0; i < TILE_LENGTH; i++)
            value += tile_a[threadIdx.y][i] * tile_b[i][threadIdx.x];

        __syncthreads();
    }

    mat_c[y * N + x] = value;
}

void mult_tiled_32x32_conventional(const float* mat_a, const float* mat_b, float* mat_c, int N)
{
    dim3 gridDim(N / BLOCK_SIZE_X, N / BLOCK_SIZE_Y, 1);
    dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    mult_tiled_32x32_conventional_kernel<<<gridDim, blockDim>>>(mat_a, mat_b, mat_c, N);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());
}

