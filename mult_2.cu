#include "mult_2.cuh"

#include "util.cuh"
#include <cstdio>

#include "tables.cuh"

inline __device__ float calc_h(const float *a, const float *b, int h_idx)
{
    float sum_a = 0, sum_b = 0;

    for (int8_t *it = a_pos[h_idx]; *it != -1; it++) sum_a += a[*it];
    for (int8_t *it = a_neg[h_idx]; *it != -1; it++) sum_a -= a[*it];
    for (int8_t *it = b_pos[h_idx]; *it != -1; it++) sum_b += b[*it];
    for (int8_t *it = b_neg[h_idx]; *it != -1; it++) sum_b -= b[*it];

    return sum_a * sum_b;
}

inline __device__ float calc_c(const float h[76], int c_idx)
{
    float res = 0.0f;
    for (int8_t *it = h_pos[c_idx]; *it != -1; it++) res += h[*it];
    for (int8_t *it = h_neg[c_idx]; *it != -1; it++) res -= h[*it];
    return res;
}

__global__ void mult_2_kernel(
    const float* mat_a,
    const float* mat_b,
    float* mat_c,
    int N)
{
    int tile_idx_x = blockIdx.x;
    int tile_idx_y = blockIdx.y * 3;

    __shared__ float tile_a[4 * 3][5];
    __shared__ float tile_b[5][5];

    float value = 0.0f;

    __shared__ float h[3][76];

    clock_t start_time, stop_time;

    constexpr bool measure_time = false;

    int y = threadIdx.y;
    int x = threadIdx.x;
    for (int offset = 0; offset < N / 5; offset++)
    {
        if constexpr (measure_time) start_time = clock();
        // Copy tiles of mat_a to shared memory
        tile_a[y][x] = mat_a[(tile_idx_y * 4 + y) * N + (offset * 5 + x)];

        // Copy tile of mat_b to shared memory
        if (y < 5)
        {
            tile_b[y][x] = mat_b[(offset * 5 + y) * N + (tile_idx_x * 5 + x)];
        }

        __syncthreads();

        if constexpr (measure_time)
        {
            stop_time = clock();
            if (x == 0 && y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
                printf("Time copy tile to shared = %ld\n", stop_time - start_time);
            start_time = clock();
        }

        // if (y == 0 && x < 3)
        // {
        //     deepmatmul_only_h(tile_a, tile_b, h[x]);
        //
        // }
       
        int tile = y / 4; // 0,1,2 (id del tile dentro del bloque de threads de cuda)
        int tile_local_id = (y % 4) * blockDim.x + x;
        // printf("%d\n", tile_local_id);

        // Hay 76 hs entonces si utilizamos solo 19 threads por tile (19 de 20)
        // podemos calcular 4 hs por thread
        if (tile_local_id < 19)
        {
            #pragma unroll 4
            for (int i = 0; i < 4; i++)
            {
                int h_idx = tile_local_id * 4 + i;
                // printf("%d\n", h_idx);
                h[tile][h_idx] = calc_h((const float *)tile_a, (const float *)tile_b, h_idx);
            }
        }

        // correct: -3 -2 -0 -6 0 -6 -2 -0 -2 2 -0 0 6 2 -2 0 2 3 -3 -0 1 -1 1 -1 -0 1 2 3 3 1 6 -6 -2 1 -3 -4 -1 -8 -8 0 -3 1 2 0 -7 2 0 0 -0 -0 1 3 -1 -2 0 0 12 4 0 -8 -2 0 12 -2 0 -2 0 -0 -0 -0 -0 -12 -4 0 -2 0
        // if (x == 0 && y == 0 && tile_idx_x == 0 && tile_idx_y == 0 && offset == 0)
        // {
        //     for (int i = 0; i < 76; i++)
        //         printf("%1.0f ", h[0][i]);
        //     printf("\n");
        // }

        __syncthreads();

        if constexpr (measure_time)
        {
            stop_time = clock();
            if (x == 0 && y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
                printf("Time calc h[] = %ld\n", stop_time - start_time);
            start_time = clock();
        }

        // if (y == 0 && x < 3)
        // {
        //     deepmatmul_only_c(h[x], &tile_c[x*4]);
        // }
        
        value += calc_c(h[tile], tile_local_id);
        
        __syncthreads();

        if constexpr (measure_time)
        {
            stop_time = clock();
            if (x == 0 && y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
                printf("Time calc_c = %ld\n", stop_time - start_time);
        }
    }

    mat_c[(tile_idx_y * 4 + y) * N + (tile_idx_x * 5 + x)] = value;
}

void mult_2(const float* mat_a, const float* mat_b, float* mat_c, int N)
{
    constexpr int TILE_SIZE_X = 5;
    constexpr int TILE_SIZE_Y = 4;
    dim3 gridDim(N / 5, N / (4 * 3), 1);
    dim3 blockDim(TILE_SIZE_X, TILE_SIZE_Y * 3, 1);
    mult_2_kernel<<<gridDim, blockDim>>>(mat_a, mat_b, mat_c, N);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());
}
