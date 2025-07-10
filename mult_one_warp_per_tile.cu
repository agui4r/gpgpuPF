#include "mult_one_warp_per_tile.cuh"

#include "util.cuh"
#include <cassert>
#include <cstdio>

#include "tables.cuh"

using MultIdxT = int64_t;

inline __device__ float calc_h(const float *a, const float *b, int h_idx)
{
    float sum_a = 0.0f, sum_b = 0.0f;

    #pragma unroll
    for (int8_t i = 0; i < sizeof(a_pos[h_idx]) / sizeof(MultIdxT); i++)
    {
        MultIdxT four_indices = ((MultIdxT*)a_pos[h_idx])[i];
        #pragma unroll
        for (int offset = 0; offset < 8 * sizeof(MultIdxT); offset+=8)
        {
            int8_t index = (four_indices >> offset) & 0xff;
            sum_a += index >= 0
                ? a[index]
                : 0.0f;
        }
    }

    #pragma unroll
    for (int8_t i = 0; i < sizeof(a_neg[h_idx]) / sizeof(MultIdxT); i++)
    {
        MultIdxT four_indices = ((MultIdxT*)a_neg[h_idx])[i];
        #pragma unroll
        for (int offset = 0; offset < 8 * sizeof(MultIdxT); offset+=8)
        {
            int8_t index = (four_indices >> offset) & 0xff;
            sum_a -= index >= 0
                ? a[index]
                : 0.0f;
        }
    }

    #pragma unroll
    for (int8_t i = 0; i < sizeof(b_pos[h_idx]) / sizeof(MultIdxT); i++)
    {
        MultIdxT four_indices = ((MultIdxT*)b_pos[h_idx])[i];
        #pragma unroll
        for (int offset = 0; offset < 8 * sizeof(MultIdxT); offset+=8)
        {
            int8_t index = (four_indices >> offset) & 0xff;
            sum_b += index >= 0
                ? b[index]
                : 0.0f;
        }
    }

    #pragma unroll
    for (int8_t i = 0; i < sizeof(b_neg[h_idx]) / sizeof(MultIdxT); i++)
    {
        MultIdxT four_indices = ((MultIdxT*)b_neg[h_idx])[i];
        #pragma unroll
        for (int offset = 0; offset < 8 * sizeof(MultIdxT); offset+=8)
        {
            int8_t index = (four_indices >> offset) & 0xff;
            sum_b -= index >= 0
                ? b[index]
                : 0.0f;
        }
    }

    return sum_a * sum_b;
}

inline __device__ float calc_c(const float h[76], int c_idx)
{
    float res = 0.0f;
    
    #pragma unroll
    for (int8_t i = 0; i < sizeof(h_pos[c_idx]) / sizeof(MultIdxT); i++)
    {
        MultIdxT four_indices = ((MultIdxT*)h_pos[c_idx])[i];
        #pragma unroll
        for (int offset = 0; offset < 8 * sizeof(MultIdxT); offset+=8)
        {
            int8_t index = (four_indices >> offset) & 0xff;
            res += index >= 0 ? h[index] : 0.0f;
        }
    }
    
    #pragma unroll
    for (int8_t i = 0; i < sizeof(h_neg[c_idx]) / sizeof(MultIdxT); i++)
    {
        MultIdxT four_indices = ((MultIdxT*)h_neg[c_idx])[i];
        #pragma unroll
        for (int offset = 0; offset < 8 * sizeof(MultIdxT); offset+=8)
        {
            int8_t index = (four_indices >> offset) & 0xff;
            res -= index >= 0 ? h[index] : 0.0f;
        }
    }

    return res;
}

__global__ void mult_one_warp_per_tile_kernel(
    const float * __restrict__ mat_a,
    const float * __restrict__ mat_b,
    float * __restrict__ mat_c,
    int N)
{
    __shared__ float AccShared[20];
    __shared__ float Areg[20], Breg[25];
    int tileX = blockIdx.x;
    int tileY = blockIdx.y;
    // 20 + 25 + 20 = 65 floats  → caben en registros ?
    int row = threadIdx.x / 5;
    int col = threadIdx.x % 5;
    
    const float *Ap = mat_a + (tileY*4)*N;
    const float *Bp = mat_b + (tileX*5);
    float       *Cp = mat_c + (tileY*4)*N + tileX*5;

    // Inicializo AccShared en 0
    if (threadIdx.x < 20) {
        AccShared[threadIdx.x] = 0.0f;
    }
    __syncwarp();
    
    #pragma unroll 1
    for (int offset = 0; offset < N; offset += 5)
    {
        {
        // Timer t("Copiar tile a registros");
        // Traigo el tile de A a los registros
        if (threadIdx.x < 20)
        {
           Areg[threadIdx.x] = Ap[row * N + col + offset];
        }
        //__syncwarp();

        // Traigo el tile de B a los registros
        if (threadIdx.x < 25)
        {
            Breg[threadIdx.x] = Bp[(row + offset) * N + col];
        }
        __syncwarp();
        }

        {
        // Timer t("Calcular C");
        // Calculo el tile de C de esta iteracion, repartiendo entre los hilos
        __shared__ float h[76];
        for (int r = threadIdx.x; r < 76; r += 32)
        {
            h[r] = calc_h(Areg, Breg, r);
            
            // float a_dot_P = 0.f, b_dot_Q = 0.f;
            //
            // #pragma unroll
            // for (int t = 0; t < 20; t++)
            // {
            //     a_dot_P += float(P[t][r]) * Areg[t];
            // }
            //
            // #pragma unroll
            // for (int t = 0; t < 25; t++)
            // {
            //     b_dot_Q += float(Q[t][r]) * Breg[t];
            // }
            //
            // float m = a_dot_P * b_dot_Q;
            //
            // #pragma unroll
            // for (int t = 0; t < 20; t++)
            // {
            // Acc[t] += float(R[t][r]) * m;
            // }
        }
        __syncwarp();
        if (threadIdx.x < 20)
        {
            AccShared[threadIdx.x] += calc_c(h, threadIdx.x);
        }
        }

        // {
        // Timer t("Combinar Acc");
        // // Hago __shfl_down_sync para acumular todos los acc locales de cada thread en el acc local del thread 0
        // for (int shuffle_offset = 16; shuffle_offset; shuffle_offset >>= 1)
        // {
        //     #pragma unroll
        //     for (int t = 0; t < 20; t++)
        //     {
        //         Acc[t] += __shfl_down_sync(0xffffffff, Acc[t], shuffle_offset);
        //     }
        // }
        // }
        //
        // {
        // Timer t("guardar en shared");
        // // Como solo el thread 0 tiene en su `Acc` el resultado de esta iteracion, solo lo puede copiar el
        // if (threadIdx.x == 0)
        // {
        //     #pragma unroll
        //     for (int t = 0; t < 20; t++)
        //     {
        //         AccShared[t] += Acc[t];
        //     }
        // }
        // }
    }

    {
    // Timer t("Escribir en c global");
    if (threadIdx.x < 20)
    {
        Cp[row * N + col] = AccShared[threadIdx.x];
    }
    __syncwarp();
    }
}

void mult_one_warp_per_tile(const float* mat_a, const float* mat_b, float* mat_c, int N)
{
    dim3 gridDim(N/5, N/4);
    mult_one_warp_per_tile_kernel<<<gridDim, 32>>>(mat_a, mat_b, mat_c, N);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());
}
