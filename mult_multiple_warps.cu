// En esta implementación buscamos realizar el cálculo de 4 tiles de C de forma paralela.
// Nos traemos un tile de A y cuatro tiles de B, con estos tiles podemos calcular cuatro 
// tiles de C.

// Etapas:
//  1) Cargar tile de A a memoria compartida y cargar tiles de B. Vamos a comenzar probando con 128 threads, los primeros 20 threads cargan A, luego los siguientes
//  100 cargan los cuatro tiles de B de 5x5.
//  2) Calculo de h´s.
//  3) Calculo de c´s.
#include "mult_multiple_warps.cuh"
#include "tables.cuh"
#include "util.cuh"
#include <cassert>
#include <cstdio>

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 5
#define TILES_PER_BLOCK 4

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

__global__ void mult_multiple_warps_kernel(
    const float *mat_a,
    const float *mat_b,
    float *mat_c,
    int N
) {
    __shared__ float tile_a[4][5];                                  // Un tile de A.
    __shared__ float tiles_b[TILES_PER_BLOCK][5][5];                // Multiples tiles de B.
    __shared__ float h_shared[TILES_PER_BLOCK][76];                 

    __shared__ float tiles_c[TILES_PER_BLOCK][4][5];
    for (int i = 0; i < TILES_PER_BLOCK; i++)
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 5; x++)
                tiles_c[i][y][x] = 0.0f;

    int tileX = blockIdx.x;
    int tileY = blockIdx.y;
    const float *Ap = mat_a + (tileY*4)*N;
    const float *Bp = mat_b + (tileX*5*4);
    float *Cp = mat_c + (tileY * 4) * N + tileX * 5 * 4;

    // Etapa 1)
    
    // Pase estos calculos para arriba del for porque siempre dan lo mismo asi no se hacen en todos los for y aparte lo puedo usar abajo
    // en la etapa 4.
    int global_idx = threadIdx.y * blockDim.x + threadIdx.x;    //0..159
    int tile_id = global_idx / 40;                              // 0..3
    int index = global_idx % 40;                                // 0..39

    for (int offset = 0; offset < N; offset += 5) {
        // Cargar tile de A: threads 0..4 de cada fila
        if (threadIdx.x < 5 && threadIdx.y < 4) {
            int a_col = offset + threadIdx.x; 
            tile_a[threadIdx.y][threadIdx.x] = Ap[threadIdx.y * N + a_col];
        }
        // Cargar 4 tiles de B: threads 5..24 
        else if (threadIdx.x >= 5 && threadIdx.x < 25 && threadIdx.y < 5) {
            int id_thread = threadIdx.x - 5;                        //0..19
            int tile_b_id = id_thread / 5;     
            int col_in_b = id_thread % 5;     

            tiles_b[tile_b_id][threadIdx.y][col_in_b] = Bp[(threadIdx.y+offset) * N + id_thread];
        }
        
        __syncthreads();

        // Etapa 2) Calc h's

        #pragma unroll
        for (int r = index; r < 76; r += 40) {
            h_shared[tile_id][r] = calc_h((float*)tile_a, (float*)tiles_b[tile_id],r);
        }
        __syncthreads();

        // Etapa 3) Calc c's

        if (global_idx < 80) {
            int c_tile_id = global_idx / 20;                                        // 0..3
            int c_idx = global_idx % 20;                                            // 0..19   
            int row = c_idx / 5;                                                    // 0..4
            int col = c_idx % 5;                                                    // 0..4
            tiles_c[c_tile_id][row][col] += calc_c(h_shared[c_tile_id], c_idx); 
        }

        __syncthreads();
    }

    // Etapa 4) Escribir resultados finales a mat_c

    if(global_idx < 80) {
        int c_tile_id = global_idx / 20;                                        // 0..3
        int c_idx = global_idx % 20;                                            // 0..19   
        int row = c_idx / 5;                                                    // 0..3
        int col = c_idx % 5;                                                    // 0..4

        Cp[row * N + col + (c_tile_id * 5)] = tiles_c[c_tile_id][row][col];
    }
}


void mult_multiple_warps(const float *mat_a, const float *mat_b, float *mat_c, int N) {
    // Cada bloque de la matriz toma cuatro tiles de B y un tile de A. Luego, calcula 4 tiles temporales de C.
    dim3 gridDim((N / 5) / 4, N / 4, 1);   
    dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    mult_multiple_warps_kernel<<<gridDim, blockDim>>>(mat_a, mat_b, mat_c, N);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());
}
