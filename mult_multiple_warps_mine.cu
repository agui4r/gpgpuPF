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

constexpr int BLOCK_SIZE_X = 32;
constexpr int BLOCK_SIZE_Y = 4;

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

__global__ void mult_multiple_warps_kernel(
    const float *mat_a,
    const float *mat_b,
    float *mat_c,
    int N
) {
    __shared__ float tile_a[4][5];                              // Un tile de A.
    __shared__ float tiles_b[BLOCK_SIZE_X][5][5];               // Multiples tiles de B.
    __shared__ float h_shared[BLOCK_SIZE_X][76];                 

    float tiles_c[BLOCK_SIZE_X][4][5];
    for (int i = 0; i < BLOCK_SIZE_X; i++)
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 5; x++)
                tiles_c[i][y][x] = 0.0f;

    //Etapa 1)

    for (int offset = 0; offset < N; offset += 5) {
        // Cargar tile de A: threads 0..4 de cada fila
        if (threadIdx.x < 5 && threadIdx.y < 4) {
            int a_row = blockIdx.y * 4 + threadIdx.y;
            int a_col = offset + threadIdx.x;
            tile_a[threadIdx.y][threadIdx.x] = mat_a[a_row * N + a_col];
        }
        // Cargar 4 tiles de B: threads 5..24 de cada fila
        else if (threadIdx.x >= 5 && threadIdx.x < 25 && threadIdx.y < 4) {
            int tile_idx = (threadIdx.x - 5) / 5;                                           // 0..3 (cuál de los 4 tiles de B)
            int local_b_row = (threadIdx.x - 5) % 5;                                        // 0..4 (fila dentro del tile)
            int b_row = offset + local_b_row;
            int b_col = blockIdx.x * 4 * 5 + tile_idx * 5;
            tiles_b[tile_idx][threadIdx.y][local_b_row] = mat_b[b_row * N + (b_col + threadIdx.y)];
        }
        
        __syncthreads();
    
    // Etapa 2) Calculo de h's usando look up tables

        int lid = threadIdx.y * BLOCK_SIZE_X + threadIdx.x;         // 0..127 
        int warp_id = lid >> 5;                                     // 0..3
        int lane = lid & 31;                                        // 0..31

        const float *a_ptr = &tile_a[0][0];
        const float *b_ptr = &tiles_b[warp_id][0][0];

        // cada hilo del warp calcula varios h’s
        for (int idx = lane; idx < 76; idx += 32) {
            h_shared[warp_id][idx] = calc_h(a_ptr, b_ptr, idx);
        }
        __syncthreads();

    // Etapa 3) Calculo de c's

        for (int c_idx = lane; c_idx < 20; c_idx += 32) {
            int row = c_idx / 5;
            int col = c_idx % 5;
            tiles_c[warp_id][row][col] += calc_c(h_shared[warp_id], c_idx);
        }
        __syncthreads();
    }

    // Etapa 4) Escribir resultados finales a mat_c
    
    int lid = threadIdx.y * BLOCK_SIZE_X + threadIdx.x;
    int warp_id = lid >> 5;
    int lane = lid & 31;
    for (int c_idx = lane; c_idx < 20; c_idx += 32) {
        int row = c_idx / 5;
        int col = c_idx % 5;
        int global_row = blockIdx.y*4 + row;
        int global_col = (blockIdx.x*BLOCK_SIZE_X + warp_id)*5 + col;
        mat_c[global_row * N + global_col] = tiles_c[warp_id][row][col];
    }
}


void mult_multiple_warps(const float *mat_a, const float *mat_b, float *mat_c, int N) {
    // Cada bloque de la matriz toma cuatro tiles de B y un tile de A. Luego, calcula 4 tiles temporales de C.
    dim3 gridDim((N / 5) / BLOCK_SIZE_X, N / 4, 1);
    dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    mult_multiple_warps_kernel<<<gridDim, blockDim>>>(mat_a, mat_b, mat_c, N);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());
}