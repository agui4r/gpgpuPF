// En esta implementación buscamos realizar el cálculo de 4 tiles de C de forma paralela.
// Nos traemos un tile de A y cuatro tiles de B, con estos tiles podemos calcular cuatro 
// tiles de C.

// Etapas:
//  1) Cargar tile de A a memoria compartida y cargar tiles de B. Vamos a comenzar probando con 128 threads, los primeros 20 threads cargan A, luego los siguientes
//  100 cargan los cuatro tiles de B de 5x5.
//  2) Calculo de h´s.
//  3) Calculo de c´s.
#include "mult_multiple_warps.cuh"

#include "util.cuh"


constexpr int BLOCK_SIZE_X = 4;
constexpr int BLOCK_SIZE_Y = 32;

__global__ void mult_multiple_warps_kernel(
    const float *mat_a,
    const float *mat_b,
    float *mat_c,
    int N
) {
    __shared__ float tile_a[4][5];                              // Un tile de A.
    __shared__ float tiles_b[BLOCK_SIZE_X][5][5];               // Multiples tiles de B.

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
    
    // Etapa 2)
        
        //Calculo de h's usando look up tables or switches

    // Etapa 3)

        //Calculo de c's
        
    }
    

}


void mult_multiple_warps(
    const float *mat_a,
    const float *mat_b,
    float *mat_c,
    int N
) {
#ifdef TIME
    cudaFuncSetCacheConfig(mult_one_thread_per_tile_in_c_mat_kernel, cudaFuncCachePreferShared);
#endif
    // Cada bloque de la matriz toma cuatro tiles de B y un tile de A.
    dim3 gridDim((N / 5) / BLOCK_SIZE_X, N / 4, 1);
    dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    mult_multiple_warps_kernel<<<gridDim, blockDim>>>(mat_a, mat_b, mat_c, N);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());
}