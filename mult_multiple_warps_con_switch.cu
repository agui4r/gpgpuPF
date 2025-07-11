// En esta implementación buscamos realizar el cálculo de 4 tiles de C de forma paralela.
// Nos traemos un tile de A y cuatro tiles de B, con estos tiles podemos calcular cuatro 
// tiles de C.

// Etapas:
//  1) Cargar tile de A a memoria compartida y cargar tiles de B. Vamos a comenzar probando con 128 threads, los primeros 20 threads cargan A, luego los siguientes
//  100 cargan los cuatro tiles de B de 5x5.
//  2) Calculo de h´s.
//  3) Calculo de c´s.
#include "mult_multiple_warps_con_switch.cuh"
#include "util.cuh"
#include <cassert>
#include <cstdio>

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 5
#define TILES_PER_BLOCK 4

__device__ float calculation_h(const float a[4][5], const float b[5][5], int h_idx) {
    switch (h_idx) {
        case 0: return a[2][1] * ( -b[1][0] - b[1][4] - b[2][0] );
        case 1: return (a[1][1] + a[1][4] - a[2][4]) * ( -b[1][4] - b[4][0] );
        case 2: return (-a[2][0] - a[3][0] + a[3][1]) * ( -b[0][0] + b[1][4] );
        case 3: return (a[0][1] + a[0][3] + a[2][3]) * ( -b[1][4] - b[3][0] );
        case 4: return (a[0][4] + a[1][1] + a[1][4]) * ( -b[1][3] + b[4][0] );
        case 5: return (-a[1][1] - a[1][4] - a[3][4]) * ( b[1][2] + b[4][0] );
        case 6: return (-a[0][0] + a[3][0] - a[3][1]) * ( b[0][0] + b[1][3] );
        case 7: return (a[2][1] - a[2][2] - a[3][2]) * ( -b[1][2] + b[2][0] );
        case 8: return (-a[0][1] - a[0][3] + a[3][3]) * ( b[1][2] + b[3][0] );
        case 9: return (a[1][1] + a[1][4]) * b[4][0];
        case 10: return (-a[1][0] - a[3][0] + a[3][1]) * ( -b[0][0] + b[1][1] );
        case 11: return (a[3][0] - a[3][1]) * b[0][0];
        case 12: return (a[0][1] + a[0][3] + a[1][3]) * ( b[1][1] + b[3][0] );
        case 13: return (a[0][2] - a[2][1] + a[2][2]) * ( b[1][3] + b[2][0] );
        case 14: return (-a[0][1] - a[0][3]) * b[3][0];
        case 15: return (-a[2][1] + a[2][2]) * b[2][0];
        case 16: return (a[0][1] + a[0][3] - a[1][0] + a[1][1] - a[1][2] + a[1][3] - a[2][1] + a[2][2] - a[3][0] + a[3][1]) * b[1][1];
        case 17: return a[1][0] * ( b[0][0] + b[0][1] + b[4][1] );
        case 18: return -a[1][2] * ( b[2][0] + b[2][1] + b[4][1] );
        case 19: return (-a[0][4] + a[1][0] + a[1][2] - a[1][4]) * ( -b[0][0] - b[0][1] + b[0][3] - b[4][1] );
        case 20: return (a[1][0] + a[1][2] - a[1][4]) * b[4][1];
        case 21: return (a[0][2] - a[0][3] - a[1][3]) * ( b[0][0] + b[0][1] - b[0][3] - b[2][0] - b[2][1] + b[2][3] + b[3][3] );
        case 22: return a[0][2] * ( -b[2][0] + b[2][3] + b[3][3] );
        case 23: return a[0][4] * ( -b[3][3] - b[4][0] + b[4][3] );
        case 24: return -a[0][0] * ( b[0][0] - b[0][3] );
        case 25: return (-a[0][2] + a[0][3] + a[0][4]) * b[3][3];
        case 26: return (a[0][2] - a[2][0] + a[2][2]) * ( b[0][0] - b[0][3] + b[0][4] + b[2][4] );
        case 27: return -a[2][3] * ( -b[2][4] - b[3][0] - b[3][4] );
        case 28: return a[2][0] * ( b[0][0] + b[0][4] + b[2][4] );
        case 29: return (a[2][0] - a[2][2] + a[2][3]) * b[2][4];
        case 30: return (-a[0][3] - a[0][4] - a[2][3]) * ( -b[3][3] - b[4][0] + b[4][3] - b[4][4] );
        case 31: return (a[1][0] + a[3][0] + a[3][3]) * ( b[0][2] - b[3][0] - b[3][1] - b[3][2] );
        case 32: return a[3][2] * ( -b[2][0] - b[2][2] );
        case 33: return a[3][3] * ( -b[0][2] + b[3][0] + b[3][2] );
        case 34: return -a[3][4] * ( b[0][2] + b[4][0] + b[4][2] );
        case 35: return (a[1][2] - a[1][4] - a[3][4]) * ( b[2][0] + b[2][1] + b[2][2] + b[4][1] );
        case 36: return (-a[3][0] - a[3][3] + a[3][4]) * b[0][2];
        case 37: return (-a[1][2] - a[2][0] + a[2][2] - a[2][3]) * ( b[2][4] + b[3][0] + b[3][1] + b[3][4] );
        case 38: return (-a[2][0] - a[3][0] - a[3][3] + a[3][4]) * ( b[0][2] + b[4][0] + b[4][2] + b[4][4] );
        case 39: return (-a[0][2] + a[0][3] + a[0][4] - a[3][3]) * ( -b[2][0] - b[2][2] + b[2][3] + b[3][3] );
        case 40: return (-a[0][0] + a[3][0] - a[3][4]) * ( b[0][2] + b[2][0] + b[2][2] - b[2][3] + b[4][0] + b[4][2] - b[4][3] );
        case 41: return (-a[1][0] + a[1][4] - a[2][4]) * ( -b[0][0] - b[0][1] - b[0][4] + b[3][0] + b[3][1] + b[3][4] - b[4][1] );
        case 42: return a[1][3] * ( b[3][0] + b[3][1] );
        case 43: return (a[1][2] + a[2][1] - a[2][2]) * ( b[1][1] - b[2][0] );
        case 44: return (-a[2][2] + a[2][3] - a[3][2]) * ( b[2][4] + b[3][0] + b[3][2] + b[3][4] + b[4][0] + b[4][2] + b[4][4] );
        case 45: return -a[2][4] * ( -b[4][0] - b[4][4] );
        case 46: return (a[1][0] - a[1][4] - a[2][0] + a[2][4]) * ( b[0][0] + b[0][1] + b[0][4] - b[3][0] - b[3][1] - b[3][4] );
        case 47: return (-a[1][2] + a[2][2]) * ( b[1][1] + b[2][1] + b[2][4] + b[3][0] + b[3][1] + b[3][4] );
        case 48: return (-a[0][0] - a[0][2] + a[0][3] + a[0][4] - a[1][0] - a[1][2] + a[1][3] + a[1][4]) * ( -b[0][0] - b[0][1] + b[0][3] );
        case 49: return (-a[0][3] - a[1][3]) * ( b[1][1] - b[2][0] - b[2][1] + b[2][3] - b[3][1] + b[3][3] );
        case 50: return a[1][1] * ( b[1][0] + b[1][1] - b[4][0] );
        case 51: return a[3][1] * (b[0][0] + b[1][0] + b[1][2]);
        case 52: return -a[0][1] * (-b[1][0] + b[1][3] + b[3][0]);
        case 53: return (a[0][1] + a[0][3] - a[1][1] - a[1][4] - a[2][1] + a[2][2] - a[3][1] + a[3][2] - a[3][3] - a[3][4]) * b[1][2];
        case 54: return (a[0][3] - a[3][3]) * (-b[1][2] + b[2][0] + b[2][2] - b[2][3] + b[3][2] - b[3][3]);
        case 55: return (a[0][0] - a[0][4] - a[3][0] + a[3][4]) * (b[2][0] + b[2][2] - b[2][3] + b[4][0] + b[4][2] - b[4][3]);
        case 56: return (-a[2][0] - a[3][0]) * (-b[0][2] - b[0][4] - b[1][4] - b[4][0] - b[4][2] - b[4][4]);
        case 57: return (-a[0][3] - a[0][4] - a[2][3] - a[2][4]) * (-b[4][0] + b[4][3] - b[4][4]);
        case 58: return (-a[2][2] + a[2][3] - a[3][2] + a[3][3]) * (b[3][0] + b[3][2] + b[3][4] + b[4][0] + b[4][2] + b[4][4]);
        case 59: return (a[1][4] + a[3][4]) * (b[1][2] - b[2][0] - b[2][1] - b[2][2] - b[4][1] - b[4][2]);
        case 60: return (a[0][3] + a[2][3]) * (b[0][0] - b[0][3] + b[0][4] - b[1][4] - b[3][3] + b[3][4] - b[4][0] + b[4][3] - b[4][4]);
        case 61: return (a[1][0] + a[3][0]) * (b[0][1] + b[0][2] + b[1][1] - b[3][0] - b[3][1] - b[3][2]);
        case 62: return (-a[2][2] - a[3][2]) * (-b[1][2] - b[2][2] - b[2][4] - b[3][0] - b[3][2] - b[3][4]);
        case 63: return (a[0][0] - a[0][2] - a[0][3] + a[2][0] - a[2][2] - a[2][3]) * (b[0][0] - b[0][3] + b[0][4]);
        case 64: return (-a[0][0] + a[3][0]) * (-b[0][2] + b[0][3] + b[1][3] - b[4][0] - b[4][2] + b[4][3]);
        case 65: return (a[0][0] - a[0][1] + a[0][2] - a[0][4] - a[1][1] - a[1][4] - a[2][1] + a[2][2] - a[3][0] + a[3][1]) * b[1][3];
        case 66: return (a[1][4] - a[2][4]) * (b[0][0] + b[0][1] + b[0][4] - b[1][4] - b[3][0] - b[3][1] - b[3][4] + b[4][1] + b[4][4]);
        case 67: return (a[0][0] + a[0][2] - a[0][3] - a[0][4] - a[3][0] - a[3][2] + a[3][3] + a[3][4]) * (-b[2][0] - b[2][2] + b[2][3]);
        case 68: return (-a[0][2] + a[0][3] - a[1][2] + a[1][3]) * (-b[1][3] - b[2][0] - b[2][1] + b[2][3] - b[4][1] + b[4][3]);
        case 69: return (a[1][2] - a[1][4] + a[3][2] - a[3][4]) * (-b[2][0] - b[2][1] - b[2][2]);
        case 70: return (-a[2][0] + a[2][2] - a[2][3] + a[2][4] - a[3][0] + a[3][2] - a[3][3] + a[3][4]) * (-b[4][0] - b[4][2] - b[4][4]);
        case 71: return (-a[1][0] - a[1][3] - a[3][0] - a[3][3]) * (b[3][0] + b[3][1] + b[3][2]);
        case 72: return (a[0][2] - a[0][3] - a[0][4] + a[1][2] - a[1][3] - a[1][4]) * (b[0][0] + b[0][1] - b[0][3] + b[1][3] + b[4][1] - b[4][3]);
        case 73: return (a[1][0] - a[1][2] + a[1][3] - a[2][0] + a[2][2] - a[2][3]) * (b[3][0] + b[3][1] + b[3][4]);
        case 74: return - (a[0][1] + a[0][3] - a[1][1] - a[1][4] - a[2][0] + a[2][1] + a[2][3] + a[2][4] - a[3][0] + a[3][1]) * b[1][4];
        case 75: return (a[0][2] + a[2][2]) * (-b[0][0] + b[0][3] - b[0][4] + b[1][3] + b[2][3] - b[2][4]);
    }
    assert(false);
    return NAN;
}

__device__ float calculation_c(const float h[76], int c_idx)
{
    switch (c_idx) {
        case 0: return -h[9]  + h[11] + h[13] - h[14] - h[15] + h[52] + h[4]  - h[65] - h[6];
        case 1: return  h[12] + h[14] + h[19] + h[20] - h[21] + h[22] + h[24] - h[42] + h[48] + h[49];
        case 2: return  h[14] + h[22] + h[23] + h[33] - h[36] + h[39] - h[40] + h[54] - h[55] - h[8];
        case 3: return -h[9]  + h[11] + h[13] - h[15] + h[22] + h[23] + h[24] + h[25] + h[4]  - h[65] - h[6];
        case 4: return  h[14] + h[23] + h[24] + h[26] - h[27] + h[29] + h[30] - h[3]  + h[60] + h[63];
        case 5: return  h[9]  + h[10] - h[11] + h[12] + h[14] + h[15] - h[16] - h[43] + h[50];
        case 6: return -h[10] + h[11] - h[12] - h[14] - h[15] + h[16] + h[17] - h[18] - h[20] + h[42] + h[43];
        case 7: return -h[9]  + h[18] + h[31] + h[34] + h[35] + h[36] - h[42] - h[59] - h[5]  - h[71];
        case 8: return  h[9]  + h[17] - h[18] + h[19] - h[21] - h[23] - h[25] - h[4]  - h[68] + h[72];
        case 9: return -h[9]  - h[17] - h[1]  - h[29] - h[37] + h[41] - h[42] + h[45] + h[66] + h[73];
        case 10: return  h[9]  - h[11] + h[14] + h[15] - h[0]  + h[1]  + h[2]  - h[3]  + h[74];
        case 11: return -h[15] - h[18] - h[20] - h[27] - h[28] - h[37] + h[41] + h[43] - h[46] + h[47];
        case 12: return -h[15] - h[27] + h[32] + h[36] - h[38] + h[44] - h[45] + h[62] - h[70] - h[7];
        case 13: return -h[13] + h[15] - h[22] - h[25] + h[26] + h[28] + h[30] + h[45] - h[57] + h[75];
        case 14: return -h[9]  + h[11] - h[14] + h[27] + h[28] - h[1]  - h[29] - h[2]  + h[45] + h[3] - h[74];
        case 15: return -h[9]  + h[11] - h[14] - h[15] + h[51] + h[53] - h[5]  - h[7]  + h[8];
        case 16: return  h[10] - h[11] - h[17] + h[20] - h[31] + h[32] - h[33] - h[35] + h[61] - h[69];
        case 17: return  h[9]  + h[14] + h[15] - h[32] + h[33] - h[34] - h[36] - h[53] + h[5]  + h[7] - h[8];
        case 18: return  h[11] + h[24] + h[25] - h[32] - h[34] - h[39] + h[40] + h[64] - h[67] - h[6];
        case 19: return -h[11] - h[28] + h[29] - h[33] + h[34] + h[38] + h[2]  - h[44] + h[56] + h[58];
    }
    assert(false);
    return NAN;
}

__global__ void mult_multiple_warps_con_switch_kernel(
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
            // int a_row = blockIdx.y * 4 + threadIdx.y;
            // int a_col = offset + threadIdx.x;
            // tile_a[threadIdx.y][threadIdx.x] = mat_a[a_row * N + a_col];


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

        // if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 && offset == 0){
        //     for (int i = 0; i < TILES_PER_BLOCK; i++){
        //         for (int y = 0; y < 5; y++){
        //             for (int x = 0; x < 5; x++)
        //                 printf("%1.1f ", tiles_b[i][y][x]);
        //             printf("\n");
        //         }
        //         printf("\n");    
        //     }
        // }
    
        // Etapa 2) Calc h's

        #pragma unroll
        for (int r = index; r < 76; r += 40) {
            h_shared[tile_id][r] = calculation_h(tile_a, tiles_b[tile_id],r);
        }
        __syncthreads();

        // Etapa 3) Calc c's

        if (global_idx < 80) {
            int c_tile_id = global_idx / 20;                                        // 0..3
            int c_idx = global_idx % 20;                                            // 0..19   
            int row = c_idx / 5;                                                    // 0..4
            int col = c_idx % 5;                                                    // 0..4
            tiles_c[c_tile_id][row][col] += calculation_c(h_shared[c_tile_id], c_idx); 
        }

        // if(global_idx < 80) {
        //     tiles_c[threadIdx.x / 20][global_idx % 20] = calc_c(h_shared[tile_id], global_idx % 20);
        // }
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
        
    // if (threadIdx.x < 20) {
    //     Cp[row * N + col] = AccShared[threadIdx.x];
    // }
    // __syncwarp();


        // for (int i = 0; i < 2; i++) {
        //     constexpr int global_idx = threadIdx.y * blockDim.x + threadIdx.x;
        //     constexpr int tile_id = global_idx / 76;
        //     constexpr int index = global_idx mod 76;
        //     h_shared[tile_id][index] = calc_h(tile_a, tiles_b[tile_id], )
        // }
        // calc_h(tile_a, tiles_b[]);


    // // Etapa 2) Calculo de h's usando look up tables

    //     int lid = threadIdx.y * BLOCK_SIZE_X + threadIdx.x;         // 0..127 
    //     int warp_id = lid >> 5;                                     // 0..3
    //     int lane = lid & 31;                                        // 0..31

    //     const float *a_ptr = &tile_a[0][0];
    //     const float *b_ptr = &tiles_b[warp_id][0][0];

    //     // cada hilo del warp calcula varios h’s
    //     for (int idx = lane; idx < 76; idx += 32) {
    //         h_shared[warp_id][idx] = calc_h(a_ptr, b_ptr, idx);
    //     }
    //     __syncthreads();

    // // Etapa 3) Calculo de c's

    //     for (int c_idx = lane; c_idx < 20; c_idx += 32) {
    //         int row = c_idx / 5;
    //         int col = c_idx % 5;
    //         tiles_c[warp_id][row][col] += calc_c(h_shared[warp_id], c_idx);
    //     }
    //     __syncthreads();
    //}

    // Etapa 4) Escribir resultados finales a mat_c
    
    // int lid = threadIdx.y * BLOCK_SIZE_X + threadIdx.x;
    // int warp_id = lid >> 5;
    // int lane = lid & 31;
    // for (int c_idx = lane; c_idx < 20; c_idx += 32) {
    //     int row = c_idx / 5;
    //     int col = c_idx % 5;
    //     int global_row = blockIdx.y*4 + row;
    //     int global_col = (blockIdx.x*BLOCK_SIZE_Y + warp_id)*5 + col;
    //     mat_c[global_row * N + global_col] = tiles_c[warp_id][row][col];
    // }
}


void mult_multiple_warps_con_switch(const float *mat_a, const float *mat_b, float *mat_c, int N) {
    // Cada bloque de la matriz toma cuatro tiles de B y un tile de A. Luego, calcula 4 tiles temporales de C.
    dim3 gridDim((N / 5) / 4, N / 4, 1);   
    dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    mult_multiple_warps_con_switch_kernel<<<gridDim, blockDim>>>(mat_a, mat_b, mat_c, N);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());
}
