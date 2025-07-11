#include "mult_one_warp_per_tile_con_switch.cuh"

#include "util.cuh"
#include <cassert>
#include <cstdio>

__device__ float calc_h(const float a[4][5], const float b[5][5], int h_idx) {
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

__device__ float calc_c(const float h[76], int c_idx)
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

__global__ void mult_one_warp_per_tile_con_switch_kernel(
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
            h[r] = calc_h((const float (*)[5])Areg, (const float (*)[5])Breg, r);
            
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

void mult_one_warp_per_tile_con_switch(const float* mat_a, const float* mat_b, float* mat_c, int N)
{
    dim3 gridDim(N/5, N/4);
    mult_one_warp_per_tile_con_switch_kernel<<<gridDim, 32>>>(mat_a, mat_b, mat_c, N);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());
}
