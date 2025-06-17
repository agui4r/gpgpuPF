#include <cstdio>
#include <nvtx3/nvToolsExt.h>

#include <stdio.h>

#ifndef MATRIX_SIZE_A
#define MATRIX_SIZE_A 1024
#endif

#ifndef MATRIX_SIZE_B
#define MATRIX_SIZE_B 1024
#endif

#ifndef BLOCK_SIZE_X
#define BLOCK_SIZE_X 32
#endif

#ifndef BLOCK_SIZE_Y
#define BLOCK_SIZE_Y 4
#endif

#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// __host__ __device__ void deepmatmul(const float a[4][5], const float b[5][5], float c[4][5]) {
__host__ __device__ void deepmatmul(const float *a[4], const float *b[5], float *c[4]) {
    float h[76];

    h[0] = a[2][1] * ( -b[1][0] - b[1][4] - b[2][0] );
    h[1] = (a[1][1] + a[1][4] - a[2][4]) * ( -b[1][4] - b[4][0] );
    h[2] = (-a[2][0] - a[3][0] + a[3][1]) * ( -b[0][0] + b[1][4] );
    h[3] = (a[0][1] + a[0][3] + a[2][3]) * ( -b[1][4] - b[3][0] );
    h[4] = (a[0][4] + a[1][1] + a[1][4]) * ( -b[1][3] + b[4][0] );
    h[5] = (-a[1][1] - a[1][4] - a[3][4]) * ( b[1][2] + b[4][0] );
    h[6] = (-a[0][0] + a[3][0] - a[3][1]) * ( b[0][0] + b[1][3] );
    h[7] = (a[2][1] - a[2][2] - a[3][2]) * ( -b[1][2] + b[2][0] );
    h[8] = (-a[0][1] - a[0][3] + a[3][3]) * ( b[1][2] + b[3][0] );
    h[9] = (a[1][1] + a[1][4]) * b[4][0];
    h[10] = (-a[1][0] - a[3][0] + a[3][1]) * ( -b[0][0] + b[1][1] );
    h[11] = (a[3][0] - a[3][1]) * b[0][0];
    h[12] = (a[0][1] + a[0][3] + a[1][3]) * ( b[1][1] + b[3][0] );
    h[13] = (a[0][2] - a[2][1] + a[2][2]) * ( b[1][3] + b[2][0] );
    h[14] = (-a[0][1] - a[0][3]) * b[3][0];
    h[15] = (-a[2][1] + a[2][2]) * b[2][0];
    h[16] = (a[0][1] + a[0][3] - a[1][0] + a[1][1] - a[1][2] + a[1][3] - a[2][1] + a[2][2] - a[3][0] + a[3][1]) * b[1][1];
    h[17] = a[1][0] * ( b[0][0] + b[0][1] + b[4][1] );
    h[18] = -a[1][2] * ( b[2][0] + b[2][1] + b[4][1] );
    h[19] = (-a[0][4] + a[1][0] + a[1][2] - a[1][4]) * ( -b[0][0] - b[0][1] + b[0][3] - b[4][1] );
    h[20] = (a[1][0] + a[1][2] - a[1][4]) * b[4][1];
    h[21] = (a[0][2] - a[0][3] - a[1][3]) * ( b[0][0] + b[0][1] - b[0][3] - b[2][0] - b[2][1] + b[2][3] + b[3][3] );
    h[22] = a[0][2] * ( -b[2][0] + b[2][3] + b[3][3] );
    h[23] = a[0][4] * ( -b[3][3] - b[4][0] + b[4][3] );
    h[24] = -a[0][0] * ( b[0][0] - b[0][3] );
    h[25] = (-a[0][2] + a[0][3] + a[0][4]) * b[3][3];
    h[26] = (a[0][2] - a[2][0] + a[2][2]) * ( b[0][0] - b[0][3] + b[0][4] + b[2][4] );
    h[27] = -a[2][3] * ( -b[2][4] - b[3][0] - b[3][4] );
    h[28] = a[2][0] * ( b[0][0] + b[0][4] + b[2][4] );
    h[29] = (a[2][0] - a[2][2] + a[2][3]) * b[2][4];
    h[30] = (-a[0][3] - a[0][4] - a[2][3]) * ( -b[3][3] - b[4][0] + b[4][3] - b[4][4] );
    h[31] = (a[1][0] + a[3][0] + a[3][3]) * ( b[0][2] - b[3][0] - b[3][1] - b[3][2] );
    h[32] = a[3][2] * ( -b[2][0] - b[2][2] );
    h[33] = a[3][3] * ( -b[0][2] + b[3][0] + b[3][2] );
    h[34] = -a[3][4] * ( b[0][2] + b[4][0] + b[4][2] );
    h[35] = (a[1][2] - a[1][4] - a[3][4]) * ( b[2][0] + b[2][1] + b[2][2] + b[4][1] );
    h[36] = (-a[3][0] - a[3][3] + a[3][4]) * b[0][2];
    h[37] = (-a[1][2] - a[2][0] + a[2][2] - a[2][3]) * ( b[2][4] + b[3][0] + b[3][1] + b[3][4] );
    h[38] = (-a[2][0] - a[3][0] - a[3][3] + a[3][4]) * ( b[0][2] + b[4][0] + b[4][2] + b[4][4] );
    h[39] = (-a[0][2] + a[0][3] + a[0][4] - a[3][3]) * ( -b[2][0] - b[2][2] + b[2][3] + b[3][3] );
    h[40] = (-a[0][0] + a[3][0] - a[3][4]) * ( b[0][2] + b[2][0] + b[2][2] - b[2][3] + b[4][0] + b[4][2] - b[4][3] );
    h[41] = (-a[1][0] + a[1][4] - a[2][4]) * ( -b[0][0] - b[0][1] - b[0][4] + b[3][0] + b[3][1] + b[3][4] - b[4][1] );
    h[42] = a[1][3] * ( b[3][0] + b[3][1] );
    h[43] = (a[1][2] + a[2][1] - a[2][2]) * ( b[1][1] - b[2][0] );
    h[44] = (-a[2][2] + a[2][3] - a[3][2]) * ( b[2][4] + b[3][0] + b[3][2] + b[3][4] + b[4][0] + b[4][2] + b[4][4] );
    h[45] = -a[2][4] * ( -b[4][0] - b[4][4] );
    h[46] = (a[1][0] - a[1][4] - a[2][0] + a[2][4]) * ( b[0][0] + b[0][1] + b[0][4] - b[3][0] - b[3][1] - b[3][4] );
    h[47] = (-a[1][2] + a[2][2]) * ( b[1][1] + b[2][1] + b[2][4] + b[3][0] + b[3][1] + b[3][4] );
    h[48] = (-a[0][0] - a[0][2] + a[0][3] + a[0][4] - a[1][0] - a[1][2] + a[1][3] + a[1][4]) * ( -b[0][0] - b[0][1] + b[0][3] );
    h[49] = (-a[0][3] - a[1][3]) * ( b[1][1] - b[2][0] - b[2][1] + b[2][3] - b[3][1] + b[3][3] );
    h[50] = a[1][1] * ( b[1][0] + b[1][1] - b[4][0] );
    h[51] = a[3][1] * (b[0][0] + b[1][0] + b[1][2]);
    h[52] = -a[0][1] * (-b[1][0] + b[1][3] + b[3][0]);
    h[53] = (a[0][1] + a[0][3] - a[1][1] - a[1][4] - a[2][1] + a[2][2] - a[3][1] + a[3][2] - a[3][3] - a[3][4]) * b[1][2];
    h[54] = (a[0][3] - a[3][3]) * (-b[1][2] + b[2][0] + b[2][2] - b[2][3] + b[3][2] - b[3][3]);
    h[55] = (a[0][0] - a[0][4] - a[3][0] + a[3][4]) * (b[2][0] + b[2][2] - b[2][3] + b[4][0] + b[4][2] - b[4][3]);
    h[56] = (-a[2][0] - a[3][0]) * (-b[0][2] - b[0][4] - b[1][4] - b[4][0] - b[4][2] - b[4][4]);
    h[57] = (-a[0][3] - a[0][4] - a[2][3] - a[2][4]) * (-b[4][0] + b[4][3] - b[4][4]);
    h[58] = (-a[2][2] + a[2][3] - a[3][2] + a[3][3]) * (b[3][0] + b[3][2] + b[3][4] + b[4][0] + b[4][2] + b[4][4]);
    h[59] = (a[1][4] + a[3][4]) * (b[1][2] - b[2][0] - b[2][1] - b[2][2] - b[4][1] - b[4][2]);
    h[60] = (a[0][3] + a[2][3]) * (b[0][0] - b[0][3] + b[0][4] - b[1][4] - b[3][3] + b[3][4] - b[4][0] + b[4][3] - b[4][4]);
    h[61] = (a[1][0] + a[3][0]) * (b[0][1] + b[0][2] + b[1][1] - b[3][0] - b[3][1] - b[3][2]);
    h[62] = (-a[2][2] - a[3][2]) * (-b[1][2] - b[2][2] - b[2][4] - b[3][0] - b[3][2] - b[3][4]);
    h[63] = (a[0][0] - a[0][2] - a[0][3] + a[2][0] - a[2][2] - a[2][3]) * (b[0][0] - b[0][3] + b[0][4]);
    h[64] = (-a[0][0] + a[3][0]) * (-b[0][2] + b[0][3] + b[1][3] - b[4][0] - b[4][2] + b[4][3]);
    h[65] = (a[0][0] - a[0][1] + a[0][2] - a[0][4] - a[1][1] - a[1][4] - a[2][1] + a[2][2] - a[3][0] + a[3][1]) * b[1][3];
    h[66] = (a[1][4] - a[2][4]) * (b[0][0] + b[0][1] + b[0][4] - b[1][4] - b[3][0] - b[3][1] - b[3][4] + b[4][1] + b[4][4]);
    h[67] = (a[0][0] + a[0][2] - a[0][3] - a[0][4] - a[3][0] - a[3][2] + a[3][3] + a[3][4]) * (-b[2][0] - b[2][2] + b[2][3]);
    h[68] = (-a[0][2] + a[0][3] - a[1][2] + a[1][3]) * (-b[1][3] - b[2][0] - b[2][1] + b[2][3] - b[4][1] + b[4][3]);
    h[69] = (a[1][2] - a[1][4] + a[3][2] - a[3][4]) * (-b[2][0] - b[2][1] - b[2][2]);
    h[70] = (-a[2][0] + a[2][2] - a[2][3] + a[2][4] - a[3][0] + a[3][2] - a[3][3] + a[3][4]) * (-b[4][0] - b[4][2] - b[4][4]);
    h[71] = (-a[1][0] - a[1][3] - a[3][0] - a[3][3]) * (b[3][0] + b[3][1] + b[3][2]);
    h[72] = (a[0][2] - a[0][3] - a[0][4] + a[1][2] - a[1][3] - a[1][4]) * (b[0][0] + b[0][1] - b[0][3] + b[1][3] + b[4][1] - b[4][3]);
    h[73] = (a[1][0] - a[1][2] + a[1][3] - a[2][0] + a[2][2] - a[2][3]) * (b[3][0] + b[3][1] + b[3][4]);
    h[74] = - (a[0][1] + a[0][3] - a[1][1] - a[1][4] - a[2][0] + a[2][1] + a[2][3] + a[2][4] - a[3][0] + a[3][1]) * b[1][4];
    h[75] = (a[0][2] + a[2][2]) * (-b[0][0] + b[0][3] - b[0][4] + b[1][3] + b[2][3] - b[2][4]);

    c[0][0] += -h[9] + h[11] + h[13] - h[14] - h[15] + h[52] + h[4] - h[65] - h[6];
    c[0][1] += h[12] + h[14] + h[19] + h[20] - h[21] + h[22] + h[24] - h[42] + h[48] + h[49];
    c[0][2] += h[14] + h[22] + h[23] + h[33] - h[36] + h[39] - h[40] + h[54] - h[55] - h[8];
    c[0][3] += -h[9] + h[11] + h[13] - h[15] + h[22] + h[23] + h[24] + h[25] + h[4] - h[65] - h[6];
    c[0][4] += h[14] + h[23] + h[24] + h[26] - h[27] + h[29] + h[30] - h[3] + h[60] + h[63];
    c[1][0] += h[9] + h[10] - h[11] + h[12] + h[14] + h[15] - h[16] - h[43] + h[50];
    c[1][1] += -h[10] + h[11] - h[12] - h[14] - h[15] + h[16] + h[17] - h[18] - h[20] + h[42] + h[43];
    c[1][2] += -h[9] + h[18] + h[31] + h[34] + h[35] + h[36] - h[42] - h[59] - h[5] - h[71];
    c[1][3] += h[9] + h[17] - h[18] + h[19] - h[21] - h[23] - h[25] - h[4] - h[68] + h[72];
    c[1][4] += -h[9] - h[17] - h[1] - h[29] - h[37] + h[41] - h[42] + h[45] + h[66] + h[73];
    c[2][0] += h[9] - h[11] + h[14] + h[15] - h[0] + h[1] + h[2] - h[3] + h[74];
    c[2][1] += -h[15] - h[18] - h[20] - h[27] - h[28] - h[37] + h[41] + h[43] - h[46] + h[47];
    c[2][2] += -h[15] - h[27] + h[32] + h[36] - h[38] + h[44] - h[45] + h[62] - h[70] - h[7];
    c[2][3] += -h[13] + h[15] - h[22] - h[25] + h[26] + h[28] + h[30] + h[45] - h[57] + h[75];
    c[2][4] += -h[9] + h[11] - h[14] + h[27] + h[28] - h[1] - h[29] - h[2] + h[45] + h[3] - h[74];
    c[3][0] += -h[9] + h[11] - h[14] - h[15] + h[51] + h[53] - h[5] - h[7] + h[8];
    c[3][1] += h[10] - h[11] - h[17] + h[20] - h[31] + h[32] - h[33] - h[35] + h[61] - h[69];
    c[3][2] += h[9] + h[14] + h[15] - h[32] + h[33] - h[34] - h[36] - h[53] + h[5] + h[7] - h[8];
    c[3][3] += h[11] + h[24] + h[25] - h[32] - h[34] - h[39] + h[40] + h[64] - h[67] - h[6];
    c[3][4] += -h[11] - h[28] + h[29] - h[33] + h[34] + h[38] + h[2] - h[44] + h[56] + h[58];
}

__global__ void mult(const float* mat_a, const float* mat_b, float* mat_c, int N)
{
    int tile_idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tile_idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    const float *tile_a[4];
    const float *tile_b[5];
    float *tile_c[4];

    for (int y = 0; y < 4; y++)
    {
        tile_c[y] = &mat_c[(tile_idx_y * 4 + y) * N + (tile_idx_x * 5)];
    }

    for (int offset = 0; offset < N / 5; offset++)
    {
        for (int y = 0; y < 4; y++)
        {
            tile_a[y] = &mat_a[(tile_idx_y * 4 + y) * N + (offset * 5)];
        }
        
        for (int y = 0; y < 5; y++)
        {
            tile_b[y] = &mat_b[(offset * 5 + y) * N + (tile_idx_x * 5)];
        }

        deepmatmul(tile_a, tile_b, tile_c);
    }
}

__global__ void naive_mult(const float* mat_a, const float* mat_b, float* mat_c, int N)
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

int main(int argc, char *argv[])
{
    int N = 5*32*10;
    int array_size = N * N;
    float *h_mat_a = (float *)malloc(array_size * sizeof(float));
    float *h_mat_b = (float *)malloc(array_size * sizeof(float));
    float *h_mat_c = (float *)malloc(array_size * sizeof(float));
    float *h_mat_c_naive_result = (float *)malloc(array_size * sizeof(float));

    // srand(1231323);
    for (int i = 0; i < N * N; i++)
    {
        h_mat_a[i] = 1.0f; //rand() % 11;
        h_mat_b[i] = 1.0f; //rand() % 11;
        h_mat_c[i] = 0.0f;
    } 

    float *d_mat_a, *d_mat_b, *d_mat_c, *d_mat_c_naive_result;
    cudaMalloc((void **)&d_mat_a, array_size * sizeof(float));
    cudaMalloc((void **)&d_mat_b, array_size * sizeof(float));
    cudaMalloc((void **)&d_mat_c, array_size * sizeof(float));
    cudaMalloc((void **)&d_mat_c_naive_result, array_size * sizeof(float));
    //Copio array al device
    CUDA_CHK(cudaMemcpy(d_mat_a, h_mat_a, array_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_mat_b, h_mat_b, array_size * sizeof(float), cudaMemcpyHostToDevice));

    dim3 gridDim((N / 5) / BLOCK_SIZE_X, (N / 4) / BLOCK_SIZE_Y, 1);
    dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);

    printf("gridDim = {%d, %d, %d}\n", gridDim.x, gridDim.y, gridDim.z);
    printf("blockDim = {%d, %d, %d}\n", blockDim.x, blockDim.y, blockDim.z);
    
    for (int i = 0; i < 10; i++)
    {
        CUDA_CHK(cudaMemcpy(d_mat_c, h_mat_c, array_size * sizeof(float), cudaMemcpyHostToDevice));
        nvtxRangePush("Multiplicacion matrices");
        mult<<<gridDim, blockDim>>>(d_mat_a, d_mat_b, d_mat_c, N);
        CUDA_CHK(cudaGetLastError());
        CUDA_CHK(cudaDeviceSynchronize());
        nvtxRangePop();
    }

    for (int i = 0; i < 10; i++)
    {
        CUDA_CHK(cudaMemcpy(d_mat_c_naive_result, h_mat_c, array_size * sizeof(float), cudaMemcpyHostToDevice));
        nvtxRangePush("Naive mult");
        naive_mult<<<
            dim3(N / BLOCK_SIZE_X, N / BLOCK_SIZE_Y, 1),
            dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1)
        >>>(d_mat_a, d_mat_b, d_mat_c_naive_result, N);
        CUDA_CHK(cudaGetLastError());
        CUDA_CHK(cudaDeviceSynchronize());
        nvtxRangePop();
    }

    CUDA_CHK(cudaMemcpy(h_mat_c, d_mat_c, array_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHK(cudaMemcpy(h_mat_c_naive_result, d_mat_c_naive_result, array_size * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHK(cudaFree(d_mat_a));
    CUDA_CHK(cudaFree(d_mat_b));
    CUDA_CHK(cudaFree(d_mat_c));
    CUDA_CHK(cudaFree(d_mat_c_naive_result));

    bool correct = true;
    for (int i = 0; i < array_size; i++)
    {
        if (h_mat_c_naive_result[i] != h_mat_c[i])
        {
            correct = false;
            break;
        }
    }

    if (correct)
    {
        printf("CORRECT RESULT\n");
    }
    else
    {
        printf("INCORRECT RESULT\n");
    }

#ifdef PRINT
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
    free(h_mat_c_naive_result);

    return correct ? 0 : 1;
}
