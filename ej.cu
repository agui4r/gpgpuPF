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
#define BLOCK_SIZE_Y 32
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

#include "deepmatmul.cu"

__global__ void mult(const float* mat_a, const float* mat_b, float* mat_c, int N)
{
    int idx_tile_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_tile_y = blockIdx.y * blockDim.y + threadIdx.y;

    float tile_a[4][5];
    float tile_b[5][5];
    float tile_c[4][5] = {0};

    for (int y = 0; y < 4; y++)
    {
        for (int x = 0; x < 5; x++)
        {
            tile_a[y][x] = mat_a[(idx_tile_y * 4 + y) * N + x];
        }
    }
    
    for (int y = 0; y < 5; y++)
    {
        for (int x = 0; x < 5; x++)
        {
            tile_b[y][x] = mat_b[y * N + (idx_tile_x * 5 + x)];
        }
    }

    deepmatmul(tile_a, tile_b, tile_c);
    
    for (int y = 0; y < 4; y++)
    {
        for (int x = 0; x < 5; x++)
        {
            mat_c[(idx_tile_y * 4 + y) * N + (idx_tile_x * 5 + x)] = tile_c[y][x];
        }
    }
}

int main(int argc, char *argv[])
{
    int N = /*4**/5*32;
    int array_size = N * N;
    float *h_mat_a = (float *)malloc(array_size * sizeof(float));
    float *h_mat_b = (float *)malloc(array_size * sizeof(float));
    float *h_mat_c = (float *)malloc(array_size * sizeof(float));

    // srand(1231323);
    for (int i = 0; i < N * N; i++)
    {
        h_mat_a[i] = 1.0f; //rand() % 11;    
        h_mat_b[i] = 1.0f; //rand() % 11;    
        h_mat_c[i] = 0.0f; //rand() % 11;    
    } 

    float *d_mat_a, *d_mat_b, *d_mat_c;
    cudaMalloc((void **)&d_mat_a, array_size * sizeof(float));
    cudaMalloc((void **)&d_mat_b, array_size * sizeof(float));
    cudaMalloc((void **)&d_mat_c, array_size * sizeof(float));
    //Copio array al device
    CUDA_CHK(cudaMemcpy(d_mat_a, h_mat_a, array_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_mat_b, h_mat_b, array_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_mat_c, h_mat_c, array_size * sizeof(float), cudaMemcpyHostToDevice));
    
    for (int i = 0; i < 10; i++)
    {
        nvtxRangePush("Multiplicacion matrices");
        mult<<<
            dim3((N / 5) / BLOCK_SIZE_X, (N / 4) / BLOCK_SIZE_Y, 1),
            dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1)
        >>>(d_mat_a, d_mat_b, d_mat_c, N);
        // TODO: Agregar el otro check
        CUDA_CHK(cudaDeviceSynchronize());
        nvtxRangePop();
    }

#ifndef DONT_PRINT
    cudaMemcpy(h_mat_c, d_mat_c, array_size * sizeof(float), cudaMemcpyDeviceToHost);
    printf("============== C =================\n");
    for (int y = 0; y < N; y++)
    {
        for (int x = 0; x < N; x++)
        {
            printf("%1.0f ", h_mat_c[y * N + x]);
        }
        printf("\n");
    }
    printf("\n");
#endif

    cudaFree(d_mat_a);
    cudaFree(d_mat_b);
    cudaFree(d_mat_c);
    free(h_mat_a);
    free(h_mat_b);
    free(h_mat_c);
}
