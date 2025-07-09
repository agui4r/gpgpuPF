#include "mult_cublas.cuh"

#include <cublas_v2.h>  
#include <cuda_runtime.h>

#include "util.cuh"

#define CUBLAS_CHECK(ans) { gpuAssertCUBLAS((ans), __FILE__, __LINE__); }
inline void gpuAssertCUBLAS(cublasStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CUBLAS_STATUS_SUCCESS)
   {
      fprintf(stderr,"cuBLAS error: %s %s %d\n", cublasGetStatusString(code), file, line);
      if (abort) exit(code);
   }
}

void mult_cublas(const float* mat_a, const float* mat_b, float* mat_c, int N)
{
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_T;


    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));
    
    float *mat_c_transposed;
    CUDA_CHK(cudaMalloc((void **)&mat_c_transposed, N*N * sizeof(float)));

    /* step 3: compute in native FP32 */
    CUBLAS_CHECK(
        cublasSgemm_v2(
            cublasH,
            transa,
            transb,
            N,
            N,
            N,
            &alpha,
            mat_a,
            N,
            mat_b,
            N,
            &beta,
            mat_c_transposed,
            N
        )
    );

    CUDA_CHK(cudaStreamSynchronize(stream));
    
    alpha = 1.0f;
    beta = 0.0f;

    CUBLAS_CHECK(
        cublasSgeam(
            cublasH,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            N,
            N,
            &alpha,
            mat_c_transposed,
            N,
            &beta,
            mat_c,
            N,
            mat_c,
            N
        )
    );
    
    CUDA_CHK(cudaFree(mat_c_transposed));

    CUDA_CHK(cudaStreamSynchronize(stream));

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHK(cudaStreamDestroy(stream));
}

