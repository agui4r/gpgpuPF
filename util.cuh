#include <stdio.h>

#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ class Timer {
public:
    __device__ inline Timer(const char *label) : label(label), start(clock()) {};
    __device__ inline ~Timer() {
        if (threadIdx.x == 0 && threadIdx.y ==0 && blockIdx.x == 0 && blockIdx.y == 0)
            printf("%s: %ld\n", label, clock() - start);
    }
private:
    const char *label;
    clock_t start;
};
