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

int main(int argc, char *argv[])
{

}
