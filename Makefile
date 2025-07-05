CC = nvcc

NVCC_FLAGS = -std=c++17 -Xcompiler -ftree-vectorize -Xcompiler -fopenmp -O3 -w -m64 -g -Wno-deprecated-gpu-targets \
#             -gencode=arch=compute_61,code=sm_61 \
			 -arch=sm_61  \
             -Xptxas -dlcm=cg

CUDA_INSTALL_PATH = /usr/local/cuda
MKLROOT = /local/gpgpu/software/intel/mkl/2021.4.0

INCLUDES = -I$(CUDA_INSTALL_PATH)/include -I./include -I${MKLROOT}/include

LIBS = -lcublas

MAIN = main
OBJS = main.o mult_one_thread_per_tile_in_c_mat.o mult_naive.o mult_2.o \
	   mult_tiled_32x32_conventional.o mult_cublas.o mult_one_warp_per_tile.o \
	   mult_one_warp_per_tile_2.o

# TEMPLATE
#
# file.o: util.cuh file.cuh
#     $(CC) $(NVCC_FLAGS) -c -o $@ file.cu

all: $(MAIN)

$(MAIN): $(OBJS)
	$(CC) $(NVCC_FLAGS) -o $@ $(LIBS) $^

main.o: \
	main.cu util.cuh mult_one_thread_per_tile_in_c_mat.cuh mult_naive.cuh mult_2.cuh \
	mult_tiled_32x32_conventional.cuh mult_cublas.cuh mult_one_warp_per_tile.cuh \
	mult_one_warp_per_tile_2.cuh
	$(CC) $(NVCC_FLAGS) -c -o $@ main.cu

mult_one_thread_per_tile_in_c_mat.o: util.cuh mult_one_thread_per_tile_in_c_mat.cuh mult_one_thread_per_tile_in_c_mat.cu
	$(CC) $(NVCC_FLAGS) -c -o $@ mult_one_thread_per_tile_in_c_mat.cu

mult_cublas.o: util.cuh mult_cublas.cuh mult_cublas.cu
	$(CC) $(NVCC_FLAGS) -c -o $@ mult_cublas.cu

mult_naive.o: util.cuh mult_naive.cuh mult_naive.cu
	$(CC) $(NVCC_FLAGS) -c -o $@ mult_naive.cu

mult_tiled_32x32_conventional.o: util.cuh mult_tiled_32x32_conventional.cuh mult_tiled_32x32_conventional.cu
	$(CC) $(NVCC_FLAGS) -c -o $@ mult_tiled_32x32_conventional.cu

mult_2.o: util.cuh mult_2.cuh mult_2.cu
	$(CC) $(NVCC_FLAGS) -c -o $@ mult_2.cu

mult_one_warp_per_tile.o: util.cuh mult_one_warp_per_tile.cuh mult_one_warp_per_tile.cu
	$(CC) $(NVCC_FLAGS) -c -o $@ mult_one_warp_per_tile.cu

mult_one_warp_per_tile_2.o: util.cuh mult_one_warp_per_tile_2.cuh mult_one_warp_per_tile_2.cu
	$(CC) $(NVCC_FLAGS) -c -o $@ mult_one_warp_per_tile_2.cu

clean:
	rm -f $(MAIN) *.o
