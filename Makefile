CC = nvcc

NVCC_FLAGS = -Xcompiler -ftree-vectorize -Xcompiler -fopenmp -O2 -w -m64 -g -Wno-deprecated-gpu-targets \
#             -gencode=arch=compute_61,code=sm_61 \
			 -arch=sm_61  \
             -Xptxas -dlcm=cg

CUDA_INSTALL_PATH = /usr/local/cuda
MKLROOT = /local/gpgpu/software/intel/mkl/2021.4.0

INCLUDES = -I$(CUDA_INSTALL_PATH)/include -I./include -I${MKLROOT}/include

MKL_LIBS = -lpthread -lm
CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib -lcudart -lcuda -lcusparse -lnvidia-ml
LIBS = $(CUDA_LIBS) $(MKL_LIBS)

MAIN = main
OBJS = main.o mult_one_thread_per_tile_in_c_mat.o mult_naive.o mult_2.o

# TEMPLATE
#
# file.o: util.cuh file.cuh
#     $(CC) $(NVCC_FLAGS) -c -o $@ file.cu

all: $(MAIN)

$(MAIN): $(OBJS)
	$(CC) $(NVCC_FLAGS) -o $@ $^

main.o: main.cu util.cuh mult_one_thread_per_tile_in_c_mat.cuh mult_naive.cuh mult_2.cuh
	$(CC) $(NVCC_FLAGS) -c -o $@ main.cu

mult_one_thread_per_tile_in_c_mat.o: util.cuh mult_one_thread_per_tile_in_c_mat.cuh mult_one_thread_per_tile_in_c_mat.cu
	$(CC) $(NVCC_FLAGS) -c -o $@ mult_one_thread_per_tile_in_c_mat.cu

mult_naive.o: util.cuh mult_naive.cuh mult_naive.cu
	$(CC) $(NVCC_FLAGS) -c -o $@ mult_naive.cu

mult_2.o: util.cuh mult_2.cuh mult_2.cu
	$(CC) $(NVCC_FLAGS) -c -o $@ mult_2.cu

clean:
	rm -f $(MAIN) *.o
