
BUILD_TYPE?=BUILD_USER

all: clean seq_quad seq_lin cuda_n cuda_half

seq_quad:
	nvcc -O3 -D $(BUILD_TYPE) -D SEQ_QUAD  execution/exec.c seq_quadratic/ElfColCnt.c   -o $@

seq_lin:
	nvcc -O3 -D $(BUILD_TYPE) -D SEQ_LIN   execution/exec.c seq_linear/ElfColCnt.c      -o $@

cuda_n:
	nvcc -O3 -D $(BUILD_TYPE) -D CUDA_N    execution/exec.c cuda_nsteps/ElfColCnt.cu    -o $@

cuda_half:
	nvcc -O3 -D $(BUILD_TYPE) -D CUDA_HALF execution/exec.c cuda_halfsteps/ElfColCnt.cu -o $@

test:
	make BUILD_TYPE=BUILD_TEST

clean:
	find -name "*~" -exec rm -vf '{}' \;
	rm -vf seq_quad seq_lin cuda_n cuda_half
