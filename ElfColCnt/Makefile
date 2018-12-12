
BUILD_TYPE?=BUILD_USER

all: seq_quad seq_lin cuda_n cuda_half

seq_quad: dummy
	gcc -Wall -O3 -D $(BUILD_TYPE) -D SEQ_QUAD  includes/exec.c Sequential_Quadratic/ElfColCnt.c   -o $@ -lm

seq_lin: dummy
	gcc -Wall -O3 -D $(BUILD_TYPE) -D SEQ_LIN   includes/exec.c Sequential_Linear/ElfColCnt.c      -o $@ -lm

cuda_n: dummy
	nvcc -O3      -I ./includes/ -D $(BUILD_TYPE) -D CUDA_N    includes/exec.c CUDA_Usual_Alg/ElfColCnt.cu    -o $@

cuda_half: dummy
	nvcc -O3      -I ./includes/ -D $(BUILD_TYPE) -D CUDA_HALF includes/exec.c CUDA_Proposed_Alg/ElfColCnt.cu -o $@

test: dummy
	make BUILD_TYPE=BUILD_TEST

dummy:

clean:
	find -name "*~" -exec rm -vf '{}' \;
	rm -vf seq_quad seq_lin cuda_n cuda_half
