
BUILD_TYPE?=BUILD_USER

all: seq_quad seq_lin cuda_usual cuda_proposed

seq_quad: dummy
	gcc -Wall -O3 -D $(BUILD_TYPE) -D SEQ_QUAD  src/exec.c src/Sequential_Quadratic.c   -o $@ -lm

seq_lin: dummy
	gcc -Wall -O3 -D $(BUILD_TYPE) -D SEQ_LIN   src/exec.c src/Sequential_Linear.c      -o $@ -lm

cuda_usual: dummy
	nvcc -O3      -D $(BUILD_TYPE) -D CUDA_N    src/exec.c src/CUDA_Usual_Alg.cu    -o $@

cuda_proposed: dummy
	nvcc -O3      -D $(BUILD_TYPE) -D CUDA_HALF src/exec.c src/CUDA_Proposed_Alg.cu -o $@

test: dummy
	make BUILD_TYPE=BUILD_TEST

dummy:

clean:
	find -name "*~" -exec rm -vf '{}' \;
	rm -vf seq_quad seq_lin cuda_n cuda_half
