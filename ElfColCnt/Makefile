
all: seq_quad seq_lin cuda_usual cuda_proposed
test: test_seq_quad test_seq_lin test_cuda_proposed test_cuda_usual


seq_quad: dummy
	gcc -Wall -O3 -D SEQ_QUAD  src/exec.c src/Sequential_Quadratic.c   -o $@ -lm

seq_lin: dummy
	gcc -Wall -O3 -D SEQ_LIN   src/exec.c src/Sequential_Linear.c      -o $@ -lm

cuda_usual: dummy
	nvcc -O3      -D CUDA_USUAL    src/exec.c src/CUDA_Usual_Alg.cu    -o $@

cuda_proposed: dummy
	nvcc -O3      -D CUDA_PROPOSED src/exec.c src/CUDA_Proposed_Alg.cu -o $@


test_seq_quad: dummy
	gcc -Wall -O3 -D SEQ_QUAD  src/test.c src/Sequential_Quadratic.c   -o $@ -lm

test_seq_lin: dummy
	gcc -Wall -O3 -D SEQ_LIN   src/test.c src/Sequential_Linear.c      -o $@ -lm

test_cuda_usual: dummy
	nvcc -O3      -D CUDA_USUAL    src/test.c src/CUDA_Usual_Alg.cu    -o $@

test_cuda_proposed: dummy
	nvcc -O3      -D CUDA_PROPOSED src/test.c src/CUDA_Proposed_Alg.cu -o $@

dummy:

clean:
	find -name "*~" -exec rm -vf '{}' \;
	rm -vf seq_quad seq_lin cuda_proposed cuda_usual
	rm -vf test_seq_quad test_seq_lin test_cuda_proposed test_cuda_usual
