all: clean seq_quad seq_lin ns_sr ns_mr hs_sr ss_at ss_ht

seq_quad:
	nvcc -O3 -D SEQUENTIAL_QUADRATIC    collision.cu Sequential_Quadratic/ElfColCnt.cu    -o seq_quad

seq_lin:
	nvcc -O3 -D SEQUENTIAL_LINEAR       collision.cu Sequential_Linear/ElfColCnt.cu       -o seq_lin

ns_sr:
	nvcc -O3 -D NSTEPS_SINGLEROW        collision.cu NSteps_SingleRow/ElfColCnt.cu        -o ns_sr

ns_mr:
	nvcc -O3 -D NSTEPS_MULTIROW         collision.cu NSteps_MultiRow/ElfColCnt.cu         -o ns_mr

hs_sr:
	nvcc -O3 -D HALFSTEPS_SINGLEROW     collision.cu HalfSteps_SingleRow/ElfColCnt.cu     -o hs_sr

ss_at:
	nvcc -O3 -D SINGLESTEPS_ALLTHREADS  collision.cu SingleSteps_AllThreads/ElfColCnt.cu  -o ss_at

ss_ht:
	nvcc -O3 -D SINGLESTEPS_HALFTHREADS collision.cu SingleSteps_HalfThreads/ElfColCnt.cu -o ss_ht

clean:
	find -name "*~" -exec rm -vf '{}' \;
	rm -vf seq_quad seq_lin ns_sr ns_mr hs_sr ss_at ss_ht
