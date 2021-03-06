#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>

/*
 * MACROS for defining which collision count method to test.
 */
#ifdef SEQ_QUAD
	#define SEQ_QUAD 1
	#include "Sequential_Quadratic.h"
#endif
#ifdef SEQ_LIN
	#define SEQ_LIN 1
	#include "Sequential_Linear.h"
#endif
#ifdef CUDA_PROPOSED
	#define CUDA_PROPOSED 1
	#include "CUDA_Proposed_Alg.h"
#endif
#ifdef CUDA_USUAL
	#define CUDA_USUAL 1
	#include "CUDA_Usual_Alg.h"
#endif
#if SEQ_QUAD != 1 && SEQ_LIN != 1 && CUDA_PROPOSED != 1 && CUDA_USUAL != 1
	#error "Must specify method using \"-D [method]\". [method] can be SEQ_QUAD SEQ_LIN CUDA_PROPOSED or CUDA_USUAL"
#endif

// Utilities for creating vectors and tests
#include "utils.h"


#ifdef SEQ_LIN
void run(int vecSize, int iters, double std, int rrate){
	int i;
	srand(72);

	int beg = clock();
	int res = 0;
	AXISTYPE *space3d = NULL;
	int axisSize = (vecSize + 1) * 2;
	unsigned long memSize = sizeof(AXISTYPE) * axisSize * axisSize * (unsigned long) axisSize;

	// Perform many iterations to collect the time spent
	for(i = 0; i < iters; i++){
		// Reallocate space array
		if(i%rrate == 0){
			free(space3d);
			space3d = (AXISTYPE *) malloc(memSize);
		}

		// Create a new random vector
		// ElfInt3d *vec = vector_protein(vecSize);
		ElfInt3d *vec = vector_rnorm(vecSize, vecSize, std);
		res = count_collisions(vec, vecSize, space3d, axisSize);
		free(vec); // Free vector
	}
	free(space3d);

	printf("CPU time: %lf ms\n", (clock() - beg) / (double) CLOCKS_PER_SEC * 1000);
	printf("Collisions [Linear]: %d\n", res);
}
#endif

#ifdef SEQ_QUAD
void run(int vecSize, int iters, double std, int rrate){
	int i;
	srand(72);

	int beg = clock();
	int res = 0;

	for(i = 0; i < iters; i++){
		// Create a new random vector
		// ElfInt3d *vec = vector_protein(vecSize);
		ElfInt3d *vec = vector_rnorm(vecSize, vecSize, std);
		res = count_collisions(vec, vecSize);
		free(vec); // Free vector
	}

	printf("CPU Time: %lf ms\n", (clock() - beg) / (double) CLOCKS_PER_SEC * 1000);
	printf("Collisions [Quadratic]: %d\n", res);
}
#endif

#ifdef CUDA_PROPOSED
void run(int vecSize, int iters, double std, int rrate){
	int i, res;
	struct CollisionCountPromise promise;

	ElfFloat3d *vec = vector_rand_f(vecSize);

	int beg = clock();
	for(i = 0; i < iters; i++){
		promise = count_collisions_launch(vec, vecSize);
		res = count_collisions_fetch(promise);
	}

	printf("CPU Time: %lf ms\n", (clock() - beg) / (double) CLOCKS_PER_SEC * 1000);
	printf("Collisions [CUDA Proposed]: %d\n", res);

	free(vec);
}
#endif

#ifdef CUDA_USUAL
void run(int vecSize, int iters, double std, int rrate){
	int i, res;
	struct CollisionCountPromise promise;

	ElfFloat3d *vec = vector_rand_f(vecSize);

	int beg = clock();
	for(i = 0; i < iters; i++){
		promise = count_collisions_launch(vec, vecSize);
		res = count_collisions_fetch(promise);
	}

	printf("CPU Time: %lf ms\n", (clock() - beg) / (double) CLOCKS_PER_SEC * 1000);
	printf("Collisions [CUDA Usual]: %d\n", res);

	free(vec);
}
#endif

int main(int argc, char *argv[]){
	int vecSize = 32 * 16 * 1024;
	int iters   = 1;
	double std  = 1;
	int rrate   = INT_MAX;
	
	switch(argc){
		case 1:
			break;
		case 2:
			vecSize = atoi(argv[1]);
			break;
		case 3:
			vecSize = atoi(argv[1]);
			iters   = atoi(argv[2]);
			break;
		case 4:
			vecSize = atoi(argv[1]);
			iters   = atoi(argv[2]);
			std     = atof(argv[3]);
			break;
		case 5:
			vecSize = atoi(argv[1]);
			iters   = atoi(argv[2]);
			std     = atof(argv[3]);
			rrate   = atoi(argv[4]);
			break;
		default:
			fprintf(stderr, "Usage: %s [problem_size] [no. iterations] [std deviation] [realloc_rate]\n", argv[0]);
			return 1;
	}
	
#if BUILD_TEST == 1
	test();
#else
	run(vecSize, iters, std, rrate);
#endif

	return 0;
}
