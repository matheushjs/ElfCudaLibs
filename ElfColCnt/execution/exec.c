#include <stdio.h>
#include <stdlib.h>

/*
 * MACROS for building programs BUILD_TEST mode or 'run' mode
 *    TEST    - when run, the program will execute the batch of tests we defined
 *    nothing - when run, the program will count collisions for a vector of size provided as program argument
 * If nothing is specified, 'run' is assumed.
 */
#ifdef BUILD_TEST
	#define BUILD_TEST 1
#endif

/*
 * MACROS for defining which collision count method to test.
 */
#ifdef SEQ_QUAD
	#define SEQ_QUAD 1
	#include "../seq_quadratic/test.h"
#endif
#ifdef SEQ_LIN
	#define SEQ_LIN 1
	#include "../seq_linear/test.h"
#endif
#ifdef CUDA_HALF
	#define CUDA_HALF 1
	#include "../cuda_halfsteps/test.h"
#endif
#ifdef CUDA_N
	#define CUDA_N 1
	#include "../cuda_nsteps/test.h"
#endif
#if SEQ_QUAD != 1 && SEQ_LIN != 1 && CUDA_HALF != 1 && CUDA_N != 1
	#error "Must specify method!"
#endif

// Utilities for creating vectors and tests
#include "utils.h"

void run(int vecSize, int iters){
	// int vecSize = 1000;
	// int iters = 10000;

	#if SEQ_LIN == 1 || SEQ_QUAD == 1
		int3d *vec = vector_rand(vecSize);
		test_count(vec, vecSize, iters);
		free(vec);
	#else
		float3d *vec = vector_rand_f(vecSize);
		test_count(vec, vecSize, iters);
		free(vec);
	#endif
}

void test(){
	int size = 16 * 1024;
	int gold, res;

	// SEQ_LIN cannot handle big vectors
	#if SEQ_LIN == 1
		goto small;
	#endif

	// Allocate the vectors we need
	#if SEQ_QUAD == 1 || SEQ_LIN == 1
		int3d *seq             = vector_seq(size);
		int3d *neigh_paircolls = vector_neigh_paircolls(size);
		int3d *rand_paircolls  = vector_rand_paircolls(size);
	#else
		float3d *seq             = vector_seq_f(size);
		float3d *neigh_paircolls = vector_neigh_paircolls_f(size);
		float3d *rand_paircolls  = vector_rand_paircolls_f(size);
	#endif

	gold = 0;
	res = test_count(seq, size, 1);
	printf("Size %d: %s\n", size, gold == res ? "SUCCESS!" : "FAILURE");

	gold = size / 2;
	res = test_count(neigh_paircolls, size, 1);
	printf("Size %d: %s\n", size, gold == res ? "SUCCESS!" : "FAILURE");

	gold = size / 2;
	res = test_count(rand_paircolls, size, 1);
	printf("Size %d: %s\n", size, gold == res ? "SUCCESS!" : "FAILURE");

	free(seq);
	free(neigh_paircolls);
	free(rand_paircolls);

	// Then we repeat the above, with vectors of more irregular size
	size = 16 * 1024 + 220;

	#if SEQ_QUAD == 1 || SEQ_LIN == 1
		seq             = vector_seq(size);
		neigh_paircolls = vector_neigh_paircolls(size);
		rand_paircolls  = vector_rand_paircolls(size);
	#else
		seq             = vector_seq_f(size);
		neigh_paircolls = vector_neigh_paircolls_f(size);
		rand_paircolls  = vector_rand_paircolls_f(size);
	#endif

	gold = 0;
	res = test_count(seq, size, 1);
	printf("Size %d: %s\n", size, gold == res ? "SUCCESS!" : "FAILURE");

	gold = size / 2;
	res = test_count(neigh_paircolls, size, 1);
	printf("Size %d: %s\n", size, gold == res ? "SUCCESS!" : "FAILURE");

	gold = size / 2;
	res = test_count(rand_paircolls, size, 1);
	printf("Size %d: %s\n", size, gold == res ? "SUCCESS!" : "FAILURE");

	free(seq);
	free(neigh_paircolls);
	free(rand_paircolls);

small:

	// Then we repeat the above, with a small vector
	size = 220;

	#if SEQ_QUAD == 1 || SEQ_LIN == 1
		seq             = vector_seq(size);
		neigh_paircolls = vector_neigh_paircolls(size);
		rand_paircolls  = vector_rand_paircolls(size);
	#else
		seq             = vector_seq_f(size);
		neigh_paircolls = vector_neigh_paircolls_f(size);
		rand_paircolls  = vector_rand_paircolls_f(size);
	#endif

	gold = 0;
	res = test_count(seq, size, 1);
	printf("Size %d: %s\n", size, gold == res ? "SUCCESS!" : "FAILURE");

	gold = size / 2;
	res = test_count(neigh_paircolls, size, 1);
	printf("Size %d: %s\n", size, gold == res ? "SUCCESS!" : "FAILURE");

	gold = size / 2;
	res = test_count(rand_paircolls, size, 1);
	printf("Size %d: %s\n", size, gold == res ? "SUCCESS!" : "FAILURE");

	free(seq);
	free(neigh_paircolls);
	free(rand_paircolls);
}


int main(int argc, char *argv[]){
	int vecSize = 32 * 16 * 1024;
	int iters   = 1;
	
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
		default:
			fprintf(stderr, "Usage: %s [problem_size] [no. iterations]\n", argv[0]);
			return 1;
	}
	
#if BUILD_TEST == 1
	test();
#else
	run(vecSize, iters);
#endif

	return 0;
}
