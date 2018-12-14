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

static
void verify(int problemSize, int answer, int truth){
	printf("Size %d: %s\n", problemSize, answer == truth ? "SUCCESS!" : "FAILURE");
}

#ifdef SEQ_LIN
void test(){
	int sizes[] = { 256, 257 }; // Largest one must be to the left
	int nCases = sizeof(sizes) / sizeof(int);
	int i;

	AXISTYPE *space3d = NULL;
	int axisSize = (sizes[1] + 1) * 2;
	unsigned long memSize = sizeof(AXISTYPE) * axisSize * axisSize * (unsigned long) axisSize;
	space3d = (AXISTYPE *) malloc(memSize);

	for(i = 0; i < nCases; i++){
		int size = sizes[i];

		ElfInt3d *seq             = vector_seq(size);
		ElfInt3d *neigh_paircolls = vector_neigh_paircolls(size);
		ElfInt3d *rand_paircolls  = vector_rand_paircolls(size);

		verify(size, // Beads placed in different places of the space
				count_collisions(seq, size, space3d, axisSize),
				0);
		verify(size, // beads 1 and 2 collide, 3 and 4 collide, and so on
				count_collisions(neigh_paircolls, size, space3d, axisSize),
				size/2);
		verify(size, // Pairs of beads collide, but the are randomly located in the vector
				count_collisions(rand_paircolls, size, space3d, axisSize),
				size/2);

		free(seq);
		free(neigh_paircolls);
		free(rand_paircolls);
	}

	free(space3d);
}
#endif

#ifdef SEQ_QUAD
void test(){
	int sizes[] = { 16*1024, 16*1024+220, 220,  16*1024+1, 16*1024+221, 221};
	int nCases = sizeof(sizes) / sizeof(int);
	int i;

	for(i = 0; i < nCases; i++){
		int size = sizes[i];

		ElfInt3d *seq             = vector_seq(size);
		ElfInt3d *neigh_paircolls = vector_neigh_paircolls(size);
		ElfInt3d *rand_paircolls  = vector_rand_paircolls(size);

		verify(size, // Beads placed in different places of the space
				count_collisions(seq, size),
				0);
		verify(size, // beads 1 and 2 collide, 3 and 4 collide, and so on
				count_collisions(neigh_paircolls, size),
				size/2);
		verify(size, // Pairs of beads collide, but the are randomly located in the vector
				count_collisions(rand_paircolls, size),
				size/2);

		free(seq);
		free(neigh_paircolls);
		free(rand_paircolls);
	}
}
#endif

#ifdef CUDA_PROPOSED
void test(){
	int sizes[] = { 16*1024, 16*1024+220, 220, 16*1024+1, 16*1024+221, 221 };
	int nCases = sizeof(sizes) / sizeof(int);
	int i;
	struct CollisionCountPromise promise[3];

	for(i = 0; i < nCases; i++){
		int size = sizes[i];

		ElfFloat3d *seq             = vector_seq_f(size);
		ElfFloat3d *neigh_paircolls = vector_neigh_paircolls_f(size);
		ElfFloat3d *rand_paircolls  = vector_rand_paircolls_f(size);

		promise[0] = count_collisions_launch(seq, size);
		promise[1] = count_collisions_launch(neigh_paircolls, size);
		promise[2] = count_collisions_launch(rand_paircolls, size);

		verify(size, // Beads placed in different places of the space
				count_collisions_fetch(promise[0]),
				0);
		verify(size, // beads 1 and 2 collide, 3 and 4 collide, and so on
				count_collisions_fetch(promise[1]),
				size/2);
		verify(size, // Pairs of beads collide, but the are randomly located in the vector
				count_collisions_fetch(promise[2]),
				size/2);

		free(seq);
		free(neigh_paircolls);
		free(rand_paircolls);
	}
}
#endif

#ifdef CUDA_USUAL
void test(){
	int sizes[] = { 16*1024, 16*1024+220, 220, 16*1024+1, 16*1024+221, 221 };
	int nCases = sizeof(sizes) / sizeof(int);
	int i;
	struct CollisionCountPromise promise[3];

	for(i = 0; i < nCases; i++){
		int size = sizes[i];

		ElfFloat3d *seq             = vector_seq_f(size);
		ElfFloat3d *neigh_paircolls = vector_neigh_paircolls_f(size);
		ElfFloat3d *rand_paircolls  = vector_rand_paircolls_f(size);

		promise[0] = count_collisions_launch(seq, size);
		promise[1] = count_collisions_launch(neigh_paircolls, size);
		promise[2] = count_collisions_launch(rand_paircolls, size);

		verify(size, // Beads placed in different places of the space
				count_collisions_fetch(promise[0]),
				0);
		verify(size, // beads 1 and 2 collide, 3 and 4 collide, and so on
				count_collisions_fetch(promise[1]),
				size/2);
		verify(size, // Pairs of beads collide, but the are randomly located in the vector
				count_collisions_fetch(promise[2]),
				size/2);

		free(seq);
		free(neigh_paircolls);
		free(rand_paircolls);
	}
}
#endif

int main(int argc, char *argv[]){
	test();

	return 0;
}
