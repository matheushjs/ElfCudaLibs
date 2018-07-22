#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#ifndef METHOD
	#define METHOD 1
#endif

#if METHOD == 0
	#include "ElfColCnt_seq_quad.cuh" // Sequential Quadratic
#elif METHOD == 1
	#include "ElfColCnt_seq_lin.cuh"  // Sequential Linear
#elif METHOD == 2
	#include "ElfColCnt_ns_sr.cuh"    // NSteps SingleRow
#elif METHOD == 3
	#include "ElfColCnt_ns_mr.cuh"    // NSteps MultiRow
#elif METHOD == 4
	#include "ElfColCnt_hs_sr.cuh"    // HalfSteps SingleRow
#elif METHOD == 5
	#include "ElfColCnt_ss_ht.cuh"    // SingleSteps HalfThreads
#else
	#error "Fix method mate."
#endif

int3 dummy[] = {
		{0, 0, 0}, // 0
		{0, 0, 0}, // 1
		{0, 0, 1}, // 2
		{0, 0, 2}, // 3
		{0, 0, 3}, // 4
		{0, 1, 3}, // 5
		{0, 2, 3}, // 6
		{0, 2, 2}, // 7
		{0, 2, 1}, // 8
		{0, 2, 0}, // 9
		{0, 1, 0}, // 10
		{0, 0, 0}, // 11
		{-1,0, 0}, // 12
		{-2,0, 0}, // 13
		{-2,-1,0}, // 14
		{-1,-1,0}, // 15
		{0 ,-1,0}, // 16
		{0, 0, 0}, // 17
		{0, 0, 0}  // 18
}; // There are 5 {0,0,0}, meaning 4 + 3 + 2 + 1 = 10 collisions

int3 dummy2[] = {
	{0, 0, 0},
	{0, 0, 0},
	{1, 0, 0},
	{1, 0, 0}
}; // 2 collision

int3 *create_vector(int size){
	int i;
	int3 *result = (int3 *) malloc(sizeof(int3) * size);

	for(i = 0; i < size; i++){
		result[i].x = 0;
		result[i].y = 0;
		result[i].z = (i % (size/2)) * 4 - size;

		// Randomize due to caching effects
		// result[i].x = rand()%(2*size) - size;
		// result[i].y = rand()%(2*size) - size;
		// result[i].z = rand()%(2*size) - size;
	}

	printf("Collisions expected: %d\n", size/2);

	return result;
}

// Creates random vector with size/2 collisions
int3 *random_vector(int size){
	int i;
	int3 *result = (int3 *) malloc(sizeof(int3) * size);

	// Generate beads with size/2 collisions
	for(i = 0; i < size; i += 2){
		result[i].x = (i*2) - size; // Guarantees different positions among pairs of beads
		result[i].y = rand()%(2*size) - size;
		result[i].z = rand()%(2*size) - size;

		// Next bead is the same as previous
		result[i+1] = result[i];
	}

	// Randomize bead positions
	for(i = 0; i < size; i++){
		int a = rand()%size;
		int b = rand()%size;

		int3 aux = result[a];
		result[a] = result[b];
		result[b] = aux;
	}

	printf("Collisions expected: %d\n", size/2);

	return result;
}

int3 *sequential_vector(int size){
	int3 *result = (int3 *) malloc(sizeof(int3) * size);

	for(int i = 0; i < size; i++){
		result[i] = (int3) {i, i, i};
	}

	return result;
}

void t1(){
	int dummySize = sizeof(dummy) / sizeof(int3);
	test_count(dummy, dummySize, 1);
}

void t2(int vecSize, int iters){
	// int vecSize = 1000;
	// int iters = 10000;

	int3 *vec = random_vector(vecSize);

	test_count(vec, vecSize, iters);
	free(vec);
}

void t3(){
	int size = 16 * 1024;

	// First we create a vector where neighbors have collisions
	int3 *vec = sequential_vector(size);
	for(int i = 0; i < size; i += 2){
		vec[i] = vec[i+1];
	}
	printf("\nExpected: %d\n", size / 2);
	test_count(vec, size, 1);
	free(vec);

	// Then we create a vector where all elements are colliding
	vec = sequential_vector(size);
	for(int i = 0; i < size; i++){
		vec[i] = vec[0];
	}
	printf("\nExpected: %d\n", size * (size - 1) / 2);
	test_count(vec, size, 1);
	free(vec);

	// Finally, no collisions at all
	vec = sequential_vector(size);
	printf("\nExpected: %d\n", 0);
	test_count(vec, size, 1);
	free(vec);

	// Then we repeat the above, with vectors of more irregular size
	size = 16 * 1024 + 220;

	// First we create a vector where neighbors have collisions
	vec = sequential_vector(size);
	for(int i = 0; i < size; i += 2){
		vec[i] = vec[i+1];
	}
	printf("\nExpected: %d\n", size / 2);
	test_count(vec, size, 1);
	free(vec);

	// Then we create a vector where all elements are colliding
	vec = sequential_vector(size);
	for(int i = 0; i < size; i++){
		vec[i] = vec[0];
	}
	printf("\nExpected: %d\n", size * (size - 1) / 2);
	test_count(vec, size, 1);
	free(vec);

	// Finally, no collisions at all
	vec = sequential_vector(size);
	printf("\nExpected: %d\n", 0);
	test_count(vec, size, 1);
	free(vec);
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
			fprintf(stderr, "Usage: %s [problem_size]\n", argv[0]);
			return 1;
	}
	
	t2(vecSize, iters);
	
	return 0;
}
