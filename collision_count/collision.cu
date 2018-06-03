#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#ifndef METHOD
	#define METHOD 1
#endif

#if METHOD == 0
	#include "collision_count_nsteps.cuh"
#elif METHOD == 1
	#include "collision_count_halfsteps.cuh"
#elif METHOD == 2
	#include "collision_count_singlestep.cuh"
#elif METHOD == 3
	#include "collision_count_sequential.cuh"
#elif METHOD == 4
	#include "collision_count_sequential_linear.cuh"
#elif METHOD == 10
	#include "collision_count_nsteps_grid.cuh"
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

void t1(){
	int dummySize = sizeof(dummy) / sizeof(int3);
	test_count(dummy, dummySize, 1);
}

void t2(){
	// int vecSize = 1000;
	// int iters = 10000;

	int vecSize = 100000;
	int iters = 1;

	int3 *vec = create_vector(vecSize);
	test_count(vec, vecSize, iters);
	free(vec);
}

int main(int argc, char *argv[]){
	t2();
	return 0;
}
