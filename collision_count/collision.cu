#include <stdio.h>
#include <cuda.h>

#define METHOD 1

#if METHOD == 0
	#include "collision_count_nsteps.cuh"
#elif METHOD == 1
	#include "collision_count_halfsteps.cuh"
#elif METHOD == 2
	#include "collision_count_sequential.cuh"
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

int3 *create_vector(int size){
	int i;
	int3 *result = (int3 *) malloc(sizeof(int3) * size);

	for(i = 0; i < size; i++){
		result[i].x = 0;
		result[i].y = 0;
		result[i].z = i;
	}

	return result;
}

int main(int argc, char *argv[]){
	int vecSize = 1000;
	int iters = 10000;

	int3 *vec = create_vector(vecSize);
	test_count(vec, vecSize, iters);
	free(vec);

	return 0;
}
