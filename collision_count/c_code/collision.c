#include <stdio.h>
#include <stdlib.h>

#ifndef METHOD
	#define METHOD 1
#endif

typedef int int3[3];

#if METHOD == 0
	#include "collision_count_sequential.h"
#elif METHOD == 1
	#include "collision_count_sequential_linear.h"
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
		result[i][0] = 0;
		result[i][1] = 0;
		result[i][2] = i % (size/2);

		// Randomize due to caching effects
		// result[i][0] = rand() % (size/2);
		// result[i][1] = rand() % (size/2);
		// result[i][2] = rand() % (size/2);
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

	int vecSize = 1000;
	int iters = 10000;

	int3 *vec = create_vector(vecSize);
	test_count(vec, vecSize, iters);
	free(vec);
}

int main(int argc, char *argv[]){
	t2();
	return 0;
}
