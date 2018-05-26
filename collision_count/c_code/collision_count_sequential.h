#ifndef COLLISION_COUNT_SEQUENTIAL_H_
#define COLLISION_COUNT_SEQUENTIAL_H_

#include <time.h>
#include <stdio.h>

#include "quadratic.h"

void test_count(int3 *vector, int size, int iters){
	int i, res;

	int beg = clock();
	for(i = 0; i < iters; i++){
		res = count_collisions(vector, size);
	}

	printf("Elapsed: %lf ms\n", (clock() - beg) / (double) CLOCKS_PER_SEC * 1000);
	printf("Collisions: %d\n", res);
}

#endif /* COLLISION_COUNT_SEQUENTIAL_H_ */
