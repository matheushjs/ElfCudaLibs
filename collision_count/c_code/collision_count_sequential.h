#ifndef COLLISION_COUNT_SEQUENTIAL_H_
#define COLLISION_COUNT_SEQUENTIAL_H_

#include <time.h>
#include <stdio.h>

#include "quadratic.h"

void test_count(int3 *vector, int size, int iters){
	int i;

	int beg = clock();
	for(i = 0; i < iters; i++){
		int res = count_collisions(vector, size);
	}
	printf("Elapsed: %lf ms\n", (clock() - beg) / (double) CLOCKS_PER_SEC * 1000);
}

#endif /* COLLISION_COUNT_SEQUENTIAL_H_ */
