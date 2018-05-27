#ifndef COLLISION_COUNT_SEQUENTIAL_LINEAR_H
#define COLLISION_COUNT_SEQUENTIAL_LINEAR_H

#include <time.h>
#include <stdio.h>

#include "linear.h"

void test_count(int3 *vector, int size, int iters){
	int i;

	int beg = clock();
	
	int axisSize = (size + 1) * 2;
	unsigned long memSize = sizeof(AXISTYPE) * axisSize * axisSize * (unsigned long int) axisSize;
	AXISTYPE *space3d = (AXISTYPE *) malloc(memSize);
	printf("Memsize: %lf Gb, Pointer: %p\n", memSize / (double) 1024 / 1024 / 1024, space3d);
	for(i = 0; i < iters; i++){
		int res = count_collisions(vector, size, space3d);
	}

	free(space3d);
	
	printf("Elapsed: %lf ms\n", (clock() - beg) / (double) CLOCKS_PER_SEC * 1000);
}


#endif // COLLISION_COUNT_SEQUENTIAL_LINEAR_H
