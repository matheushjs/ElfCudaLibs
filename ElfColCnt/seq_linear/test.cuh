#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "ElfColCnt.cuh"

int test_count(int3d *vector, int size, int iters){
	int i, res;

	int beg = clock();
	
	int axisSize = (size + 1) * 2;
	unsigned long memSize = sizeof(AXISTYPE) * axisSize * axisSize * (unsigned long) axisSize;
	AXISTYPE *space3d = (AXISTYPE *) malloc(memSize);
	printf("Memsize: %lf Gb, Pointer: %p\n", memSize / (double) 1024 / 1024 / 1024, space3d);
	for(i = 0; i < iters; i++){
		res = count_collisions(vector, size, space3d, axisSize);
	}

	free(space3d);
	
	printf("Elapsed: %lf ms\n", (clock() - beg) / (double) CLOCKS_PER_SEC * 1000);
	printf("Collisions [Linear]: %d\n", res);

	return res;
}
