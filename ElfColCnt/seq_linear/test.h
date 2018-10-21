#include <stdlib.h>
#include <stdio.h>

#include "ElfColCnt.h"

AXISTYPE *space3d = NULL;

int test_count(ElfInt3d *vector, int size, int iters){
	int i, res = 0;
	int axisSize = (size + 1) * 2;

	if(space3d == NULL){
		unsigned long memSize = sizeof(AXISTYPE) * axisSize * axisSize * (unsigned long) axisSize;
		space3d = (AXISTYPE *) malloc(memSize);
		printf("Memsize: %lf Gb, Pointer: %p\n", memSize / (double) 1024 / 1024 / 1024, space3d);
	}
	if(vector == NULL){
		free(space3d);
	}

	for(i = 0; i < iters; i++){
		res = count_collisions(vector, size, space3d, axisSize);
	}

	return res;
}
