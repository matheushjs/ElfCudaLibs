#include <stdlib.h>
#include <stdio.h>

#include "ElfColCnt.h"

int test_count(ElfInt3d *vector, int size, int iters){
	int i, res = 0;

	for(i = 0; i < iters; i++){
		res = count_collisions(vector, size);
	}

	return res;
}
