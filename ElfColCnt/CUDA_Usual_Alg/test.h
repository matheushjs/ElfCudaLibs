#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "ElfColCnt.h"

#ifdef BATCH
	#define BATCH 1
#else
	#define BATCH 0
#endif

int test_count(ElfFloat3d *vector, int size, int iters){
	struct CollisionCountPromise *promises;
	promises = (struct CollisionCountPromise *) malloc(sizeof(struct CollisionCountPromise) * iters);

	int beg = clock();

	int i, res;
#if BATCH == 1
	for(i = 0; i < iters; i++){
		promises[i] = count_collisions_launch(vector, size);
	}
	for(i = 0; i < iters; i++){
		res = count_collisions_fetch(promises[i]);
	}
#else
	for(i = 0; i < iters; i++){
		promises[i] = count_collisions_launch(vector, size);
		res = count_collisions_fetch(promises[i]);
	}
#endif

	printf("Elapsed: %lf ms\n", (clock() - beg) / (double) CLOCKS_PER_SEC * 1000);
	printf("Collisions [N Steps Grid 1Row %s]: %d\n", BATCH ? "Batch" : "Non-Batch", res);

	return res;
}
