#ifndef _ELF_COL_CNT_CUH_
#define _ELF_COL_CNT_CUH_

struct CollisionCountPromise {
	int *d_toReduce;
	int *d_reduced;
};

struct CollisionCountPromise
	count_collisions_launch(int3 *vector, int size);

int count_collisions_fetch(struct CollisionCountPromise promise);

#endif
