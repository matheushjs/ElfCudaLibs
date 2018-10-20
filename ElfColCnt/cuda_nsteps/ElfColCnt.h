#ifndef _ELF_COL_CNT_CUH_
#define _ELF_COL_CNT_CUH_

struct CollisionCountPromise {  // Vectors of # of collisions
	int *d_toReduce;
	int *d_reduced;
};

typedef struct {
	float x;
	float y;
	float z;
} float3d;

/*
Launches the GPU procedure for counting collisions in 'vector' which has size 'size'.

This function does not wait until the GPU procedure is finished.
It returns a "Promise" structure which represents a promise for a future return value.
The return value can be fetched with the _fetch corresponding function.
*/
struct CollisionCountPromise
	count_collisions_launch(float3d *vector, int size);

/*
Returns the number of collisions associated with the given "Promise" structure.

The "Promise" structure is a promise for a future return value, which is returned
  by the non-blocking _launch function.
*/
int count_collisions_fetch(struct CollisionCountPromise promise);

#endif
