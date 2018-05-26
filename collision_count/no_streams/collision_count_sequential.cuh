#ifndef COLLISION_COUNT_SEQUENTIAL_H_
#define COLLISION_COUNT_SEQUENTIAL_H_

#include <time.h>
#include <stdio.h>

/* Given a vector with 3D coordinates of points in the space,
 *   this function calculates the number of collisions among
 *   points.
 */
int count_collisions(int3 *vector, int size){
	int i, j, collisions;

	collisions = 0;
	for(i = 0; i < size-1; i++){
		for(j = i+1; j < size; j++){
			collisions += (
					vector[i].x == vector[j].x
					&& vector[i].y == vector[j].y
					&& vector[i].z == vector[j].z
			);
		}
	}

	return collisions;
}

void test_count(int3 *vector, int size, int iters){
	int i;

	int beg = clock();
	for(i = 0; i < iters; i++){
		int res = count_collisions(vector, size);

		volatile int a;
		a = res;
	}
	printf("Elapsed: %lf ms\n", (clock() - beg) / (double) CLOCKS_PER_SEC * 1000);
}

#endif /* COLLISION_COUNT_SEQUENTIAL_H_ */
