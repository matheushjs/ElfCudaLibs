#ifndef COLLISION_COUNT_SEQUENTIAL_LINEAR_H
#define COLLISION_COUNT_SEQUENTIAL_LINEAR_H

#include <time.h>
#include <stdio.h>

#define COORD3D(V, AXIS) COORD(V[0], V[1], V[2], AXIS)
#define COORD(X, Y, Z, AXIS) ( (Z+AXIS/2) * (AXIS*AXIS) + (Y+AXIS/2) * (AXIS) + (X+AXIS/2))

#define AXISTYPE char

/* Given a vector with 3D coordinates of points in the space,
 *   this function calculates the number of collisions among
 *   points.
 *
 * 'space3d' is expected to be a vector with size:
 *    { [ (size + 1) * 2 ] ** 3 } * sizeof(AXISTYPE)
 */
int count_collisions(int3 *vector, int size, AXISTYPE *space3d){
	int i, collisions;
	int axisSize = (size + 1) * 2;

	collisions = 0;

	// Reset space
	for(i = 0; i < size; i++){
		int idx = COORD3D(vector[i], axisSize);
		space3d[idx] = 0;
	}

	// Place beads in the space (actually calculate the collisions at the same time)
	for(i = 0; i < size; i++){
		int idx = COORD3D(vector[i], axisSize);
		if(space3d[idx]){
			collisions += space3d[idx];
		}
		space3d[idx]++;
	}

	return collisions;
}

void test_count(int3 *vector, int size, int iters){
	int i;

	int beg = clock();
	
	int axisSize = (size + 1) * 2;
	unsigned long memSize = sizeof(AXISTYPE) * axisSize * axisSize * axisSize;
	AXISTYPE *space3d = (AXISTYPE *) malloc(memSize);
	printf("Memsize: %lf Gb, Pointer: %p\n", memSize / (double) 1024 / 1024 / 1024, space3d);
	for(i = 0; i < iters; i++){
		int res = count_collisions(vector, size, space3d);
	}

	free(space3d);
	
	printf("Elapsed: %lf ms\n", (clock() - beg) / (double) CLOCKS_PER_SEC * 1000);
}


#endif // COLLISION_COUNT_SEQUENTIAL_LINEAR_H
