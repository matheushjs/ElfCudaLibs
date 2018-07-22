#include <stdio.h>

#include "ElfColCnt.cuh"

#define COORD3D(V, AXIS) COORD(V.x, V.y, V.z, AXIS)
#define COORD(X, Y, Z, AXIS) ( (Z+AXIS/2) * ((long int) AXIS*AXIS) + (Y+AXIS/2) * ((long int) AXIS) + (X+AXIS/2))

/* Given a vector with 3D coordinates of points in the space,
 *   this function calculates the number of collisions among
 *   points.
 *
 * 'space3d' is expected to be a vector with size:
 *    { [ (size + 1) * 2 ] ** 3 } * sizeof(AXISTYPE)
 */
int count_collisions(int3 *vector, int size, AXISTYPE *space3d, int axisSize){
	int i, collisions;

	collisions = 0;

	// Reset space
	for(i = 0; i < size; i++){
		long int idx = COORD3D(vector[i], axisSize);
		space3d[idx] = 0;
	}

	// Place beads in the space (actually calculate the collisions at the same time)
	for(i = 0; i < size; i++){
		long int idx = COORD3D(vector[i], axisSize);
		if(space3d[idx]){
			collisions += space3d[idx];
		}
		space3d[idx]++;
	}

	return collisions;
}


