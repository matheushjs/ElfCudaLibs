#include <stdio.h>

#include "Sequential_Linear.h"

#define COORD3D(V, AXIS) COORD(V.x, V.y, V.z, AXIS)
#define COORD(X, Y, Z, AXIS) ( (Z+AXIS/2) * ((long int) AXIS*AXIS) + (Y+AXIS/2) * ((long int) AXIS) + (X+AXIS/2))

/* Documented in header file */
int count_collisions(ElfInt3d *vector, int size, AXISTYPE *space3d, int axisSize){
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
		int beadCount = space3d[idx];
		collisions += beadCount;
		space3d[idx]++;
	}

	return collisions;
}


