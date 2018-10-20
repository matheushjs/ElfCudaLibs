#include <stdio.h>

/* Documented in header file */
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
