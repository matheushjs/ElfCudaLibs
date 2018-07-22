#include "quadratic.h"

/* Given a vector with 3D coordinates of points in the space,
 *   this function calculates the number of collisions among
 *   points.
 */
int count_collisions(int3 *vector, int size){
	int i, j, collisions;

	collisions = 0;
	for(i = 0; i < size-1; i++){
		int3 buf = { vector[i][0], vector[i][1], vector[i][2], };
		for(j = i+1; j < size; j++){
			collisions += (
					buf[0] == vector[j][0]
					&& buf[1] == vector[j][1]
					&& buf[2] == vector[j][2]
			);
		}
	}

	return collisions;
}
