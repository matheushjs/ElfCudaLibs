#ifndef _ELF_COL_CNT_H_
#define _ELF_COL_CNT_H_

/*
This represents the type that will be used to hold the number of collisions
  in each point of 'space3d'.

Setting it to 'char' saves memory, but limits the number of collisions in a single point to 127.

Don't touch unless you are sure of what you're doing.
*/
#define AXISTYPE char

typedef struct {
	int x;
	int y;
	int z;
} int3d;

/*
Calculates the number of collisions among the 'size' beads in 'vector'.

'space3d' is an allocated memory that represents the discrete space in which the beads are located.
Each element in the array represents a point in the discrete space; so, for example, space3d[x][y][z]
  would represent the coordinate (x,y,z) in the discrete space.
This array should be big enough so that all beads in 'vector' have coordinates within the allocated space.

'space3d' has 3 axes, each of which has size 'axisSize'.
If 'axisSize' is 8, we assume each axis contains the values {-4, -3, -2, 1, 0, 1, 2, 3}
*/
int count_collisions(int3d *vector, int size, AXISTYPE *space3d, int axisSize);

#endif
