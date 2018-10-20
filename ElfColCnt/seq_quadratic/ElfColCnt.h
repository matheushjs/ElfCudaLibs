#ifndef _ELF_COL_CNT_H_
#define _ELF_COL_CNT_H_

typedef struct {
	int x;
	int y;
	int z;
} int3d;

/*
Given a vector 'vector' with 'size' beads, returns the number of collisions among all beads.
*/
int count_collisions(int3d *vector, int size);

#endif
