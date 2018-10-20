#ifndef _ELF_COL_CNT_H_
#define _ELF_COL_CNT_H_

typedef struct {
	int x;
	int y;
	int z;
} ElfInt3d;

typedef struct {
	float x;
	float y;
	float z;
} ElfFloat3d;

/*
Given a vector 'vector' with 'size' beads, returns the number of collisions among all beads.
*/
int count_collisions(ElfInt3d *vector, int size);

#endif
