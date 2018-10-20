#ifndef _UTILS_H
#define _UTILS_H

/*
 * We don't need to declare ElfInt3d and ElfFloat3d because all ElfColCnt.h declare them.
 */

#include <stdlib.h>

// Creates a vector whose beads follow each other along the vector (1,1,1)
ElfInt3d *vector_seq(int size){
	int i;
	ElfInt3d *result = (ElfInt3d *) malloc(sizeof(ElfInt3d) * size);

	for(i = 0; i < size; i++){
		result[i].x = i;
		result[i].y = i;
		result[i].z = i;
	}

	return result;
}

// Creates a vector whose beads are in random coordinates
ElfInt3d *vector_rand(int size){
	int i;
	ElfInt3d *result = (ElfInt3d *) malloc(sizeof(ElfInt3d) * size);

	for(i = 0; i < size; i++){
		result[i].x = rand()%(2*size) - size;
		result[i].y = rand()%(2*size) - size;
		result[i].z = rand()%(2*size) - size;
	}

	return result;
}

// Creates a vector where each bead collides with 1 other bead
// The colliding beads are neighbors in the vector
ElfInt3d *vector_neigh_paircolls(int size){
	int i;
	ElfInt3d *result = (ElfInt3d *) malloc(sizeof(ElfInt3d) * size);

	// Generate beads with size/2 collisions
	for(i = 0; i < size; i += 2){
		result[i].x = i;
		result[i].y = i;
		result[i].z = i;

		// Next bead is the same as previous
		result[i+1].x = i;
		result[i+1].y = i;
		result[i+1].z = i;
	}

	return result;
}

// Creates a vector where each bead collides with 1 other bead
// The bead positions are randomized
ElfInt3d *vector_rand_paircolls(int size){
	int i;
	ElfInt3d *result = vector_neigh_paircolls(size);

	// Randomize bead positions
	for(i = 0; i < size*2; i++){
		int a = rand()%size;
		int b = rand()%size;

		ElfInt3d aux;
		aux.x       = result[a].x; aux.y       = result[a].y; aux.z       = result[a].z;
		result[a].x = result[b].x; result[a].y = result[b].y; result[a].z = result[b].z;
		result[b].x = aux.x;       result[b].y = aux.y;       result[b].z = aux.z;
	}
	
	return result;
}

// Creates a vector whose beads follow each other along the vector (1,1,1)
ElfFloat3d *vector_seq_f(int size){
	int i;
	ElfFloat3d *result = (ElfFloat3d *) malloc(sizeof(ElfFloat3d) * size);

	for(i = 0; i < size; i++){
		result[i].x = i;
		result[i].y = i;
		result[i].z = i;
	}

	return result;
}

// Creates a vector whose beads are in random coordinates
ElfFloat3d *vector_rand_f(int size){
	int i;
	ElfFloat3d *result = (ElfFloat3d *) malloc(sizeof(ElfFloat3d) * size);

	for(i = 0; i < size; i++){
		result[i].x = rand()%(2*size) - size;
		result[i].y = rand()%(2*size) - size;
		result[i].z = rand()%(2*size) - size;
	}

	return result;
}

// Creates a vector where each bead collides with 1 other bead
// The colliding beads are neighbors in the vector
ElfFloat3d *vector_neigh_paircolls_f(int size){
	int i;
	ElfFloat3d *result = (ElfFloat3d *) malloc(sizeof(ElfFloat3d) * size);

	// Generate beads with size/2 collisions
	for(i = 0; i < size; i += 2){
		result[i].x = i;
		result[i].y = i;
		result[i].z = i;

		// Next bead is the same as previous
		result[i+1].x = i;
		result[i+1].y = i;
		result[i+1].z = i;
	}

	return result;
}

// Creates a vector where each bead collides with 1 other bead
// The bead positions are randomized
ElfFloat3d *vector_rand_paircolls_f(int size){
	int i;
	ElfFloat3d *result = vector_neigh_paircolls_f(size);

	// Randomize bead positions
	for(i = 0; i < size*2; i++){
		int a = rand()%size;
		int b = rand()%size;

		ElfFloat3d aux;
		aux.x       = result[a].x; aux.y       = result[a].y; aux.z       = result[a].z;
		result[a].x = result[b].x; result[a].y = result[b].y; result[a].z = result[b].z;
		result[b].x = aux.x;       result[b].y = aux.y;       result[b].z = aux.z;
	}

	return result;
}

#endif
