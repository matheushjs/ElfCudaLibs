#include <stdio.h>
#include <cuda.h>

#define METHOD 0

#if METHOD == 0
	#include "collision_count_nsteps.h"
#elif METHOD == 1
	#include "collision_count_sequential.h"
#else
	#error "Fix method mate."
#endif

int3 dummy[] = {
		{0, 0, 0}, // 0
		{0, 0, 0}, // 1
		{0, 0, 1}, // 2
		{0, 0, 2}, // 3
		{0, 0, 3}, // 4
		{0, 1, 3}, // 5
		{0, 2, 3}, // 6
		{0, 2, 2}, // 7
		{0, 2, 1}, // 8
		{0, 2, 0}, // 9
		{0, 1, 0}, // 10
		{0, 0, 0}, // 11
		{-1,0, 0}, // 12
		{-2,0, 0}, // 13
		{-2,-1,0}, // 14
		{-1,-1,0}, // 15
		{0 ,-1,0}, // 16
		{0, 0, 0}, // 17
		{0, 0, 0}  // 18
}; // There are 5 {0,0,0}, meaning 4 + 3 + 2 + 1 = 10 collisions

int main(int argc, char *argv[]){
	test_count(dummy, sizeof(dummy) / (3 * sizeof(int)), 100000);

	return 0;
}
