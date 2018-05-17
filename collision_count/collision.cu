#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>

#include "gpu_timer.h"

/*
 * Collision Count procedure implemented in CUDA.
 *
 * This procedure parallelizes the sequential algorithm:
 * for i in 0:N-2
 *   for j in i+1:N-1
 *     collisions += (bead[i] == bead[j])
 * by performing just the outer 'for' in parallel.
 *
 * The outer 'for' has N-1 iterations, hence we allocate N-1 threads.
 * Assumptions:
 *   - All threads are in a single block
 *   - The number of threads allocated equals exactly N-1
 *   - N-1 is lower than the maximum number of threads per block
 *
 * Required Shared Memory: threadsAllocated integer elements.
 */
__global__
void count_collisions_cu(int3 *coords, int *result, int lower2Power){
	// We get our ID
	int tid = threadIdx.x;

	// The total number of elements in the vector
	// Since we allocated N-1 threads, blockDim.x+1 gives us N
	int N = blockDim.x + 1;


	// Count collisions
	int collisions = 0;
	for(int j = tid + 1; j < N; j++)
		collisions += (
				coords[tid].x == coords[j].x
				&& coords[tid].y == coords[j].y
				&& coords[tid].z == coords[j].z
			);


	// Fill shared memory
	extern __shared__ int sdata[];
	sdata[tid] = collisions;
	__syncthreads();

	// Apply one reduce iteration, to force the element pool to have a power of 2 size.
	if(tid < lower2Power && tid+lower2Power < blockDim.x)
		sdata[tid] += sdata[tid+lower2Power];
	__syncthreads();

	// Reduce
	for(int stride = lower2Power >> 1; stride > 0; stride >>= 1){
		if(tid < stride)
			sdata[tid] += sdata[tid+stride];

		__syncthreads();
	}

	// Export result
	if(tid == 0)
		*result = sdata[0];
}


int count_collisions(int3 *vector, int size){
	// We find the power of 2 immediately below 'size'
	int pow2 = 1;
	while(pow2 < size) pow2 <<= 1;
	pow2 >>= 1;

	int3 *d_vector;
	int *d_result;

	// Allocate cuda vector for the 3D coordinates
	cudaMalloc(&d_vector, sizeof(int3) * size);
	cudaMemcpy(d_vector, vector, sizeof(int3) * size, cudaMemcpyHostToDevice);

	// Allocate cuda memory for the number of collisions
	cudaMalloc(&d_result, sizeof(int));

	// Launch kernel
	int nThreads = size - 1;
	int nShMem = nThreads * sizeof(int);

	GpuTimer timer;
	timer.start();
	for(int i = 0; i < 1; i++)
		count_collisions_cu<<<1, nThreads, nShMem>>>(d_vector, d_result, pow2);
	timer.stop();
	printf("Elapsed: %lf\n", timer.elapsed());

	// Fetch result
	int result;
	cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

	return result;
}



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
	int coll = count_collisions(dummy, sizeof(dummy) / (3 * sizeof(int)));
	printf("Collisions: %d\n", coll);
	return 0;
}
