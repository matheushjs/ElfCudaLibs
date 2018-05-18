#ifndef COLLISION_COUNT_NSTEPS_H_
#define COLLISION_COUNT_NSTEPS_H_

#include <cuda.h>
#include <stdlib.h>

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
 * Required Shared Memory (in bytes): threadsAllocated * sizeof(integer)
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

	// Apply one reduce iteration that forces the working vector
	//   size to be a power of 2.
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


struct CollisionCountPromise {
	int3 *d_vector;
	int *d_result;
};

/* Given a vector with 3D coordinates of points in the space,
 *   this function calculates the number of collisions among
 *   points, using CUDA-enable GPU.
 *
 * This functions just launches the kernel, returning a
 *   structure that can later be used to fetch the result
 *   back from the device memory.
 */
struct CollisionCountPromise
count_collisions_launch(int3 *vector, int size){
	// Allocate cuda streams in the first execution
	const int nStreams = 8;
	static cudaStream_t streams[nStreams];
	static int streamInit = 0;
	if(streamInit == 0){
		streamInit = 1;
		for(int i = 0; i < nStreams; i++){
			cudaStreamCreate(&streams[i]);
		}
	}
	static unsigned int launches = 0;
	launches++;

	int3 *d_vector;
	int *d_result;

	// Allocate cuda vector for the 3D coordinates
	cudaMalloc(&d_vector, sizeof(int3) * size);
	cudaMemcpyAsync(d_vector, vector, sizeof(int3) * size, cudaMemcpyHostToDevice, streams[launches%nStreams]);

	// Allocate cuda memory for the number of collisions
	cudaMalloc(&d_result, sizeof(int));


	// Prepare to launch kernel
	int nThreads = size - 1;
	int nShMem = nThreads * sizeof(int);

	// We find the power of 2 immediately below 'nThreads'
	// It is more efficient if its strictly below, and not equal 'nThreads'
	// We calculate this here to avoid calculating it into the GPU
	int pow2 = 1;
	while(pow2 < nThreads) pow2 <<= 1;
	pow2 >>= 1;

	// Finally launch kernel
	count_collisions_cu<<<1, nThreads, nShMem, streams[launches%nStreams]>>>(d_vector, d_result, pow2);
	launches++;

	const struct CollisionCountPromise ret = { d_vector, d_result };
	return ret;
}

/* This procedure fetches the result from the call to the
 *   _launch correspondent.
 * The pointers within the promise structure are freed, so
 *   it shouldn't be used anywhere after a call to this function.
 */
int count_collisions_fetch(struct CollisionCountPromise promise){
	int result;
	cudaMemcpy(&result, promise.d_result, sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(&promise.d_result);
	cudaFree(&promise.d_vector);

	return result;
}

void test_count(int3 *vector, int size, int iters){
	struct CollisionCountPromise *promises;
	promises = (struct CollisionCountPromise *) malloc(sizeof(struct CollisionCountPromise) * iters);

	GpuTimer timer;
	timer.start();

	int i;
	for(i = 0; i < iters; i++){
		promises[i] = count_collisions_launch(vector, size);
	}

	for(i = 0; i < iters; i++){
		int res = count_collisions_fetch(promises[i]);
	}

	timer.stop();
	printf("Elapsed: %lf\n", timer.elapsed());
}

#endif /* COLLISION_COUNT_NSTEPS_H_ */
