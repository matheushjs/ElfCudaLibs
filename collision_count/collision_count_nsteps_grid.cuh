#ifndef COLLISION_COUNT_NSTEPS_GRID_H_
#define COLLISION_COUNT_NSTEPS_GRID_H_

#include <cuda.h>
#include <stdlib.h>
#include <time.h>

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
 * TODO Assumptions:
 *   - All threads are in a single block
 *   - The number of threads allocated equals exactly N-1
 *   - N-1 is lower than the maximum number of threads per block
 *
 * Required Shared Memory (in bytes): nCoords * sizeof(integer) * 3
 */
__global__
void count_collisions_cu(int3 *coords, int *result, int nCoords, int lower2Power){
	// We get our horizontal ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// We read our element in a register
	int3 buf;
	if(tid < nCoords)
		buf = coords[tid];

	// In the coords vector, how many elements are to be processed (rounds up)?
	int dataBlock = (nCoords + gridDim.y - 1) / gridDim.y;
	
	// And what are the begin/end indexes?
	int beg = blockIdx.y * dataBlock;
	int endx = beg + dataBlock; // x stands for 'excluded'
	if((tid + 1) > beg) beg = tid + 1;
	if(endx > nCoords) endx = nCoords;

	// Count collisions
	int collisions = 0;
	for(int j = beg; j < endx; j++){
		collisions += (
				buf.x == coords[j].x
				&& buf.y == coords[j].y
				&& buf.z == coords[j].z
			);
	}

	// Fill shared memory with collisions
	extern __shared__ int sdata[];
	sdata[threadIdx.x] = collisions;
	__syncthreads();

	// Apply one reduce iteration that forces the working vector
	//   size to be a power of 2.
	if(threadIdx.x < lower2Power && threadIdx.x+lower2Power < blockDim.x)
		sdata[threadIdx.x] += sdata[threadIdx.x+lower2Power];
	__syncthreads();

	// Reduce
	for(int stride = lower2Power >> 1; stride > 0; stride >>= 1){
		if(threadIdx.x < stride)
			sdata[threadIdx.x] += sdata[threadIdx.x+stride];

		__syncthreads();
	}

	// Export result
	if(threadIdx.x == 0)
		atomicAdd(result, sdata[0]);
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
	cudaMemsetAsync(d_result, 0, sizeof(int), streams[launches%nStreams]);

	// Prepare to launch kernel
	int elemInShmem = 1024; // TODO: Fine tune elements in sh mem
	dim3 dimBlock(1024, 1);
	dim3 dimGrid(
			(size + dimBlock.x - 1) / dimBlock.x,
			(size + elemInShmem - 1) / elemInShmem
		);
	int nShMem = elemInShmem * sizeof(int) * 3;

	if(launches == 1)
		printf("Grid: (%d, %d); Global mem: %lfMb; Shared mem: %lfKB\n",
				dimGrid.x, dimGrid.y,
				(sizeof(int3) * size + sizeof(int)) / (double) 1E6,
				nShMem / (double) 1E3);

	// We find the power of 2 immediately below 'nThreads'
	// It is more efficient if its strictly below, and not equal 'nThreads'
	// We calculate this here to avoid calculating it into the GPU
	int pow2 = 1;
	while(pow2 < dimBlock.x) pow2 <<= 1;
	pow2 >>= 1;

	// Finally launch kernel
	count_collisions_cu<<<dimGrid, dimBlock, nShMem, streams[launches%nStreams]>>>(d_vector, d_result, size, pow2);

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

	int beg = clock();

	int i, res;
	for(i = 0; i < iters; i++){
		promises[i] = count_collisions_launch(vector, size);
	}

	for(i = 0; i < iters; i++){
		res = count_collisions_fetch(promises[i]);
	}

	printf("Elapsed: %lf ms\n", (clock() - beg) / (double) CLOCKS_PER_SEC * 1000);
	printf("Collisions [N Steps Grid]: %d\n", res);
}

#endif /* COLLISION_COUNT_NSTEPS_H_ */
