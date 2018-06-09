#ifndef COLLISION_COUNT_NSTEPS_GRID_H_
#define COLLISION_COUNT_NSTEPS_GRID_H_

#include <cuda.h>
#include <stdlib.h>
#include <time.h>

/* Multi-block reduce.
 * Accepts only vectors that are power of 2.
 */
__global__
void reduce(int *vec, int *result){
	extern __shared__ int sdata[];

	int tid = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = vec[idx];
	__syncthreads();

	// Reduce
	for(int stride = blockDim.x >> 1; stride > 0; stride >>= 1){
		if(threadIdx.x < stride)
			sdata[threadIdx.x] += sdata[threadIdx.x+stride];

		__syncthreads();
	}

	result[blockIdx.x] = sdata[0];
}


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
		result[blockIdx.x + blockDim.y * gridDim.x] = sdata[0];

	//TODO: implement reduce in another function.
	//XXX: Must not use more shared memory!! On reduce, re-use the coords vector.
}


struct CollisionCountPromise {
	int3 *d_vector;
	int *d_result;
};


cudaStream_t get_next_stream(){
	const int nStreams = 8;
	static cudaStream_t streams[nStreams];
	static unsigned int launches = 0;

	// Allocate cuda streams in the first execution
	static int streamInit = 0;
	if(streamInit == 0){
		streamInit = 1;
		for(int i = 0; i < nStreams; i++){
			cudaStreamCreate(&streams[i]);
		}
	}

	launches++;
	return streams[launches%nStreams];
}

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
	int3 *d_vector;
	int *d_result;
	cudaStream_t stream = get_next_stream();

	// Allocate cuda vector for the 3D coordinates
	cudaMalloc(&d_vector, sizeof(int3) * size);
	cudaMemcpyAsync(d_vector, vector, sizeof(int3) * size, cudaMemcpyHostToDevice, stream);

	// Prepare to launch kernel
	int elemInShmem = 2048;
	dim3 dimBlock(1024, 1);
	dim3 dimGrid(
			(size + dimBlock.x - 1) / dimBlock.x,
			(size + elemInShmem - 1) / elemInShmem
		);
	int nShMem = elemInShmem * sizeof(int) * 3;

	// Allocate cuda memory for the number of collisions
	// This will also be used as a working vector for reducing among blocks
	int resultSize, t;
	for(t = dimGrid.x * dimGrid.y, resultSize = 1; resultSize < t; resultSize <<= 1); // Find power of 2 immediately above what is needed
	cudaMalloc(&d_result, sizeof(int) * resultSize);
	cudaMemsetAsync(d_result, 0, sizeof(int) * resultSize, stream); // Reset is needed due to size overestimation

	// We find the power of 2 immediately below 'nThreads'
	// It is more efficient if its strictly below, and not equal 'nThreads'
	// We calculate this here to avoid calculating it into the GPU
	int pow2 = 1;
	while(pow2 < dimBlock.x) pow2 <<= 1;
	pow2 >>= 1;

	// Finally launch kernels
	count_collisions_cu<<<dimGrid, dimBlock, nShMem, stream>>>(d_vector, d_result, size, pow2);

	while(resultSize > 1024){
		int nBlocks = resultSize / 1024;

		printf("Reducing from %d to %d\n", resultSize, nBlocks);
		reduce<<<resultSize/1024, 1024, 1024*sizeof(int), stream>>>(d_result, (int *) d_vector);

		resultSize = nBlocks;
		int *aux = d_result;
		d_result = (int *) d_vector;
		d_vector = (int3 *) aux;
	}

	reduce<<<1, resultSize, resultSize*sizeof(int), stream>>>(d_result, (int *) d_vector);

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

#include <math.h>
void test_reduce(){
	/* We create a vector of size 2**X  */
	const int SIZE = 4096;
	int *vector = (int *) malloc(sizeof(int) * SIZE);
	int i;

	/* We fill it with 1..size */
	for(i = 0; i < SIZE; i++){
		vector[i] = i;
	}

	/* We reduce it sequentially */
	int gold = 0;
	for(i = 0; i < SIZE; i++)
		gold += vector[i];

	/* We send the vector to the cuda memory */
	int *d_vector;
	cudaMalloc(&d_vector, sizeof(int) * SIZE);
	cudaMemcpy(d_vector, vector, sizeof(int) * SIZE, cudaMemcpyHostToDevice);

	/* We create a vector for holding the result */
	int *d_result;
	cudaMalloc(&d_result, sizeof(int) * SIZE);
	cudaMemset(d_result, 0, sizeof(int) * SIZE);

	/* We reduce the vector in the GPU */
	int workSize, nBlocks;
	workSize = SIZE;
	nBlocks = workSize/1024;
	while(true){
		reduce<<<nBlocks, 1024, sizeof(int) * 1024>>>(d_vector, d_result);

		workSize = nBlocks;
		nBlocks = workSize/1024;

		int *aux = d_vector;
		d_vector = d_result;
		d_result = aux;

		if(nBlocks == 0){
			reduce<<<1, workSize, sizeof(int) * workSize>>>(d_vector, d_result);
			break;
		}
	}

	/* We get the result vector */
	cudaMemcpy(vector, d_result, sizeof(int) * SIZE, cudaMemcpyDeviceToHost);

	/* We print it */
	for(i = 0; i < SIZE; i++){
		printf("%5d ", vector[i]);
	}

	/* We check whether it was successful */
	if(gold != vector[0]){
		printf("Result is wrong, %d != %d!\n", gold, vector[0]);
	} else {
		printf("Result is correct! %d sum.\n", gold);
	}

	/* We free resources */
	cudaFree(d_result);
	cudaFree(d_vector);
	free(vector);
}

void test_count(int3 *vector, int size, int iters){
	test_reduce();
	return;

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
