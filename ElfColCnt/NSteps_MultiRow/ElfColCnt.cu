#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

#include "ElfColCnt.cuh"
#include "utils.h"

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
 * Assumptions:
 *   - The grid of blocks has dimension HxW (height x width)
 *   - Horizontally, 'coords' is split  among each each block (not necessarily evenly), meaning each
 *       block is in charge of some number N of elements.
 *   - The amount of shared memory required is:  N * sizeof(int3)
 */
__global__
void count_collisions_cu(int3 *coords, int *result, int nCoords, int lower2Power){
	/* Get the most important parameters for deciding what to compute */
	int threadBaseIdx = blockIdx.x * blockDim.x;              // Id of the first thread in our block
	int horizontalId = blockIdx.x * blockDim.x + threadIdx.x; // Our horizontal ID
	int blockBaseIdx = blockIdx.y * 2048; // Index in 'coords' where our datablock begins
	int blockBaseEndx = min(blockBaseIdx + 2048, nCoords);   // Index in 'coords' where our datablock ends (exlusive)

	// Get rid of unused blocks
	if(threadBaseIdx >= blockBaseEndx) return;

	// We read our element in a register
	int3 buf = coords[horizontalId % nCoords];
	
	// Read the 2 blocks into shared memory (surplus threads read anything)
	extern __shared__ int3 sCoords[];
	sCoords[threadIdx.x] = coords[(blockBaseIdx + threadIdx.x) % nCoords];
	sCoords[threadIdx.x + 1024] = coords[(blockBaseIdx + threadIdx.x + 1024) % nCoords];
	__syncthreads();

	/*
	if(horizontalId == 0 && blockIdx.y == 1){
		for(int i = 0; i < 2048; i++){
			printf("%d ", sCoords[i].z);
		}
		printf("\n");
	}
	__syncthreads();
	*/

	// Count collisions
	int collisions = 0;
	// Get index in 'coords' and 'sCoords' of the element we are processing
	int offset;
	if(threadBaseIdx == blockBaseIdx) {
		offset = threadIdx.x + 1;
	} else if(threadBaseIdx == blockBaseIdx + 1024){
		offset = threadIdx.x + 1025;
	} else /* if(threadBaseIdx <= blockBaseIdx + 2048) */{
		offset = 0;
	}

/*
	if(threadIdx.x == 0){
		printf("Block: (%d, %d), end: %d\n", blockIdx.x, blockIdx.y, blockBaseEndx);
	}
	__syncthreads();
*/

	int elementInScoords = offset;
	int elementInCoords  = blockBaseIdx + offset;
	while(elementInCoords < blockBaseEndx){
		collisions += (
			buf.x == sCoords[elementInScoords].x
			& buf.y == sCoords[elementInScoords].y
			& buf.z == sCoords[elementInScoords].z
		);
		elementInScoords++;
		elementInCoords++;
	}
	__syncthreads();

	// Fill shared memory with collisions (surplus threads are ignored)
	extern __shared__ int sdata[];
	sdata[threadIdx.x] = collisions * (horizontalId < nCoords);
	__syncthreads();

	// Reduce 1024 elements
	for(int stride = 512; stride > 0; stride >>= 1){
		if(threadIdx.x < stride)
			sdata[threadIdx.x] += sdata[threadIdx.x+stride];

		__syncthreads();
	}
	
	// Export result
	if(threadIdx.x == 0){
		result[blockIdx.x + blockIdx.y * gridDim.x] = sdata[0];
	}
}

/* Gets the next cuda stream in the circular list of streams.
 */
static
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

	// Prepare kernel launch parameters
	const int elemInShmem = 2048; // 2048 allows 2 blocks to use the whole shared memory available.
	dim3 dimBlock(1024, 1); // We allocate maximum number of threads per block.
	dim3 dimGrid(
			divisionCeil(size, dimBlock.x), // Width depends on size / threadsPerBlock
			divisionCeil(size, elemInShmem) // Height depends on size / elementsInShmemPerBlock
		);
	int nShMem = elemInShmem * sizeof(int3); // Shared memory required

	// Allocate cuda memory for the number of collisions
	// This will also be used as a working vector for reducing among blocks
	int resultSize = higherEqualPow2(dimGrid.x * dimGrid.y);
	cudaMalloc(&d_result, sizeof(int) * resultSize);
	cudaMemsetAsync(d_result, 0, sizeof(int) * resultSize, stream); // Reset is needed due to size overestimation

	// Allocate cuda vector for the 3D coordinates.
	// Maximum because we'll need extra space for reduction later. This is because there are too many blocks.
	int vectorBytes = max(sizeof(int3) * size, (sizeof(int) * resultSize)/1024);
	cudaMalloc(&d_vector, vectorBytes);
	cudaMemcpyAsync(d_vector, vector, vectorBytes, cudaMemcpyHostToDevice, stream);

	// We find the power of 2 immediately below 'nThreads'
	// We calculate this here to avoid calculating it into the GPU
	int pow2 = lowerStrictPow2(dimBlock.x);

	// Finally launch kernels
	count_collisions_cu<<<dimGrid, dimBlock, nShMem, stream>>>(d_vector, d_result, size, pow2);
	
/*
	int res[resultSize];
	cudaMemcpy(res, d_result, sizeof(int) * resultSize, cudaMemcpyDeviceToHost);
	for(int i = 0; i < resultSize; i++) printf("%d ", res[i]);
	printf("\n\n");
*/

	// Reduce the result vector
	int workSize = resultSize;
	int nBlocks = resultSize/1024;
	int *d_toReduce = d_result;
	int *d_reduced  = (int *) d_vector; // must have size at least 'nBlocks'
	while(true){
		if(nBlocks == 0){
			reduce<<<1, workSize, sizeof(int) * workSize, stream>>>(d_toReduce, d_reduced);
			break;
		}

		reduce<<<nBlocks, 1024, sizeof(int) * 1024, stream>>>(d_toReduce, d_reduced);

/*
		int res[nBlocks];
		cudaMemcpy(res, d_reduced, sizeof(int) * nBlocks, cudaMemcpyDeviceToHost);
		for(int i = 0; i < nBlocks; i++) printf("%d ", res[i]);
		printf("\n\n");
*/

		// For the next run, vectors should be swapped
		int *aux = d_reduced;
		d_reduced = d_toReduce;
		d_toReduce = aux;

		// For the next run, the workSize and nBlocks are lower
		workSize = nBlocks;
		nBlocks = workSize/1024;
	}

	const struct CollisionCountPromise ret = { d_toReduce, d_reduced };
	return ret;
}

/* This procedure fetches the result from the call to the
 *   _launch correspondent.
 * The pointers within the promise structure are freed, so
 *   it shouldn't be used anywhere after a call to this function.
 */
int count_collisions_fetch(struct CollisionCountPromise promise){
	/*
	int result;
	cudaMemcpy(&result, promise.d_result, sizeof(int), cudaMemcpyDeviceToHost);
	*/

	const int n = 1;
	int result[n];
	cudaMemcpy(&result, promise.d_reduced, sizeof(int) * n, cudaMemcpyDeviceToHost);

	cudaFree(promise.d_toReduce);
	cudaFree(promise.d_reduced);

	return result[0];
}

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
