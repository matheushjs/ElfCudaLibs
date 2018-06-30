#ifndef COLLISION_COUNT_NSTEPS_GRID_SINGLEROW_H_
#define COLLISION_COUNT_NSTEPS_GRID_SINGLEROW_H_

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
 * TODO: Assumptions:
 *   - The grid of blocks has dimension HxW (height x width)
 *   - Horizontally, 'coords' is split  among each each block (not necessarily evenly), meaning each
 *       block is in charge of some number N of elements.
 *   - The amount of shared memory required is:  N * sizeof(int3)
 */
__global__
void count_collisions_cu(int3 *coords, int *result, int nCoords){
	int baseIdx = blockIdx.x * 1024;
	int maxIterations = nCoords - baseIdx;
	int horizontalId = threadIdx.x + blockIdx.x * blockDim.x;

	// We read our element in a register
	int3 buf = coords[horizontalId % nCoords];
	
	// Read the 2 blocks into shared memory (surplus threads read anything)
	extern __shared__ int3 sCoords[];
	sCoords[threadIdx.x] = coords[ (baseIdx + threadIdx.x) % nCoords ];
	sCoords[threadIdx.x + 1024] = coords[ (baseIdx + threadIdx.x + 1024) % nCoords ];
	__syncthreads();

	// Move our base index
	baseIdx = baseIdx + 2048; // We could use modulus here, but doesn't seem necessary

	/*
	if(horizontalId == 0){
		for(int i = 0; i < 2048; i++){
			printf("%d ", sCoords[i].z);
		}
	}
	__syncthreads();
	*/

	// Count collisions
	int iterations = 0;
	int offset = 1;
	int collisions = 0;
	while(iterations < maxIterations){
		
		// horizontalId + iterations + 1 is the element we are comparing to
		if(horizontalId + iterations + 1 < nCoords){
			collisions += (
				buf.x == sCoords[threadIdx.x + offset].x
				& buf.y == sCoords[threadIdx.x + offset].y
				& buf.z == sCoords[threadIdx.x + offset].z
			);
		}
		offset++;
		iterations++;
		
		// Change blocks in shared memory when needed
		if(offset == 1025 && baseIdx < nCoords){
			// Unfortunately we need to synchronize threads here
			__syncthreads();
			
			// Rewrite older block with earlier block
			sCoords[threadIdx.x] = sCoords[threadIdx.x + 1024];
			// Read new block
			sCoords[threadIdx.x + 1024] = coords[ (baseIdx + threadIdx.x) % nCoords ];

			// We also have to sync here
			__syncthreads();
			
			// Move base index
			baseIdx += 1024;

			offset = 1;
		}

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
		result[blockIdx.x] = sdata[0];
	}
}

struct CollisionCountPromise {
	int *d_toReduce;
	int *d_reduced;
};

/* Gets the next cuda stream in the circular list of streams.
 */
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

/* Divides 'dividend' by 'divisor', rounding up.
 */
static inline
int divisionCeil(int dividend, int divisor){
	return (dividend + divisor - 1) / divisor;
}

// Returns the first power of 2 that is >= 'base'.
static inline
int higherEqualPow2(int base){
	int result = 1;
	while(result < base) result <<= 1;
	return result;
}

// Returns the last power of 2 that is < 'base'
static inline
int lowerStrictPow2(int base){
	int result = 1;
	while(result < base) result <<= 1; // Get a result such that result >= base
	return result >> 1; // Then divide the result by 2 so that result < base
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
	if(size%2 != 0){
		fprintf(stderr, "Error: Vector size must be even.\n");
		exit(1);
	}

	int3 *d_vector;
	int *d_result;
	cudaStream_t stream = get_next_stream();

	// Allocate cuda vector for the 3D coordinates
	cudaMalloc(&d_vector, sizeof(int3) * size);
	cudaMemcpyAsync(d_vector, vector, sizeof(int3) * size, cudaMemcpyHostToDevice, stream);

	// Prepare kernel launch parameters
	const int elemInShmem = 2048; // 2048 because we need 2 blocks of 1024 elements in shmem.
	int nThreads = 1024;          // We allocate maximum number of threads per block.
	int nBlocks = divisionCeil(size, nThreads);
	int nShMem = elemInShmem * sizeof(int3); // Shared memory required

	// Allocate cuda memory for the number of collisions
	// This will also be used as a working vector for reducing among blocks
	int resultSize = higherEqualPow2(nBlocks);
	cudaMalloc(&d_result, sizeof(int) * resultSize);
	cudaMemsetAsync(d_result, 0, sizeof(int) * resultSize, stream); // Reset is needed due to size overestimation

	// Finally launch kernels
	count_collisions_cu<<<nBlocks, nThreads, nShMem, stream>>>(d_vector, d_result, size);
	
/*
	int res[resultSize];
	cudaMemcpy(res, d_result, sizeof(int) * resultSize, cudaMemcpyDeviceToHost);
	for(int i = 0; i < resultSize; i++) printf("%d ", res[i]);
	printf("\n\n");
*/

	// Reduce the result vector
	nBlocks = resultSize/1024;
	int workSize = resultSize;
	int *d_toReduce = d_result;
	int *d_reduced  = (int *) d_vector;
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
	printf("Collisions [N Steps Grid 1Row]: %d\n", res);
}

#endif /* COLLISION_COUNT_NSTEPS_H_ */
