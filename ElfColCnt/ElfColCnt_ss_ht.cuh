#ifndef COLLISION_COUNT_SINGLESTEP_H_
#define COLLISION_COUNT_SINGLESTEP_H_

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
 * Assumptions
 * - For now we assume threadsPerBlock is a power of 2.
 * 
 * Required shared memory: threadsPerBlock * sizeof(int)
 */
__global__
void count_collisions_cu(int3 *coords, int *result, int nCoords, int N){
	// Get our thread number
	unsigned long int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	// Get our position in the imaginary matrix
	unsigned long int vecId = (tid * (N + 1)) % (N*N);
	unsigned long int row = vecId / N;
	unsigned long int col = vecId % N;

	if(threadIdx.x == 0 && blockIdx.x == gridDim.x -1)
		printf("%ld %ld\n", vecId, N+1);

	// Transpose if we are on lower triangle
	if(row > col){
		row = col;
		col = vecId / N;
	}

	// if(tid < 32) printf("(tid, row, col) == (%d, %d, %d)\n", tid, row, col);

	// Calculate collision
	extern __shared__ int sdata[];
	
	int3 bead1 = coords[row];
	int3 bead2 = coords[col+1];

	int collision = (
			bead1.x == bead2.x
			& bead1.y == bead2.y
			& bead1.z == bead2.z
		);

	// if(tid < 32) printf("(tid, bead1, bead2, collision, lim) == (%d, %d, %d, %d, %d)\n",
	//		tid, row, col+1, collision, (tid < (N*N + N)/2));
	sdata[threadIdx.x] = collision
	                     * (tid < (N*N + N)/2); // thread out of bounds
	__syncthreads();
	
	// Reduce shared memory into the first element
	for(int stride = blockDim.x >> 1; stride > 0; stride >>= 1){
		if(threadIdx.x < stride)
			sdata[threadIdx.x] += sdata[threadIdx.x+stride];

		__syncthreads();
	}

	// Export result
	if(threadIdx.x == 0){
		result[blockIdx.x] = sdata[0];
	}
}

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

// Returns the first power of 2 that is >= 'base'.
static inline
int higherEqualPow2(int base){
	int result = 1;
	while(result < base) result <<= 1;
	return result;
}

/* Divides 'dividend' by 'divisor', rounding up.
 */
static inline
int divisionCeil(int dividend, int divisor){
	return (dividend + divisor - 1) / divisor;
}

struct CollisionCountPromise {
	int *d_toReduce;
	int *d_reduced;
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
	int3 *d_vector;
	int *d_result;
	cudaStream_t stream = get_next_stream();

	// Prepare to launch kernel
	// Get the imaginary matrix dimension
	int N = size - 1;
	// Count total number of threads
	int nThreads = (N*N + N) / 2; // Always divisible by 2.

	// Count number of threads per block and number of blocks
	int threadsPerBlock = 1024;
	int nBlocks = divisionCeil(nThreads, threadsPerBlock); //Rounds up nThreads/thrPerBlock

	// Allocate cuda memory for the number of collisions.
	// Each block will write to an element, the final result (after reduce) will be in result[0].
	int resultSize = higherEqualPow2(nBlocks);
	cudaMalloc(&d_result, sizeof(int) * resultSize);
	cudaMemsetAsync(d_result, 0, sizeof(int) * resultSize, stream); // Reset is needed due to size overestimation

	// Allocate cuda vector for the 3D coordinates.
	// Maximum because we'll need extra space for reduction later. This is because there are too many blocks.
	int vectorBytes = max(sizeof(int3) * size, (sizeof(int) * resultSize)/1024);
	cudaMalloc(&d_vector, vectorBytes);
	cudaMemcpyAsync(d_vector, vector, vectorBytes, cudaMemcpyHostToDevice, stream);

	// Calculate amount of shared memory
	int nShMem = threadsPerBlock * sizeof(int);

	// Finally launch kernel
	count_collisions_cu<<<nBlocks, threadsPerBlock, nShMem, stream>>>(d_vector, d_result, nThreads, N);

	// Reduce the result vector
	int workSize = resultSize;
	nBlocks = resultSize/1024;
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
	const int n = 1;
	int result[n];
	cudaMemcpy(&result, promise.d_reduced, sizeof(int) * n, cudaMemcpyDeviceToHost);

	cudaFree(promise.d_toReduce);
	cudaFree(promise.d_reduced);

	return result[0];
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
	printf("Collisions [SingleSteps HalfThreads]: %d\n", res);
}

#endif /* COLLISION_COUNT_SINGLESTEP_H_ */
