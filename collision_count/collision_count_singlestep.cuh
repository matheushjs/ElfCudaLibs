#ifndef COLLISION_COUNT_SINGLESTEP_H_
#define COLLISION_COUNT_SINGLESTEP_H_

#include <cuda.h>
#include <stdlib.h>
#include <time.h>

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
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	// Get our position in the imaginary matrix
	int vecId = (tid * (N + 1)) % (N*N);
	int row = vecId / N;
	int col = vecId % N;

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
			&& bead1.y == bead2.y
			&& bead1.z == bead2.z
		) * (tid < (N*N + N)/2); // 0 if thread is out of bounds

	// if(tid < 32) printf("(tid, bead1, bead2, collision, lim) == (%d, %d, %d, %d, %d)\n",
	//		tid, row, col+1, collision, (tid < (N*N + N)/2));
	sdata[threadIdx.x] = collision;
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


struct CollisionCountPromise {
	int3 *d_vector;
	int *d_result;
	int resultSize;
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

	// Allocate cuda vector for the 3D coordinates
	int3 *d_vector;
	cudaMalloc(&d_vector, sizeof(int3) * size);
	cudaMemcpyAsync(d_vector, vector, sizeof(int3) * size, cudaMemcpyHostToDevice, streams[launches%nStreams]);


	// Prepare to launch kernel
	// Get the imaginary matrix dimension
	int N = size - 1;
	// Count total number of threads
	int nThreads = (N*N + N) / 2; // Always divisible by 2.

	// Count number of threads per block and number of blocks
	int threadsPerBlock = 1024;
	int nBlocks = (nThreads + threadsPerBlock - 1) / threadsPerBlock; //Rounds up nThreads/thrPerBlock

	// Allocate cuda memory for the number of collisions.
	// Each block will write to an element, the final result (after reduce) will be in result[0].
	int *d_result;
	cudaMalloc(&d_result, sizeof(int) * nBlocks);

	// Calculate amount of shared memory
	int nShMem = threadsPerBlock * sizeof(int);

	// Finally launch kernel
	count_collisions_cu<<<nBlocks, threadsPerBlock, nShMem, streams[launches%nStreams]>>>(d_vector, d_result, nThreads, N);

	const struct CollisionCountPromise ret = { d_vector, d_result, nBlocks };
	return ret;
}

/* This procedure fetches the result from the call to the
 *   _launch correspondent.
 * The pointers within the promise structure are freed, so
 *   it shouldn't be used anywhere after a call to this function.
 */
int count_collisions_fetch(struct CollisionCountPromise promise){
	
	int result[promise.resultSize];
	cudaMemcpy(result, promise.d_result, sizeof(int) * promise.resultSize, cudaMemcpyDeviceToHost);

	cudaFree(&promise.d_result);
	cudaFree(&promise.d_vector);

	int i;
	for(i = 1; i < promise.resultSize; i++)
		result[0] += result[i];

	return result[0];
}

void test_count(int3 *vector, int size, int iters){
	struct CollisionCountPromise *promises;
	promises = (struct CollisionCountPromise *) malloc(sizeof(struct CollisionCountPromise) * iters);

	int beg = clock();

	int i;
	for(i = 0; i < iters; i++){
		promises[i] = count_collisions_launch(vector, size);
	}

	for(i = 0; i < iters; i++){
		int res = count_collisions_fetch(promises[i]);
	}

	printf("Elapsed: %lf ms\n", (clock() - beg) / (double) CLOCKS_PER_SEC * 1000);
}

#endif /* COLLISION_COUNT_SINGLESTEP_H_ */
