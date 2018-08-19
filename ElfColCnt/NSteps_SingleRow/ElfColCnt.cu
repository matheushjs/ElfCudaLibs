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
 */
__global__
void count_collisions_cu(int3 *coords, int *result, int nCoords){
	int baseIdx = blockIdx.x * 1024;
	int horizontalId = threadIdx.x + blockIdx.x * blockDim.x;

	// Calculate number of iterations to execute
	// If we have 2048 nCoords and baseIdx is 0, we must execute 2048 iterations.
	int maxIterations = nCoords - baseIdx;

	// We read our element in a register (surplus threads will read anything)
	int3 buf = coords[horizontalId % nCoords];
	
	// Read the first block into shared memory (surplus threads read anything)
	extern __shared__ int3 sCoords[];
	sCoords[threadIdx.x] = coords[ (baseIdx + threadIdx.x) % nCoords ];
	__syncthreads();

	// Count collisions on first block, which is a problematic block
	int offset = 0;
	int collisions = 0;
	int iterations = 0;
	int limit = min(1024, maxIterations);
	for(; iterations < limit; iterations++){
		// Check collision
		int collision = (
			buf.x   == sCoords[offset].x
			& buf.y == sCoords[offset].y
			& buf.z == sCoords[offset].z
		);

		// Assert our comparison element is after the base element in 'buf'
		collision *= (baseIdx + offset > horizontalId);

		// Sum on global collisions
		collisions += collision;

		offset++;
	}

	baseIdx += 1024;
	offset  = 0;

	// Now do the rest of the blocks
	while(iterations < maxIterations){
		// Read 2 blocks. Modulus prevents invallid memory accesses.
		__syncthreads();
		sCoords[threadIdx.x] = coords[ (baseIdx + threadIdx.x) % nCoords ];
		sCoords[threadIdx.x + 1024] = coords[ (baseIdx + threadIdx.x + 1024) % nCoords ];
		__syncthreads();

		// Do 2048 iterations, or maybe less
		limit = min(iterations + 2048, maxIterations);
		for(; iterations < limit; iterations++){
			// Check collision
			collisions += (
				buf.x   == sCoords[offset].x
				& buf.y == sCoords[offset].y
				& buf.z == sCoords[offset].z
			);

			offset++;
		}
		
		baseIdx += 2048;
		offset  = 0;
	}

	// Sync before reducing collisions on shared memory
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
