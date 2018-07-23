Elf Collision Count
===

CUDA Procedures for collision counting on discrete tridimensional space.

Index
---

Introduction
---

In this directory, I offer a bunch of procedures that count the number of collisions within a vector of beads in the integer tridimensional space. Some of the procedures are sequential, and some were implemented in parallel using the CUDA programming model.

The most basic sequential code is as follows
```c
int collisions = 0;
for(i = 0; i < vecSize-1; i++){
	for(j = i+1; j < vecSize; j++){
		isEqual(vector[i], vector[j]){
			collisions += 1;
		}
	}
}
```
where `vector` is an array of beads in the tridimensional integer space; that is, each element in the vector is a structure containing 3 integers, one for each coordinate on each axis.

In the following sections, each paralellization is explained in further detail.


NSteps SingleRow
---

This was the first CUDA paralellization proposed, and it is derived directly from the sequential code presented earlier. The parallelization is achieved by simply implementing the outer `for` loop to be executed in parallel, which means we launch `N` threads, and each thread `i` evaluates the collision of the bead `vector[i]` with all beads `vector[j]` where `j > i`. After all threads finish, we apply a reduce operation to accumulate all collisions calculated by each thread individually.

To implement this in CUDA, the initial vector with `N` beads is virtually split into "segments" of 1024 beads each. Then we launch a block of 1024 threads that will take care of each segment, and within a block each thread takes care of only one bead. Thread `i` in block `j` takes care of bead `i` in segment `j`; this means this thread compares its bead B with all beads that come after B in the initial vector. This distribution can be better visualized in the figure below.

![]()

To use the GPU memory efficiently, each thread reads in a register the element it is in charge of. Each block of threads processes segments sequentially; block `i` has to process segments `i`, `i+1`, `i+2` and so on. For efficiency, we use **shared memory as a cache for segments**. Initially the block reads segments `i` and `i+1` in shared memory; when it finishes processing all elements in segment `i`, it reads segment `i+2` over the memory freed by dumping the segment `i`. This goes on until all segments are processed.

The first thread among all threads launched has to compare bead `0` with all beads following it, giving a total of `N-1` operations. Hence, the depth of this algorithm is `N-1`, which explains the first part of this approache's name *NSteps*. It is also called *SingleRow* because the grid of blocks of threads is 1-dimensional.



NSteps MultiRow
---


