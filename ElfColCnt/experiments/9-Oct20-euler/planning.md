Experiment Planning
===

Binaries
---

### seq_quad [problemSize] [iterations]

### seq_lin [problemSize] [iterations]

The above count collisions in a vector with `problemSize` beads using the sequential approaches.

- Vector of beads generated in a protein-like manner

- The counting is performed `iterations` times

- For each of these iterations, the vector is re-generated

### cuda_n [problemSize] [iterations]

### cuda_half [problemSize] [iterations]

The above count collisions in a vector with `problemSize` beads using the CUDA approaches.

- Vector of beads generated randomly

- The counting is performed `iterations` times

- For each of these iterations, the same randomized vector is re-copied to the GPU device


Current Experiments
---

### Sequential Approaches

```python
for problemSize in [64, 128, ..., 8192]:
	./program problemSize 5 # warmup run
	./program problemSize 1000
```

### CUDA Approaches

```python
for problemSize in [1024, 66560, ..., 1049600]:
	./program problemSize 5 # warmup run
	./program problemSize 100
```
