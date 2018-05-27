Experiments
===


Experimental Conditions
---

- 10K iterations

- Vector of size 1000

```
 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "GeForce GTX 650"
  CUDA Driver Version / Runtime Version          9.0 / 8.0
  CUDA Capability Major/Minor version number:    3.0
  Total amount of global memory:                 978 MBytes (1026031616 bytes)
  ( 2) Multiprocessors, (192) CUDA Cores/MP:     384 CUDA Cores
  GPU Max Clock rate:                            1202 MHz (1.20 GHz)
  Memory Clock rate:                             2500 Mhz
  Memory Bus Width:                              128-bit
  L2 Cache Size:                                 262144 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(65536), 2D=(65536, 65536), 3D=(4096, 4096, 4096)
  Maximum Layered 1D Texture Size, (num) layers  1D=(16384), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(16384, 16384), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 1 copy engine(s)
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 2 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 9.0, CUDA Runtime Version = 8.0, NumDevs = 1, Device0 = GeForce GTX 650
Result = PASS
```


N Steps
---

- 21423.849609 ms
	- **warp-efficient strides**
	- **use of shared memory**
	- **fitting the vector to a power of 2**
	- **launch & fetch separation**
	- After Julio's modifications
		- NVCC from CUDA 8: 10164.731445ms
		- NVCC from CUDA 9: 8502.455078ms


- 6183.920410 ms
	- warp-efficient strides
	- use of shared memory
	- fitting the vector to a power of 2
	- launch & fetch separation
	- **usage of 8 cuda streams, with asynchronous memcpy and kernel launch**
	- After Julio's modifications
		- NVCC from CUDA 8: 4518.114000 ms
		- NVCC from CUDA 9: 3473.578000 ms
	- Single execution information (After Julio + CUDA 9):
		- Average kernel time (nvprof): 1.2817ms
		- Serialized total (10K iterations): 12.8164s
		- Average kernel time without reduce (fishy): 606ns

Half Steps
---

- 4121.961426 ms
	- use of shared memory
	- fitting the vector to a power of 2
	- launch & fetch separation
	- usage of 8 cuda streams, with asynchronous memcpy and kernel launch
	- After Julio's modifications
		- NVCC from CUDA 8: 4149.953000 ms
		- NVCC from CUDA 9: 4149.414000 ms
	- Single execution information (After Julio + CUDA 9):
		- Average kernel time (nvprof): 1.6148ms
		- Serialized total (10K iterations): 16.1466s

- 4143.978000 ms
	- Optimizations to reduce registers used. From 25 to 21.
	- Average kernel time (nvprof): 1.6144ms

- 4150.637000 ms
	- Optimization to reduce instructions.
	- Registers reduced from 21 to 20.
	- Average kernel time (nvprof): 1.6164ms
	- Average kernel time without reduce (fishy): 605ns

Single Steps
---

- 10023.894531 ms
	- use of shared memory
	- assumption that number of threads per block is a power of 2
	- launch & fetch separation
	- usage of 8 cuda streams, with asynchronous memcpy and kernel launch
	- usage of multiple blocks
	- reduce happening in the CPU
	- After Julio's modifications
		- NVCC from CUDA 8: 10127.355000 ms
		- NVCC from CUDA 9: 10112.933000 ms

- Single execution information (After Julio + CUDA 9):
	- Average kernel time (nvprof): 979.36us
	- Serialized total (10K iterations): 9.79394s

Sequential
---

- 25222.143000 ms

- 4600.959000 ms
	- **With O3**
	- **Volatile trick to prevent loop removal**

- 3816.593000 ms
	- **With O3 only for the collision count function**
	- **No more volatile trick**
	- **Beads positioned along the z-axis, sequentially in the interval [0, size/2], in a circular fashion.**
	- Average collision count time: 0.3816593us

- 1779.082000 ms
	- With O3 only for the collision count function
	- No more volatile trick
	- **Beads positioned randomly along all the space**
	- Average collision count time: 0.1779082us

Sequential Linear
---

- 152.499000 ms 
	- Using size/2 due to memory allocation problem
	- Randomizing vector to prevent cache effects

- 90.898000 ms
	- Using size/2 due to memory allocation problem
	- Randomizing vector to prevent cache effects
	- **With O3**

- 289.722000 ms
	- Randomizing vector to prevent cache effects
	- **Using full memory required**
	- **No O3**

- 150.919000 ms
	- Randomizing vector to prevent cache effects
	- Using full memory required
	- **With O3**
