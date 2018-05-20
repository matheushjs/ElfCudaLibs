Experiments
===


Experimental Conditions
---

- 10K iterations

- Vector of size 1000


N Steps
---

- 21423.849609 ms
	- **warp-efficient strides**
	- **use of shared memory**
	- **fitting the vector to a power of 2**
	- **launch & fetch separation**


- 6183.920410 ms
	- warp-efficient strides
	- use of shared memory
	- fitting the vector to a power of 2
	- launch & fetch separation
	- **usage of 8 cuda streams, with asynchronous memcpy and kernel launch**
	

Half Steps
---

- 4121.961426 ms
	- use of shared memory
	- fitting the vector to a power of 2
	- launch & fetch separation
	- usage of 8 cuda streams, with asynchronous memcpy and kernel launch

Single Steps
---

- 10023.894531 ms
	- use of shared memory
	- assumption that number of threads per block is a power of 2
	- launch & fetch separation
	- usage of 8 cuda streams, with asynchronous memcpy and kernel launch
	- usage of multiple blocks
	- reduce happening in the CPU

Sequential
---

- 25222.143000 ms

- 4600.959000 ms
	- **With O3**
	- **Volatile trick to prevent loop removal**

Sequential Linear
---

- 152.499000 ms 
	- Using size/2 due to memory allocation problem
	- Randomizing vector to prevent cache effects

- 90.898000 ms
	- Using size/2 due to memory allocation problem
	- Randomizing vector to prevent cache effects
	- **With O3**

